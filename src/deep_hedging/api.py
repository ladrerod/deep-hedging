"""HTTP API: the no-transaction band, the next hedge and the indifference price from trained NTB checkpoints."""

import itertools
import math
import threading
from pathlib import Path
from typing import Literal

import torch
from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, RedirectResponse
from pydantic import BaseModel, ConfigDict, Field, model_validator

from deep_hedging.core import DT, STRIKE, greeks, pnl, simulate
from deep_hedging.train import load, policy

PRICE_SEED = 1234  # fixed, so the same request always returns the same price
PRICE_SLOTS = 8  # ponytail: at most 8 pricing requests wait or run per process, the rest get 503; scale with replicas


class Option(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)

    derivative: Literal["european", "lookback"]
    spot: float = Field(gt=0)
    strike: float = Field(gt=0)
    days_to_maturity: int = Field(ge=1)
    volatility: float = Field(gt=0)
    cost: float = Field(ge=0, lt=1)
    current_hedge: float = Field(default=0.0, ge=-10, le=10)
    max_spot: float | None = Field(default=None, gt=0, description="lookback only: running maximum including today")

    @model_validator(mode="after")
    def _check_max_spot(self):
        if self.max_spot is not None and self.derivative == "european":
            raise ValueError("max_spot applies to lookback options only")
        if self.max_spot is not None and self.max_spot < self.spot:
            raise ValueError("max_spot must be >= spot (the running maximum includes today)")
        return self


def load_models(models_dir):
    """{(derivative, sigma): {"days": int, "costs": {cost: (checkpoint, policy)}}} from NTB checkpoints."""
    best = {}
    for f in sorted(Path(models_dir).glob("*.pt")):
        ckpt = load(f)
        c = ckpt["config"]
        key = (c["derivative"], c["sigma"], c["days"], c["cost"])
        if c["method"] == "ntb" and (key not in best or ckpt["metrics"]["price"] < best[key]["metrics"]["price"]):
            best[key] = ckpt
    if not best:
        raise SystemExit(f"no NTB checkpoints in {models_dir}; train some with: deep-hedging sweep --preset quick")
    groups = {}
    for (d, sigma, days, cost), ckpt in best.items():
        group = groups.setdefault((d, sigma), {"days": days, "costs": {}})
        if group["days"] != days:
            raise SystemExit(f"{models_dir}: {d} models with sigma={sigma} have several horizons; keep one")
        group["costs"][cost] = (ckpt, policy(ckpt))
    return groups


def select(models, opt):
    """Pick the model for a request: same derivative and volatility, nearest trained cost on a log scale."""
    group = next(
        (
            g
            for (d, sig), g in models.items()
            if d == opt.derivative and math.isclose(sig, opt.volatility, rel_tol=1e-9)
        ),
        None,
    )
    if group is None:
        vols = sorted(sig for d, sig in models if d == opt.derivative)
        raise HTTPException(
            404,
            f"no {opt.derivative} model for volatility {opt.volatility}; available: {vols}. "
            f"Train one: deep-hedging train --derivative {opt.derivative} --sigma "
            f"{opt.volatility} --cost <c>",
        )
    if opt.days_to_maturity > group["days"]:
        raise HTTPException(422, f"days_to_maturity must be <= {group['days']}, the model's horizon")
    costs = sorted(group["costs"])
    if opt.cost == 0:
        if 0.0 not in group["costs"]:
            raise HTTPException(422, f"no zero-cost model; trained costs: {costs}")
        return group, group["costs"][0.0]
    positive = [c for c in costs if c > 0]
    logs = [math.log(c) for c in positive]
    tol = max([0.125] + [(b - a) / 2 for a, b in itertools.pairwise(logs)])
    nearest = min(positive, key=lambda c: abs(math.log(c / opt.cost)), default=None)
    if nearest is None or abs(math.log(nearest / opt.cost)) > tol:
        raise HTTPException(
            422, f"cost {opt.cost} is not within {tol:.3f} (log scale) of a trained cost; trained costs: {costs}"
        )
    return group, group["costs"][nearest]


def state(days, opt):
    """Request -> model features (s, m), rejecting states outside what training simulated."""
    s0 = -math.log(STRIKE[opt.derivative])  # training start: spot 1, strike K_train
    s = math.log(opt.spot) - math.log(opt.strike)
    m = max(math.log(opt.max_spot or opt.spot) - math.log(opt.strike), s0) if opt.derivative == "lookback" else s
    bound = 4 * opt.volatility * math.sqrt((days - opt.days_to_maturity + 1) * DT)
    if abs(s - s0) > bound or m - s0 > bound:
        lo, hi = opt.strike * math.exp(s0 - bound), opt.strike * math.exp(s0 + bound)
        what = "spot and max_spot" if opt.derivative == "lookback" else "spot"
        raise HTTPException(
            422,
            f"{what} must lie in [{lo:.4f}, {hi:.4f}] for strike {opt.strike} with "
            f"{opt.days_to_maturity} days left (the range the model was trained on)",
        )
    return s, m


def control(derivative, paths, m0, sigma):
    """sum_i delta_i (S_{i+1} - S_i): exactly zero-mean on the simulated martingale, correlated with the payoff."""
    n = paths.shape[0] - 1
    m = torch.clamp(torch.cummax(paths, 0).values, min=float(m0))
    tau = (n - torch.arange(n)) * DT
    delta = greeks(derivative, paths[:-1], m[:-1], tau[:, None], sigma)[1]
    return (delta * torch.diff(STRIKE[derivative] * torch.exp(paths), dim=0)).sum(0)


def price_state(derivative, net, s, m, days, cost, prev, sigma, n_paths, seed=PRICE_SEED):
    """Entropic price (model units) of the remaining hedged liability, with a control variate. Returns (price, se)."""
    paths = simulate(n_paths, days, sigma, s, torch.Generator().manual_seed(seed))
    with torch.no_grad():
        x = torch.exp(-pnl(net, derivative, paths, cost, sigma, prev0=prev, m0=m).double())
        y = control(derivative, paths, m, sigma).double()
    var_y = y.var()
    beta = torch.cov(torch.stack([x, y]))[0, 1] / var_y if var_y > 1e-12 else 0.0
    z = x - beta * y
    mean = z.mean().item()
    return math.log(mean), z.std().item() / (math.sqrt(n_paths) * mean)


def info(ckpt):
    c = ckpt["config"]
    return {
        "derivative": c["derivative"],
        "cost": c["cost"],
        "volatility": c["sigma"],
        "days": c["days"],
        "trained_strike": c["strike"],
        "preset": c["preset"],
    }


def create_app(models_dir, price_paths=20_000):
    torch.set_num_threads(1)
    models = load_models(models_dir)
    lock = threading.Lock()
    app = FastAPI(title="deep-hedging", description="No-transaction band network (Imaki et al. 2021)")
    app.state.price_slots = threading.BoundedSemaphore(PRICE_SLOTS)  # lock waiters must not fill the threadpool

    @app.exception_handler(RequestValidationError)
    async def _invalid(request, exc):  # never echo the input: a bare NaN in it would turn the 422 into a 500
        return JSONResponse(
            status_code=422, content={"detail": [{"loc": list(e["loc"]), "msg": e["msg"]} for e in exc.errors()]}
        )

    @app.get("/", include_in_schema=False)
    async def root():
        return RedirectResponse("/docs")

    @app.get("/healthz")
    async def healthz():  # async: answered on the event loop even while pricing threads are busy
        return {"status": "ok", "models": sum(len(g["costs"]) for g in models.values())}

    @app.get("/v1/models")
    def list_models():
        return [
            {
                **info(ckpt),
                "price": ckpt["metrics"]["price"],
                "price_se": ckpt["metrics"]["price_se"],
                "utility": ckpt["metrics"]["utility"],
            }
            for g in models.values()
            for ckpt, _ in g["costs"].values()
        ]

    @app.post("/v1/hedge")
    def hedge(opt: Option):
        group, (ckpt, net) = select(models, opt)
        s, m = state(group["days"], opt)
        sigma, tau, t = ckpt["config"]["sigma"], torch.tensor(opt.days_to_maturity * DT), torch.tensor
        with torch.no_grad():
            delta = greeks(opt.derivative, t([s]), t([m]), tau, sigma)[1].item()
            lo, hi = (x.item() for x in net.band(t([s]), t([m]), tau, sigma))
        if lo > hi:
            lo = hi = (lo + hi) / 2
        nxt = min(max(opt.current_hedge, lo), hi)  # Python floats: inside the band the trade is exactly 0.0
        return {
            "bs_delta": delta,
            "band": {"lower": lo, "upper": hi},
            "next_hedge": nxt,
            "trade": nxt - opt.current_hedge,
            "model": info(ckpt),
        }

    @app.post("/v1/price")
    def price(opt: Option):
        group, (ckpt, net) = select(models, opt)
        s, m = state(group["days"], opt)
        if not app.state.price_slots.acquire(blocking=False):
            raise HTTPException(503, "pricing is busy; retry shortly", headers={"Retry-After": "1"})
        try:
            with lock:  # ponytail: one pricing per process bounds memory; scale with replicas
                p, se = price_state(
                    opt.derivative,
                    net,
                    s,
                    m,
                    opt.days_to_maturity,
                    opt.cost,
                    opt.current_hedge,
                    ckpt["config"]["sigma"],
                    price_paths,
                )
        finally:
            app.state.price_slots.release()
        scale = opt.strike / STRIKE[opt.derivative]
        return {"price": p * scale, "std_error": se * scale, "paths": price_paths, "model": info(ckpt)}

    return app
