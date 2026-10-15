"""One training run, presets, checkpoints and the resumable parallel sweep."""

import math
import multiprocessing
import os
import statistics
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from datetime import UTC, datetime
from pathlib import Path

import torch

from deep_hedging import __version__
from deep_hedging.core import STRIKE, Hedger, pnl, price, simulate, utility, whalley_wilmott

LR = 1e-3
COSTS = [0.0] + [math.exp(-10 + 0.25 * i) for i in range(21)]
DERIVATIVES = ("european", "lookback")
METHODS = ("ntb", "ffn", "ww")
PRESETS = {
    "smoke": {
        "costs": [0.0, math.exp(-7.5)],
        "seeds": [0],
        "iters": 5,
        "paths": 1_000,
        "eval_batches": 2,
        "eval_paths": 1_000,
    },
    "quick": {
        "costs": [0.0] + [math.exp(k) for k in (-10, -8.75, -7.5, -6.25, -5)],
        "seeds": [0],
        "iters": 300,
        "paths": 10_000,
        "eval_batches": 20,
        "eval_paths": 10_000,
    },
    "paper": {
        "costs": COSTS,
        "seeds": [0, 1, 2, 3, 4],
        "iters": 1_000,
        "paths": 50_000,
        "eval_batches": 20,
        "eval_paths": 50_000,
    },
}


def evaluate(policy, derivative, cost, sigma, days, seed, batches, n_paths):
    """Paper protocol: mean and standard error over batches. Paths depend only on the seed (common random numbers)."""
    gen = torch.Generator().manual_seed(10_000 + seed)
    s0 = -math.log(STRIKE[derivative])
    prices, utils = [], []
    with torch.no_grad():
        for _ in range(batches):
            wealth = pnl(policy, derivative, simulate(n_paths, days, sigma, s0, gen), cost, sigma)
            prices.append(price(wealth))
            utils.append(utility(wealth))
    return {
        "price": statistics.mean(prices),
        "price_se": statistics.stdev(prices) / math.sqrt(batches),
        "utility": statistics.mean(utils),
        "utility_se": statistics.stdev(utils) / math.sqrt(batches),
    }


def run(derivative, method, cost, seed, iters, paths, eval_batches, eval_paths, days=28, sigma=0.2, preset="custom"):
    """Train (ntb, ffn) or just evaluate (ww) one strategy; returns a self-contained checkpoint dict."""
    start = time.time()
    torch.manual_seed(seed)
    history, state = [], {}
    if method == "ww":
        policy = whalley_wilmott(derivative, cost)
    else:
        policy = Hedger(derivative, method)
        opt = torch.optim.Adam(policy.parameters(), lr=LR)
        gen, s0 = torch.Generator().manual_seed(seed), -math.log(STRIKE[derivative])
        for _ in range(iters):
            loss = torch.exp(-pnl(policy, derivative, simulate(paths, days, sigma, s0, gen), cost, sigma)).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            history.append(loss.item())
        state = policy.state_dict()
    metrics = evaluate(policy, derivative, cost, sigma, days, seed, eval_batches, eval_paths)
    config = {
        "preset": preset,
        "derivative": derivative,
        "method": method,
        "cost": cost,
        "sigma": sigma,
        "days": days,
        "strike": STRIKE[derivative],
        "lr": LR,
        "iters": iters,
        "paths": paths,
        "eval_batches": eval_batches,
        "eval_paths": eval_paths,
        "seed": seed,
        "eval_seed": 10_000 + seed,
    }
    meta = {"version": __version__, "torch": str(torch.__version__), "created": datetime.now(UTC).isoformat()}
    return {
        "config": config,
        "metrics": {**metrics, "seconds": time.time() - start},
        "history": history,
        "state_dict": state,
        "meta": meta,
    }


def run_name(derivative, method, cost, seed, iters, paths, days=28, sigma=0.2, **_):
    return f"{derivative}-{method}-c{cost:.4e}-sig{sigma:g}-d{days}-i{iters}-p{paths}-s{seed}"


def save(ckpt, path):
    """Atomic write: a killed process leaves at most a *.pt.tmp, never a truncated *.pt."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".pt.tmp")
    torch.save(ckpt, tmp)
    os.replace(tmp, path)


def load(path):
    """Safe load: weights_only refuses pickled code."""
    return torch.load(path, weights_only=True)


def policy(ckpt):
    """Rebuild the hedging policy stored in a checkpoint."""
    c = ckpt["config"]
    if c["method"] == "ww":
        return whalley_wilmott(c["derivative"], c["cost"])
    net = Hedger(c["derivative"], c["method"])
    net.load_state_dict(ckpt["state_dict"])
    return net.eval()


def tasks(preset):
    """Seed-major order: an interrupted paper sweep already holds complete single-seed results."""
    p = PRESETS[preset]
    return [
        {
            "derivative": d,
            "method": m,
            "cost": c,
            "seed": s,
            "iters": p["iters"],
            "paths": p["paths"],
            "eval_batches": p["eval_batches"],
            "eval_paths": p["eval_paths"],
            "preset": preset,
        }
        for s in p["seeds"]
        for d in DERIVATIVES
        for c in p["costs"]
        for m in METHODS
    ]


def _work(task, path):
    torch.set_num_threads(1)
    save(run(**task), path)
    return path.name


def sweep(preset, out, workers):
    """Run every task of a preset into out/<preset>/runs, skipping finished runs (resumable)."""
    runs = Path(out) / preset / "runs"
    todo = [(t, runs / f"{run_name(**t)}.pt") for t in tasks(preset)]
    todo = [(t, p) for t, p in todo if not p.exists()]
    print(f"{preset}: {len(todo)} runs to do", flush=True)
    if workers == 1:
        for t, p in todo:
            print(_work(t, p), flush=True)
        return
    try:
        with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("spawn")) as pool:
            for f in as_completed([pool.submit(_work, t, p) for t, p in todo]):
                print(f.result(), flush=True)
    except BrokenProcessPool:
        raise SystemExit("a worker died (out of memory?); rerun with fewer --workers; finished runs are kept")
