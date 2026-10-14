import inspect
import json
import math
import shutil

import pytest
import torch
from fastapi.testclient import TestClient

from deep_hedging import core, train
from deep_hedging.api import PRICE_SLOTS, create_app, price_state
from deep_hedging.report import report


@pytest.fixture(scope="module")
def smoke(tmp_path_factory):
    out = tmp_path_factory.mktemp("out")
    train.sweep("smoke", out, workers=2)
    report(out / "smoke")
    return out / "smoke"


def test_sweep_writes_every_run_and_resumes(smoke, capsys):
    runs = sorted((smoke / "runs").glob("*.pt"))
    assert len(runs) == 12  # 2 derivatives x 2 costs x 3 methods x 1 seed
    for p in runs:
        ckpt = train.load(p)
        assert all(math.isfinite(x) for x in ckpt["history"])
        assert math.isfinite(ckpt["metrics"]["price"])
    (smoke / "runs" / "stray.pt.tmp").write_bytes(b"partial write from a killed run")
    capsys.readouterr()
    train.sweep("smoke", smoke.parent, workers=1)
    assert "smoke: 0 runs to do" in capsys.readouterr().out


def test_checkpoint_reproduces_its_recorded_price(smoke):
    ckpt = train.load(next((smoke / "runs").glob("lookback-ntb-*.pt")))
    c = ckpt["config"]
    again = train.evaluate(
        train.policy(ckpt),
        c["derivative"],
        c["cost"],
        c["sigma"],
        c["days"],
        c["seed"],
        c["eval_batches"],
        c["eval_paths"],
    )
    assert again["price"] == pytest.approx(ckpt["metrics"]["price"], abs=1e-9)


def test_short_training_lowers_the_loss():
    h = train.run("european", "ntb", 1e-3, seed=0, iters=30, paths=5_000, eval_batches=2, eval_paths=1_000)["history"]
    assert sum(h[-5:]) / 5 < sum(h[:5]) / 5


def test_report_writes_tables_figures_and_models(smoke):
    rep = smoke / "report"
    for name in (
        "fig2_european.png",
        "fig5_lookback.png",
        "fig3_band.png",
        "fig4_european_history.png",
        "fig6_lookback_history.png",
    ):
        assert (rep / name).stat().st_size > 10_000, name
    tables = (rep / "tables.md").read_text()
    assert "| 0.000553 |" in tables and "paper NTB" in tables
    assert len((rep / "results.csv").read_text().strip().splitlines()) == 13
    assert len(list((smoke / "models").glob("*.pt"))) == 4  # NTB per derivative x cost


def test_report_removes_stale_model_picks(smoke):
    stale = smoke / "models" / "stale.pt"
    stale.write_bytes(b"left over from an earlier report")
    report(smoke)
    assert not stale.exists()


def test_report_serves_the_lowest_price_seed_and_shows_the_seed_sd(tmp_path):
    base = train.run("european", "ntb", 0.0005, seed=0, iters=1, paths=100, eval_batches=2, eval_paths=100)
    for seed, price in ((0, 0.0230), (1, 0.0220)):
        ckpt = {**base, "config": {**base["config"], "seed": seed}, "metrics": {**base["metrics"], "price": price}}
        train.save(ckpt, tmp_path / "runs" / f"{train.run_name(**ckpt['config'])}.pt")
    report(tmp_path)
    best = train.run_name(**{**base["config"], "seed": 1}) + ".pt"
    assert [p.name for p in (tmp_path / "models").glob("*.pt")] == [best]
    assert "| 0.000500 | 0.022500 ± 0.000707 |" in (tmp_path / "report" / "tables.md").read_text()


EURO = {
    "derivative": "european",
    "spot": 100.0,
    "strike": 100.0,
    "days_to_maturity": 20,
    "volatility": 0.2,
    "cost": 0.0005,
}


@pytest.fixture(scope="module")
def client(smoke):
    return TestClient(create_app(smoke / "models"))


def test_health_models_and_docs(client):
    assert client.get("/healthz").json() == {"status": "ok", "models": 4}
    assert {m["derivative"] for m in client.get("/v1/models").json()} == {"european", "lookback"}
    assert client.get("/", follow_redirects=False).headers["location"] == "/docs"


def test_hedge_stays_in_band_and_trades_exactly_zero_inside(client):
    r = client.post("/v1/hedge", json=EURO).json()
    lo, hi = r["band"]["lower"], r["band"]["upper"]
    assert lo <= r["next_hedge"] <= hi and r["model"]["preset"] == "smoke"
    inside = client.post("/v1/hedge", json={**EURO, "current_hedge": (lo + hi) / 2}).json()
    assert inside["trade"] == 0.0


def test_price_is_finite_with_a_standard_error(client):
    r = client.post("/v1/price", json=EURO).json()
    assert r["price"] > 0 and 0 < r["std_error"] < r["price"] and r["paths"] == 20_000


def test_control_variate_agrees_with_the_plain_estimator(smoke):
    ckpt = train.load(next((smoke / "models").glob("lookback-*-c5.5308e-04-*.pt")))
    net, s0 = train.policy(ckpt), -math.log(1.03)
    cv, cv_se = price_state("lookback", net, s0, s0, 28, ckpt["config"]["cost"], 0.0, 0.2, 200_000)
    paths = core.simulate(200_000, 28, 0.2, s0, torch.Generator().manual_seed(99))
    with torch.no_grad():
        x = torch.exp(-core.pnl(net, "lookback", paths, ckpt["config"]["cost"], 0.2).double())
    plain, plain_se = math.log(x.mean()), (x.std() / (math.sqrt(len(x)) * x.mean())).item()
    assert abs(cv - plain) <= 3 * math.hypot(cv_se, plain_se)


@pytest.mark.parametrize(
    "change,status",
    [
        ({"spot": -1.0}, 422),
        ({"days_to_maturity": True}, 422),
        ({"days_to_maturity": "15"}, 422),
        ({"days_to_maturity": 0}, 422),
        ({"days_to_maturity": 29}, 422),
        ({"cost": 0.5}, 422),
        ({"spot": 130.0}, 422),
        ({"max_spot": 150.0}, 422),
        ({"current_hedge": 11.0}, 422),
        ({"unknown": 1}, 422),
        ({"volatility": 0.3}, 404),
        ({"derivative": "lookback", "strike": 103.0, "max_spot": 99.0}, 422),
    ],
)
def test_bad_requests_are_rejected(client, change, status):
    assert client.post("/v1/hedge", json={**EURO, **change}).status_code == status


@pytest.mark.parametrize("raw", ["NaN", "Infinity"])
def test_non_finite_numbers_are_422_not_500(client, raw):
    body = json.dumps(EURO).replace('"spot": 100.0', f'"spot": {raw}')
    assert client.post("/v1/hedge", content=body, headers={"content-type": "application/json"}).status_code == 422


@pytest.mark.parametrize(
    "body",
    [
        {**EURO, "days_to_maturity": 1},  # gamma blows up at the money near maturity
        {**EURO, "derivative": "lookback", "spot": 95.0, "strike": 103.0, "max_spot": 104.0, "days_to_maturity": 10},
    ],
)
def test_edge_states_give_finite_answers(client, body):
    h = client.post("/v1/hedge", json=body).json()
    p = client.post("/v1/price", json=body).json()
    assert all(math.isfinite(v) for v in (h["band"]["lower"], h["band"]["upper"], p["price"], p["std_error"]))


def test_only_healthz_runs_on_the_event_loop(client):
    endpoints = {r.path: r.endpoint for r in client.app.routes if hasattr(r, "endpoint")}
    assert inspect.iscoroutinefunction(endpoints["/healthz"])
    assert not inspect.iscoroutinefunction(endpoints["/v1/price"])
    assert not inspect.iscoroutinefunction(endpoints["/v1/hedge"])


def test_startup_refuses_empty_or_mixed_horizon_model_dirs(smoke, tmp_path):
    with pytest.raises(SystemExit, match="no NTB checkpoints"):
        create_app(tmp_path)
    for f in (smoke / "models").glob("*.pt"):
        shutil.copy(f, tmp_path)
    other = train.run("european", "ntb", 0.0005, seed=0, iters=1, paths=100, eval_batches=2, eval_paths=100, days=10)
    train.save(other, tmp_path / "custom.pt")
    with pytest.raises(SystemExit, match="several horizons"):
        create_app(tmp_path)


def test_price_sheds_load_with_503_when_its_queue_is_full(client):
    slots = client.app.state.price_slots
    for _ in range(PRICE_SLOTS):
        assert slots.acquire(blocking=False)
    try:
        busy = client.post("/v1/price", json=EURO)
        assert busy.status_code == 503 and busy.headers["retry-after"] == "1"
        assert client.post("/v1/hedge", json=EURO).status_code == 200  # the fast path stays available
    finally:
        for _ in range(PRICE_SLOTS):
            slots.release()
    assert client.post("/v1/price", json=EURO).status_code == 200
