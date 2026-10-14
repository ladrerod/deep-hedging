import math

import pytest

from deep_hedging import train


@pytest.fixture(scope="module")
def smoke(tmp_path_factory):
    out = tmp_path_factory.mktemp("out")
    train.sweep("smoke", out, workers=2)
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
