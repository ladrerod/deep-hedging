"""Command line: deep-hedging train | sweep | report | serve."""

import argparse
import os
import signal
from pathlib import Path


def main(argv=None):
    out = os.environ.get("DEEP_HEDGING_OUT", "out")
    p = argparse.ArgumentParser(prog="deep-hedging", description="No-transaction band network: train, reproduce, serve")
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train", help="train (or evaluate, for ww) one model into OUT/custom/runs")
    t.add_argument("--derivative", choices=["european", "lookback"], required=True)
    t.add_argument("--cost", type=float, required=True)
    t.add_argument("--method", choices=["ntb", "ffn", "ww"], default="ntb")
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--iters", type=int, default=1000)
    t.add_argument("--paths", type=int, default=50_000)
    t.add_argument("--days", type=int, default=28)
    t.add_argument("--sigma", type=float, default=0.2)
    t.add_argument("--out", default=out)

    s = sub.add_parser("sweep", help="run a preset into OUT/<preset> (resumable), then report")
    s.add_argument("--preset", choices=["smoke", "quick", "paper"], default="quick")
    s.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    s.add_argument("--out", default=out)

    r = sub.add_parser("report", help="re-render DIR/report and DIR/models from DIR/runs")
    r.add_argument("dir")

    v = sub.add_parser("serve", help="start the HTTP API")
    v.add_argument("--models", default=os.environ.get("DEEP_HEDGING_MODELS", "models"))
    v.add_argument("--host", default=os.environ.get("DEEP_HEDGING_HOST", "127.0.0.1"))
    v.add_argument("--port", type=int, default=8000)

    a = p.parse_args(argv)
    if a.cmd == "serve":
        import uvicorn

        from deep_hedging.api import create_app

        uvicorn.run(create_app(a.models), host=a.host, port=a.port)
        return

    signal.signal(signal.SIGTERM, lambda *_: os._exit(143))  # pod deletion / docker stop: runs are atomic
    from deep_hedging import train
    from deep_hedging.report import report

    if a.cmd == "train":
        ckpt = train.run(a.derivative, a.method, a.cost, a.seed, a.iters, a.paths, 20, a.paths, a.days, a.sigma)
        path = Path(a.out) / "custom" / "runs" / f"{train.run_name(**ckpt['config'])}.pt"
        train.save(ckpt, path)
        m = ckpt["metrics"]
        print(f"{path}: price {m['price']:.6f} ± {m['price_se']:.6f}, utility {m['utility']:.6f}, {m['seconds']:.0f}s")
    elif a.cmd == "sweep":
        train.sweep(a.preset, a.out, a.workers)
        report(Path(a.out) / a.preset)
    else:
        report(a.dir)
