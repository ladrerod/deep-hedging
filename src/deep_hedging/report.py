"""Figures, tables (ours next to the paper's), results.csv and the servable model set from a run directory."""

import csv
import math
import shutil
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from deep_hedging.core import european, ww_band
from deep_hedging.train import COSTS, load, policy

# Paper Tables B2 (European) and B4 (lookback): price (mean, sd over 5 seeds) per cost, in COSTS order.
PAPER_PRICES = {
    "european": {
        "ntb": [
            (0.022101, 0.000004),
            (0.022160, 0.000005),
            (0.022173, 0.000006),
            (0.022189, 0.000007),
            (0.022208, 0.000007),
            (0.022230, 0.000008),
            (0.022257, 0.000009),
            (0.022288, 0.000010),
            (0.022321, 0.000011),
            (0.022362, 0.000012),
            (0.022408, 0.000015),
            (0.022457, 0.000019),
            (0.022512, 0.000020),
            (0.022571, 0.000022),
            (0.022662, 0.000034),
            (0.022675, 0.000032),
            (0.022675, 0.000032),
            (0.022675, 0.000032),
            (0.022675, 0.000032),
            (0.022675, 0.000032),
            (0.022675, 0.000032),
            (0.022675, 0.000032),
        ],
        "ffn": [
            (0.022101, 0.000004),
            (0.022174, 0.000005),
            (0.022189, 0.000007),
            (0.022210, 0.000008),
            (0.022234, 0.000009),
            (0.022263, 0.000007),
            (0.022296, 0.000004),
            (0.022340, 0.000009),
            (0.022386, 0.000007),
            (0.022444, 0.000006),
            (0.022491, 0.000011),
            (0.022543, 0.000019),
            (0.022618, 0.000012),
            (0.022694, 0.000039),
            (0.022785, 0.000039),
            (0.022891, 0.000061),
            (0.022976, 0.000048),
            (0.023066, 0.000068),
            (0.023229, 0.000152),
            (0.023404, 0.000115),
            (0.023688, 0.000102),
            (0.023872, 0.000237),
        ],
        "ww": [
            (0.022101, 0.000004),
            (0.022162, 0.000007),
            (0.022176, 0.000008),
            (0.022191, 0.000008),
            (0.022211, 0.000008),
            (0.022235, 0.000009),
            (0.022265, 0.000009),
            (0.022300, 0.000010),
            (0.022344, 0.000011),
            (0.022397, 0.000012),
            (0.022461, 0.000013),
            (0.022539, 0.000014),
            (0.022631, 0.000015),
            (0.022742, 0.000017),
            (0.022872, 0.000018),
            (0.023024, 0.000020),
            (0.023201, 0.000022),
            (0.023415, 0.000024),
            (0.023673, 0.000026),
            (0.023989, 0.000028),
            (0.024378, 0.000030),
            (0.024859, 0.000032),
        ],
    },
    "lookback": {
        "ntb": [
            (0.018040, 0.000005),
            (0.018115, 0.000007),
            (0.018128, 0.000009),
            (0.018145, 0.000011),
            (0.018165, 0.000013),
            (0.018187, 0.000016),
            (0.018215, 0.000021),
            (0.018240, 0.000032),
            (0.018265, 0.000038),
            (0.018294, 0.000049),
            (0.018328, 0.000063),
            (0.018373, 0.000087),
            (0.018416, 0.000097),
            (0.018406, 0.000031),
            (0.018482, 0.000147),
            (0.018523, 0.000228),
            (0.018587, 0.000354),
            (0.018660, 0.000500),
            (0.018721, 0.000621),
            (0.018800, 0.000780),
            (0.018915, 0.001010),
            (0.019124, 0.001427),
        ],
        "ffn": [
            (0.018035, 0.000005),
            (0.018139, 0.000008),
            (0.018156, 0.000011),
            (0.018173, 0.000012),
            (0.018198, 0.000011),
            (0.018224, 0.000014),
            (0.018258, 0.000018),
            (0.018298, 0.000019),
            (0.018341, 0.000028),
            (0.018393, 0.000031),
            (0.018446, 0.000038),
            (0.018525, 0.000034),
            (0.018604, 0.000021),
            (0.018715, 0.000042),
            (0.018834, 0.000056),
            (0.018967, 0.000046),
            (0.019210, 0.000076),
            (0.019529, 0.000176),
            (0.019853, 0.000144),
            (0.020239, 0.000244),
            (0.020766, 0.000267),
            (0.021625, 0.000334),
        ],
        "ww": [
            (0.018053, 0.000008),
            (0.018133, 0.000005),
            (0.018150, 0.000004),
            (0.018170, 0.000004),
            (0.018193, 0.000005),
            (0.018222, 0.000006),
            (0.018256, 0.000007),
            (0.018297, 0.000008),
            (0.018346, 0.000009),
            (0.018403, 0.000010),
            (0.018470, 0.000012),
            (0.018547, 0.000013),
            (0.018635, 0.000015),
            (0.018731, 0.000018),
            (0.018839, 0.000020),
            (0.018964, 0.000022),
            (0.019113, 0.000023),
            (0.019296, 0.000025),
            (0.019533, 0.000027),
            (0.019849, 0.000028),
            (0.020256, 0.000029),
            (0.020775, 0.000029),
        ],
    },
}

LABELS = {"ntb": "No-transaction band network", "ffn": "Feed-forward network", "ww": "Asymptotic (Whalley-Wilmott)"}
HISTORY_COSTS = (math.exp(-7.5), math.exp(-5))  # the paper's Figures 4 and 6


def _on_grid(cost, grid):
    return next((i for i, c in enumerate(grid) if math.isclose(c, cost, rel_tol=1e-9, abs_tol=1e-15)), None)


def report(run_dir):
    """Read run_dir/runs/*.pt; write run_dir/report/ (figures, tables.md, results.csv) and run_dir/models/."""
    run_dir = Path(run_dir)
    ckpts = [(f, load(f)) for f in sorted((run_dir / "runs").glob("*.pt"))]
    out = run_dir / "report"
    out.mkdir(parents=True, exist_ok=True)
    _results_csv(out / "results.csv", ckpts)
    stats = _stats([c for _, c in ckpts])
    tables = _tables(stats)
    (out / "tables.md").write_text(tables)
    for d, n in (("european", 2), ("lookback", 5)):
        _fig_cost_sweep(stats, d, out / f"fig{n}_{d}.png")
    _fig_band([c for _, c in ckpts], out / "fig3_band.png")
    for d, n in (("european", 4), ("lookback", 6)):
        _fig_history([c for _, c in ckpts], d, out / f"fig{n}_{d}_history.png")
    _select_models(ckpts, run_dir / "models")
    print(tables)


def _results_csv(path, ckpts):
    rows = [{"file": f.name, **c["config"], **c["metrics"]} for f, c in ckpts]
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]) if rows else ["file"])
        writer.writeheader()
        writer.writerows(rows)


def _stats(ckpts):
    """(derivative, method, cost) -> mean and error of price and utility; error = sd over seeds, or se for one seed."""
    groups = {}
    for c in ckpts:
        cfg = c["config"]
        groups.setdefault((cfg["derivative"], cfg["method"], cfg["cost"]), []).append(c["metrics"])

    def agg(ms, key):
        xs = [m[key] for m in ms]
        return statistics.mean(xs), statistics.stdev(xs) if len(xs) > 1 else ms[0][key + "_se"]

    return {k: {"price": agg(ms, "price"), "utility": agg(ms, "utility")} for k, ms in groups.items()}


def _tables(stats):
    fmt = "{0[0]:.6f} ± {0[1]:.6f}".format
    lines = []
    for d in ("european", "lookback"):
        costs = sorted({k[2] for k in stats if k[0] == d})
        if not costs:
            continue
        lines += [
            f"## {d}",
            "",
            "| cost | NTB | paper NTB | FFN | paper FFN | WW | paper asym. |",
            "|---|---|---|---|---|---|---|",
        ]
        for c in costs:
            i = _on_grid(c, COSTS)
            cells = [f"{c:.6f}"]
            for m in ("ntb", "ffn", "ww"):
                cells.append(fmt(stats[(d, m, c)]["price"]) if (d, m, c) in stats else "")
                cells.append(fmt(PAPER_PRICES[d][m][i]) if i is not None else "")
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")
    lines += [
        "Indifference prices log E[exp(-P)], in units of the strike (European) or of the initial spot (lookback).",
        "Ours: mean ± sd over seeds (± standard error with one seed). Paper: Tables B2 and B4, mean ± sd over 5 seeds.",
        (
            "Not comparable (see README, Deviations): the paper's lookback asymptotic column, and its European"
            " asymptotic column at c >= e^-6. The paper's Table B3 duplicates Table B1 and is not used."
        ),
        "",
    ]
    return "\n".join(lines)


def _fig_cost_sweep(stats, d, path):
    p0 = stats.get((d, "ww", 0.0))
    costs = sorted(c for (dd, m, c) in stats if dd == d and m == "ntb" and c > 0)
    if p0 is None or not costs:
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    for m in ("ntb", "ffn", "ww"):
        cs = [c for c in costs if (d, m, c) in stats]
        u = [stats[(d, m, c)]["utility"] for c in cs]
        ax1.errorbar(cs, [x[0] for x in u], yerr=[x[1] for x in u], label=LABELS[m], marker="o", ms=3, capsize=2)
        sp = [(c, stats[(d, m, c)]["price"][0] - p0["price"][0], stats[(d, m, c)]["price"][1]) for c in cs]
        sp = [x for x in sp if x[1] > 0]  # log axis
        ax2.errorbar([x[0] for x in sp], [x[1] for x in sp], yerr=[x[2] for x in sp], marker="o", ms=3, capsize=2)
    ax1.set(xscale="log", xlabel="Cost rate", ylabel="Utility", title=f"Utility, {d} option")
    ax2.set(
        xscale="log",
        yscale="log",
        xlabel="Cost rate",
        ylabel="Price spread p(c) - p(0)",
        title=f"Price spread, {d} option",
    )
    ax1.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _fig_band(ckpts, path):
    """European NTB band vs log-moneyness at 15 days, smallest and largest positive cost (paper Figure 3)."""
    best = {}
    for c in ckpts:
        cfg = c["config"]
        mine = cfg["derivative"] == "european" and cfg["method"] == "ntb" and cfg["cost"] > 0
        if mine and (cfg["cost"] not in best or c["metrics"]["price"] < best[cfg["cost"]]["metrics"]["price"]):
            best[cfg["cost"]] = c
    if not best:
        return
    picks = sorted({min(best), max(best)})
    fig, axes = plt.subplots(1, len(picks), figsize=(5.5 * len(picks), 4), squeeze=False)
    s, tau = torch.linspace(-0.1, 0.1, 201), torch.tensor(15 / 365)
    for ax, cost in zip(axes[0], picks):
        sigma = best[cost]["config"]["sigma"]
        with torch.no_grad():
            lo, hi = policy(best[cost]).band(s, s, tau, sigma)
            wlo, whi = ww_band("european", cost, s, s, tau, sigma)
            delta = european(s, tau, sigma)[1]
        ax.fill_between(s, torch.minimum(lo, hi), torch.maximum(lo, hi), alpha=0.3, label=LABELS["ntb"])
        ax.plot(s, delta, "k--", label="Black-Scholes delta")
        ax.plot(s, wlo, "C2", label=LABELS["ww"])
        ax.plot(s, whi, "C2")
        ax.set(xlabel="Log moneyness", ylabel="Hedge ratio", title=f"Band at 15 days to maturity, c = {cost:.1e}")
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _fig_history(ckpts, d, path):
    """Training loss (20-iteration rolling mean) at the paper's small and large cost, seed 0 (Figures 4 and 6)."""
    runs = {}
    for c in ckpts:
        cfg = c["config"]
        i = _on_grid(cfg["cost"], HISTORY_COSTS)
        if cfg["derivative"] == d and cfg["seed"] == 0 and i is not None:
            runs[(cfg["method"], i)] = c
    idx = sorted({i for (m, i) in runs if m == "ntb"})
    if not idx:
        return
    fig, axes = plt.subplots(1, len(idx), figsize=(5.5 * len(idx), 4), squeeze=False)
    for ax, i in zip(axes[0], idx):
        for m in ("ntb", "ffn"):
            if (m, i) in runs:
                h = runs[(m, i)]["history"]
                ax.plot([statistics.mean(h[max(0, j - 19) : j + 1]) for j in range(len(h))], label=LABELS[m])
        if ("ww", i) in runs:
            ax.axhline(-runs[("ww", i)]["metrics"]["utility"], color="C2", ls="--", label=LABELS["ww"])
        ax.set(xlabel="Training iteration", ylabel="Loss E[exp(-P)]", title=f"{d}, c = {HISTORY_COSTS[i]:.1e}")
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _select_models(ckpts, models_dir):
    """Copy the lowest-price NTB checkpoint per (derivative, sigma, days, cost); stale picks are removed first."""
    models_dir.mkdir(parents=True, exist_ok=True)
    for old in models_dir.glob("*.pt"):
        old.unlink()
    best = {}
    for f, c in ckpts:
        cfg = c["config"]
        key = (cfg["derivative"], cfg["sigma"], cfg["days"], cfg["cost"])
        if cfg["method"] == "ntb" and (key not in best or c["metrics"]["price"] < best[key][1]["metrics"]["price"]):
            best[key] = (f, c)
    for f, _ in best.values():
        shutil.copy2(f, models_dir / f.name)
