# No-Transaction Band Deep Hedging Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a public repo that trains, reproduces and serves the paper's No-Transaction Band Network, using one CPU container image for Docker and Kubernetes.

**Architecture:**
- One small PyTorch package, `deep_hedging`:
  - `core` holds the math;
  - `train` runs training, checkpoints and the resumable sweep;
  - `report` writes figures and tables;
  - `api` is the FastAPI server;
  - `cli` is the command line.
- One image whose entrypoint is the CLI: `serve` by default, or `sweep` / `train` / `report`.
- Four Kubernetes objects in three plain YAML files, plus one Kustomize patch that switches the API to models trained onto the volume.
- GitHub Actions runs the tests, a kind end-to-end check, and a multi-arch publish on version tags.

**Tech Stack:** Python 3.12, uv, PyTorch 2.14 (CPU), FastAPI 0.143 + uvicorn, matplotlib, pytest + httpx2, ruff 0.17, Docker (buildx), Kubernetes with kubectl/Kustomize, GitHub Actions, GHCR.

**Spec:** `docs/superpowers/specs/2026-10-14-ntb-deep-hedging-design.md`. Executors read the spec next to this plan; section numbers below (§) refer to it.

**How this plan was made:**
- It follows ECC `/plan` (requirements, risks, complexity, validation) under ponytail's rules (the leanest code that fully works).
- Every code block below was run before the plan was written, in a scratch copy of the repo:
  - all 53 tests pass, and `ruff check` and `ruff format` are clean;
  - the image was built and served read-only as an arbitrary user;
  - the manifests passed server-side dry-runs on OrbStack;
  - the workflow passed actionlint;
  - the quick preset ran in 77 s.
- Expected outputs quoted in steps come from those runs.

## Requirements restatement (ECC)

1. **Train and reproduce.** Train the NTB, the feed-forward baseline and the Whalley–Wilmott band for a European and a lookback call across the paper's 22 costs. Regenerate the paper's figures and tables, with ours printed next to the paper's numbers. Paper conventions follow §2, including 28 daily steps and closed-form lookback greeks.
2. **Serve.** An HTTP API returns the band, the next hedge and the indifference price for a requested option and cost. It ships pre-trained models, validates strictly and rejects anything outside the trained domain.
3. **Run anywhere.**
   - Docker: `docker run` serves immediately, and `docker run … sweep` trains.
   - Kubernetes: `kubectl apply -f k8s/api.yaml` serves; a PVC plus Job trains; `kubectl apply -k k8s` serves what was trained.
4. **Public quality.**
   - CI covers tests and a real-cluster end-to-end run.
   - Tagged releases publish multi-arch images to GHCR.
   - The README states deviations from the paper and gives proper attribution.
   - MIT license.

## Global Constraints

Copied from the spec. Every task implicitly includes these:
- **Names:** repo `ladrerod/deep-hedging`, package `deep_hedging`, CLI `deep-hedging`, image `ghcr.io/ladrerod/deep-hedging`. The image tag equals the pyproject `version` with no `v` (`0.1.0`).
- **Python and dependencies:** Python `>=3.12`. Runtime dependencies are exactly `torch`, `fastapi`, `uvicorn`, `matplotlib`; dev dependencies are `pytest`, `ruff`, `httpx2`. On Linux, torch comes from `https://download.pytorch.org/whl/cpu`, so there are no CUDA wheels.
- **Market conventions:** S0 = 1, K = 1 (European) or 1.03 (lookback), σ = 0.2, λ = 1 on absolute P&L, `DT = 1/365`, **28** daily increments, `tau_i = (28 - i)/365`. Exact-martingale GBM.
- **Model and training:**
  - MLP 4 × 32 ReLU. NTB outputs `LeakyReLU(0.01)` offsets around the Black–Scholes delta, with midpoint clamping for inverted bands.
  - Adam with lr `1e-3` (a constant). Fresh paths every iteration.
  - Evaluation: 20 batches with `eval_seed = 10_000 + seed`, price reduced in float64.
- **Checkpoints:** one self-contained `.pt` per run, written atomically, read only with `torch.load(weights_only=True)`.
- **Image:** non-root uid/gid `10001`, read-only root filesystem plus a writable `/tmp`.
- **API:**
  - Strict pydantic validation (`extra="forbid", strict=True, allow_inf_nan=False`).
  - POST handlers are plain `def`; only `/healthz` is `async def`.
  - No authentication; the Service is ClusterIP only.
- **License:** standard MIT, © 2026 Daniel Ladrero Meca. No code is copied from pfnet-research's repos; they are credited as consulted.
- **Ponytail:**
  - No abstractions, config or files beyond what a task lists.
  - Deliberate ceilings carry a `# ponytail:` comment.
  - One focused test per piece of non-trivial logic.

## Review Focus

These inputs and failure modes are implied by the spec but easy to break. Each is pinned by a test in the task that owns the code:
1. **Near-maturity at-the-money requests** (`days_to_maturity: 1`, spot = strike), where gamma blows up. The API must return a finite band and price. Pinned by `test_edge_states_give_finite_answers` (Task 6).
2. **A lookback in deep drawdown** (spot well below `max_spot`, near the edge of the trained range) must give finite answers, not NaN. Pinned by the same test (Task 6).
3. **A sweep killed mid-run** leaves a `*.pt.tmp`. Resuming must redo that run, and the server must ignore the file. Pinned by `test_sweep_writes_every_run_and_resumes` (Task 4); the API's `*.pt` glob ignores `.tmp` (Task 6).
4. **Pricing load starving the liveness probe:** `/healthz` must stay on the event loop while pricing runs in threads. Pinned by `test_only_healthz_runs_on_the_event_loop` (Task 6).
5. **A models directory that is empty or mixes horizons** must fail at startup with a clear message, not answer wrongly. Pinned by `test_startup_refuses_empty_or_mixed_horizon_model_dirs` (Task 6).

## Patterns to mirror (ECC)

The repo is greenfield: it has only the spec, so there is no existing pattern to mirror. The plan's own conventions are:
- **Naming:** files and functions are snake_case, and run files are named by `train.run_name`.
- **Errors:** CLI and server startup failures use `raise SystemExit("<what to do>")`. Request errors use `HTTPException(422|404, "<reason with the accepted range>")`.
- **Tests:** two pytest files with plain asserts. Module-scoped fixtures (`smoke`, `client`) share one smoke sweep.

## File Structure

| File | Task | Responsibility |
|---|---|---|
| `pyproject.toml`, `uv.lock`, `.gitignore`, `LICENSE`, `README.md` (stub), `src/deep_hedging/__init__.py` | 1 | packaging, single-sourced version |
| `src/deep_hedging/core.py` | 1–3 | the paper's math: greeks, clamp, paths, P&L, entropic price, Whalley–Wilmott, networks |
| `tests/test_core.py` | 1–3 | math against closed forms, reference doctests, paper anchors |
| `src/deep_hedging/train.py` | 4 | one run, presets, atomic save / safe load, resumable parallel sweep |
| `src/deep_hedging/report.py` | 5 | tables next to the paper's, figures, results.csv, servable model set |
| `src/deep_hedging/api.py` | 6 | FastAPI app: validation, model selection, domain checks, hedge, price |
| `src/deep_hedging/cli.py` | 7 | `train | sweep | report | serve` |
| `tests/test_e2e.py` | 4–8, 10 | sweep, report, API, CLI, bundled models, manifest tags |
| `models/` | 8 → 13 | bundled NTB checkpoints (quick stand-ins, then the paper preset) |
| `Dockerfile`, `.dockerignore` | 9 | CPU image, non-root, entrypoint = CLI |
| `k8s/api.yaml`, `k8s/data.yaml`, `k8s/sweep.yaml`, `k8s/kustomization.yaml` | 10 | Deployment + Service, PVC, Job, trained-model patch |
| `.github/workflows/ci.yml` | 11 | test, kind end-to-end, tag-only publish |
| `README.md`, `docs/results/` | 12 → 13 | user docs and reproduced results |

## ECC skills and reviewers per task

- Tasks 1–3: `ecc:pytorch-patterns`, `ecc:python-testing`; reviewer `ecc:python-reviewer`.
- Tasks 4–5 and 13: `ecc:mle-workflow` (reproducibility, artifact provenance); reviewer `ecc:mle-reviewer`.
- Task 6: `ecc:fastapi-patterns`; reviewers `ecc:fastapi-reviewer` and `ecc:security-reviewer`.
- Task 7: reviewer `ecc:python-reviewer`.
- Task 9: `ecc:docker-patterns`; reviewer `ecc:security-reviewer`.
- Task 10: `ecc:kubernetes-patterns`; reviewer `ecc:security-reviewer`.
- Task 11: `ecc:github-ops`, `ecc:deployment-patterns`.
- Task 12: `ecc:doc-updater`.
- Whole branch before Task 14: `ecc:code-reviewer`, plus `ecc:silent-failure-hunter` on `train.py` and `api.py`.

---

### Task 1: Project scaffold, Black–Scholes greeks and the clamp

**Files:**
- Create: `pyproject.toml`, `.gitignore`, `LICENSE`, `README.md`, `src/deep_hedging/__init__.py`, `src/deep_hedging/core.py`, `tests/test_core.py`
- Generated: `uv.lock` (by `uv lock`)

**Interfaces:**
- Consumes: nothing.
- Produces (`deep_hedging.core`):
  - `DT = 1/365` and `STRIKE = {"european": 1.0, "lookback": 1.03}`;
  - `european(s, tau, sigma)` and `lookback(s, m, tau, sigma)`, each returning `(price_k, delta, gamma_k)` tensors;
  - `greeks(derivative, s, m, tau, sigma)`;
  - `clamp(x, lo, hi)`.
- Produces: `deep_hedging.__version__` (from package metadata).

- [ ] **Step 1: Create the packaging files**

Create `pyproject.toml`:

```toml
[project]
name = "deep-hedging"
version = "0.1.0"
description = "No-Transaction Band Network (Imaki et al. 2021): train, reproduce and serve deep hedgers on Docker and Kubernetes"
readme = "README.md"
license = "MIT"
requires-python = ">=3.12"
dependencies = [
    "torch>=2.14",
    "fastapi>=0.143",
    "uvicorn>=0.54",
    "matplotlib>=3.11",
]

[project.scripts]
deep-hedging = "deep_hedging.cli:main"

[dependency-groups]
dev = ["pytest>=9.1", "ruff>=0.17", "httpx2>=2.13"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.uv.sources]
torch = [{ index = "pytorch-cpu", marker = "sys_platform == 'linux'" }]

[[tool.uv.index]]
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cpu"
explicit = true

[tool.ruff]
line-length = 120
```

Create `.gitignore`:

```gitignore
out/
.venv/
__pycache__/
*.pt.tmp
.pytest_cache/
.ruff_cache/
```

Create `LICENSE`:

```text
MIT License

Copyright (c) 2026 Daniel Ladrero Meca

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

Create `README.md` (Task 12 replaces it; hatchling needs it to build):

```markdown
# deep-hedging

The No-Transaction Band Network (Imaki et al., arXiv:2103.01775) as a product: train, reproduce and serve deep hedgers on Docker and Kubernetes.
```

Create `src/deep_hedging/__init__.py`:

```python
from importlib.metadata import version

__version__ = version("deep-hedging")
```

- [ ] **Step 2: Lock and install**

Run: `uv lock && uv sync`
Expected:
- `uv.lock` is created;
- `grep -c 'download.pytorch.org/whl/cpu' uv.lock` prints `3`;
- `grep -ci nvidia uv.lock` prints `0`.

- [ ] **Step 3: Write the failing tests**

Create `tests/test_core.py`:

```python
import math

import pytest
import torch

from deep_hedging import core


def t64(*xs):
    return [torch.tensor(x, dtype=torch.float64) for x in xs]


def test_european_greeks_match_reference_and_closed_form():
    s, tau = t64([-0.01, 0.0, 0.01], 0.1)
    _, delta, _ = core.european(s, tau, 0.2)
    assert torch.allclose(delta, torch.tensor([0.4497, 0.5126, 0.5752], dtype=torch.float64), atol=1e-4)
    price, _, _ = core.european(*t64(0.0, 28 / 365), 0.2)
    assert price.item() == pytest.approx(0.0220962, abs=1e-7)


@pytest.mark.parametrize(
    "S,M,K,days,expected",
    [
        (1.0, 1.0, 1.03, 30, (0.0228074, 0.637942, 12.6734)),
        (1.02, 1.05, 1.03, 15, (0.0318061, 0.493096, 15.3960)),
        (1.05, 1.05, 1.03, 10, (0.0480230, 1.026689, 23.4338)),
    ],
)
def test_lookback_greeks_match_verified_vectors(S, M, K, days, expected):
    s, m, tau = t64(math.log(S / K), math.log(M / K), days / 365)
    price_k, delta, gamma_k = core.lookback(s, m, tau, 0.2)
    assert K * price_k.item() == pytest.approx(expected[0], abs=5e-7)
    assert delta.item() == pytest.approx(expected[1], abs=5e-7)
    assert gamma_k.item() / K == pytest.approx(expected[2], rel=1e-4)


@pytest.mark.parametrize("m_over_k", [0.98, 1.04])
def test_lookback_delta_and_gamma_are_derivatives_of_the_price(m_over_k):
    # away from S == M, analytic greeks must equal autograd of the closed-form price (in absolute units, K = 1.03)
    K = 1.03
    spot = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
    m, tau = t64(math.log(m_over_k * 1.05 / K), 20 / 365)
    price_k, delta, gamma_k = core.lookback(torch.log(spot / K), m, tau, 0.2)
    (d_auto,) = torch.autograd.grad(K * price_k, spot, create_graph=True)
    (g_auto,) = torch.autograd.grad(d_auto, spot)
    assert d_auto.item() == pytest.approx(delta.item(), abs=1e-10)
    assert g_auto.item() == pytest.approx(gamma_k.item() / K, rel=1e-8)


def test_clamp_matches_reference_including_inverted_band():
    x = torch.linspace(-2, 12, 15) * 0.1
    assert torch.equal(core.clamp(x, torch.tensor(0.0), torch.tensor(1.0)), x.clamp(0.0, 1.0))
    out = core.clamp(torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0]), torch.tensor(0.0))
    assert torch.equal(out, torch.tensor([0.0, 0.5]))
```

- [ ] **Step 4: Run them to verify they fail**

Run: `uv run pytest tests/test_core.py -q`
Expected: collection error `ImportError: cannot import name 'core' from 'deep_hedging'`.

- [ ] **Step 5: Implement the greeks and the clamp**

Create `src/deep_hedging/core.py`:

```python
"""The paper's math: Black-Scholes greeks, hedging strategies, paths, P&L and the entropic price.

Units follow the paper: spot starts at 1, strike K = STRIKE[derivative], risk aversion 1 on absolute P&L.
Greeks return (price/K, delta, gamma*K), so everything is scale-free in log-moneyness s = log(S/K).
"""

import math

import torch
from torch import nn

DT = 1 / 365
STRIKE = {"european": 1.0, "lookback": 1.03}


def _ncdf(x):
    return torch.special.ndtr(x)


def _npdf(x):
    return torch.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)


def european(s, tau, sigma):
    """Black-Scholes European call, zero rates. Returns (price/K, delta, gamma*K)."""
    a = sigma * torch.sqrt(tau)
    d1 = s / a + 0.5 * a
    es = torch.exp(s)
    return es * _ncdf(d1) - _ncdf(d1 - a), _ncdf(d1), _npdf(d1) / (es * a)


def lookback(s, m, tau, sigma):
    """Fixed-strike lookback call, zero rates, continuous monitoring; m = log(running max / K).

    Returns (price/K, delta, gamma*K). At S == max the gamma is the one-sided value (S below the max);
    autograd through max() returns gamma/4 there on torch 2.x, hence the closed form.
    """
    h = torch.maximum(m, s).clamp(min=0.0)
    a = sigma * torch.sqrt(tau)
    d1 = (s - h) / a + 0.5 * a
    n1, p1 = _ncdf(d1), _npdf(d1)
    g = d1 * n1 + p1
    es, eh = torch.exp(s), torch.exp(h)
    price = (eh - 1.0) + es * n1 - eh * _ncdf(d1 - a) + es * a * g
    return price, 2.0 * n1 + a * g, (2.0 * p1 / a + n1) / es


def greeks(derivative, s, m, tau, sigma):
    return european(s, tau, sigma) if derivative == "european" else lookback(s, m, tau, sigma)


def clamp(x, lo, hi):
    """Paper eq. 13; an inverted band (lo > hi) gives the midpoint."""
    return torch.where(lo < hi, torch.minimum(torch.maximum(x, lo), hi), (lo + hi) / 2)
```

- [ ] **Step 6: Run the tests**

Run: `uv run pytest tests/test_core.py -q && uv run ruff check .`
Expected: `7 passed`, `All checks passed!`

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml uv.lock .gitignore LICENSE README.md src tests
git commit -m "feat: package scaffold, Black-Scholes greeks and the band clamp"
```

---

### Task 2: Paths, P&L, entropic price, Whalley–Wilmott, paper anchors

**Files:**
- Modify: `src/deep_hedging/core.py` (append)
- Test: `tests/test_core.py` (append)

**Interfaces:**
- Consumes: `greeks`, `clamp`, `STRIKE`, `DT` (Task 1).
- Produces:
  - `simulate(n_paths, days, sigma, s0=0.0, generator=None) -> Tensor (days+1, n_paths)`;
  - `pnl(policy, derivative, s, cost, sigma, prev0=0.0, m0=None) -> Tensor (n_paths,)`, where `policy(s_i, m_i, tau_i, sigma, prev) -> hedge`;
  - `price(wealth, lam=1.0) -> float` and `utility(wealth, lam=1.0) -> float`;
  - `ww_band(derivative, cost, s, m, tau, sigma, lam=1.0) -> (lo, hi)` and `whalley_wilmott(derivative, cost) -> policy`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_core.py`:

```python
def test_simulate_uses_the_exact_martingale_drift():
    sigma, s0 = 0.2, -0.1
    s = core.simulate(1000, 28, sigma, s0, torch.Generator().manual_seed(0))
    z = torch.randn(28, 1000, generator=torch.Generator().manual_seed(0))
    expected = s0 + torch.cumsum(sigma * math.sqrt(core.DT) * z - 0.5 * sigma**2 * core.DT, 0)
    assert s.shape == (29, 1000)
    assert torch.all(s[0] == s0)
    assert torch.allclose(s[1:], expected, atol=1e-6)


def test_pnl_on_a_hand_written_path():
    s = torch.log(torch.tensor([[1.0], [1.1], [0.9], [1.2]]))  # European, K = 1
    zero = core.pnl(lambda *a: torch.zeros(1), "european", s, 0.01, 0.2)
    assert zero.item() == pytest.approx(-0.2, abs=1e-6)  # minus the payoff
    one = core.pnl(lambda *a: torch.ones(1), "european", s, 0.01, 0.2)
    assert one.item() == pytest.approx(1.2 - 1.0 - 0.01 * 1.0 - 0.2, abs=1e-6)  # cost only on the first trade


def test_lookback_payoff_includes_the_past_maximum():
    s = torch.log(torch.tensor([[1.0], [1.01], [0.99]]) / 1.03)
    past_max = math.log(1.10 / 1.03)
    wealth = core.pnl(lambda *a: torch.zeros(1), "lookback", s, 0.0, 0.2, m0=past_max)
    assert wealth.item() == pytest.approx(-(1.10 - 1.03), abs=1e-6)


def test_entropic_price_and_utility_match_reference():
    wealth = -torch.arange(4.0)
    assert core.utility(wealth) == pytest.approx(-7.7982, abs=1e-4)
    assert core.price(wealth) == pytest.approx(2.0539, abs=1e-4)


def test_whalley_wilmott_half_width():
    s, m, tau = t64(0.0, 0.0, 15 / 365)
    _, _, gamma_k = core.european(s, tau, 0.2)
    lo, hi = core.ww_band("european", 1e-3, s, m, tau, 0.2)
    assert ((hi - lo) / 2).item() == pytest.approx((1.5 * 1e-3 * gamma_k.item() ** 2) ** (1 / 3), rel=1e-12)
    lo0, hi0 = core.ww_band("european", 0.0, s, m, tau, 0.2)
    assert lo0.item() == hi0.item()


def no_hedge(s, m, tau, sigma, prev):
    return torch.zeros_like(s)


@pytest.mark.parametrize(
    "derivative,policy,expected,tol",
    [
        ("european", core.whalley_wilmott("european", 0.0), 0.022101, 3e-5),
        ("lookback", core.whalley_wilmott("lookback", 0.0), 0.01804, 6e-5),
        ("european", no_hedge, 0.022673, 2.4e-4),
        ("lookback", no_hedge, 0.018408, 2.0e-4),
    ],
)
def test_paper_anchors(derivative, policy, expected, tol):
    # zero-cost delta hedge vs Tables B2/B4 c=0 rows; no-hedge prices measured with 10M paths (28 daily steps)
    s = core.simulate(200_000, 28, 0.2, -math.log(core.STRIKE[derivative]), torch.Generator().manual_seed(1))
    with torch.no_grad():
        assert core.price(core.pnl(policy, derivative, s, 0.0, 0.2)) == pytest.approx(expected, abs=tol)
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/test_core.py -q`
Expected: the 9 new tests fail with `AttributeError: module 'deep_hedging.core' has no attribute 'simulate'` (and similar); the 7 from Task 1 pass.

- [ ] **Step 3: Implement**

Append to `src/deep_hedging/core.py`:

```python
def simulate(n_paths, days, sigma, s0=0.0, generator=None):
    """Driftless GBM in log-moneyness, shape (days + 1, n_paths), exact per-step drift -sigma^2 dt / 2."""
    z = torch.randn(days, n_paths, generator=generator)
    steps = sigma * math.sqrt(DT) * z - 0.5 * sigma**2 * DT
    return torch.cat([torch.full((1, n_paths), float(s0)), s0 + torch.cumsum(steps, 0)])


def pnl(policy, derivative, s, cost, sigma, prev0=0.0, m0=None):
    """Seller's final wealth (paper eq. 4) on paths s of shape (n + 1, N), in units where K = STRIKE[derivative].

    policy(s_i, m_i, tau_i, sigma, prev) returns the hedge held over [t_i, t_{i+1}]. Each trade costs
    cost * S_i * |change|; liquidation at maturity is free. m0 is a running maximum observed before s[0].
    """
    k = STRIKE[derivative]
    n = s.shape[0] - 1
    m = torch.cummax(s, 0).values
    if m0 is not None:
        m = torch.clamp(m, min=float(m0))
    spot = k * torch.exp(s)
    prev = torch.full_like(s[0], float(prev0))
    wealth = torch.zeros_like(s[0])
    for i in range(n):
        hedge = policy(s[i], m[i], torch.tensor((n - i) * DT), sigma, prev)
        wealth = wealth + hedge * (spot[i + 1] - spot[i]) - cost * spot[i] * (hedge - prev).abs()
        prev = hedge
    payoff = spot[-1] - k if derivative == "european" else k * torch.exp(m[-1]) - k
    return wealth - payoff.clamp(min=0.0)


def price(wealth, lam=1.0):
    """Entropic risk measure = indifference price: log E[exp(-lam P)] / lam, reduced in float64."""
    return math.log(torch.exp(-lam * wealth.double()).mean().item()) / lam


def utility(wealth, lam=1.0):
    """Expected exponential utility E[-exp(-lam P)]."""
    return -torch.exp(-lam * wealth.double()).mean().item()


def ww_band(derivative, cost, s, m, tau, sigma, lam=1.0):
    """Whalley-Wilmott asymptotic band (paper eq. 20): delta -/+ (1.5 c S gamma^2 / lam)^(1/3)."""
    _, delta, gamma_k = greeks(derivative, s, m, tau, sigma)
    w = (1.5 * cost * torch.exp(s) * gamma_k**2 / (STRIKE[derivative] * lam)) ** (1 / 3)
    return delta - w, delta + w


def whalley_wilmott(derivative, cost):
    """Hedging policy that keeps the hedge inside the Whalley-Wilmott band (a pure delta hedge at zero cost)."""
    return lambda s, m, tau, sigma, prev: clamp(prev, *ww_band(derivative, cost, s, m, tau, sigma))
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_core.py -q`
Expected: `16 passed`.

The anchor tests reproduce the paper's zero-cost prices at seed 1 and 200k paths: European 0.022119 (tolerance ±3e-5 around 0.022101) and lookback 0.018064 (±6e-5 around 0.01804). If an anchor fails, the step convention, units or payoff is wrong; do not widen the tolerance.

- [ ] **Step 5: Commit**

```bash
git add src/deep_hedging/core.py tests/test_core.py
git commit -m "feat: GBM paths, hedging P&L, entropic price and Whalley-Wilmott band"
```

---

### Task 3: The hedging networks (NTB and feed-forward)

**Files:**
- Modify: `src/deep_hedging/core.py` (append)
- Test: `tests/test_core.py` (append)

**Interfaces:**
- Consumes: `greeks`, `clamp` (Task 1).
- Produces:
  - `mlp(n_in, n_out) -> nn.Sequential`;
  - `class Hedger(nn.Module)`, constructed as `Hedger(derivative, method)` with method `"ntb"` or `"ffn"`. Methods: `band(s, m, tau, sigma) -> (lo, hi)` (NTB) and `forward(s, m, tau, sigma, prev) -> next_hedge`. A `Hedger` is a valid `pnl` policy.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_core.py`:

```python
@pytest.mark.parametrize(
    "derivative,method,n_params",
    [("european", "ntb", 3362), ("lookback", "ntb", 3394), ("european", "ffn", 3361), ("lookback", "ffn", 3393)],
)
def test_network_sizes_match_the_paper(derivative, method, n_params):
    assert sum(p.numel() for p in core.Hedger(derivative, method).parameters()) == n_params


def test_ntb_clamps_into_its_band_and_ignores_the_previous_hedge():
    torch.manual_seed(0)
    net = core.Hedger("lookback", "ntb")
    s = torch.linspace(-0.1, 0.05, 50)
    m, tau = s.clamp(min=-0.03), torch.tensor(10 / 365)
    lo, hi = net.band(s, m, tau, 0.2)
    for prev in (torch.full_like(s, -5.0), torch.full_like(s, 5.0), (lo + hi) / 2):
        out = net(s, m, tau, 0.2, prev)
        assert torch.all(out >= torch.minimum(lo, hi) - 1e-6) and torch.all(out <= torch.maximum(lo, hi) + 1e-6)
    inside = (lo + hi) / 2
    assert torch.equal(net(s, m, tau, 0.2, inside), torch.where(lo < hi, inside, (lo + hi) / 2))


def test_ffn_depends_on_the_previous_hedge():
    torch.manual_seed(0)
    net = core.Hedger("european", "ffn")
    s, tau = torch.zeros(4), torch.tensor(10 / 365)
    assert not torch.equal(net(s, s, tau, 0.2, torch.zeros(4)), net(s, s, tau, 0.2, torch.ones(4)))
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/test_core.py -q`
Expected: the 6 new tests fail with `AttributeError: module 'deep_hedging.core' has no attribute 'Hedger'`.

- [ ] **Step 3: Implement**

Append to `src/deep_hedging/core.py`:

```python
def mlp(n_in, n_out):
    """4 hidden layers x 32 units, ReLU (paper Section 5)."""
    layers = []
    for i in range(4):
        layers += [nn.Linear(n_in if i == 0 else 32, 32), nn.ReLU()]
    return nn.Sequential(*layers, nn.Linear(32, n_out))


class Hedger(nn.Module):
    """'ntb': no-transaction band network (paper eq. 12). 'ffn': feed-forward baseline (eq. 9)."""

    def __init__(self, derivative, method):
        super().__init__()
        self.derivative, self.method = derivative, method
        n_in = 3 if derivative == "european" else 4
        self.net = mlp(n_in, 2) if method == "ntb" else mlp(n_in + 1, 1)

    def features(self, s, m, tau, sigma):
        cols = [s] if self.derivative == "european" else [s, m]
        return torch.stack([*cols, tau.expand_as(s), torch.full_like(s, sigma)], -1)

    def band(self, s, m, tau, sigma):
        delta = greeks(self.derivative, s, m, tau, sigma)[1]
        o = nn.functional.leaky_relu(self.net(self.features(s, m, tau, sigma)), 0.01)
        return delta - o[..., 0], delta + o[..., 1]

    def forward(self, s, m, tau, sigma, prev):
        if self.method == "ntb":
            return clamp(prev, *self.band(s, m, tau, sigma))
        delta = greeks(self.derivative, s, m, tau, sigma)[1]
        x = torch.cat([self.features(s, m, tau, sigma), prev.unsqueeze(-1)], -1)
        return delta + torch.tanh(self.net(x)[..., 0])
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_core.py -q && uv run ruff check .`
Expected: `22 passed`, `All checks passed!`

- [ ] **Step 5: Commit**

```bash
git add src/deep_hedging/core.py tests/test_core.py
git commit -m "feat: no-transaction band network and feed-forward baseline"
```

---

### Task 4: Training run, checkpoints and the resumable sweep

**Files:**
- Create: `src/deep_hedging/train.py`
- Test: `tests/test_e2e.py` (create)

**Interfaces:**
- Consumes: `core.STRIKE`, `Hedger`, `pnl`, `price`, `simulate`, `utility`, `whalley_wilmott`; `deep_hedging.__version__`.
- Produces:
  - `LR`, `COSTS` (22 floats, index-aligned with the paper's tables), `DERIVATIVES`, `METHODS`, `PRESETS` (keys `smoke`, `quick`, `paper`);
  - `evaluate(policy, derivative, cost, sigma, days, seed, batches, n_paths) -> {price, price_se, utility, utility_se}`;
  - `run(derivative, method, cost, seed, iters, paths, eval_batches, eval_paths, days=28, sigma=0.2, preset="custom") -> checkpoint dict` (layout as in §6: `config`, `metrics`, `history`, `state_dict`, `meta`);
  - `run_name(**config) -> str`, `save(ckpt, path)`, `load(path)`, `policy(ckpt)`;
  - `tasks(preset) -> list[dict]` and `sweep(preset, out, workers)`, which writes `out/<preset>/runs/*.pt`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_e2e.py`:

```python
import math

import pytest

from deep_hedging import train


@pytest.fixture(scope="module")
def smoke(tmp_path_factory):
    out = tmp_path_factory.mktemp("out")
    train.sweep("smoke", out, workers=2)
    return out / "smoke"
```

and append:

```python
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
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/test_e2e.py -q`
Expected: collection error `ImportError: cannot import name 'train' from 'deep_hedging'`.

- [ ] **Step 3: Implement**

Create `src/deep_hedging/train.py`:

```python
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
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_e2e.py -q && uv run ruff check .`
Expected: `3 passed`. The fixture uses `workers=2`, so the spawn process pool runs; the resume check uses `workers=1`. The 30-iteration training check measured loss 1.02457 → 1.02361.

- [ ] **Step 5: Commit**

```bash
git add src/deep_hedging/train.py tests/test_e2e.py
git commit -m "feat: training run, atomic checkpoints and resumable parallel sweep"
```

---

### Task 5: Report — tables next to the paper's, figures, servable model set

**Files:**
- Create: `src/deep_hedging/report.py`
- Test: `tests/test_e2e.py` (modify the imports and fixture, append tests)

**Interfaces:**
- Consumes: `train.COSTS`, `train.load`, `train.policy`; `core.european`, `core.ww_band`.
- Produces:
  - `PAPER_PRICES[derivative][method] -> list[(mean, sd)]`, aligned with `COSTS`;
  - `report(run_dir)`, which writes `run_dir/report/{fig2_european,fig5_lookback,fig3_band,fig4_european_history,fig6_lookback_history}.png`, `tables.md` and `results.csv`, plus `run_dir/models/*.pt`;
  - `_stats(ckpts)` and `_on_grid(cost, grid)`, used by the Task 13 checks snippet.

- [ ] **Step 1: Write the failing tests**

In `tests/test_e2e.py`, add the import `from deep_hedging.report import report` under `from deep_hedging import train`. Replace the `smoke` fixture with:

```python
@pytest.fixture(scope="module")
def smoke(tmp_path_factory):
    out = tmp_path_factory.mktemp("out")
    train.sweep("smoke", out, workers=2)
    report(out / "smoke")
    return out / "smoke"
```

Then append:

```python
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
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/test_e2e.py -q`
Expected: collection error `ModuleNotFoundError: No module named 'deep_hedging.report'`.

- [ ] **Step 3: Implement**

Create `src/deep_hedging/report.py`. `PAPER_PRICES` was parsed programmatically from the paper's Tables B2 and B4 (22 rows × 3 methods × (mean, sd)); keep it byte for byte.

```python
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
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_e2e.py -q && uv run ruff check .`
Expected: `5 passed`, `All checks passed!`

- [ ] **Step 5: Commit**

```bash
git add src/deep_hedging/report.py tests/test_e2e.py
git commit -m "feat: report with paper columns, figures and best-seed model selection"
```

---

### Task 6: HTTP API

**Files:**
- Create: `src/deep_hedging/api.py`
- Test: `tests/test_e2e.py` (replace the imports, append tests)

**Interfaces:**
- Consumes: `core.DT`, `STRIKE`, `greeks`, `pnl`, `simulate`; `train.load`, `train.policy`.
- Produces:
  - `create_app(models_dir, price_paths=20_000) -> FastAPI`;
  - `price_state(derivative, net, s, m, days, cost, prev, sigma, n_paths, seed=PRICE_SEED) -> (price_model, se_model)`;
  - `class Option` (the request model);
  - `load_models`, `select`, `state`, `control`, `info`.
- Endpoints: `GET /` (redirect to `/docs`), `GET /healthz`, `GET /v1/models`, `POST /v1/hedge`, `POST /v1/price`, with request and response shapes as in §7.

- [ ] **Step 1: Write the failing tests**

Replace the import block at the top of `tests/test_e2e.py` with:

```python
import inspect
import json
import math
import shutil

import pytest
import torch
from fastapi.testclient import TestClient

from deep_hedging import core, train
from deep_hedging.api import create_app, price_state
from deep_hedging.report import report
```

Then append:

```python
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
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/test_e2e.py -q`
Expected: collection error `ModuleNotFoundError: No module named 'deep_hedging.api'`.

- [ ] **Step 3: Implement**

Create `src/deep_hedging/api.py`:

```python
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
        scale = opt.strike / STRIKE[opt.derivative]
        return {"price": p * scale, "std_error": se * scale, "paths": price_paths, "model": info(ckpt)}

    return app
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_e2e.py -q && uv run ruff check .`
Expected: `27 passed` with **no** `StarletteDeprecationWarning`, because httpx2 is the test client's HTTP library, and `All checks passed!`.

- [ ] **Step 5: Commit**

```bash
git add src/deep_hedging/api.py tests/test_e2e.py
git commit -m "feat: HTTP API with strict validation, domain checks and control-variate pricing"
```

---

### Task 7: Command line

**Files:**
- Create: `src/deep_hedging/cli.py`
- Test: `tests/test_e2e.py` (append)

**Interfaces:**
- Consumes: `train.run`, `run_name`, `save`, `sweep`; `report.report`; `api.create_app`.
- Produces:
  - `main(argv=None)` with subcommands `train`, `sweep`, `report`, `serve`;
  - the console script `deep-hedging` (already declared in `pyproject.toml`).
- Environment variables: `DEEP_HEDGING_OUT` (default `out`), `DEEP_HEDGING_MODELS` (default `models`), `DEEP_HEDGING_HOST` (default `127.0.0.1`).

- [ ] **Step 1: Write the failing test**

Append to `tests/test_e2e.py`:

```python
def test_cli_train_then_report(tmp_path, capsys):
    from deep_hedging.cli import main

    main(
        [
            "train",
            "--derivative",
            "european",
            "--cost",
            "0.001",
            "--method",
            "ww",
            "--paths",
            "1000",
            "--out",
            str(tmp_path),
        ]
    )
    assert "price 0.0" in capsys.readouterr().out
    assert len(list((tmp_path / "custom" / "runs").glob("european-ww-*.pt"))) == 1
    main(["report", str(tmp_path / "custom")])
    assert "| 0.001000 |" in (tmp_path / "custom" / "report" / "tables.md").read_text()
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_e2e.py -q -k cli`
Expected: FAIL with `ModuleNotFoundError: No module named 'deep_hedging.cli'`.

- [ ] **Step 3: Implement**

Create `src/deep_hedging/cli.py`:

```python
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
```

- [ ] **Step 4: Run the tests and the console script**

Run: `uv run pytest -q && uv run deep-hedging --help`
Expected: `50 passed`, then the usage line `usage: deep-hedging [-h] {train,sweep,report,serve} ...`.

- [ ] **Step 5: Commit**

```bash
git add src/deep_hedging/cli.py tests/test_e2e.py
git commit -m "feat: deep-hedging CLI (train, sweep, report, serve)"
```

---

### Task 8: Quick preset, its quality check, and the bundled stand-in models

**Files:**
- Create: `models/*.pt` (12 files, generated)
- Test: `tests/test_e2e.py` (add an import, append tests)

**Interfaces:**
- Consumes: the `sweep`, `report` and `api` modules.
- Produces: `models/` with the NTB checkpoint per derivative × cost of the quick preset, preset `"quick"`. Task 13 replaces them with the paper preset's 44.

- [ ] **Step 1: Run the quick preset**

Run: `uv run deep-hedging sweep --preset quick --workers 4`
Expected: about 80 s on an M5 Pro. It prints `quick: 36 runs to do`, 36 file names, then `tables.md`. In the measured run the European NTB column read 0.022162 / 0.022262 / 0.022475 / 0.022676 / 0.022676 for the five positive costs, against the paper's 0.022160 / 0.022257 / 0.022457 / 0.022675 / 0.022675.

- [ ] **Step 2: Check the quick preset shows the paper's effect (§13)**

Run:

```bash
uv run python -c "
import math
from deep_hedging.report import _stats
from deep_hedging.train import load
from pathlib import Path
st = _stats([load(p) for p in Path('out/quick/runs').glob('*.pt')])
for d in ('european', 'lookback'):
    for c in sorted({c for (dd, m, c) in st if dd == d and c > 0}):
        (n, ns), (f, fs) = st[(d, 'ntb', c)]['price'], st[(d, 'ffn', c)]['price']
        assert n <= f + 2 * math.hypot(ns, fs), (d, c, n, f)
print('quick preset: NTB <= FFN + 2 se at every positive cost')
"
```

Expected: `quick preset: NTB <= FFN + 2 se at every positive cost`. If it fails, raise the quick preset to `iters=500, paths=20_000, eval_paths=20_000` in `PRESETS`, re-run Steps 1–2, and record the new values in the spec §13 and the README.

- [ ] **Step 3: Write the failing tests**

In `tests/test_e2e.py`, add `from pathlib import Path` under `import shutil`, then append:

```python
ROOT = Path(__file__).resolve().parents[1]
BUNDLED = sorted((ROOT / "models").glob("*.pt"))


def test_bundled_models_load_safely_and_reproduce():
    assert BUNDLED, "models/ is empty: run a sweep and copy OUT/<preset>/models/*.pt into models/"
    ckpts = [train.load(p) for p in BUNDLED]  # weights_only=True for every file
    c = ckpts[0]["config"]
    again = train.evaluate(
        train.policy(ckpts[0]),
        c["derivative"],
        c["cost"],
        c["sigma"],
        c["days"],
        c["seed"],
        c["eval_batches"],
        c["eval_paths"],
    )
    assert abs(again["price"] - ckpts[0]["metrics"]["price"]) <= 0.5 * ckpts[0]["metrics"]["price_se"]


def test_price_endpoint_maps_request_units_like_training():
    # spot 100 / strike 103 is the training state (spot 1, strike 1.03) scaled by 100, so the price must be
    # 100 x the recorded one; simulating in strike units instead (spot 1/1.03, strike 1) would be ~3% off
    ckpt = next(train.load(p) for p in BUNDLED if p.name.startswith("lookback-ntb-c5.5308e-04"))
    cfg, rec = ckpt["config"], ckpt["metrics"]
    body = {
        "derivative": "lookback",
        "spot": 100.0,
        "strike": 103.0,
        "days_to_maturity": cfg["days"],
        "volatility": cfg["sigma"],
        "cost": cfg["cost"],
    }
    r = TestClient(create_app(ROOT / "models")).post("/v1/price", json=body).json()
    assert abs(r["price"] / 100 - rec["price"]) <= 4 * math.hypot(r["std_error"] / 100, rec["price_se"])
```

- [ ] **Step 4: Run them to verify they fail**

Run: `uv run pytest tests/test_e2e.py -q -k "bundled or units"`
Expected: `test_bundled_models_load_safely_and_reproduce` fails with `AssertionError: models/ is empty`, and the unit test fails with `StopIteration`.

- [ ] **Step 5: Bundle the quick models**

Run: `mkdir -p models && find models -name '*.pt' -delete && cp out/quick/models/*.pt models/ && ls models | wc -l`
Expected: `12` (288 KB). Use `find … -delete`, not `rm models/*.pt`: zsh aborts a glob that matches nothing.

- [ ] **Step 6: Run the tests**

Run: `uv run pytest -q`
Expected: `52 passed`.

In the measured run, the unit-mapping test saw API price/100 = 0.018410 against a recorded 0.018361. Its tolerance is 3.6e-4, and a 3% unit error would be 5.5e-4, so it catches that error.

- [ ] **Step 7: Commit**

```bash
git add models tests/test_e2e.py
git commit -m "feat: bundle quick-preset models; bundled-model and unit-mapping tests"
```

---

### Task 9: Docker image

**Files:**
- Create: `Dockerfile`, `.dockerignore`

**Interfaces:**
- Consumes: `pyproject.toml`, `uv.lock`, `src/`, `models/`, `README.md`.
- Produces: image `ghcr.io/ladrerod/deep-hedging:0.1.0`. Entrypoint `deep-hedging`, default command `serve`. Environment: `DEEP_HEDGING_MODELS=/app/models`, `DEEP_HEDGING_OUT=/data`, `DEEP_HEDGING_HOST=0.0.0.0`. uid/gid 10001.

- [ ] **Step 1: Write the image files**

Create `Dockerfile`:

```dockerfile
# syntax=docker/dockerfile:1
FROM python:3.12-slim
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_NO_DEV=1 PATH=/app/.venv/bin:$PATH \
    PYTHONUNBUFFERED=1 MPLCONFIGDIR=/tmp \
    DEEP_HEDGING_MODELS=/app/models DEEP_HEDGING_OUT=/data DEEP_HEDGING_HOST=0.0.0.0
WORKDIR /app
RUN --mount=from=ghcr.io/astral-sh/uv:0.13,source=/uv,target=/bin/uv \
    --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-install-project
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN --mount=from=ghcr.io/astral-sh/uv:0.13,source=/uv,target=/bin/uv \
    --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-editable
COPY models ./models
RUN mkdir /data && chown 10001:10001 /data
USER 10001:10001
EXPOSE 8000
ENTRYPOINT ["deep-hedging"]
CMD ["serve"]
```

Create `.dockerignore`:

```text
.git
.github
.venv
out
tests
docs
k8s
**/__pycache__
**/*.pt.tmp
```

- [ ] **Step 2: Build**

Run: `docker build -t ghcr.io/ladrerod/deep-hedging:0.1.0 . && docker image ls ghcr.io/ladrerod/deep-hedging:0.1.0 --format '{{.Size}}'`
Expected: the build succeeds, with about 30 s for a warm uv cache; the image is about 1.46 GB on disk (arm64).

- [ ] **Step 3: Smoke-test it read-only as an arbitrary user**

Run:

```bash
docker run -d --name dh --read-only --tmpfs /tmp --user 12345:12345 -p 127.0.0.1:8000:8000 ghcr.io/ladrerod/deep-hedging:0.1.0
get() { curl -fsS --retry 30 --retry-delay 1 --retry-all-errors "$@"; echo; }
body='{"derivative":"european","spot":100,"strike":100,"days_to_maturity":20,"volatility":0.2,"cost":0.0005}'
get localhost:8000/healthz
get localhost:8000/v1/hedge -H 'content-type: application/json' -d "$body"
get localhost:8000/v1/price -H 'content-type: application/json' -d "$body"
docker rm -f dh
```

Expected:
- `{"status":"ok","models":12}`;
- a hedge response with `"preset":"quick"`;
- a price response with a price around 1.9 and `"paths":20000`.

The logs show `Uvicorn running on http://0.0.0.0:8000`, because the host comes from the image environment and not from `CMD`.

- [ ] **Step 4: Train inside the container into a host folder**

Run: `mkdir -p out-docker && docker run --rm -u "$(id -u):$(id -g)" -v "$PWD/out-docker:/data" ghcr.io/ladrerod/deep-hedging:0.1.0 sweep --preset smoke --workers 2 | tail -2 && ls out-docker/smoke/models | wc -l && rm -r out-docker`
Expected: the tables footer, then `4`.

- [ ] **Step 5: Commit**

```bash
git add Dockerfile .dockerignore
git commit -m "feat: CPU-only container image, non-root, read-only friendly"
```

---

### Task 10: Kubernetes manifests, verified live on the local cluster

**Files:**
- Create: `k8s/api.yaml`, `k8s/data.yaml`, `k8s/sweep.yaml`, `k8s/kustomization.yaml`
- Test: `tests/test_e2e.py` (add imports, append the manifest-tag test)

**Interfaces:**
- Consumes: image `ghcr.io/ladrerod/deep-hedging:0.1.0` (Task 9; OrbStack's cluster shares the Docker image store); CLI `serve` / `sweep`.
- Produces:
  - Deployment and Service `deep-hedging`, selector `{app.kubernetes.io/name: deep-hedging, app.kubernetes.io/component: api}`;
  - PVC `deep-hedging-data`;
  - Job `deep-hedging-sweep`, writing `/data/quick/…`;
  - a Kustomize patch serving `/data/quick/models`.

- [ ] **Step 1: Write the failing test**

In `tests/test_e2e.py`, add `import re` under `import math` and `import tomllib` under `import shutil`, then append:

```python
def test_manifest_image_tags_match_the_package_version():
    version = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"]
    text = "".join(p.read_text() for p in (ROOT / "k8s").glob("*.yaml"))
    tags = re.findall(r"image: ghcr\.io/ladrerod/deep-hedging:(\S+)", text)
    assert len(tags) == 2 and set(tags) == {version}
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_e2e.py -q -k manifest`
Expected: FAIL with `AssertionError` (no manifests, so `len(tags) == 0`).

- [ ] **Step 3: Write the manifests**

Create `k8s/api.yaml`:

```yaml
# Stateless API serving the models built into the image.
apiVersion: apps/v1
kind: Deployment
metadata:
  name: deep-hedging
  labels: {app.kubernetes.io/name: deep-hedging, app.kubernetes.io/component: api}
spec:
  replicas: 1
  selector:
    matchLabels: {app.kubernetes.io/name: deep-hedging, app.kubernetes.io/component: api}
  template:
    metadata:
      labels: {app.kubernetes.io/name: deep-hedging, app.kubernetes.io/component: api}
    spec:
      automountServiceAccountToken: false
      securityContext:
        runAsNonRoot: true
        runAsUser: 10001
        runAsGroup: 10001
        fsGroup: 10001
        seccompProfile: {type: RuntimeDefault}
      containers:
        - name: api
          image: ghcr.io/ladrerod/deep-hedging:0.1.0
          args: ["serve"]
          ports: [{name: http, containerPort: 8000}]
          startupProbe: {httpGet: {path: /healthz, port: http}, periodSeconds: 2, failureThreshold: 30}
          readinessProbe: {httpGet: {path: /healthz, port: http}, periodSeconds: 5}
          livenessProbe: {httpGet: {path: /healthz, port: http}, periodSeconds: 10, timeoutSeconds: 2}
          resources:
            requests: {cpu: 250m, memory: 512Mi}
            limits: {memory: 1Gi}
          securityContext:
            readOnlyRootFilesystem: true
            allowPrivilegeEscalation: false
            capabilities: {drop: [ALL]}
          volumeMounts: [{name: tmp, mountPath: /tmp}]
      volumes: [{name: tmp, emptyDir: {}}]
---
apiVersion: v1
kind: Service
metadata:
  name: deep-hedging
  labels: {app.kubernetes.io/name: deep-hedging, app.kubernetes.io/component: api}
spec:
  selector: {app.kubernetes.io/name: deep-hedging, app.kubernetes.io/component: api}
  ports: [{name: http, port: 80, targetPort: http}]
```

Create `k8s/data.yaml`:

```yaml
# Results of the training Job live here. Its own file, so re-running or deleting the Job never deletes them.
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: deep-hedging-data
  labels: {app.kubernetes.io/name: deep-hedging}
spec:
  accessModes: [ReadWriteOnce]
  resources:
    requests: {storage: 1Gi}
```

Create `k8s/sweep.yaml`:

```yaml
# Train and reproduce on the cluster: writes /data/<preset>/{runs,report,models} on the PVC from data.yaml.
# Re-run: kubectl delete job deep-hedging-sweep && kubectl apply -f k8s/sweep.yaml (finished runs are skipped).
apiVersion: batch/v1
kind: Job
metadata:
  name: deep-hedging-sweep
  labels: {app.kubernetes.io/name: deep-hedging, app.kubernetes.io/component: sweep}
spec:
  backoffLimit: 3
  template:
    metadata:
      labels: {app.kubernetes.io/name: deep-hedging, app.kubernetes.io/component: sweep, deep-hedging/volume: data}
    spec:
      restartPolicy: Never
      automountServiceAccountToken: false
      securityContext:
        runAsNonRoot: true
        runAsUser: 10001
        runAsGroup: 10001
        fsGroup: 10001
        seccompProfile: {type: RuntimeDefault}
      affinity:  # every pod that mounts the ReadWriteOnce volume lands on the volume's node
        podAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
            - labelSelector: {matchLabels: {deep-hedging/volume: data}}
              topologyKey: kubernetes.io/hostname
      containers:
        - name: sweep
          image: ghcr.io/ladrerod/deep-hedging:0.1.0
          args: ["sweep", "--preset=quick", "--workers=2"]  # paper preset: ["sweep", "--preset=paper", "--workers=4"]
          resources:
            requests: {cpu: 500m, memory: 1Gi}
            limits: {cpu: "2", memory: 3Gi}  # paper preset: {cpu: "4", memory: 6Gi}
          securityContext:
            readOnlyRootFilesystem: true
            allowPrivilegeEscalation: false
            capabilities: {drop: [ALL]}
          volumeMounts: [{name: data, mountPath: /data}, {name: tmp, mountPath: /tmp}]
      volumes:
        - {name: data, persistentVolumeClaim: {claimName: deep-hedging-data}}
        - {name: tmp, emptyDir: {}}
```

Create `k8s/kustomization.yaml`:

```yaml
# kubectl apply -k k8s: the API from api.yaml, now serving the models the Job trained onto the PVC.
# Back to the built-in models: kubectl apply -f k8s/api.yaml
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization
resources: [api.yaml]
patches:
  - target: {kind: Deployment, name: deep-hedging}
    patch: |-
      apiVersion: apps/v1
      kind: Deployment
      metadata: {name: deep-hedging}
      spec:
        template:
          metadata:
            labels: {deep-hedging/volume: data}
          spec:
            affinity:
              podAffinity:
                requiredDuringSchedulingIgnoredDuringExecution:
                  - labelSelector: {matchLabels: {deep-hedging/volume: data}}
                    topologyKey: kubernetes.io/hostname
            containers:
              - name: api
                env: [{name: DEEP_HEDGING_MODELS, value: /data/quick/models}]  # paper preset: /data/paper/models
                volumeMounts: [{name: data, mountPath: /data, readOnly: true}]
            volumes:
              - {name: data, persistentVolumeClaim: {claimName: deep-hedging-data, readOnly: true}}
```

- [ ] **Step 4: Run the test and the server-side dry-runs**

Run: `uv run pytest -q && kubectl apply --dry-run=server -f k8s/api.yaml -f k8s/data.yaml -f k8s/sweep.yaml && kubectl kustomize k8s | kubectl apply --dry-run=server -f -`
Expected: `53 passed`, then `created (server dry run)` for the Deployment, Service, PVC and Job, and again for the Service and Deployment.

- [ ] **Step 5: Verify the whole newcomer path live, under the restricted Pod Security profile**

Run (needs the image from Task 9 and the `orbstack` context):

```bash
kubectl create namespace dh-verify
kubectl label namespace dh-verify pod-security.kubernetes.io/enforce=restricted
get() { curl -fsS --retry 30 --retry-delay 1 --retry-all-errors "$@"; echo; }
body='{"derivative":"european","spot":100,"strike":100,"days_to_maturity":20,"volatility":0.2,"cost":0.0005}'
kubectl -n dh-verify apply -f k8s/api.yaml && kubectl -n dh-verify rollout status deployment/deep-hedging --timeout=180s
kubectl -n dh-verify port-forward service/deep-hedging 8000:80 & pf=$!
get localhost:8000/v1/hedge -H 'content-type: application/json' -d "$body"; kill $pf
kubectl -n dh-verify apply -f k8s/data.yaml -f k8s/sweep.yaml
kubectl -n dh-verify wait --for=condition=complete job/deep-hedging-sweep --timeout=15m
kubectl -n dh-verify logs job/deep-hedging-sweep | tail -5
kubectl -n dh-verify apply -k k8s && kubectl -n dh-verify rollout status deployment/deep-hedging --timeout=180s
kubectl -n dh-verify port-forward service/deep-hedging 8000:80 & pf=$!
get localhost:8000/v1/models | head -c 300; echo; kill $pf
kubectl delete namespace dh-verify
```

Expected:
- no Pod Security warnings;
- the first hedge response says `"preset":"quick"` (built-in models);
- the Job completes in about 2–3 minutes;
- after `apply -k`, `/v1/models` lists models served from the PVC;
- the namespace is deleted at the end.

If the Job pod stays `Pending`, run `kubectl describe pod`: it is usually CPU requests against a small cluster.

- [ ] **Step 6: Commit**

```bash
git add k8s tests/test_e2e.py
git commit -m "feat: Kubernetes manifests (API, PVC, training Job, trained-model patch)"
```

---

### Task 11: CI/CD workflow

**Files:**
- Create: `.github/workflows/ci.yml`

**Interfaces:**
- Consumes: `uv.lock`, the test suite, the `Dockerfile`, and `k8s/*` with the literal strings `ghcr.io/ladrerod/deep-hedging:<version>` and `quick`, which the job rewrites with `sed`.
- Produces: jobs `test`, `image` (kind end-to-end) and `publish` (tags `v*` only: `ghcr.io/ladrerod/deep-hedging:<version>` and `:latest`, linux/amd64 + linux/arm64).

- [ ] **Step 1: Write the workflow**

Create `.github/workflows/ci.yml`. The action major versions were checked on 2026-10-14 with `gh api repos/<owner>/<action>/releases/latest`.

```yaml
name: ci

on:
  push:
    branches: [main]
    tags: ["v*"]
  pull_request:

permissions:
  contents: read

jobs:
  test:
    runs-on: ubuntu-24.04
    steps:
      - uses: actions/checkout@v7
      - uses: astral-sh/setup-uv@v10
      - run: uv sync --locked
      - run: uv run ruff check .
      - run: uv run pytest -q

  image:
    needs: test
    runs-on: ubuntu-24.04
    steps:
      - uses: actions/checkout@v7
      - uses: docker/setup-buildx-action@v4
      - uses: docker/build-push-action@v7
        with:
          context: .
          load: true
          tags: deep-hedging:ci
          cache-from: type=gha
          cache-to: type=gha,mode=max
      - name: Kubernetes end-to-end on kind (the runner ships kind and kubectl)
        run: |
          set -euo pipefail
          get() { curl -fsS --retry 30 --retry-delay 1 --retry-all-errors "$@"; echo; }
          body='{"derivative":"european","spot":100,"strike":100,"days_to_maturity":20,"volatility":0.2,"cost":0.0005}'
          kind create cluster --wait 120s
          kind load docker-image deep-hedging:ci
          sed -i 's#ghcr.io/ladrerod/deep-hedging:[^ ]*#deep-hedging:ci#' k8s/api.yaml k8s/sweep.yaml
          sed -i 's/quick/smoke/' k8s/sweep.yaml k8s/kustomization.yaml
          kubectl apply -f k8s/api.yaml
          kubectl rollout status deployment/deep-hedging --timeout=180s
          kubectl port-forward service/deep-hedging 8000:80 & pf=$!
          get localhost:8000/healthz
          get localhost:8000/v1/hedge -H 'content-type: application/json' -d "$body"
          get localhost:8000/v1/price -H 'content-type: application/json' -d "$body"
          kill "$pf"
          kubectl apply -f k8s/data.yaml -f k8s/sweep.yaml
          kubectl wait --for=condition=complete job/deep-hedging-sweep --timeout=600s
          kubectl logs job/deep-hedging-sweep | tail -20
          kubectl apply -k k8s
          kubectl rollout status deployment/deep-hedging --timeout=180s
          kubectl port-forward service/deep-hedging 8000:80 &
          get localhost:8000/v1/hedge -H 'content-type: application/json' -d "$body"
      - name: Cluster state on failure
        if: failure()
        run: |
          kubectl get all,pvc -o wide || true
          kubectl describe pods || true
          kubectl logs -l app.kubernetes.io/name=deep-hedging --all-containers --tail=100 || true

  publish:
    if: startsWith(github.ref, 'refs/tags/v')
    needs: image
    runs-on: ubuntu-24.04
    permissions:
      contents: read
      packages: write
    steps:
      - uses: actions/checkout@v7
      - name: Tag must match the pyproject version
        run: test "v$(sed -n 's/^version = "\(.*\)"/\1/p' pyproject.toml)" = "$GITHUB_REF_NAME"
      - uses: docker/setup-qemu-action@v4
      - uses: docker/setup-buildx-action@v4
      - uses: docker/login-action@v4
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}
      - id: meta
        uses: docker/metadata-action@v6
        with:
          images: ghcr.io/ladrerod/deep-hedging
          tags: type=semver,pattern={{version}}
      - uses: docker/build-push-action@v7
        with:
          context: .
          platforms: linux/amd64,linux/arm64
          push: true
          tags: ${{ steps.meta.outputs.tags }}
          labels: ${{ steps.meta.outputs.labels }}
```

- [ ] **Step 2: Lint it**

Run: `docker run --rm -v "$PWD:/repo" -w /repo rhysd/actionlint:latest -color=false; echo "exit=$?"`
Expected: no findings, `exit=0`. The real run happens on the first push (Task 14).

- [ ] **Step 3: Commit**

```bash
git add .github/workflows/ci.yml
git commit -m "ci: tests, kind end-to-end, tag-only multi-arch publish to GHCR"
```

---

### Task 12: README and the first results

**Files:**
- Modify: `README.md` (replace the stub)
- Create: `docs/results/` (figures, `tables.md`, `results.csv`, copied from `out/quick/report/`)

**Interfaces:**
- Consumes: CLI flags (Task 7), endpoints (Task 6), manifests (Task 10), image tag (Task 9), presets and timings (spec §13).
- Produces: the public README. Its "Reproducing the paper" section embeds `docs/results/*.png` and links `docs/results/tables.md`.

- [ ] **Step 1: Copy the quick results**

Run: `mkdir -p docs/results && cp out/quick/report/* docs/results/ && ls docs/results`
Expected: five PNGs, `results.csv` and `tables.md`. These are stand-ins; Task 13 replaces them with the paper preset's output before any release.

- [ ] **Step 2: Write the README**

Replace `README.md` with:

````markdown
# deep-hedging

[![ci](https://github.com/ladrerod/deep-hedging/actions/workflows/ci.yml/badge.svg)](https://github.com/ladrerod/deep-hedging/actions/workflows/ci.yml)

Train, reproduce and serve the **No-Transaction Band Network** for hedging options under transaction costs, with one container image for Docker and Kubernetes.

![Price spread vs transaction cost, European option](docs/results/fig2_european.png)

## What it is

An independent re-implementation of Imaki, Imajo, Ito, Minami and Nakagawa,
[*No-Transaction Band Network: A Neural Network Architecture for Efficient Deep Hedging*](https://arxiv.org/abs/2103.01775) (2021):

- A small network outputs a band `[b_l, b_u]` around the Black-Scholes delta, and the hedger only trades when its position leaves the band. This avoids the "action dependence" that makes ordinary deep hedging slow to train.
- `deep-hedging sweep` trains the band network, a feed-forward deep hedger and the Whalley-Wilmott asymptotic band across 22 transaction costs for a European and a lookback call. It then regenerates the paper's figures and tables.
- `deep-hedging serve` exposes trained band networks over HTTP: the band, the next hedge and the indifference price for your option and cost.

Not affiliated with the authors or Preferred Networks. Research code, not investment advice.

## Quickstart: Docker

```sh
docker run --rm -p 127.0.0.1:8000:8000 ghcr.io/ladrerod/deep-hedging:0.1.0
curl -s localhost:8000/v1/hedge -H 'content-type: application/json' \
  -d '{"derivative":"european","spot":100,"strike":100,"days_to_maturity":20,"volatility":0.2,"cost":0.0005,"current_hedge":0.3}'
```

The image serves 44 pre-trained models (22 costs × 2 options). Interactive docs are at http://localhost:8000/docs.

To train your own (the quick preset takes about 1–2 minutes on 4 cores) and serve them:

```sh
mkdir -p out
docker run --rm -u "$(id -u):$(id -g)" -v "$PWD/out:/data" ghcr.io/ladrerod/deep-hedging:0.1.0 sweep --preset quick
docker run --rm -p 127.0.0.1:8000:8000 -v "$PWD/out:/data:ro" ghcr.io/ladrerod/deep-hedging:0.1.0 serve --models /data/quick/models
```

`out/quick/report/` then holds the figures and `tables.md`. On Linux, `-u` lets the non-root container write to your folder.

## Quickstart: Kubernetes

Any cluster works: OrbStack, kind, minikube, Docker Desktop or a cloud cluster.

```sh
kubectl apply -f https://raw.githubusercontent.com/ladrerod/deep-hedging/v0.1.0/k8s/api.yaml
kubectl rollout status deployment/deep-hedging
kubectl port-forward service/deep-hedging 8000:80 &
sleep 2
curl -s localhost:8000/v1/hedge -H 'content-type: application/json' \
  -d '{"derivative":"european","spot":100,"strike":100,"days_to_maturity":20,"volatility":0.2,"cost":0.0005}'
```

Train on the cluster, then serve what was trained:

```sh
git clone --depth 1 --branch v0.1.0 https://github.com/ladrerod/deep-hedging && cd deep-hedging
kubectl apply -f k8s/data.yaml -f k8s/sweep.yaml          # PVC + Job (quick preset)
kubectl wait --for=condition=complete job/deep-hedging-sweep --timeout=60m
kubectl logs job/deep-hedging-sweep | tail -40             # the results tables
kubectl apply -k k8s && kubectl rollout status deployment/deep-hedging
kubectl port-forward service/deep-hedging 8000:80 &        # restart: the old one ended with its pod
kubectl exec deploy/deep-hedging -- tar -C /data/quick -cf - report | tar -xf -   # copy the figures out
```

- **Re-run the Job:** `kubectl delete job deep-hedging-sweep && kubectl apply -f k8s/sweep.yaml`. Finished runs are skipped, and results stay on the PVC (`k8s/data.yaml`), so never delete that file's claim unless you mean it.
- **Back to the built-in models:** `kubectl apply -f k8s/api.yaml`.
- **Paper preset:** edit the two commented lines in `k8s/sweep.yaml` and the one in `k8s/kustomization.yaml`.
- **Storage:** every pod that mounts the ReadWriteOnce volume is pinned to its node by pod affinity. With a ReadWriteMany storage class you can drop the affinity blocks.
- **Locally built image:** build with `docker build -t ghcr.io/ladrerod/deep-hedging:0.1.0 .`. OrbStack and Docker Desktop see it directly; kind needs `kind load docker-image ghcr.io/ladrerod/deep-hedging:0.1.0`; minikube needs `minikube image load ghcr.io/ladrerod/deep-hedging:0.1.0`.

The API has no authentication: keep the Service cluster-internal, or add auth and TLS in front of it.

## API

| Endpoint | Returns |
|---|---|
| `GET /healthz` | `{"status": "ok", "models": 44}` |
| `GET /v1/models` | every model: derivative, cost, volatility, days, trained strike, preset, recorded price |
| `POST /v1/hedge` | `bs_delta`, `band.lower`, `band.upper`, `next_hedge`, `trade` (microseconds) |
| `POST /v1/price` | `price`, `std_error` from 20,000 Monte Carlo paths with a control variate (~0.1–0.2 s) |

**Request body:**

```json
{"derivative": "european", "spot": 101.5, "strike": 100, "days_to_maturity": 15,
 "volatility": 0.2, "cost": 0.0001, "current_hedge": 0.3}
```

- Lookback requests may add `max_spot`: the running maximum over daily closes including today (default `spot`).
- `price` is the indifference price of the remaining hedged liability, starting from `current_hedge`. For a new option (`current_hedge` 0) it is the paper's indifference price.

**Validity limits.** Requests outside these return 422 or 404 with the accepted range:

- **Volatility** must equal a trained one (0.2 for the built-in models).
- **Maturity:** `days_to_maturity` ≤ 28.
- **Moneyness:** spot within the range training simulated for that time. That is ±4σ√((elapsed days + 1)/365) around the start, for example within about 4% of the strike (European) for a brand-new 28-day option.
- **Cost:** within half a grid step (log scale) of a trained cost. Cost 0 uses only the zero-cost model.

**Units.** Risk aversion λ = 1 applies to P&L per unit of `strike / K_train`, where `K_train` is 1 (European) or 1.03 (lookback). Equivalently `λ = K_train / strike`. With this low risk aversion the bands are wide, and above roughly 0.15% cost the European network learns not to hedge at all. That matches the paper, but it surprises people.

**Served models** are the best of 5 seeds, so their recorded prices are optimistic by about one standard error.

## Reproducing the paper

| Preset | Costs × seeds | Iterations × paths | Time (Apple M5 Pro, 4 workers) |
|---|---|---|---|
| `smoke` | 2 × 1 | 5 × 1,000 | seconds (tests and CI) |
| `quick` (default) | 6 × 1 | 300 × 10,000 | about 1–2 minutes |
| `paper` | 22 × 5 | 1,000 × 50,000 | about 7 hours |

- Run `deep-hedging sweep --preset paper --workers 4`. It resumes if interrupted.
- One training iteration at 50,000 paths takes about 0.17 s on one core. More than 4 workers did not help: the work is memory-bandwidth bound.

The figures and tables below come from the paper preset: [`docs/results/tables.md`](docs/results/tables.md) lists our prices next to the paper's Tables B2 and B4.

| | |
|---|---|
| ![](docs/results/fig5_lookback.png) | ![](docs/results/fig3_band.png) |
| ![](docs/results/fig4_european_history.png) | ![](docs/results/fig6_lookback_history.png) |

### Deviations from the paper

- **28 daily steps, not 30.** The paper's tables only match 28: the zero-cost European price is 0.022101 with 28 steps (paper 0.022101) but 0.022880 with 30. The authors' code builds `int((30/365)/(1/365)) = 29` time points.
- **Exactly driftless paths.** The reference code's path generator has a drift of about -1.1e-4 over the horizon.
- **Closed-form lookback greeks** instead of automatic differentiation, which on current PyTorch returns a quarter of the gamma whenever the spot equals its running maximum.
- **Not reproducible from the paper itself:**
  - Table B3 is a copy of Table B1.
  - The lookback "Asymptotic" column (for example 0.018731 at c = e^-7, where exact greeks give 0.018464 and the authors' own library gives 0.018445).
  - The European "Asymptotic" column at c ≥ e^-6, which matches only a one-day-shifted time-to-maturity input.
  - The lookback band-network prices at high cost, which exceed the price of not hedging (0.0191 vs 0.0184). Ours level off at the no-hedge price.
- **Learning curves** show the per-iteration training loss (20-iteration rolling mean) instead of 20 validation simulations per point.
- **Reproducible within Monte Carlo error,** not bit-for-bit across CPUs or PyTorch versions.

## Train your own model

```sh
deep-hedging train --derivative lookback --cost 0.001 --sigma 0.25 --days 21
deep-hedging serve --models out/custom/runs
```

A models directory may hold one horizon (`--days`) per derivative and volatility.

## Development

```sh
uv sync && uv run pytest -q && uv run ruff check .
docker build -t ghcr.io/ladrerod/deep-hedging:0.1.0 .
```

- **Bundled models and figures:**
  ```sh
  deep-hedging sweep --preset paper --workers 4
  find models -name '*.pt' -delete && cp out/paper/models/*.pt models/
  cp out/paper/report/* docs/results/
  ```
- **Release:**
  1. Bump `version` in `pyproject.toml` and the image tag in `k8s/api.yaml` and `k8s/sweep.yaml` (a test checks they agree).
  2. Commit, tag `vX.Y.Z` and push.
  3. CI publishes the multi-arch image to GHCR. After the first release, check that the package is public under GitHub → Packages.

## Limitations

One volatility (0.2), driftless geometric Brownian motion, European and lookback calls only, proportional costs, a fixed risk aversion, CPU only, one-node training on Kubernetes.

## Credits

The paper: Shota Imaki, Kentaro Imajo, Katsuya Ito, Kentaro Minami, Kei Nakagawa.

The authors' [`pfnet-research/NoTransactionBandNetwork`](https://github.com/pfnet-research/NoTransactionBandNetwork) and [`pfnet-research/pfhedge`](https://github.com/pfnet-research/pfhedge) (MIT, © 2021 pfnet-research) were consulted for conventions and test values. No code from them is included.

```bibtex
@article{imaki2021ntb,
  title   = {No-Transaction Band Network: A Neural Network Architecture for Efficient Deep Hedging},
  author  = {Imaki, Shota and Imajo, Kentaro and Ito, Katsuya and Minami, Kentaro and Nakagawa, Kei},
  journal = {arXiv preprint arXiv:2103.01775},
  year    = {2021}
}
```

License: MIT.
````

- [ ] **Step 3: Check every command in the README against the code**

Run: `uv run deep-hedging train --help && uv run deep-hedging sweep --help && uv run deep-hedging serve --help && uv run pytest -q`
Expected:
- the flags named in the README (`--derivative --cost --sigma --days --preset --workers --models`) all appear;
- `53 passed`.

- [ ] **Step 4: Commit**

```bash
git add README.md docs/results
git commit -m "docs: README with quickstarts, API, reproduction notes and credits"
```

---

### Task 13: Paper preset: bundled models, results and reproduction checks

Start Step 1 as early as Task 8 is committed: it runs about 7 hours in the background while Tasks 9–12 proceed.

**Files:**
- Modify: `models/` (replaced by 44 paper checkpoints), `docs/results/` (replaced), `README.md` (adds a "Reproduction checks" subsection)

**Interfaces:**
- Consumes: `sweep`/`report` (Tasks 4–5) and `report._stats`, `report._on_grid`, `report.PAPER_PRICES`.
- Produces: the release content. The `/healthz` model count becomes 44.

- [ ] **Step 1: Run the paper sweep in the background**

Run: `mkdir -p out && nohup uv run deep-hedging sweep --preset paper --workers 4 > out/paper.log 2>&1 &`
Expected:
- `out/paper.log` starts with `paper: 660 runs to do`, and file names accumulate there;
- `ls out/paper/runs | wc -l` reaches 660 after about 7 hours on an M5 Pro;
- the log ends with `tables.md`.

If the sweep is interrupted, re-run the same command: finished runs are skipped.

- [ ] **Step 2: Compute the reproduction checks**

Save as a scratch file (not committed) and run it with `uv run python checks.py out/paper`:

```python
import math
import statistics
import sys
from pathlib import Path

from deep_hedging.report import PAPER_PRICES, _on_grid, _stats
from deep_hedging.train import COSTS, load

run_dir = Path(sys.argv[1])
ckpts = [load(p) for p in sorted((run_dir / "runs").glob("*.pt"))]
st = _stats(ckpts)
n_seeds = len({c["config"]["seed"] for c in ckpts})
pos = sorted({c for (_, _, c) in st if c > 0})
ok = lambda b: "yes" if b else "**no**"  # noqa: E731


def beats(d):
    return all(st[(d, "ntb", c)]["price"][0] <= st[(d, "ffn", c)]["price"][0] for c in pos)


near = []
for c in (c for c in pos if c <= 1e-3):
    ours, paper = st[("european", "ntb", c)]["price"], PAPER_PRICES["european"]["ntb"][_on_grid(c, COSTS)]
    near.append(abs(ours[0] - paper[0]) <= 3 * math.hypot(ours[1], paper[1]) / math.sqrt(5))
plateau = [abs(st[("european", "ntb", c)]["price"][0] - 0.022673) <= 3 * max(st[("european", "ntb", c)]["price"][1], 3e-5)
           for c in pos if c >= 1.5e-3]


def flat_by_100(method):
    h = next(c["history"] for c in ckpts if c["config"]["derivative"] == "european" and c["config"]["method"] == method
             and c["config"]["seed"] == 0 and math.isclose(c["config"]["cost"], math.exp(-7.5)))
    roll = lambda j: statistics.mean(h[max(0, j - 19) : j + 1])  # noqa: E731
    return (roll(19) - roll(100)) / (roll(19) - roll(len(h) - 1))


print(f"- {n_seeds} seed(s), {len(pos)} positive costs.")
print(f"- European: band network price <= feed-forward at every positive cost: {ok(beats('european'))}.")
print(f"- European: band network within 3 combined sd of Table B2 at costs <= 1e-3: {sum(near)}/{len(near)}.")
print(f"- European: band network plateaus at the no-hedge price 0.02267 for c >= 1.5e-3: {sum(plateau)}/{len(plateau)}.")
print(f"- Lookback: band network price <= feed-forward at every positive cost: {ok(beats('lookback'))}.")
print(f"- Share of the loss improvement reached by iteration 100 (European, c = e^-7.5, seed 0): band network "
      f"{flat_by_100('ntb'):.0%}, feed-forward {flat_by_100('ffn'):.0%}.")
```

Expected: six bullet lines. On the quick preset they read:
- European band network ≤ feed-forward: yes;
- within 3 combined sd of Table B2: 3/3;
- plateau: 2/2;
- lookback band network ≤ feed-forward: yes;
- share of the loss improvement reached by iteration 100: band network 88%, feed-forward 6%.

The paper run's numbers replace these. Any line that reads `**no**` is a finding to report, not a test to tune.

- [ ] **Step 3: Replace the bundled models and results**

Run: `find models -name '*.pt' -delete && cp out/paper/models/*.pt models/ && cp out/paper/report/* docs/results/ && ls models | wc -l`
Expected: `44`.

- [ ] **Step 4: Add the checks to the README**

In `README.md`, directly above `### Deviations from the paper`, insert:

```markdown
### Reproduction checks (paper preset)

<the six bullet lines printed in Step 2, verbatim>
```

The bullets are command output, pasted exactly as printed.

- [ ] **Step 5: Re-run everything that depends on the bundle**

Run: `uv run pytest -q && docker build -t ghcr.io/ladrerod/deep-hedging:0.1.0 . && docker run --rm -d --name dh -p 127.0.0.1:8000:8000 ghcr.io/ladrerod/deep-hedging:0.1.0 && curl -fsS --retry 30 --retry-delay 1 --retry-all-errors localhost:8000/healthz; docker rm -f dh`
Expected: `53 passed`, then `{"status":"ok","models":44}`.

- [ ] **Step 6: Commit**

```bash
git add models docs/results README.md
git commit -m "feat: bundle paper-preset models and reproduced results"
```

---

### Task 14: Merge, publish and release v0.1.0 (requires explicit user approval)

Stop and ask the user before Step 2. Creating a public repository and pushing are outward-facing.

**Files:** none changed (git and GitHub operations only).

**Interfaces:**
- Consumes: the whole branch.
- Produces:
  - the public `https://github.com/ladrerod/deep-hedging` with `main` and tag `v0.1.0`;
  - images `ghcr.io/ladrerod/deep-hedging:0.1.0` and `:latest` (amd64 + arm64).

- [ ] **Step 1: Whole-branch review and integration**

Use superpowers:finishing-a-development-branch. Run the reviewers listed under "ECC skills and reviewers" on the full diff, fix what they confirm, and run `uv run pytest -q && uv run ruff check .` (expect `53 passed`). Then merge `feat/ntb-v0.1` into `main`.

- [ ] **Step 2: Create the public repository and push** (after the user approves)

Run: `gh repo create ladrerod/deep-hedging --public --source . --remote origin --push --description "No-Transaction Band Network (Imaki et al. 2021): train, reproduce and serve deep hedgers on Docker and Kubernetes"`
Expected: `main` is pushed, and the `ci` workflow starts.

Watch it with `gh run watch`. Expected: `test` and `image` succeed, and the kind job's log shows the three curls and the Job completion.

- [ ] **Step 3: Tag and publish**

Run: `git tag v0.1.0 && git push origin v0.1.0 && gh run watch`
Expected: `publish` succeeds and pushes `ghcr.io/ladrerod/deep-hedging:0.1.0` and `:latest`.

- [ ] **Step 4: Make the package public if needed**

Run: `gh api /users/ladrerod/packages/container/deep-hedging -q .visibility`
Expected: `public`. If it prints `private`, ask the user to set it to public under GitHub → Packages → deep-hedging → Package settings; there is no REST endpoint for this.

- [ ] **Step 5: Verify the stranger's path from the registry**

Run:

```bash
docker image rm ghcr.io/ladrerod/deep-hedging:0.1.0
docker run --rm -d --name dh -p 127.0.0.1:8000:8000 ghcr.io/ladrerod/deep-hedging:0.1.0
curl -fsS --retry 60 --retry-delay 1 --retry-all-errors localhost:8000/healthz; docker rm -f dh
kubectl create namespace dh-release
kubectl -n dh-release apply -f https://raw.githubusercontent.com/ladrerod/deep-hedging/v0.1.0/k8s/api.yaml
kubectl -n dh-release rollout status deployment/deep-hedging --timeout=300s
kubectl delete namespace dh-release
```

Expected: the image pulls anonymously, `{"status":"ok","models":44}` is returned, and the rollout succeeds from the raw manifest URL.

---

## Risks (ECC)

| Risk | Likelihood | Mitigation |
|---|---|---|
| Lookback results differ from parts of Table B4 (by design: §2.6) and readers assume a bug | High | README "Deviations" with evidence; `tables.md` footnotes; the checks list is printed as measured |
| Paper sweep (about 7 h) interrupted or slower on other machines | Medium | Resumable and seed-major; started early in the background (Task 13) |
| kind end-to-end flaky on CI runners (slow image load, pod start) | Medium | Curl retries, generous timeouts, a diagnostics step on failure |
| GHCR package private after the first publish, so quickstarts fail to pull | Medium | Explicit visibility check (Task 14, Step 4) |
| Upstream wheel or action changes break the lock or CI | Low | `uv sync --locked`, pinned action majors, CPU index scoped to Linux |
| QEMU arm64 build slow | Low | Multi-arch build only on tags |

## Complexity (ECC)

Medium. Roughly:
- 4–6 hours of implementation and review for Tasks 1–12;
- about 7 hours of unattended compute for Task 13;
- about 30 minutes for Task 14, plus CI time.

## Validation (ECC)

```bash
uv sync --locked && uv run ruff check . && uv run pytest -q          # 53 passed
docker build -t ghcr.io/ladrerod/deep-hedging:0.1.0 .                # image builds
kubectl apply --dry-run=server -f k8s/api.yaml -f k8s/data.yaml -f k8s/sweep.yaml
kubectl kustomize k8s | kubectl apply --dry-run=server -f -
docker run --rm -v "$PWD:/repo" -w /repo rhysd/actionlint:latest     # workflow lint
```

## Acceptance (ECC)

- [ ] Tasks 1–13 complete, and every step's expected output was observed.
- [ ] `53 passed`, `ruff check` clean, image builds, manifests dry-run clean, actionlint clean.
- [ ] Live Kubernetes path verified under the restricted Pod Security profile (Task 10, Step 5).
- [ ] README reproduction checks come from the paper preset (Task 13).
- [ ] Task 14 performed only after explicit user approval; the stranger's path verified from GHCR.
