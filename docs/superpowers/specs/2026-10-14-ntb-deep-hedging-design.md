# No-Transaction Band Network: train + serve on Docker and Kubernetes — design

- **Date:** 2026-10-14
- **Status:** approved in conversation (approach, Section 1, Section 2, Section 3); written spec pending review
- **Paper:** Imaki, Imajo, Ito, Minami, Nakagawa, "No-Transaction Band Network: A Neural Network Architecture for Efficient Deep Hedging", arXiv:2103.01775v1 (2021)
- **Names:** GitHub repo `ladrerod/deep-hedging`, Python package `deep_hedging`, CLI `deep-hedging`, image `ghcr.io/ladrerod/deep-hedging`

## 1. Goal and non-goals

**Goal.** A public repo that anyone can use with Docker or Kubernetes to:
1. **Train** the paper's no-transaction band network (NTB) and its baselines, and regenerate the paper's figures and tables across the cost sweep (batch: `docker run … sweep`, Kubernetes Job).
2. **Serve** trained hedgers over HTTP: the no-transaction band, the next hedge and the indifference price for a given option and transaction cost (`docker run`, Kubernetes Deployment + Service). It works out of the box with pre-trained models built into the image.

**User decisions.** Train + serve from one image. A clean PyTorch implementation, with no dependency on `pfhedge`. CPU is the target. Approach A ("leanest working design") with the grafts listed in this spec.

**Non-goals (v0.1).** GPU or MPS support. Market models other than driftless GBM. Derivatives other than the European and fixed-strike lookback call. Configurable risk aversion. Multi-node sweeps. Ingress, authentication or TLS. Serving the baseline strategies. PyPI publishing.

**Engineering rule.** Build the minimum that fully delivers the above. There are no speculative abstractions, and each file and Kubernetes object must justify itself. The following are never cut: input validation at the API boundary, protection against data loss, security basics, and one focused test per piece of non-trivial logic.

## 2. Reproduction protocol

Every number in this section was measured during design with throwaway scripts (10M-path Monte Carlo unless noted). Units are the paper's: S0 = 1, K = 1 (European) or 1.03 (lookback), σ = 0.2, λ = 1 on absolute P&L.

### 2.1 Market simulation
- `dt = 1/365`. The number of daily increments is `n = days` (default **28**; see 2.6). The price path has `n+1` points.
- The path is exact driftless GBM: `S[0] = 1`, `S[i+1] = S[i] * exp(σ*sqrt(dt)*z_i - σ²*dt/2)`, `z_i ~ N(0,1)`, all in float32.
- The code works in log-moneyness `s_i = log(S_i/K)`. For the lookback, `m_i = cummax(s)[i]` includes the current point, so `m_i >= s_i` holds bitwise.
- Time to maturity at hedging step `i = 0..n-1` is `tau_i = (n - i) * dt`, which is self-consistent with the path length.

### 2.2 Payoffs, P&L, risk
- European: `Z = max(S_n - K, 0)`. Lookback: `Z = max(max_{0..n} S - K, 0)`, where the maximum runs over all `n+1` points including `S_0`.
- When pricing from a mid-life state (Section 7), a past running maximum `m0` (in log-moneyness) enters both the features and the payoff: `m_i = max(m0, cummax(s)_i)` and the lookback `Z = max(K e^{m_n} - K, 0)`.
- The hedge starts at `h_{-1} = 0`. The policy chooses `h_i` at step `i` for the period `[t_i, t_{i+1}]`.
- P&L: `P = Σ_{i=0}^{n-1} [ h_i (S_{i+1} - S_i) - c * S_i * |h_i - h_{i-1}| ] - Z`. Liquidation at maturity is free.
- Training loss is `mean(exp(-λP))`. Utility is `-mean(exp(-λP))`. Price is `log(mean(exp(-λP)))/λ`; the final reduction is done in float64 (float32 P&L, `.double()` before `exp`).
- The indifference price is `p(c) = π(-Z) - π(0)`, with `π(0) = 0` for a driftless asset. The price spread is `p(c) - p(0)`, where `p(0)` is the zero-cost Black–Scholes-delta-hedge price of the same derivative (paper §5.1).

### 2.3 Strategies (inputs `I_i`, Black–Scholes delta `Δ_i`, gamma `Γ_i`)
- Features: European `[s_i, tau_i, σ]`; lookback `[s_i, m_i, tau_i, σ]`. σ is a constant column.
- MLP: `Linear(in,32)-ReLU` ×4, then `Linear(32,out)`. In float32 this is 3,362 parameters for the European NTB and 3,394 for the lookback NTB.
- **NTB:** `o = leaky_relu(MLP_{in→2}(I_i), 0.01)`; `lo = Δ_i - o[...,0]`, `hi = Δ_i + o[...,1]`; `h_i = clamp(h_{i-1}, lo, hi)`.
  - Clamp rule (paper eq. 13 plus the inverted-band rule): `x = min(max(h, lo), hi); h_i = where(lo < hi, x, (lo+hi)/2)`.
  - `pnl()` runs every policy (NTB, FFN, WW) step by step through `forward()`. `band()` is used only by the API and Figure 3. A batched NTB pass over `(n, N, in)` measured about 24% slower (0.218 vs 0.176 s per iteration), so there is no second code path.
- **FFN:** `h_i = Δ_i + tanh(MLP_{in+1→1}([I_i, h_{i-1}]))`.
- **WW (Whalley–Wilmott asymptotic band):** `w = (1.5 * c * S_i * Γ_i² / λ)^(1/3)`, `h_i = clamp(h_{i-1}, Δ_i - w, Δ_i + w)`. At `c = 0` this reduces to the pure delta hedge.
  - Do **not** clip the band to [0, 1]: the lookback delta reaches 1.046.

### 2.4 Black–Scholes greeks, zero rates
Notation: `a = σ√tau`, `N` the standard normal cdf (`torch.special.ndtr`), `n` its pdf. Each function returns `(price/K, Δ, Γ·K)`. They are evaluated under `torch.no_grad()`, with `tau > 0` always (hedging dates have `tau >= 1 day`).
- **European call:** `d1 = s/a + a/2`; `price/K = e^s N(d1) - N(d1 - a)`; `Δ = N(d1)`; `Γ·K = n(d1)/(e^s a)`.
- **Fixed-strike lookback call, continuous monitoring:**
  - `h = max(m, s, 0)` (in torch: `torch.maximum(m, s).clamp(min=0)`), `d1 = (s - h)/a + a/2`, `g = d1 N(d1) + n(d1)`.
  - `price/K = (e^h - 1) + e^s N(d1) - e^h N(d1 - a) + e^s a g`.
  - `Δ = 2N(d1) + a g`; `Γ·K = (2 n(d1)/a + N(d1)) / e^s`.
  - The price matches pfhedge's formula to 4e-16 and exact continuous-monitoring Monte Carlo at 9 points (|z| ≤ 1.97). Δ and Γ match pathwise Monte Carlo and finite differences.
  - The analytic Γ is used **instead of autograd**. At `S == M > K`, which is 8% of simulated states, autograd through `max` in torch 2.x returns Γ/4.
- **WW in these units:** `Γ = (Γ·K)/K` and `S = K e^s`, so `w = (1.5 * c * e^s * (Γ·K)² / (K λ))^(1/3)`.
- **Test vectors**, float64, columns `S, M, K, σ, tau_days → price, Δ, Γ` (Γ in absolute units):
  `1,1,1.03,0.2,30 → 0.0228074, 0.637942, 12.6734`;
  `1.02,1.05,1.03,0.2,15 → 0.0318061, 0.493096, 15.3960`;
  `1.05,1.05,1.03,0.2,10 → 0.0480230, 1.026689, 23.4338`.
  European Δ at `s ∈ {-0.01, 0, 0.01}`, tau = 0.1, σ = 0.2 is `[0.4497, 0.5126, 0.5752]` (reference doctest).

### 2.5 Training and evaluation
- Adam, lr 1e-3 (a module constant `LR`, not a parameter), default betas. Each iteration draws fresh training paths and takes one gradient step on the loss.
- Seeding: `torch.manual_seed(seed)` for initialisation, and a `torch.Generator(seed)` for training paths.
- Evaluation follows the paper's protocol: 20 batches of evaluation paths, generated from `eval_seed = 10_000 + seed`.
  - Within a seed, all methods and all costs share the same evaluation paths (common random numbers), so spreads are clean.
  - Across seeds the evaluation paths differ, so error bars include Monte Carlo noise, as in the paper.
- Reported price = mean over the 20 batches of the per-batch price; `price_se` = standard deviation over batches / √20. Utility is computed the same way.
- WW has no training; it is evaluated per (cost, seed) on the same evaluation paths.
- One worker process uses one torch thread (`torch.set_num_threads(1)`).

### 2.6 Paper vs reference code: decisions

| Item | Paper text | What we do | Evidence |
|---|---|---|---|
| Number of steps | 30 daily steps to T = 30/365 | **28 increments**, `tau_i = (28-i)/365` | European c=0 delta-hedge price: 28 steps 0.022101 ± 1e-6 vs paper 0.022101; 30 steps 0.022880. Lookback: 28 steps 0.018043 vs paper 0.018040–0.018053; 30 steps 0.019362. The reference code's `int((30/365)/(1/365)) = 29` points gives 28 increments. |
| GBM drift | driftless | exact martingale (`-σ²dt/2` per step) | The reference `utils.py` uses `linspace(0, T, 29)` for the drift, so `E[S_T] ≈ 1 - 1.1e-4`. pfhedge is exact. |
| Lookback greeks | "numerically computed" | closed form (2.4) | pfhedge uses autograd; torch 2.x tie rule gives Γ/4 at S = M. |
| Lookback model | 4 features, centred on the lookback delta | as the paper | The reference notebook's lookback cell reuses the 3-feature European net ("omit the cumulative maximum for simplicity"). |
| Table B3 | lookback utilities | ignored | Byte-for-byte duplicate of Table B1. |
| Table B4 "Asymptotic" column, c > 0 | e.g. 0.018731 at c = e^-7 | not a target | On our grid, exact greeks give 0.018464 ± 7e-6. The authors' own pfhedge at the arXiv date, under torch 1.9.0, gives 0.018445. At high cost the exact band never trades, so the price equals no-hedge (0.018408). |
| Table B2 "Asymptotic" column, c ≥ e^-6 | e.g. 0.024859 at c = e^-5 | not a target at high cost | Our grid's WW minus the paper is +7.6e-5 at e^-6 and +2.88e-4 at e^-5, against paper errors of about 3e-5. The paper's values match only `tau_i = (29-i)/365`, an off-by-one in the time feature. It matches within 2e-5 for c ≤ e^-7. |
| Table B4 NTB at high cost | rises to 0.019124 ± 0.001427 | expect a plateau at about 0.01840 | The no-hedge price is 0.018408 ± 9e-6, so the paper's values are worse than not hedging. Single-seed runs at e^-7 and e^-5 (published-code grid) give 0.018399, i.e. no trading. Once no path trades the gradient is exactly zero (an absorbing state). |
| Learning-history figure | validation loss, 20 simulations per point | training loss, 20-iteration rolling mean | The paper's version would roughly triple training compute. |
| Appendix A (lr 1e-2 and 1e-4) | figure A1 | skipped | Lowest-value output; it would add lr to run identity. |

Daily monitoring: the served lookback price is for a **daily-monitored** option. The continuous-monitoring Black–Scholes value is 18.8% higher. The continuous Δ remains the band centre, as in the paper; the bias costs at most 2.1e-5 in the c=0 price and the network absorbs it.

### 2.7 What counts as "reproduced"
1. **Tested (`test_core.py`, seeded, deterministic):**
   - Zero-cost delta-hedge prices: European `|p - 0.022101| ≤ 3e-5`; lookback `|p - 0.01804| ≤ 6e-5`. Both use 200k paths, and the tolerances are about 3 standard errors.
   - No-hedge prices: European 0.022673, lookback 0.018408, each within 3 standard errors at the test's path count.
2. **Reported (`tables.md`, not gated):** for every derivative and cost, our NTB / FFN / WW prices (mean ± sd over seeds) next to the paper's B2 / B4 values (mean ± sd). Footnotes cover the paper's lookback "Asymptotic" column and its European column at c ≥ e^-6 (see 2.6).
3. **Checked once at implementation time and recorded in the README:** with the paper preset,
   - the European NTB price is ≤ the FFN price at every positive cost, and agrees with Table B2 at costs ≤ 1e-3 within `3·sqrt(sd_ours² + sd_paper²)/√5`. Both sds are over 5 seeds; the paper's ± is a seed sd.
   - the European NTB plateaus at about the no-hedge price (0.02267) for c ≳ 1.5e-3;
   - the lookback NTB ≤ FFN at every positive cost;
   - the NTB's learning curve flattens within about 100 iterations.

## 3. Repository layout

```
README.md                    quickstarts, API, reproduced results, deviations, credits
LICENSE                      standard MIT, Copyright (c) 2026 Daniel Ladrero Meca
pyproject.toml               metadata, runtime deps, dev group, console script, CPU torch index on Linux
uv.lock                      cross-platform lock
Dockerfile                   single-stage CPU image, non-root, entrypoint = CLI
.dockerignore                keeps .git, .venv, out/, tests/, docs/, k8s/ out of the build context
.gitignore                   out/, .venv/, caches
.github/workflows/ci.yml     tests, kind end-to-end with the built image, tag-only multi-arch publish
src/deep_hedging/__init__.py __version__ = importlib.metadata.version("deep-hedging") (single source: pyproject)
src/deep_hedging/core.py     the paper's math (torch only, no I/O)
src/deep_hedging/train.py    one run, presets, atomic save / safe load, resumable parallel sweep
src/deep_hedging/report.py   figures, tables (ours next to the paper's), results.csv, servable model set
src/deep_hedging/api.py      FastAPI app
src/deep_hedging/cli.py      argparse: train | sweep | report | serve
tests/test_core.py           math against closed forms, reference doctests and paper anchors
tests/test_e2e.py            smoke sweep, report, API, bundled models, manifest tags
k8s/api.yaml                 Deployment + Service (stateless, built-in models)
k8s/data.yaml                PersistentVolumeClaim (results live here)
k8s/sweep.yaml               Job (train and reproduce on the cluster)
k8s/kustomization.yaml       api.yaml patched to serve models trained onto the PVC
models/                      GENERATED, committed: best NTB checkpoint per derivative × cost (44 × ~26 KB)
docs/results/                GENERATED, committed: figures, tables.md, results.csv from the paper preset
docs/superpowers/specs/      this design
```

There is no Makefile, compose file, Helm chart, config file, notebook or docs site.

## 4. Python components

Dependency direction: `core ← train ← (report, api)`. `train` never imports `report`. The `sweep` CLI command calls `train.sweep()` and then `report.report()`. `cli` imports modules lazily, so `serve` never imports matplotlib.

**`core.py`** (torch only):
- `STRIKE = {"european": 1.0, "lookback": 1.03}`, `DT = 1/365`.
- `european(s, tau, sigma) -> (price_k, delta, gamma_k)` and `lookback(s, m, tau, sigma) -> (price_k, delta, gamma_k)`, per 2.4.
- `greeks(derivative, s, m, tau, sigma)` dispatches to the above. `m` is ignored for the European.
- `clamp(x, lo, hi)`, per 2.3.
- `class Hedger(nn.Module)`, `Hedger(derivative, method)` with method in `{"ntb", "ffn"}`:
  - `band(s, m, tau, sigma) -> (lo, hi)`, NTB only;
  - `forward(s, m, tau, sigma, prev) -> next_hedge`.
- `whalley_wilmott(derivative, cost)` returns a policy with the same `forward` signature.
- `simulate(n_paths, days, sigma, s0, generator) -> s` of shape `(days+1, n_paths)`, in log-moneyness.
- `pnl(policy, derivative, s, cost, sigma, prev0=0.0, m0=None) -> (n_paths,)`, per 2.2, in the model's units (K = `STRIKE[derivative]`).
- `price(pnl, lam=1.0)` (float64 reduction) and `utility(pnl, lam=1.0)`.

**`train.py`** (torch, stdlib):
- `COSTS = [0.0] + [exp(-10 + 0.25 i) for i in range(21)]`.
- `PRESETS = {"smoke": …, "quick": …, "paper": …}`, one dict (see 13).
- `run(derivative, method, cost, seed, iters, paths, eval_batches, eval_paths, days=28, sigma=0.2, preset="custom") -> dict`. Method `"ww"` skips training and only evaluates. Returns the checkpoint dict (see 6).
- `run_name(cfg) -> str`, e.g. `european-ntb-c9.1188e-04-sig0.2-d28-i1000-p50000-s0`.
- `save(ckpt, path)` writes `path.tmp` then `os.replace`. `load(path)` uses `torch.load(path, weights_only=True)`.
- `policy(ckpt)` rebuilds a `Hedger` or WW policy from a checkpoint.
- `sweep(preset, out, workers)`:
  - expands the preset into tasks in **seed-major** order, so an interrupted paper sweep still yields complete single-seed results;
  - skips tasks whose file already exists, which makes the sweep resumable;
  - runs the rest in a `spawn` `ProcessPoolExecutor` with one torch thread per worker (in-process when `workers == 1`);
  - if a worker dies (`BrokenProcessPool`), it exits non-zero with: "a worker died (out of memory?); rerun with fewer --workers; finished runs are kept". Under Kubernetes with cgroup v2 an OOM kills the whole container instead, which shows as pod reason `OOMKilled`.
- SIGTERM: `cli.main` installs `signal.signal(SIGTERM, lambda *_: os._exit(143))` before train, sweep and report, so pod deletion and `docker stop` do not wait out the grace period. Atomic writes make a kill lose only in-flight runs.

**`report.py`** (matplotlib Agg, stdlib csv):
- `PAPER_PRICES`: Table B2 (European) and B4 (lookback) prices per cost and method as `(mean, sd)` pairs, a literal dict of 264 numbers. B3 is omitted.
- `report(run_dir)` reads `run_dir/runs/*.pt` and writes `run_dir/report/` and `run_dir/models/`, then prints `tables.md`. It first deletes `run_dir/models/*.pt`, so stale picks never linger. The report contents:
  - `fig2_european.png`, `fig5_lookback.png`: utility vs cost (log x) and price spread vs cost (log-log), with error bars = sd over seeds.
  - `fig3_band.png`: the European NTB band vs log-moneyness ∈ [-0.1, 0.1] at tau = 15 days, for the smallest positive and the largest cost, with the BS delta and the WW band.
  - `fig4_european_history.png`, `fig6_lookback_history.png`: training loss (20-iteration rolling mean) at c = e^-7.5 and e^-5, NTB vs FFN, with WW as a flat line.
  - `tables.md`: per derivative, `cost | NTB | paper NTB | FFN | paper FFN | WW | paper asym.`, plus footnotes.
  - `results.csv`: one row per run.
  - `models/`: a byte-for-byte copy of the lowest-price NTB checkpoint per (derivative, sigma, days, cost). Because the pick is the best of 5 seeds, its recorded price is optimistic by about 1 `price_se`; the README says so.
- `report` works on partial run directories, skipping figures whose runs are missing.

**`api.py`** (fastapi, pydantic, torch): `create_app(models_dir) -> FastAPI` (see 7).

**`cli.py`** (argparse, uvicorn): `main()` with four subcommands (see 5).

Runtime dependencies: `torch`, `fastapi`, `uvicorn`, `matplotlib`. Development dependencies: `pytest`, `ruff`, and the HTTP client Starlette's `TestClient` currently requires (verify at implementation time). Python ≥ 3.12.

## 5. CLI

The output root is `--out`, default `$DEEP_HEDGING_OUT`, else `./out`. The image sets it to `/data`.

```
deep-hedging train --derivative {european,lookback} --cost C [--method {ntb,ffn,ww}=ntb] [--seed 0]
                   [--iters 1000] [--paths 50000] [--days 28] [--sigma 0.2] [--out OUT]
    one run -> OUT/custom/runs/<run_name>.pt (overwrites a file of the same name);
    evaluates with 20 batches x --paths (paper protocol); prints price, utility, seconds
deep-hedging sweep [--preset {smoke,quick,paper}=quick] [--workers N=min(4, os.cpu_count())] [--out OUT]
    all tasks of the preset -> OUT/<preset>/runs/, then report -> OUT/<preset>/{report,models}/
deep-hedging report DIR
    re-render DIR/report and DIR/models from DIR/runs
deep-hedging serve [--models DIR=$DEEP_HEDGING_MODELS or ./models] [--host $DEEP_HEDGING_HOST or 127.0.0.1] [--port 8000]
    start the API; writes nothing
```

The image sets `DEEP_HEDGING_HOST=0.0.0.0`, so `serve` inside a container listens on all interfaces. This holds even when Kubernetes `args` or `docker run IMAGE serve …` replace the default command. Outside a container the default stays loopback.

## 6. Artifacts and data flow

**Checkpoint** (one self-contained file per run, `torch.save`, written atomically):
```
{"config":  {preset, derivative, method, cost, sigma, days, strike, lr, iters, paths, eval_batches, eval_paths, seed, eval_seed},
 "metrics": {price, price_se, utility, utility_se, seconds},
 "history": [training loss per iteration],          # empty for ww
 "state_dict": {...},                               # empty for ww
 "meta":    {"version": str, "torch": str(torch.__version__), "created": ISO-8601 str}}
```
- Load it only with `torch.load(weights_only=True)`, which refuses arbitrary code.
- `torch.__version__` must be stored as `str()`; otherwise the safe loader rejects the file.

**Directory layout:** `OUT/<preset>/runs/` (all runs), `OUT/<preset>/report/`, `OUT/<preset>/models/`. Each preset has its own directory, so quick and paper results never mix. Custom `train` runs go to `OUT/custom/runs/`.

**Built-in models:**
- The maintainer runs `deep-hedging sweep --preset paper` once per release, about 7 hours on an M5 Pro. They then run `find models -name '*.pt' -delete && cp out/paper/models/*.pt models/` and `cp out/paper/report/* docs/results/`, and commit. `find` is used because zsh aborts `rm models/*.pt` when nothing matches.
- The Dockerfile copies `models/` to `/app/models`.
- During initial implementation the quick preset's models stand in until the paper sweep finishes. `/v1/models` exposes which preset produced each model.

**Honesty of bundled models:**
- Each file carries its config, seeds, metrics and versions, and `docs/results/results.csv` lists every run.
- `test_e2e` loads every bundled checkpoint with `weights_only=True`.
- It re-evaluates one checkpoint with its own `eval_seed`, `eval_batches` and `eval_paths`, which must reproduce its recorded price within 0.5 `price_se`.
- It checks the request-to-model unit mapping through `/v1/price` (Section 11).
- Reproducible means within Monte Carlo tolerance, not bit-identical across CPUs or torch versions.

**Docker flow:**
- Serve built-in models: `docker run -p 127.0.0.1:8000:8000 ghcr.io/ladrerod/deep-hedging`. The API has no authentication, so the README publishes it on loopback only.
- Train:
  ```
  mkdir -p out && docker run --rm -u "$(id -u):$(id -g)" -v "$PWD/out:/data" ghcr.io/ladrerod/deep-hedging sweep --preset quick
  ```
- Serve what you trained:
  ```
  docker run --rm -p 127.0.0.1:8000:8000 -v "$PWD/out:/data:ro" ghcr.io/ladrerod/deep-hedging serve --models /data/quick/models
  ```

**Kubernetes flow:**
- `sweep.yaml`'s Job writes to the PVC at `/data/<preset>/`.
- `kubectl apply -k k8s` re-applies the Deployment with the PVC mounted read-only and `DEEP_HEDGING_MODELS=/data/quick/models`, which rolls the pod onto the trained models.
- Figures are copied out with `kubectl exec deploy/deep-hedging -- tar -C /data/quick -cf - report | tar -xf -`.

## 7. HTTP API

FastAPI + uvicorn, one process, `torch.set_num_threads(1)`. Interactive docs are at `/docs`.

| Method and path | Purpose |
|---|---|
| `GET /` | 307 redirect to `/docs` |
| `GET /healthz` | `async def`, so it never queues behind pricing: `{"status":"ok","models":44}` |
| `GET /v1/models` | each model: `derivative, cost, volatility, days, trained_strike, preset, price, price_se, utility` |
| `POST /v1/hedge` | band, delta, next hedge (one forward pass, microseconds) |
| `POST /v1/price` | Monte Carlo price from the requested state (about 0.1–0.2 s) |

**Handlers.**
- `/v1/hedge` and `/v1/price` are plain `def` handlers, so they run in FastAPI's threadpool and never block the event loop.
- Only `/v1/price` takes the pricing lock.
- `/healthz` is the only `async def`, so probes are answered even while every pricing thread is busy.

**Request body** (both POST endpoints). The pydantic model uses `ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)`: unknown fields, booleans and numeric strings are rejected, and a JSON integer is accepted for a float field.
```json
{"derivative": "european", "spot": 101.5, "strike": 100, "days_to_maturity": 15,
 "volatility": 0.2, "cost": 0.0001, "current_hedge": 0.3}
```
- `derivative ∈ {european, lookback}`; `spot > 0`; `strike > 0`; `days_to_maturity` is an integer in `[1, model days]`; `volatility > 0`; `0 ≤ cost < 1`.
- `current_hedge` is finite with `|current_hedge| ≤ 10` (default 0). Larger values overflow `exp(-λP)` through the cost term.
- Lookback only: `max_spot` (running maximum over daily observations including today; default = `spot`; must be `≥ spot`). Sending `max_spot` with `derivative = european` is a 422.

**`/v1/hedge` response:**
```json
{"bs_delta": 0.6508, "band": {"lower": 0.42, "upper": 0.88}, "next_hedge": 0.42, "trade": 0.12,
 "model": {"derivative": "european", "cost": 9.6e-05, "volatility": 0.2, "days": 28, "preset": "paper"}}
```
- Features use `s = log(spot) - log(strike)` (two logs, so extreme ratios cannot underflow to `log 0`), `m = max(log(max_spot) - log(strike), s0)` and `tau = days_to_maturity/365`. Here `s0 = log(1/K_train)` is the training start: 0 for the European, `-log 1.03` for the lookback.
- Raising a running maximum that lies below the training start up to it cannot change the payoff, because both values are below the strike.
- If the network returns `lower > upper`, both are reported as the midpoint.
- `next_hedge = min(max(current_hedge, lower), upper)` is computed with **Python floats**, so a hedge inside the band yields `trade == 0.0` exactly. (A float32 clamp produced a spurious 3e-9 trade.)

**`/v1/price` response:** `{"price": 2.53, "std_error": 0.01, "paths": 20000, "model": {...}}` (illustrative values for the request above, whose zero-cost Black–Scholes value is 2.488).
- The server simulates 20,000 paths with a fixed seed from the requested state `(s, m)` for `days_to_maturity` steps. It uses the same `pnl()` as training, the selected model's policy, the **requested** cost, `current_hedge` as `prev0` and `m` as `m0`, all in the model's units (`K_train`).
- **Control variate:** `X = exp(-λP)`, `Y = Σ_i Δ_i (S_{i+1} - S_i)`. `E[Y] = 0` exactly, because the simulated path is a discrete martingale and Δ_i is known at step i. Then `β = cov(X,Y)/var(Y)`, forced to 0 if `var(Y) < 1e-12`. The estimate is `Ê = mean(X) - β mean(Y)`, `price_model = log(Ê)/λ` and `se = sd(X - βY)/(√n · λ · Ê)`.
  - Measured over 100 × 20k paths: no bias, 8× lower variance when the model has learned not to hedge, 1.6× for a WW-like band.
- Returned price = `price_model × strike / K_train`; the standard error is scaled the same way.
- Equivalently, λ = 1 applies per unit of `strike/K_train`, i.e. `λ_abs = K_train/strike`: 1/strike for the European and 1.03/strike for the lookback. The bands are scale-free.
- With `current_hedge = 0` on a new option this is the paper's indifference price. Otherwise it is the certainty-equivalent cost of the remaining hedged liability from the current position.
- Requests are serialised by one `threading.Lock` per process, which bounds memory and CPU; scale with replicas. Admission is bounded too: at most `PRICE_SLOTS = 8` pricing requests wait or run per process, and the next one gets 503 with `Retry-After: 1`, so a backlog cannot occupy the threadpool that `/v1/hedge` and `/v1/models` share.

**Model selection and domain checks** (errors return JSON `{"detail": "..."}` with the reason):
- No model for `(derivative, volatility)`, matched within 1e-9 relative: **404**, listing the available volatilities and the `train` command.
- `days_to_maturity > model days`: **422**.
- **Cost:**
  - `cost == 0` uses the zero-cost model only.
  - `cost > 0` uses the nearest positive trained cost on a log scale, accepted only if `|ln cost - ln c_model| ≤ tol`.
  - `tol` is computed per (derivative, volatility) group: `max(0.125, half the largest gap between adjacent positive trained log-costs)`, or 0.125 with fewer than two positive costs. That gives 0.125 for the paper and smoke grids and 0.625 for the quick grid.
  - Otherwise **422**, listing the trained costs.
  - The response always names the model's cost; the price is still simulated at the requested cost.
- **Moneyness:**
  - Require `|s - s0| ≤ 4σ√((e+1)/365)`, where `e = model days - days_to_maturity` is the elapsed days. That gives ±0.042 for a new 28-day option and ±0.157 at 15 days left.
  - For the lookback, the same bound applies to `m - s0`.
  - Otherwise **422**, stating the accepted spot range for that strike and maturity.
- **Validation errors:** a custom `RequestValidationError` handler returns 422 with only `loc` and `msg`. FastAPI's default handler echoes the input and returns **500** for a bare `NaN` or `Infinity` in the JSON body.

**Startup:**
- Load every `*.pt` in the models directory with `weights_only=True` and keep `method == "ntb"`.
- If several files share a key `(derivative, sigma, days, cost)`, keep the lowest recorded price.
- If one (derivative, sigma) group has more than one `days` value, exit non-zero with a clear message, because "model days" must be unambiguous.
- If no models are found, exit non-zero with a clear message.

**Not included:** authentication, rate limiting and a body-size cap. The Service is ClusterIP only, and the README says to add these before exposing it outside a cluster.

## 8. Docker

```dockerfile
# syntax=docker/dockerfile:1
FROM python:3.12-slim
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_NO_DEV=1 PATH=/app/.venv/bin:$PATH \
    PYTHONUNBUFFERED=1 MPLCONFIGDIR=/tmp \
    DEEP_HEDGING_MODELS=/app/models DEEP_HEDGING_OUT=/data DEEP_HEDGING_HOST=0.0.0.0
WORKDIR /app
RUN --mount=from=ghcr.io/astral-sh/uv:<pinned>,source=/uv,target=/bin/uv \
    --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-install-project
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN --mount=from=ghcr.io/astral-sh/uv:<pinned>,source=/uv,target=/bin/uv \
    --mount=type=cache,target=/root/.cache/uv uv sync --locked --no-editable
COPY models ./models
RUN mkdir /data && chown 10001:10001 /data
USER 10001:10001
EXPOSE 8000
ENTRYPOINT ["deep-hedging"]
CMD ["serve"]
```
- `<pinned>` is the newest `uv` image tag verified at implementation time.
- The OCI source label is set by `docker/metadata-action` on published images (Section 10), which links the GHCR package to the repo.
- **CPU-only torch:** `[tool.uv.sources] torch = [{index = "pytorch-cpu", marker = "sys_platform == 'linux'"}]` with `[[tool.uv.index]] name = "pytorch-cpu", url = "https://download.pytorch.org/whl/cpu", explicit = true`. Linux amd64 and arm64 get `+cpu` wheels with no CUDA; macOS gets the PyPI wheel.
- **Size:** measured at 1.46 GB on disk (arm64 build of the draft), about 300 MB compressed. The image is multi-arch (`linux/amd64`, `linux/arm64`); nothing compiles.
- **User:** numeric uid/gid 10001, so `runAsNonRoot` can verify it. The image works with a read-only root filesystem plus a writable `/tmp`.
- **Linux bind mounts:** they need `-u "$(id -u):$(id -g)"` and an existing host folder; the README states this.

## 9. Kubernetes

Raw manifests plus one Kustomize file, deployed with `kubectl` only. Image: `ghcr.io/ladrerod/deep-hedging:<version>`, where `<version>` equals the pyproject version without a `v` (enforced by a test). With a pinned tag the pull policy is `IfNotPresent`.

**Labels:** every object carries `app.kubernetes.io/name: deep-hedging`.
- API pods also carry `app.kubernetes.io/component: api`, and the Deployment and Service selectors are `{app.kubernetes.io/name: deep-hedging, app.kubernetes.io/component: api}`.
- Sweep pods carry `app.kubernetes.io/component: sweep`.
- Without the component label the Service would send traffic to the running sweep pod. A Deployment's selector is immutable, so this must be right from v0.1.

1. **`k8s/api.yaml`: Deployment `deep-hedging`.**
   - replicas 1; args `["serve"]` (the host comes from the image's `DEEP_HEDGING_HOST`); port 8000.
   - Probes on `/healthz`: `startupProbe` (period 2 s × 30), `readinessProbe` (period 5 s) and `livenessProbe` (period 10 s, timeout 2 s).
   - Requests cpu 250m / memory 512Mi; memory limit 1Gi.
   - Pod: `automountServiceAccountToken: false`, `runAsNonRoot`, uid/gid/fsGroup 10001, `seccompProfile: RuntimeDefault`.
   - Container: `readOnlyRootFilesystem`, `allowPrivilegeEscalation: false`, `capabilities.drop: [ALL]`; emptyDir at `/tmp`.
   - It serves the built-in models and needs no storage.
2. **`k8s/api.yaml`: Service `deep-hedging`.** ClusterIP, port 80 → 8000.
3. **`k8s/data.yaml`: PVC `deep-hedging-data`.** ReadWriteOnce, 1Gi, default StorageClass. It sits in its own file so that deleting or re-applying the Job never deletes results.
4. **`k8s/sweep.yaml`: Job `deep-hedging-sweep`.**
   - One pod; args `["sweep", "--preset=quick", "--workers=2"]`. Output goes to `/data` via the image's `DEEP_HEDGING_OUT`.
   - Requests cpu 500m / memory 1Gi; limits cpu 2 / memory 3Gi. The requests fit a default 2-CPU minikube next to the API pod.
   - `restartPolicy: Never`, `backoffLimit: 3`. Each retry is a new pod on the same PVC that resumes the sweep, and failed pods keep their logs and `OOMKilled` status.
   - Same security context (fsGroup 10001). PVC at `/data`, emptyDir at `/tmp`.
   - Pod label `deep-hedging/volume: data` and a required `podAffinity` to that label on `kubernetes.io/hostname`.
   - Paper-preset values sit in comments: `--preset=paper --workers=4`, limits cpu 4 / memory 6Gi (4 × 1.1 GB plus the parent).
5. **`k8s/kustomization.yaml`:** `resources: [api.yaml]` plus one patch on the Deployment. The patch adds the PVC (mounted read-only at `/data`), env `DEEP_HEDGING_MODELS=/data/quick/models` (paper path in a comment), and the same volume label and affinity.

**Storage reasoning:**
- ReadWriteOnce allows several pods on one node. Every pod that mounts the claim carries the label and the required pod affinity, so all of them land on the volume's node. The scheduler's self-affinity rule lets the first such pod go anywhere, and finished Job pods no longer count.
- Multi-node parallel writers would need ReadWriteMany, which the default local-path, kind and k3s provisioners and default EKS/GKE classes do not offer.
- On a cluster with ReadWriteMany, the user can change `accessModes` and drop the affinity blocks; the README says so.

**Newcomer path** (local cluster: OrbStack, kind, minikube or Docker Desktop):
```
kubectl apply -f https://raw.githubusercontent.com/ladrerod/deep-hedging/v<version>/k8s/api.yaml
kubectl rollout status deployment/deep-hedging
kubectl port-forward service/deep-hedging 8000:80 &
sleep 2
curl -s localhost:8000/v1/hedge -H 'content-type: application/json' \
  -d '{"derivative":"european","spot":100,"strike":100,"days_to_maturity":20,"volatility":0.2,"cost":0.0005}'
# train on the cluster, then serve what was trained (needs a clone of the same release for -k):
git clone --depth 1 --branch v<version> https://github.com/ladrerod/deep-hedging && cd deep-hedging
kubectl apply -f k8s/data.yaml -f k8s/sweep.yaml
kubectl wait --for=condition=complete job/deep-hedging-sweep --timeout=60m
kubectl logs job/deep-hedging-sweep | tail -40
kubectl apply -k k8s && kubectl rollout status deployment/deep-hedging
kubectl port-forward service/deep-hedging 8000:80 &   # restart: the old one died with its pod
```
- **Locally built images:** tag the image exactly as in the manifests. OrbStack's cluster uses the Docker engine's image store (verified: node runtime `docker://29.8.2`). kind needs `kind load docker-image <tag>`; minikube needs `minikube image load <tag>`.
- **Re-run:** `kubectl delete job deep-hedging-sweep && kubectl apply -f k8s/sweep.yaml`. Job templates are immutable, and the results stay on the PVC.
- **Back to the built-in models:** `kubectl apply -f k8s/api.yaml`.

## 10. CI/CD

One workflow, `.github/workflows/ci.yml`. Action major versions are checked at implementation time.

- **`test`** (pull requests and pushes): `uv sync --locked`; `uv run ruff check .`; `uv run pytest -q`.
- **`image`** (needs `test`; `permissions: contents: read`):
  1. Build for the runner architecture with `load: true` and tag `:ci`.
  2. kind end-to-end, using the kind already installed on `ubuntu-24.04` runners:
     - `kind create cluster`, then `kind load docker-image`;
     - rewrite the image to `:ci` in `k8s/api.yaml` and `k8s/sweep.yaml`, and `quick` → `smoke` in `k8s/sweep.yaml` and `k8s/kustomization.yaml` (`sed -i`), so the API serves `/data/smoke/models`;
     - apply `api.yaml` and wait for the rollout; port-forward; curl `/healthz`, one `/v1/hedge` and one `/v1/price` against the built-in models;
     - apply `data.yaml` + `sweep.yaml` and wait for the Job to complete;
     - `kubectl apply -k k8s` and wait; restart the port-forward; curl `/v1/hedge` with cost 0.0005, which is within the smoke grid's tolerance;
     - every curl uses `curl -fsS --retry 30 --retry-delay 1 --retry-all-errors`, because the pod is still loading torch when the port opens.
     - This exercises the non-root user, the read-only root filesystem, the PVC, the probes, the selectors and the trained-model handoff.
     - A separate `docker run` smoke step would be redundant with this.
- **`publish`** (tags `v*` only; needs `image`; `permissions: contents: read, packages: write`):
  - Guard: `v$(pyproject version) == $GITHUB_REF_NAME`.
  - Log in to `ghcr.io` with `GITHUB_TOKEN`.
  - `docker/metadata-action` with `tags: type=semver,pattern={{version}}` gives `0.1.0` (matching the manifests); `latest` comes from the default `flavor: latest=auto`. It also sets the OCI labels, including `org.opencontainers.image.source`.
  - Build and push `linux/amd64,linux/arm64`.
- **Release by hand:**
  - Bump the version in `pyproject.toml` and the image tag in `k8s/api.yaml` and `k8s/sweep.yaml` (a test enforces equality).
  - Commit, tag `vX.Y.Z`, push.
  - After the first publish, check the GHCR package's visibility and set it to public if needed. GitHub's docs conflict on the default, so this is a check, not an assumption.

## 11. Testing

Two files, pytest, about 30 seconds in total.

**`tests/test_core.py`** (no training):
- European greeks: reference doctest deltas `[0.4497, 0.5126, 0.5752]`; ATM price for 28/365 is 0.0220962.
- Lookback greeks: the test vectors in 2.4; analytic Δ and Γ equal autograd derivatives of the closed-form price away from `S == M`, for running maximum both below and above the strike.
- Clamp: reference doctest values, including the inverted band (`[1.0, 0.0]`, lower `[0.0, 1.0]`, upper `0.0` → `[0.0, 0.5]`).
- P&L on a hand-written 3-step path:
  - a zero hedge gives `-Z`;
  - a constant hedge of 1 gives `S_n - S_0 - c·S_0 - Z`;
  - cost is charged only on changes.
- Entropic: `P = -[0,1,2,3]` gives `mean(exp(-P)) = 7.7982` and price 2.0539.
- WW half-width equals `(1.5 c S Γ²)^(1/3)` for a hand value, and is 0 at zero cost.
- Paper anchors (seeded, 200k paths), per 2.7:
  - European zero-cost delta-hedge price 0.022101 (±3e-5) and no-hedge price 0.022673;
  - lookback zero-cost price 0.01804 (±6e-5) and no-hedge price 0.018408.
- `simulate` uses the exact drift: with a seeded generator, its output equals `s0 + cumsum(σ√dt·z - σ²dt/2)` recomputed from the same draws, to 1e-6. A Monte Carlo mean test could not detect the reference code's 1.1e-4 drift at 200k paths.

**`tests/test_e2e.py`:**
- **Smoke sweep** into a temp dir (workers=1):
  - the expected run files exist and load with `weights_only=True`, and the losses are finite;
  - a second sweep does no work;
  - `report` writes non-empty PNGs, `tables.md` and `models/`.
- **Short training:** 30 iterations × 5k paths, European NTB at c = 1e-3. The mean of the last 5 losses must be below the mean of the first 5.
- **API over the smoke models**, with the test client:
  - `/healthz` and `/v1/models`;
  - `/v1/hedge`: `next_hedge` lies within the band, and `trade == 0.0` exactly when `current_hedge` is inside the band;
  - `/v1/price`: finite, positive, with `std_error`;
  - the control-variate price agrees with the plain estimator within 3 combined standard errors on 200k paths.
- **Error codes:**
  - 422 for: negative spot; bare `NaN` and `Infinity` in the raw JSON body; an unknown field; `days_to_maturity` of `true` or `"15"`; days 0 and 29; a cost outside the trained range; moneyness out of range; `max_spot < spot`; `max_spot` on a European request; `|current_hedge| > 10`.
  - 404 for an unknown volatility.
- **Bundled models:**
  - each checkpoint in `models/` loads with `weights_only=True`;
  - one checkpoint re-evaluated with its own `eval_seed`, `eval_batches` and `eval_paths` reproduces its recorded price within 0.5 `price_se`;
  - **unit mapping:** `POST /v1/price` for the bundled lookback model at inception (spot 100, strike 103, days = model days, cost = model cost, hedge 0) lies within 4 combined standard errors of 100 × its recorded price. This catches a wrong `s0`, `m` floor or strike scaling, such as the 3% lookback unit trap.
- **Manifest tags:** every image tag in `k8s/*.yaml` equals the pyproject version.

**Live verification during implementation** (not CI): `docker build`; the Docker smoke test; the full Kubernetes newcomer path on the local OrbStack cluster using the locally built image.

## 12. Docs, license, attribution

**README.md**, in this order:
1. Title, one sentence, CI badge, the reproduced European price-spread figure.
2. What it is: an independent re-implementation of Imaki et al. (arXiv:2103.01775), not affiliated with the authors or Preferred Networks; research code, not investment advice.
3. Quickstart, Docker.
4. Quickstart, Kubernetes, including locally built images on OrbStack, kind and minikube.
5. API: endpoints, examples, validity limits (σ, horizon, moneyness, cost range), units and the fixed λ, link to `/docs`.
6. Reproducing the paper: presets with measured runtimes, embedded figures, `tables.md` with the paper's columns.
7. Deviations from the paper (2.6), stated plainly.
8. Train your own model.
9. Development and release, including the GHCR visibility check and how `models/` and `docs/results/` are produced.
10. Limitations, and what is deliberately not included.
11. Credits and citation: the paper's BibTeX. "`pfnet-research/NoTransactionBandNetwork` and `pfnet-research/pfhedge` (MIT, © 2021 pfnet-research) were consulted for conventions and test values; no code from them is included."

**License:** standard MIT, Copyright (c) 2026 Daniel Ladrero Meca. The paper's PDF and figures are not redistributed; our figures are our own output. The README also states that served models are the best of 5 seeds, so their recorded prices are optimistic by about one standard error.

## 13. Compute plan and presets

Measured on an idle Apple M5 Pro (18 cores):
- One process, one torch thread: about 0.17 s per training iteration at 50k paths × 28 steps (NTB 0.176, FFN 0.158).
- Throughput saturates at about 18 iterations/s with 4 worker processes. The work is memory-bandwidth bound: 9, 12 and 18 workers give no more.
- Memory is about 1.1 GB per worker at 50k paths.
- MPS runs one job about 2.3× faster but gives no more throughput than 4 CPU workers, so it is not used.
- One 50k-path evaluation batch takes about 0.07 s.

| Preset | Costs | Seeds | Methods | Iterations × paths | Evaluation | Runs | Time (M5 Pro) |
|---|---|---|---|---|---|---|---|
| smoke | `0, e^-7.5` | 0 | ntb, ffn, ww | 5 × 1,000 | 2 × 1,000 | 12 | about 1 s |
| quick (default) | `0, e^-10, e^-8.75, e^-7.5, e^-6.25, e^-5` | 0 | ntb, ffn, ww | 300 × 10,000 | 20 × 10,000 | 36 | 77 s with 4 workers (measured) |
| paper | all 22 | 0–4 | ntb, ffn, ww | 1,000 × 50,000 | 20 × 50,000 | 660 (440 trained) | about 7 h with 4 workers |

- **Quick-preset check** at implementation time: at every positive cost and for both derivatives, the seed-0 NTB price must be ≤ the FFN price + `2·sqrt(price_se_ntb² + price_se_ffn²)`. The design-time prototype already showed this for the European option over 5 seeds at 300 × 10k. If the check fails, raise to 500 × 20,000 and re-check, then record the final values here and in the README.
- Default `--workers = min(4, os.cpu_count())`, since more did not help here. Inside containers `cpu_count` reports host cores, which is another reason for the cap.

## 14. Deliberately skipped (and when to add each)

- **Multi-node sweep** (Indexed Job, `--shard i/n`, ReadWriteMany): add when one node is too slow and the cluster has RWX storage.
- **Configurable risk aversion λ:** add first if users find the bands too wide. It touches the loss, the price, WW and model identity.
- **Helm chart, Kustomize overlays:** add when a second environment's values must live in git.
- **Ingress, TLS, auth, rate limiting, body cap, NetworkPolicy, autoscaling:** add before exposing the Service outside the cluster.
- **Validation-loss learning curves, appendix learning-rate study:** add for full figure fidelity, at about 3× compute.
- **Interpolation between cost models, cost-conditional networks:** add if users need arbitrary costs. Today: nearest trained cost within half a grid step.
- **Serving FFN or WW:** add only for comparison demos; the product serves the paper's method.
- **Hot reload of models:** restart the Deployment instead.
- **GPU/MPS, other dynamics, drift, other derivatives, other cost models:** future work, as listed by the paper's conclusion.
- **Slim serving image (ONNX/numpy):** add if the roughly 1.1 GB image becomes a problem.
- **Makefile, compose, pre-commit, Dependabot, image signing/SBOM, SHA-pinned actions, metrics/tracing, PyPI, notebook, docs site:** add on demand.

## 15. Risks

- **Lookback results differ from parts of Table B4:** the asymptotic column, and NTB at high cost. This is documented with evidence; readers may still read it as our bug.
- **λ = 1 makes the bands wide,** and the European NTB stops hedging above about 0.15% cost. The API will often answer "do not trade", which is faithful but surprising, so the README explains it.
- **Narrow trained domain** (σ = 0.2, ≤ 28 days, a moneyness cone around the training start). The API rejects requests outside it rather than extrapolating.
- **The paper sweep takes about 7 h** on a fast laptop and runs on one node only. Small cloud nodes will be slower.
- **GHCR visibility after the first publish** must be checked, or the quickstarts fail to pull.
- **Not bit-reproducible** across CPU architectures and torch versions; the claims are within Monte Carlo tolerance.
- **Linux bind mounts need `-u`** for the non-root container to write.
- **QEMU arm64 builds may be slow** in CI; they run only on tags.

## 16. Publishing

Creating the public GitHub repo, pushing, and tagging the first release are outward-facing. They happen only after explicit approval at the end of implementation.
