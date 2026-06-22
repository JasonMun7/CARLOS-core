# CARLOS Domain Glossary

Reference: [Continuous-time Optimal Stopping through Deep Reinforcement Learning (CARLOS)](https://arxiv.org/pdf/2606.17545)

## Benchmark

**Benchmark** — a reproducible evaluation run with fixed contract parameters, training hyperparameters, and an acceptance target. Passing means the validation price meets the target within a stated tolerance.

**Benchmark run** — an explicit, scored invocation of the full pipeline (not a smoke test). Produces pass/fail against the preset target.

_Avoid_: Using `train --dev` or ad-hoc scripts as the official benchmark run.

## B1

**B1** — the first paper benchmark: a 1D arithmetic basket put (Table 2 parameters) trained with Table 6 path counts. Acceptance target: validation price **4.592** on the finest exercise grid, within ±0.05.

**Benchmark protocol:** training seed **0**; scoring uses a **fixed validation path bank** so the reported price is comparable across runs with the same training seed.

_Avoid_: Using `--dev` runs or coarse-grid prices as the B1 score.

## Smoke Test

**Smoke test** — a fast pipeline run with reduced path counts (`dev_mode`) to verify wiring and convergence behavior. Not scored against B1.

_Avoid_: Calling smoke tests "B1," "benchmark," or comparing their price to 4.592.

## Contract Parameters

**Contract parameters** — Table 2 option definition: asset dynamics, payoff, horizon, and initial state. Independent of how CARLOS is trained.

_Avoid_: Benchmark, config (when you mean the option itself).

## Algorithm Hyperparameters

**Algorithm hyperparameters** — Table 6 CARLOS training settings: path counts, mixture weights, grid schedule, learning rate, etc. Shared across contracts unless a benchmark specifies otherwise.

## Benchmark Preset

**Benchmark preset** — a named bundle of contract parameters, algorithm hyperparameters, and an acceptance target. **B1** is the first preset.

_Avoid_: Using the benchmark name as the name for the general configuration type.

## Validation Price

**Validation price** — forward Monte Carlo estimate of option value under the learned stopping rule (Eq. 11). For benchmark scoring, use the finest exercise grid only.

**Validation path bank** — the fixed set of simulated paths used to compute validation price and to monitor grid saturation. One bank per benchmark run, derived deterministically from the training seed.

_Avoid_: Bermudan price, LSMC price (reference checks, not the benchmark score). Simulating a fresh path bank per scoring call.

## Timing Value

**Timing value** `𝒯(t, x)` — difference between continuation value and immediate payoff (Eq. 5):

`𝒯(t, x) = q̃(t, x) - h(x)`

The ADNN `R^Θ(t, x)` approximates this quantity. Stop when `R^Θ ≤ 0` and `h(x) > 0`.

## ADNN

**Aggregate Deep Neural Network (ADNN)** — single feedforward network `R^Θ: ℝ^{d+1} → ℝ` taking time and asset state as input, replacing per-timestep LSMC regressors.

## Stopping Rule

**Stopping rule** `φ(t, x)` (Eq. 11): `φ = 1` (exercise) when `R^Θ ≤ 0` and `h(x) > 0`, or at maturity `t = T`; otherwise continue (`φ = 0`).

## Regions

- **Stopping region** `𝒮` — states where exercise is optimal.
- **Continuation region** `𝒞` — states where waiting is optimal.

## Grids

- **Exercise grid** `𝕋^(ex,b)` — times at which stopping is allowed during forward evaluation.
- **Solver grid** `𝕋^(tr,b)` — times used for training the stopping policy.

## Delayed Payoff

**Delayed payoff** `y_dpf` (Eq. 29) — discounted payoff under exploratory stopping with waiting period `Δ_wait`, used to compute training targets `y = y_dpf - h(x)` (Eq. 30).

## GBM Path

**GBM path** — asset trajectory under geometric Brownian motion (Eq. 15), independent Brownian motions per dimension under the risk-neutral measure.

## LSMC

**Longstaff-Schwartz Monte Carlo (LSMC)** — backward induction with regression to estimate continuation values at each exercise date (Sec. 2.1). Stage 1a produces per-step timing-value samples; Stage 1b stacks them to train the ADNN.

## Anchor Sampling

**Anchor sampling** (Algorithm 2) — mixture of exploratory (ITM path points), exploitative +/- (boundary crossings), and terminal inputs used to build each RL training batch (Eq. 20).

## Traffic-Light Labels

**Traffic-light labels** (Algorithm 3) — yellow/green/red path states governing exploratory delayed stopping: paths originating in the stopping region may wait before exercise to expand the learned continuation region.
