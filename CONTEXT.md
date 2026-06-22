# CARLOS Domain Glossary

Reference: [Continuous-time Optimal Stopping through Deep Reinforcement Learning (CARLOS)](https://arxiv.org/pdf/2606.17545)

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
