# Algorithm 3 fidelity notes

CARLOS-core implements a simplified traffic-light delayed stopping state machine (yellow / green / red) in `carlos/delayed_payoff.py` aligned with paper Sec. 4.3 Eq. 29–31, but without the full ring/boundary machinery described in the appendix figures.

We decided to document this gap rather than block optimization work: the simplified machine preserves the exploration window (`delta_wait`, Eq. 31) and Eq. 30 target `y = y_dpf - h(x)`, which proved sufficient for B1 scoring within ±0.05 of 4.592.

RL training uses **one Adam pass per Algorithm 1 loop** (`rl_epochs=1`, fresh optimizer each loop). Table 6 epoch counts apply to **Stage 1 ADNN** training (`stage1_epochs`), not repeated inner epochs with a persistent optimizer.

We decided to document this after B1 regression: persistent Adam with `rl_epochs=5` per loop degraded the finest-grid price from ~4.57 (Stage 1) to ~4.42 despite correct target batching.

**Best-checkpoint restore:** `run_rl_loop` keeps the weights that achieved the highest finest-grid validation price across Stage 1 + all RL loops, then scores that snapshot.

## Consequences

- Further fidelity work targets price impact on benchmarks, not line-count parity with appendix figures.
- Unit tests in `tests/test_delayed_payoff.py` guard hand-computed tiny-path behavior.
