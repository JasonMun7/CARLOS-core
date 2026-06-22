# B1 benchmark scoring protocol

CARLOS-core needs a reproducible way to know whether the implementation matches the paper's B1 result (Table 3 target **4.592**). Early code printed multiple prices (coarse vs fine grid, different path banks) and allowed `--dev` runs to be compared against the same target, which made "passing B1" ambiguous.

We decided that **B1 pass/fail** means: a **benchmark run** (not a smoke test) with Table 6 path counts, training seed **0**, validation price on the **finest exercise grid** within **±0.05** of 4.592. Scoring and grid-saturation monitoring share **one validation path bank**, built deterministically from the training seed (`training_seed + 1000`). Smoke tests (`dev_mode`) verify pipeline wiring only and must not be scored against 4.592.

Configuration will be modeled as a general **`CarlosConfig`** with a **`b1_benchmark()` preset** (not a type named after a single benchmark). The official entry point will be **`python -m carlos benchmark b1`**, which enforces full path counts, fixed seeds, finest-grid scoring, and a non-zero exit code on failure. Experimental training remains on `python -m carlos train`.

## Considered Options

- **Paper-faithful evaluation (rejected):** Match Table 3 under opaque paper internals. Hard to test without reverse-engineering their evaluation harness.
- **Coarse-grid final price (rejected):** The CLI printed a 20-step `validate_price` after training while finer per-level prices differed materially.
- **Separate path banks for scoring vs Eq. 32 (rejected):** `validate_price` re-simulated with `seed=123` while grid transitions used `seed + 1000`, so metrics were not comparable.
- **`train` as the benchmark command (rejected):** Too easy to run `--dev` or misread output as the official score.

## Consequences

- Implementation must unify scoring onto `build_validation_paths` and remove ad-hoc re-simulation in benchmark reporting.
- README and CLI help must distinguish **benchmark run** vs **smoke test**.
- CI can gate on `python -m carlos benchmark b1` with a single pass/fail line.
