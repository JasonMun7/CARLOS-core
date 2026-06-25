# Paper benchmark suite scoring protocol

We decided that scored benchmarks for all seven Table 2 contracts follow the same rules as ADR 0001 (B1): training seed **0** (unless overridden), validation path bank seed **training_seed + 1000**, finest-grid validation price only, Table 6 path counts (no `dev_mode`), and pass/fail exit codes from `python -m carlos benchmark <preset>`.

Acceptance uses Table 3 **CARLOS** column targets with tolerance **max(0.05, 3 × reported_std)** from the paper. Presets live in `carlos/benchmarks.py`; CLI supports `benchmark list`, individual presets (`b1`, `b2`, `m2a`, `m2b`, `m3`, `m5a`, `m5b`), and `benchmark all`.

## Consequences

- Official scoring uses CPU for reproducibility (`force_cpu` during benchmark runs).
- README maintains a results matrix as contracts are validated.
- Smoke tests (`train --dev`) remain non-scored.
