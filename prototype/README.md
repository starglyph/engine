# Prototype

Rust workspace for the phase-1 simulator prototype.

## Crates

- `simulator-core` - domain pipeline skeleton (camera -> projection -> rendering -> degradations -> export).
- `dataset-cli` - command-line entry point for dataset generation.
- `solver-core` - phase-2 baseline solver pipeline (detection -> matching -> pose -> overlay + benchmark).
- `solver-cli` - command-line benchmark runner for solver-core.

## Dataset v1 generation

Prepare the full star catalog once:

```bash
make fetch-catalog
```

Generate train/val/test in one command:

```bash
cargo run -p dataset-cli -- \
  --seed 42 \
  --output-root artifacts/simulator/dataset-v1 \
  --catalog-csv ../data/catalogs/hyg_v3.csv \
  --train-frames 100 \
  --val-frames 20 \
  --test-frames 20 \
  --validate-reproducibility
```

If you do not have the full catalog yet, run with the built-in fallback:

```bash
cargo run -p dataset-cli -- --use-baseline-catalog
```

The command writes:

```text
artifacts/simulator/dataset-v1/
  manifest.json
  train/frame-000001/{image.png,meta.json,truth-stars.csv}
  val/frame-000001/{image.png,meta.json,truth-stars.csv}
  test/frame-000001/{image.png,meta.json,truth-stars.csv}
```

## Validation procedure

Run all phase-1 simulator gates (projection tests, visual goldens, reproducibility):

```bash
make validate-simulator
```

Run the phase-2 solver benchmark on an already generated dataset:

```bash
cargo run -p solver-cli -- \
  --dataset-root artifacts/simulator/dataset-v1 \
  --output-root artifacts/recognizer/run-latest \
  --split test
```

