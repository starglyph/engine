# Eval regression baseline

`baseline-ci.json` pins the **license-clean CI subset** of the sky-sample eval harness:

- **Frames:** `tetra3_alt40`, `tetra3_alt60` (Apache-2.0, ESA tetra3)
- **Catalog:** `data/catalogs/hyg_v42.csv.gz`
- **Config:** blind solve (no FOV hint)

`make eval-gate` runs the same configuration and compares `artifacts/eval/ci/summary.json` against this file. The gate fails on solver-track solve-rate drops or axis-angle p95 regression beyond the configured threshold.

## Regenerating the baseline

From `prototype/`:

```bash
source ~/.cargo/env
python3 ../data/samples/sky-samples/fetch_sample.py tetra3_alt40 tetra3_alt60
cargo run --release -p starglyph-cli -- eval \
  --manifest ../data/samples/sky-samples/manifest.json \
  --ids tetra3_alt40,tetra3_alt60 \
  --catalog ../data/catalogs/hyg_v42.csv.gz \
  --out-dir artifacts/eval/ci
cp artifacts/eval/ci/summary.json eval/baseline-ci.json
```

Updating the committed baseline is a deliberate act — review the diff like any code change.
