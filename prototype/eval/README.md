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

## Independent local WCS

`solve_local_wcs.py` runs installed Astrometry.net against EXIF-oriented FITS pixels
without uploading photos. `compare_local_wcs.py` compares its full WCS and external
star correspondences with `starglyph eval` outputs in `solve-reports/` (full report,
refined camera, normalized dimensions and source SHA-256). These artifacts are
written alongside the existing `per-frame/` summaries.

Setup, commands, coordinate conventions and reference-review criteria:
[docs/local-wcs.md](../../docs/local-wcs.md).

Python checks: `python3 -m unittest discover -s eval -p 'test_local_wcs.py'`
from `prototype/`, with Astropy, NumPy and Pillow installed.
