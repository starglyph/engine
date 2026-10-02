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

## Smartphone regression gate

From `prototype/`, run `make eval-smartphone-gate`. It evaluates all 17 committed
smartphone frames independently, with EXIF hints and the committed HYG v4.2
catalog. No external astrometry installation or network download is required.

`baseline-smartphone.json` pins input hashes, oriented dimensions, catalog hash,
configuration and the individual previously solved IDs. Losing one of those IDs
fails even if another frame starts solving. Missing/corrupt inputs, missing
reports, incorrect orientation and invalid numeric camera results also fail.
New successes are allowed; expected failures are not frozen as failures.

Each invocation creates a fresh directory under `artifacts/eval/smartphone/`
with full reports, debug logs and execution provenance. CI uploads these artifacts
on success or failure. Timings are diagnostic, not a shared-runner performance gate;
inspect `cache_before` and the debug log before comparing cold and warm runs.
The first run can spend several minutes generating dense pattern databases.

To verify an existing run:

```bash
python3 eval/smartphone_gate.py --run-dir artifacts/eval/smartphone/<run-directory>
python3 -m unittest discover -s eval -p 'test_smartphone_gate.py'
```

Baseline updates are manual: reproduce the complete dataset twice with the same
configuration and a warm cache, investigate lost successes, and review the exact
ID/hash diff. Never regenerate a weaker baseline automatically after a failure.
WCS candidate acceptance is separate from this recognition gate; a successful
plate solve alone does not certify full-field astrometric accuracy.
