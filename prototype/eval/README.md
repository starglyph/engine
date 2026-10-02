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

## Automatic skyline research baseline

`automatic_sky_experiment.py` compares unmasked solving, the pinned manual
sky-fill experiment, and image-only RGB skyline masks on all 17 smartphone frames.
It uses the existing CLI polygon interface; no runtime defaults or acceptance
thresholds change. Install `eval/requirements-automatic-sky.txt` in a Python
environment, then run from `prototype/`:

```bash
make test-automatic-sky
cargo build --release -p starglyph-cli --locked
python3 eval/automatic_sky_experiment.py --out-dir artifacts/automatic-sky/new-run
python3 eval/automatic_sky_experiment.py --out-dir artifacts/automatic-sky/new-run --compare-only
```

Each full run requires a nonexistent output directory. The split (six manually
annotated frames and eleven validation frames), algorithm parameters and input
hashes are frozen in `plan.json` before generation. Automatic estimation receives
pixels only, without manual annotations, frame IDs, catalog or solver feedback.
The validation subset is from the same previously studied series, not an
independent camera/site holdout. Do not retune on it and still call it held out.

`comparison.json` reports individual gains/losses, per-split counts, solver quality,
timing, overlap with conservative manual masks, and point/cloud diagnostics.
Generation includes image decode and is timed separately from solver reports;
diagnostic passes are outside reported solve timing. Brightness gradients and
clouds can truncate this skyline baseline. Abstention uses the unmasked path.
All modes use original photos, with no cropping or resampling of solver input.

Both controls must pass their existing per-ID gates. Candidate regressions remain
explicit negative results: **this experiment command is not a release gate**, and
a zero exit code means valid measurement, not permission to enable the algorithm.
`--compare-only` checks input and artifact hashes, mode/configuration isolation,
source identity, camera fields, and report completeness without running the solver.
The stored input paths must still exist; changed binaries, inputs or annotations
require a fresh run. See [results and limitations](../../docs/automatic-sky-experiment.md).

The default `--algorithm rgb_skyline_v1` remains reproducible. The follow-up
`--algorithm rgb_gradient_v2` tracks local RGB slopes instead of constant seed
color. Pin a previous v1 run with `--reference-run artifacts/automatic-sky/new-run`;
the runner checks its artifact integrity and the comparator requires the same
engine/data plus unchanged unmasked/manual control reports. Use a fresh output
directory for v2. `--compare-only` reads the algorithm and reference from its plan.
Generation metadata must match that plan's algorithm and parameters.

All 17 frames were examined in the first experiment. For v2, both groups are
development/evaluation data; the historical `validation` JSON label does **not**
mean held-out data. The model was tested on synthetic gradients, clouds and
silhouettes before its real-image evaluation. See the
[gradient experiment](../../docs/gradient-sky-experiment.md).

## Matching search replay

`matching_search_experiment.py` compares the original reduced/deep matching ladder
with extra 40/50 prefixes through an explicitly invoked, ignored Rust test. It
freezes input hashes and 17 deterministic uniform-coordinate negative controls
before execution. This tests candidates before refinement, not full solves.
`test_matching_search.py` checks reproducibility, preserved photometry and portrait
scale selection; it is included in `make test-automatic-sky`.

`compare_matching_runs.py` compares complete three-arm runs across solver builds,
checks input/artifact hashes and identical masks, rejects lost solved IDs, and
reports any changed previous camera/report (excluding timing and equal-flux ties).
Pass archived binaries with `--before-binary` / `--after-binary` when Cargo has
replaced the executables originally recorded in a plan. It never weakens or rewrites
the frozen plans. Changed prior solutions must still be reviewed separately.

The production extension is restricted to a hinted dense band, only after both
ordinary detection tiers fail; blind ladders and acceptance thresholds are unchanged.
See [method and measurements](../../docs/extended-dense-search-experiment.md).

The sky-fill gate also runs the ignored `extended_dense` integration test on the
committed `215950` photo and manual mask. It protects the new recovery independently
of the unchanged 12-ID baseline, including minimum inliers, RMS and pose stability.
