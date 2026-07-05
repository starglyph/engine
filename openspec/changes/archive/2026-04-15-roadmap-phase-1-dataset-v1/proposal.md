## Why

Roadmap phase 1 defines the first deliverable that unblocks all downstream recognition work: a reproducible synthetic dataset with trustworthy ground truth. The project currently has goals and architecture notes, but no executable dataset pipeline that can produce stable train/val/test data from a fixed seed.

## What Changes

- Add a minimal simulator baseline that projects catalog stars through a pinhole camera model and exports camera intrinsics/extrinsics per frame.
- Add star rendering behavior that converts stellar magnitude to pixel intensity and applies a baseline PSF with dynamic-range clipping.
- Add baseline image degradations (shot/read noise, light blur, JPEG artifacts) so dataset v1 is not unrealistically clean.
- Add a dataset generator command that produces train/val/test splits with deterministic output for a given seed.
- Add per-frame metadata and visible-star truth export for downstream detector/matcher evaluation.
- Add simulator validation with projection-focused unit tests and a small set of visual golden outputs.

## Capabilities

### New Capabilities
- `simulator-camera-projection`: Define deterministic camera pose/orientation projection and camera parameter export for synthetic frames.
- `simulator-star-rendering`: Define how star brightness is rendered into image space, including PSF and dynamic-range constraints.
- `simulator-dataset-generation`: Define deterministic split generation, metadata packaging, and visible-star truth outputs for dataset v1.
- `simulator-validation`: Define required correctness checks for projection math and visual regression coverage for simulator output.

### Modified Capabilities
- None.

## Impact

- Establishes the first executable data pipeline for `starglyph` and a stable input contract for phase 2 recognizer work.
- Introduces simulator-side modules, dataset output layout, and test assets under the future prototype codebase.
- Adds reproducibility and quality gates that will be used as release criteria for dataset versions.
