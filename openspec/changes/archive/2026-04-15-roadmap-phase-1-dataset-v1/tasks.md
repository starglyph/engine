## 1. Simulator scaffolding and configuration

- [x] 1.1 Create simulator module layout for camera sampling, projection, rendering, degradations, and artifact export stages.
- [x] 1.2 Define dataset generation configuration schema including split sizes, camera ranges, degradation parameters, and required global seed.
- [x] 1.3 Implement deterministic RNG strategy (global seed with per-split/per-frame derivation) and add a reproducibility utility helper.

## 2. Camera projection and truth visibility

- [x] 2.1 Implement pinhole projection with explicit intrinsics/extrinsics input and deterministic projection output for fixed inputs.
- [x] 2.2 Implement visibility classification for out-of-frustum and behind-camera stars used by renderer and truth export.
- [x] 2.3 Add frame-level camera metadata export including calibration and pose representation for every generated frame.

## 3. Star rendering and baseline degradations

- [x] 3.1 Implement documented magnitude-to-intensity mapping that preserves relative brightness ordering.
- [x] 3.2 Implement baseline PSF rasterization and dynamic-range clipping for rendered stars.
- [x] 3.3 Implement configurable degradations (shot noise, read noise, mild blur, JPEG artifacts) and persist applied settings in metadata.

## 4. Dataset generation pipeline

- [x] 4.1 Implement one-command dataset builder that generates train/val/test splits into a versioned dataset v1 output root.
- [x] 4.2 Export per-frame metadata artifacts linking image path, split, camera parameters, generation settings, and frame identifiers.
- [x] 4.3 Export visible-star truth artifacts containing star identifiers and projected coordinates for each generated frame.
- [x] 4.4 Add deterministic regression check that reruns generation with the same seed and validates output equivalence.

## 5. Validation and documentation

- [x] 5.1 Add projection unit tests using deterministic fixtures for coordinate and visibility expectations.
- [x] 5.2 Create and wire visual golden fixtures for representative rendering/degradation configurations.
- [x] 5.3 Add test/validation command wiring so projection tests, golden checks, and reproducibility checks run in CI/local workflow.
- [x] 5.4 Update `docs/` with dataset v1 artifact layout and validation procedure needed to declare phase 1 exit criteria met.
