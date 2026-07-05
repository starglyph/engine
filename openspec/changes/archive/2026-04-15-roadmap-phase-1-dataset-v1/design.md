## Context

Starglyph currently has product and architecture intent documented in `docs/`, but does not yet provide an executable simulator pipeline that emits reproducible synthetic frames and machine-readable truth data. Phase 1 in the roadmap requires a dataset v1 that can be regenerated deterministically from a fixed seed and used as the stable foundation for phase 2 detection/matching work.

This design must stay aligned with the desktop Rust prototype direction and preserve clear subsystem boundaries from AGENTS guidance: simulator outputs should be explicit artifacts that recognizer code can consume without coupling to simulator internals.

## Goals / Non-Goals

**Goals:**
- Define a deterministic simulator flow from camera pose sampling through star projection and image synthesis.
- Standardize dataset v1 output contracts (splits, per-frame metadata, visible-star truth).
- Define baseline realism knobs (noise/blur/jpeg) that are simple enough for a first implementation and reproducible under seed control.
- Define validation gates (projection unit tests and visual goldens) that protect correctness and regressions.

**Non-Goals:**
- Implementing phase 2 star detection, pattern matching, or pose solving logic.
- Modeling advanced domain-gap effects such as distortion, vignetting, chromatic aberration, or clouds (roadmap phase 3).
- Supporting mobile runtime constraints or real-time inference paths.

## Decisions

1. **Deterministic RNG as a first-class input**
   - Decision: Require a global dataset seed and deterministic per-sample derivation (for example split + frame index based stream).
   - Rationale: Reproducibility is the phase exit criterion and a prerequisite for trustworthy benchmarks.
   - Alternative considered: ad-hoc random calls from each module; rejected because subtle call-order changes would invalidate reproducibility.

2. **Pipeline decomposition by simulator stages**
   - Decision: Structure simulator behavior as clear stages: camera sampling -> projection -> radiometric rendering -> degradations -> artifact export.
   - Rationale: Makes reasoning, testing, and future realism extensions simpler while keeping simulator/recognizer boundaries clear.
   - Alternative considered: one monolithic render function; rejected because it limits testability and obscures where regressions originate.

3. **Explicit frame-level artifact contracts**
   - Decision: For each generated frame, export image plus metadata containing camera intrinsics/extrinsics, generation config, and visible-star truth list.
   - Rationale: Downstream evaluation needs exact ground truth and camera state, not only rasterized images.
   - Alternative considered: image-only dataset with implicit parameters; rejected because it blocks precise metric computation and debugging.

4. **Baseline rendering realism only**
   - Decision: In phase 1 include magnitude-to-intensity mapping, compact PSF model, dynamic-range clipping, shot/read noise, mild blur, and jpeg compression artifacts.
   - Rationale: Provides useful synthetic variability without overextending before baseline solver exists.
   - Alternative considered: physically richer optics/noise stack now; rejected as high effort with unclear value before v1 recognizer feedback.

5. **Validation split between numeric correctness and visual regression**
   - Decision: Use unit tests for projection invariants and deterministic golden image checks for end-to-end simulator drift.
   - Rationale: Numeric tests catch geometry errors; goldens catch rendering/degradation regressions that may not violate scalar assertions.
   - Alternative considered: only metric-based checks; rejected because visual failures can slip through coarse metrics.

## Risks / Trade-offs

- **[Brightness mapping oversimplifies astrophotometric behavior]** -> Start with documented baseline mapping and revisit once recognizer sensitivity data is available.
- **[Golden images can become brittle across implementation changes]** -> Keep golden set small, deterministic, and tied to explicit update procedure.
- **[Seed reproducibility can break from hidden nondeterminism]** -> Ban nondeterministic sources in generation path and test same-seed byte-level repeatability in CI.
- **[Early output schema mismatch with recognizer needs]** -> Export richer metadata now (camera + visible-star truth + config) and version the dataset manifest.
