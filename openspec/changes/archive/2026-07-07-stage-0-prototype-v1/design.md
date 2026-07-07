## Context

The prototype solver lives in `starglyph-core` (library) and `starglyph-serve` (HTTP wrapper). Legacy `solver-core` from phase 2 is superseded but retained for reference tests. Stage 0 added real-image solving, headless deployment, and evaluation against astrometry.net WCS sidecars.

## Goals / Non-Goals

- Goals: document shipped behavior as testable requirements; align OpenSpec with `prototype/crates/starglyph-*`.
- Non-Goals: re-specify simulator phase-1 (unchanged); document E1 beta frontend; specify k2/tangential distortion (not yet implemented).

## Decisions

- **Single library crate**: `starglyph-core` owns detect → solve → overlay → eval; HTTP service is a thin Axum layer with engine pooling.
- **tetra3 for pattern matching**: bootstrap DB for blind solve, dense-band DB generated on demand; magnitude depth controlled by `STARGLYPH_DB_MAG_LIMIT`.
- **Two-stage acceptance**: tetra3 hypothesis plus independent catalog verification (log-odds + hit count) before LM refinement.
- **k1 only (B5 partial)**: radial distortion refined in LM when ≥8 matches; FOV-dependent prior prevents narrow-field noise fits.
- **PII minimization**: EXIF reads focal length and timestamp only; GPS never parsed; telemetry logs anonymous aggregates.
- **Eval convention split**: TIFF vs JPEG astrometry sidecars use different orientation→roll rules (empirically calibrated in `eval.rs`).

## Risks / Trade-offs

- Specs may drift if new engine work ships without an OpenSpec change → mitigate by requiring OpenSpec changes for new capabilities.
- All specs in this public repository are Apache 2.0 — keep strategic and product-only content out of `openspec/`.

## Documentation boundary

| Layer | Location | Content |
|-------|----------|---------|
| Product / backlog | outside this repo | Epics, gates, blockers (not referenced in public docs) |
| Engine requirements | `openspec/specs/` | Capability requirements aligned with code |
