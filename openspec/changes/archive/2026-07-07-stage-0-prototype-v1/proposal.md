## Why

Stage 0 prototype work (B1–B5 partial, C1, C3, D1) was implemented in `prototype/crates/starglyph-core` and `starglyph-serve` ahead of the archived phase-1/2 OpenSpec changes. This change retroactively documents the shipped prototype capabilities so OpenSpec main specs match the code.

## What Changes

- Sync phase-2 solver specs to `openspec/specs/` with code-derived refinements (detection on real JPEGs, k1 distortion, eval gate).
- Add Stage 0 capability specs: image input, engine database lifecycle, HTTP service, anonymous telemetry.
- Archive completed changes `roadmap-phase-2-solver-v1` and this retrospective change.

## Capabilities

### New Capabilities

- `solver-image-input`: decode real photo formats and extract solve-relevant EXIF without GPS.
- `solver-engine-database`: tetra3 bootstrap/dense-band databases with mag-depth caching.
- `solver-http-service`: `starglyph-serve` POST /solve, health probes, engine pool.
- `solver-telemetry`: anonymous JSONL per-request logging (Stage 0 · D1).

### Modified Capabilities

- `solver-star-detection`: consumer JPEG robustness, 12 MP performance budget.
- `solver-pattern-matching`: tetra3 verification gate, blind FOV retry ladder.
- `solver-pose-estimation`: LM refinement with k1 radial distortion (partial B5).
- `solver-overlay-debug`: structured SolveOverlay with planets and optional grid.
- `solver-benchmark-harness`: astrometry.net WCS eval and CI baseline gate.

## Impact

- `openspec/specs/` becomes the authoritative requirement set for the prototype solver stack.
- No code changes required — documentation-only sync.
- OpenSpec tracks *what the code does*; product backlog and strategy are maintained outside this repository.
