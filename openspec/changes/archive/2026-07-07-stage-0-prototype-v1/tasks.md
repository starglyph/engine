# Tasks: stage-0-prototype-v1

Retrospective documentation sync — all items were implemented before this change was opened.

## 1. Image input and EXIF

- [x] 1.1 Load PNG/JPEG/BMP/TIFF into normalized grayscale `FrameImage`
- [x] 1.2 Extract focal-length and timestamp EXIF; never read GPS
- [x] 1.3 Derive FOV prior from 35 mm equivalent focal length

## 2. Engine database lifecycle

- [x] 2.1 Bootstrap tetra3 database (10–70°) with disk cache
- [x] 2.2 On-demand dense-band generation keyed by FOV center
- [x] 2.3 `STARGLYPH_DB_MAG_LIMIT` env for magnitude depth (B4)
- [x] 2.4 Single-flight lock for concurrent database generation

## 3. Solve pipeline (starglyph-core)

- [x] 3.1 Deep detector with JPEG robustness and rayon parallelism (B3)
- [x] 3.2 tetra3 matching with verification gate and blind FOV ladder
- [x] 3.3 LM pose refinement with k1 radial distortion (B5 partial)
- [x] 3.4 Structured `SolveReport` with overlay geometry

## 4. HTTP service (starglyph-serve)

- [x] 4.1 POST /solve multipart API with hints and overlay modes (C1)
- [x] 4.2 Engine pool with queue and solve timeouts (C3 partial)
- [x] 4.3 healthz / readyz probes and Docker packaging (C3)
- [x] 4.4 Anonymous JSONL telemetry sink (D1)

## 5. Evaluation

- [x] 5.1 WCS ground-truth parsing and pose-error metrics
- [x] 5.2 `make eval-gate` CI baseline regression check
- [x] 5.3 Detection precision/recall test on synthetic field

## 6. OpenSpec sync

- [x] 6.1 Sync phase-2 solver specs to `openspec/specs/`
- [x] 6.2 Add Stage 0 capability specs
- [x] 6.3 Archive `roadmap-phase-2-solver-v1`
- [x] 6.4 Validate all specs with `openspec validate --specs`
