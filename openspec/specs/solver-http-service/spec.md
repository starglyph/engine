# Solver HTTP service

## Purpose

Headless HTTP API (`starglyph-serve`) exposing plate solving over multipart image upload with health probes, engine pooling, timeouts, and optional overlay rendering.

## Requirements

### Requirement: POST /solve accepts image uploads
The service SHALL accept `multipart/form-data` POST requests with a required image field and optional solve hints, returning a JSON [`SolveReport`].

#### Scenario: Successful solve returns 200 with report
- **WHEN** a valid image is uploaded and the solver succeeds within the solve timeout
- **THEN** the response is HTTP 200 with a JSON body containing status `solved`, pose, quality, timing, detections, and overlay geometry

#### Scenario: Failed solve returns 200 with failure envelope
- **WHEN** the image is valid but the solver cannot produce an accepted pose
- **THEN** the response is HTTP 200 with status `failed` and a structured failure code and message

### Requirement: Service exposes health and readiness probes
The service SHALL provide liveness and readiness endpoints for orchestration.

#### Scenario: healthz always returns ok when process is up
- **WHEN** `GET /healthz` is called
- **THEN** the service responds HTTP 200 with body `ok`

#### Scenario: readyz waits for bootstrap database warmup
- **WHEN** `GET /readyz` is called before the bootstrap database is warmed
- **THEN** the service responds HTTP 503 indicating warmup in progress

### Requirement: Engine pool limits concurrent solves
The service MUST checkout exclusive warmed [`Engine`] instances from a pool and enforce queue and solve timeouts.

#### Scenario: Pool exhaustion returns 503
- **WHEN** all engines are busy and no permit becomes available within the queue timeout
- **THEN** the service responds HTTP 503 with a busy error code

#### Scenario: Solve timeout returns 504
- **WHEN** a solve exceeds the configured wall-clock timeout
- **THEN** the service responds HTTP 504; the engine is returned to the pool when the abandoned solve completes

### Requirement: Overlay can be returned as PNG or inline base64
The service SHALL support optional overlay delivery alongside the JSON report.

#### Scenario: Overlay PNG is attached on request
- **WHEN** the client requests `overlay=png` (form field or query parameter)
- **THEN** the response includes a rendered overlay PNG in addition to the JSON solve report
