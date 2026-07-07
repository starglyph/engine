## ADDED Requirements

### Requirement: POST /solve accepts image uploads
The service SHALL accept `multipart/form-data` POST requests with a required image field and optional solve hints, returning a JSON [`SolveReport`].

#### Scenario: Successful solve returns 200 with report
- **WHEN** a valid image is uploaded and the solver succeeds within the solve timeout
- **THEN** the response is HTTP 200 with a JSON body containing status `solved`, pose, quality, timing, detections, and overlay geometry

### Requirement: Service exposes health and readiness probes
The service SHALL provide liveness and readiness endpoints for orchestration.

#### Scenario: readyz waits for bootstrap database warmup
- **WHEN** `GET /readyz` is called before the bootstrap database is warmed
- **THEN** the service responds HTTP 503 indicating warmup in progress

### Requirement: Engine pool limits concurrent solves
The service MUST checkout exclusive warmed [`Engine`] instances from a pool and enforce queue and solve timeouts.

#### Scenario: Pool exhaustion returns 503
- **WHEN** all engines are busy and no permit becomes available within the queue timeout
- **THEN** the service responds HTTP 503 with a busy error code

### Requirement: Overlay can be returned as PNG or inline base64
The service SHALL support optional overlay delivery alongside the JSON report.

#### Scenario: Overlay PNG is attached on request
- **WHEN** the client requests `overlay=png` (form field or query parameter)
- **THEN** the response includes a rendered overlay PNG in addition to the JSON solve report
