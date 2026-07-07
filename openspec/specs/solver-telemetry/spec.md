# Solver telemetry

## Purpose

Anonymous per-request solve logging for Stage 0 operations monitoring. One JSON line per `/solve` request appended to an append-only log.

## Requirements

### Requirement: Telemetry records one line per request
The system SHALL append exactly one JSON record per `/solve` request to a configured log file when telemetry is enabled.

#### Scenario: Solved request records outcome and metrics
- **WHEN** a solve completes successfully
- **THEN** the telemetry record includes outcome `solved`, image dimensions, detection count, pose quality summary, and timing breakdown

#### Scenario: Rejected request records reject code
- **WHEN** a request is rejected before solving (bad input, not ready, busy, timeout)
- **THEN** the telemetry record includes outcome `rejected`, HTTP status, and a machine-readable reject code

### Requirement: Telemetry is anonymous
The system MUST NOT log user identity, client IP address, raw EXIF dumps, or GPS/location data.

#### Scenario: EXIF contribution is summarized not echoed
- **WHEN** EXIF metadata contributed to hints
- **THEN** the record notes presence and derived FOV prior or timestamp flags without serializing the EXIF block

#### Scenario: Source is filename stem only
- **WHEN** the upload includes a file name
- **THEN** only the file-name stem is logged, never a full path

### Requirement: Telemetry schema is versioned
The system SHALL include a schema version field in each record and bump it only on incompatible changes.

#### Scenario: New fields are additive
- **WHEN** optional telemetry fields are added
- **THEN** the schema version remains unchanged and new fields are omitted when absent
