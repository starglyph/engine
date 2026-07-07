## ADDED Requirements

### Requirement: Telemetry records one line per request
The system SHALL append exactly one JSON record per `/solve` request to a configured log file when telemetry is enabled.

#### Scenario: Solved request records outcome and metrics
- **WHEN** a solve completes successfully
- **THEN** the telemetry record includes outcome `solved`, image dimensions, detection count, pose quality summary, and timing breakdown

### Requirement: Telemetry is anonymous
The system MUST NOT log user identity, client IP address, raw EXIF dumps, or GPS/location data.

#### Scenario: EXIF contribution is summarized not echoed
- **WHEN** EXIF metadata contributed to hints
- **THEN** the record notes presence and derived FOV prior or timestamp flags without serializing the EXIF block

### Requirement: Telemetry schema is versioned
The system SHALL include a schema version field in each record and bump it only on incompatible changes.

#### Scenario: New fields are additive
- **WHEN** optional telemetry fields are added
- **THEN** the schema version remains unchanged and new fields are omitted when absent
