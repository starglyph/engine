## ADDED Requirements

### Requirement: Solver detects star candidates on synthetic frames
The system SHALL detect star candidates from a synthetic input frame and produce a deterministic candidate list for fixed input data and configuration.

#### Scenario: Deterministic detection for fixed seed
- **WHEN** the same synthetic frame and detector configuration are processed multiple times
- **THEN** the detector returns the same ordered list of star candidates with identical coordinates and scores

### Requirement: Detection quality is measurable against synthetic truth
The system MUST compute precision and recall for detected stars against synthetic truth data using configurable matching tolerance.

#### Scenario: Precision and recall are reported per frame
- **WHEN** detector output and per-frame truth stars are available
- **THEN** the system reports precision and recall values for that frame

#### Scenario: Aggregate detection metrics are reported for a run
- **WHEN** detection is executed on a dataset split
- **THEN** the system reports aggregate precision and recall across processed frames
