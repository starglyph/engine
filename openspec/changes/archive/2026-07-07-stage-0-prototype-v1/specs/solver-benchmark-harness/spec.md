## MODIFIED Requirements

### Requirement: Evaluation compares against astrometry.net WCS ground truth
The system SHALL parse astrometry.net-style WCS JSON sidecars and compute pose-error metrics against the solver output using container-specific orientation conventions.

#### Scenario: CI eval gate enforces baseline regression
- **WHEN** `make eval-gate` is run with the CI baseline manifest
- **THEN** the harness fails if pose-error metrics regress beyond the stored baseline thresholds
