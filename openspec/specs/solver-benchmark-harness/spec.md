# Solver benchmark harness

## Purpose

Batch evaluation of the solve pipeline against reference datasets. Supports synthetic truth metrics and astrometry.net WCS ground-truth comparison for real sky samples.

## Requirements

### Requirement: Benchmark harness runs solver on dataset splits
The system SHALL execute the solver pipeline on configured dataset manifests and produce a reproducible benchmark report.

#### Scenario: Benchmark run emits aggregate metrics
- **WHEN** benchmark is launched for a dataset manifest
- **THEN** the harness outputs aggregate metrics including solve success rate, pose error statistics, and per-frame diagnostics

### Requirement: Benchmark harness stores worst-case frames
The system MUST automatically persist worst-case frames and diagnostics according to configured ranking criteria.

#### Scenario: Worst cases are ranked and exported
- **WHEN** benchmark run completes
- **THEN** the harness exports a ranked subset of worst-performing frames with solver diagnostics

#### Scenario: Repeated run with same configuration is reproducible
- **WHEN** benchmark is re-run on identical inputs and configuration
- **THEN** the aggregate metrics and worst-case ranking remain consistent within defined numeric tolerance

### Requirement: Evaluation compares against astrometry.net WCS ground truth
The system SHALL parse astrometry.net-style WCS JSON sidecars and compute pose-error metrics against the solver output using container-specific orientation conventions.

#### Scenario: WCS ground truth is converted to solver convention
- **WHEN** a sample frame has a committed WCS sidecar
- **THEN** the evaluation harness converts astrometry orientation and parity into the solver's camera roll convention before computing angular error

#### Scenario: CI eval gate enforces baseline regression
- **WHEN** `make eval-gate` is run with the CI baseline manifest
- **THEN** the harness fails if pose-error metrics regress beyond the stored baseline thresholds
