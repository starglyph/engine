## ADDED Requirements

### Requirement: Benchmark harness runs solver on dataset splits
The system SHALL execute the solver pipeline on configured dataset splits and produce a reproducible benchmark report.

#### Scenario: Benchmark run emits aggregate metrics
- **WHEN** benchmark is launched for a dataset split
- **THEN** the harness outputs aggregate metrics including detection quality and pose-related quality indicators

### Requirement: Benchmark harness stores worst-case frames
The system MUST automatically persist worst-case frames and diagnostics according to configured ranking criteria.

#### Scenario: Worst cases are ranked and exported
- **WHEN** benchmark run completes
- **THEN** the harness exports a ranked subset of worst-performing frames with solver diagnostics

#### Scenario: Repeated run with same configuration is reproducible
- **WHEN** benchmark is re-run on identical inputs and configuration
- **THEN** the aggregate metrics and worst-case ranking remain consistent within defined numeric tolerance
