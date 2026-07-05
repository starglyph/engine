# Simulator validation

## Purpose

Phase 1 quality gates: automated projection tests, visual golden regression checks, and reproducibility verification for the synthetic dataset pipeline.

## Requirements

### Requirement: Projection correctness tests
The simulator codebase SHALL include automated unit tests that validate core projection behavior, including deterministic mapping and expected handling of non-visible stars.

#### Scenario: Projection test suite passes on deterministic fixtures
- **WHEN** automated tests are executed for simulator projection logic
- **THEN** test cases verify expected coordinates/visibility outcomes for predefined camera and star fixtures

### Requirement: Visual golden regression set
The simulator SHALL maintain a visual golden set generated from fixed seeds and configurations to detect rendering and degradation regressions.

#### Scenario: Golden comparison flags image drift
- **WHEN** simulator output for a golden fixture differs from the stored golden artifact beyond configured tolerance
- **THEN** the validation flow marks the run as failed and reports the mismatch

### Requirement: Reproducibility verification
Validation SHALL include a reproducibility check that compares repeated generation outputs under identical configuration and seed.

#### Scenario: Identical seeded runs are stable
- **WHEN** the same dataset generation config is run twice with the same seed
- **THEN** reproducibility validation confirms outputs are equivalent according to project-defined deterministic checks
