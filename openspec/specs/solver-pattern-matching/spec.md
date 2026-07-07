# Solver pattern matching

## Purpose

Match detected star patterns against a tetra3 pattern database built from the HYG catalog. Produces ranked attitude hypotheses with confidence and ambiguity signals.

## Requirements

### Requirement: Solver matches detected patterns to catalog hypotheses
The system SHALL generate catalog match hypotheses from detected stars using a tetra3 pattern-matching algorithm backed by on-disk pattern databases.

#### Scenario: Candidate hypotheses are produced from valid detections
- **WHEN** a frame has enough detected stars to build pattern features
- **THEN** the matcher returns one or more ranked catalog hypotheses

#### Scenario: Blind solve retries across FOV bands
- **WHEN** no FOV hint is available and the bootstrap database does not yield an accepted match
- **THEN** the pipeline retries matching using dense-band databases centered at configured blind FOV values

### Requirement: Matcher provides confidence and ambiguity signals
The system MUST assign confidence to each hypothesis and MUST flag ambiguous cases where multiple hypotheses are similarly likely.

#### Scenario: Ambiguous match is explicitly flagged
- **WHEN** top hypotheses are within configured confidence margin
- **THEN** the match result includes an ambiguity flag and all competing hypotheses

#### Scenario: Low-confidence match is rejected
- **WHEN** best hypothesis confidence is below acceptance threshold
- **THEN** the matcher returns a no-accept decision instead of a forced top-1 match

### Requirement: Independent verification gates acceptance
The system MUST verify an accepted tetra3 hypothesis against projected catalog stars before treating the solve as successful.

#### Scenario: Verification rejects false positives
- **WHEN** a tetra3 hypothesis fails the independent log-odds and hit-count verification thresholds
- **THEN** the solve is rejected even if tetra3 returned a top hypothesis
