## ADDED Requirements

### Requirement: Solver matches detected patterns to catalog hypotheses
The system SHALL generate catalog match hypotheses from detected stars using a baseline pattern-matching algorithm.

#### Scenario: Candidate hypotheses are produced from valid detections
- **WHEN** a frame has enough detected stars to build pattern features
- **THEN** the matcher returns one or more ranked catalog hypotheses

### Requirement: Matcher provides confidence and ambiguity signals
The system MUST assign confidence to each hypothesis and MUST flag ambiguous cases where multiple hypotheses are similarly likely.

#### Scenario: Ambiguous match is explicitly flagged
- **WHEN** top hypotheses are within configured confidence margin
- **THEN** the match result includes an ambiguity flag and all competing hypotheses

#### Scenario: Low-confidence match is rejected
- **WHEN** best hypothesis confidence is below acceptance threshold
- **THEN** the matcher returns a no-accept decision instead of a forced top-1 match
