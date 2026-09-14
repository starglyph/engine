# Solver pattern matching

## Purpose

Match detected star patterns against a tetra3 pattern database built from the HYG catalog. Produces ranked attitude hypotheses with confidence and ambiguity signals.

## Requirements

### Requirement: Large frames support bounded working images
The solver SHALL try a working image with longest edge at most 1600 pixels before
retrying at original resolution. It SHALL NOT upscale smaller inputs. Reports and
cached overlay cameras SHALL use the full EXIF-oriented image coordinates, and RMS
SHALL be expressed in those pixels. Rescaling SHALL NOT increase match confidence.

#### Scenario: The working image cannot be solved
- **WHEN** matching the reduced frame fails
- **THEN** the original frame is also attempted before returning failure

#### Scenario: Automatic EXIF FOV is misleading
- **WHEN** an EXIF-derived FOV hint yields no accepted solution
- **THEN** the solver retries without that automatic FOV hint
- **AND** explicit caller hints and the acquisition epoch remain unchanged

#### Scenario: A different photo is opened in the desktop app
- **WHEN** a new image is loaded after a successful solve
- **THEN** the previous image's inferred FOV is cleared

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
