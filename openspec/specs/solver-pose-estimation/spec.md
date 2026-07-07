# Solver pose estimation

## Purpose

Estimate camera orientation and intrinsics from matched star correspondences. Uses Levenberg–Marquardt refinement with optional radial distortion (k1) for wide fields.

## Requirements

### Requirement: Solver estimates camera pose from matched stars
The system SHALL estimate camera orientation (RA, Dec, roll), focal length, and field of view from matched star correspondences and expose the result in a structured [`SolveReport`] contract.

#### Scenario: Pose is produced from sufficient inlier correspondences
- **WHEN** a match result contains enough geometrically consistent correspondences
- **THEN** the pose estimator outputs a camera orientation with fit diagnostics (inlier count, RMS, log-odds, confidence)

### Requirement: Pose estimation is robust to outliers
The system MUST apply robust outlier filtering during pose estimation to prevent single bad correspondences from dominating the result.

#### Scenario: Outlier-heavy input still yields stable pose
- **WHEN** correspondence set contains a minority of strong outliers
- **THEN** the estimator identifies inliers and computes pose from the inlier subset

#### Scenario: Pose is withheld when robust fit fails
- **WHEN** robust filtering cannot find a stable inlier model
- **THEN** the estimator returns an explicit failure reason and no accepted pose

### Requirement: Radial distortion k1 is refined when identifiable
The system SHALL optionally refine a radial distortion coefficient k1 during LM pose refinement when enough matches are available, with FOV-dependent regularization toward zero.

#### Scenario: k1 is free on wide fields with sufficient matches
- **WHEN** at least eight verified matches are available and the initial horizontal FOV exceeds the narrow-field taper start
- **THEN** k1 is included as a free parameter in the LM refinement

#### Scenario: k1 is clamped on narrow fields
- **WHEN** the initial horizontal FOV is below the narrow-field taper start
- **THEN** k1 regularization strongly favors zero to prevent noise-driven false distortion fits

#### Scenario: Wide-field re-matching admits edge stars
- **WHEN** barrel distortion displaces edge stars outside the initial match radius
- **THEN** the pipeline re-matches under the refined model using a widening-then-tightening radius ladder before final pose output
