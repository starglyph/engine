## ADDED Requirements

### Requirement: Solver estimates camera pose from matched stars
The system SHALL estimate camera orientation from matched star correspondences and expose the resulting pose in a structured output contract.

#### Scenario: Pose is produced from sufficient inlier correspondences
- **WHEN** a match result contains enough geometrically consistent correspondences
- **THEN** the pose estimator outputs a camera orientation with fit diagnostics

### Requirement: Pose estimation is robust to outliers
The system MUST apply robust outlier filtering during pose estimation to prevent single bad correspondences from dominating the result.

#### Scenario: Outlier-heavy input still yields stable pose
- **WHEN** correspondence set contains a minority of strong outliers
- **THEN** the estimator identifies inliers and computes pose from the inlier subset

#### Scenario: Pose is withheld when robust fit fails
- **WHEN** robust filtering cannot find a stable inlier model
- **THEN** the estimator returns an explicit failure reason and no accepted pose
