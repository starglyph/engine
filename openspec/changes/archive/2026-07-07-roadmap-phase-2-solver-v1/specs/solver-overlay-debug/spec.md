## ADDED Requirements

### Requirement: Solver renders constellation overlay for accepted pose
The system SHALL render constellation line overlays on top of the source frame when an accepted pose is available.

#### Scenario: Overlay is generated for successful solve
- **WHEN** solver returns an accepted pose
- **THEN** the overlay module outputs an image layer with projected constellation lines aligned to that pose

### Requirement: Solver exposes debug visualization layers
The system MUST provide debug layers that visualize detection points and pose-estimation inliers/outliers for each processed frame.

#### Scenario: Debug layers include detection points
- **WHEN** a frame is processed through detection
- **THEN** the debug output includes rendered markers for detected star candidates

#### Scenario: Debug layers include inliers and outliers
- **WHEN** pose estimation runs with matched correspondences
- **THEN** the debug output includes separate visual markers for inlier and outlier correspondences
