# Solver overlay and debug

## Purpose

Produce structured overlay geometry and optional rendered debug layers for solved frames: constellation lines, labeled stars, planets, and an optional RA/Dec grid.

## Requirements

### Requirement: Solver renders constellation overlay for accepted pose
The system SHALL render constellation line overlays aligned to an accepted camera pose.

#### Scenario: Overlay geometry is returned for successful solve
- **WHEN** solver returns an accepted pose
- **THEN** the overlay module outputs structured geometry (constellation polylines, star markers, optional planets and grid) in the [`SolveOverlay`] contract

#### Scenario: Overlay PNG can be rendered server-side
- **WHEN** the HTTP service receives an overlay format request
- **THEN** the service may render the overlay geometry as a PNG alongside the JSON solve report

### Requirement: Solver exposes debug visualization layers
The system MUST provide debug information that visualizes detection points and pose-estimation inliers/outliers for each processed frame.

#### Scenario: Debug layers include detection points
- **WHEN** a frame is processed through detection
- **THEN** the solve report includes per-detection coordinates, flux, SNR, and inlier flags

#### Scenario: Debug layers include inliers and outliers
- **WHEN** pose estimation runs with matched correspondences
- **THEN** the solve report distinguishes inlier detections from outliers via the `inlier` flag on each [`SolveDetection`]

### Requirement: Overlay respects observation epoch
The system SHALL apply proper-motion and ephemeris corrections when an observation epoch or Julian date is available.

#### Scenario: Planet positions are included when epoch is known
- **WHEN** a solve succeeds and Julian date is resolved from EXIF or explicit hints
- **THEN** the overlay includes projected planet positions for that epoch
