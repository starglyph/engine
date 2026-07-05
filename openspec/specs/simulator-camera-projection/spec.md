# Simulator camera projection

## Purpose

Phase 1 synthetic dataset: deterministic pinhole projection from catalog stars to image coordinates, camera calibration and pose export, and visibility classification for rendering and truth.

## Requirements

### Requirement: Deterministic pinhole projection
The simulator SHALL project catalog stars into image coordinates using a pinhole camera model with explicit intrinsics and extrinsics. For identical input star catalog subset, camera parameters, and seed-derived pose sampling inputs, the projected coordinates and visibility decisions MUST be deterministic.

#### Scenario: Same inputs produce same projection output
- **WHEN** the simulator is executed twice with identical star data, camera parameters, and deterministic seed context
- **THEN** each frame contains identical projected coordinates and visible/invisible classification for all stars

### Requirement: Camera pose and calibration export
For each generated frame, the simulator SHALL export camera intrinsics and extrinsics in machine-readable metadata alongside the image artifact.

#### Scenario: Frame metadata contains camera model parameters
- **WHEN** a frame is generated in dataset v1
- **THEN** its metadata includes focal parameters, principal point, orientation representation, and camera position or equivalent extrinsic transform

### Requirement: Out-of-frustum star handling
The simulator SHALL exclude stars that are outside the camera field of view or behind the camera from rendered star placement while still allowing their status to be reasoned about during validation.

#### Scenario: Behind-camera stars are not rendered
- **WHEN** a star lies outside the camera frustum or has non-visible projection geometry
- **THEN** the star is not rendered into the output image and is marked non-visible in truth outputs
