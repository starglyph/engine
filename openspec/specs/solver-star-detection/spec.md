# Solver star detection

## Purpose

Detect star candidates from grayscale frame images (synthetic or real photos) for the blind-solve pipeline. The detector is tuned for consumer JPEGs and phone exports as well as clean synthetic frames.

## Requirements

### Requirement: Solver detects star candidates on input frames
The system SHALL detect star candidates from a grayscale input frame and produce a deterministic candidate list for fixed input data and configuration.

#### Scenario: Deterministic detection for fixed input
- **WHEN** the same frame and detector configuration are processed multiple times
- **THEN** the detector returns the same ordered list of star candidates with identical coordinates and scores

### Requirement: Detection quality is measurable against truth
The system MUST compute precision and recall for detected stars against reference truth using configurable matching tolerance.

#### Scenario: Precision and recall are reported per frame
- **WHEN** detector output and per-frame truth stars are available
- **THEN** the system reports precision and recall values for that frame

#### Scenario: Aggregate detection metrics are reported for a run
- **WHEN** detection is executed on a dataset split
- **THEN** the system reports aggregate precision and recall across processed frames

### Requirement: Detector handles denoised consumer JPEGs
The detector SHALL apply robust background estimation, adaptive thresholding, and blob filtering so that heavily compressed night-sky JPEGs do not produce thousands of noise blobs.

#### Scenario: Mask fill triggers threshold doubling
- **WHEN** the initial threshold mask covers more than the configured maximum fraction of pixels
- **THEN** the detector doubles the threshold (up to a configured cap) until the mask is sparse enough for reliable blob analysis

#### Scenario: Saturated bright stars are retained
- **WHEN** a round saturated blob exceeds the normal maximum area but meets saturation criteria
- **THEN** the blob is accepted as a star candidate rather than rejected as extended structure

### Requirement: Detection meets Stage 0 performance budget
The detector implementation SHALL process a 12 megapixel frame within the Stage 0 time budget on release builds.

#### Scenario: 12 MP detection completes within budget
- **WHEN** a 12 MP synthetic star field is detected on a release build
- **THEN** wall-clock detection time is at most 1.5 seconds
