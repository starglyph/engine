# Solver image input

## Purpose

Load real photo frames (PNG, JPEG, BMP, TIFF) into a normalized grayscale representation and extract solve-relevant EXIF metadata without collecting location data.

## Requirements

### Requirement: Normalize image orientation before solving
The system SHALL apply valid EXIF Orientation transforms before reporting dimensions,
deriving dimension-dependent FOV hints, detecting stars or displaying the image.
Missing or invalid orientation SHALL preserve the decoded row order.

#### Scenario: A phone portrait is encoded as landscape pixels
- **WHEN** a 4000 by 2252 image has EXIF Orientation 6
- **THEN** the loaded frame is rotated clockwise and reports 2252 by 4000 pixels
- **AND** detections and overlays use that same displayed coordinate system

### Requirement: Frame loader supports common photo formats
The system SHALL decode PNG, JPEG, BMP, and TIFF images into an 8-bit-normalized grayscale [`FrameImage`] suitable for detection.

#### Scenario: JPEG frame is loaded with dimensions preserved
- **WHEN** a valid JPEG file path is provided
- **THEN** the loader returns width, height, normalized grayscale pixels, and a source name stem

#### Scenario: Unsupported or corrupt files fail explicitly
- **WHEN** a file cannot be read or decoded
- **THEN** the loader returns a structured error naming the path and failure reason

### Requirement: EXIF metadata seeds solve hints
The system SHALL extract focal-length and timestamp EXIF fields when present and expose them for FOV and epoch hints.

#### Scenario: 35 mm equivalent focal length becomes FOV prior
- **WHEN** `FocalLengthIn35mmFilm` is present and in a rectilinear sanity range
- **THEN** the frame exposes a horizontal FOV prior in degrees derived from the equivalent focal length

#### Scenario: DateTimeOriginal becomes observation epoch
- **WHEN** `DateTimeOriginal` is present and parseable
- **THEN** the frame exposes a fractional-year epoch for proper-motion and planet overlay

### Requirement: GPS and location EXIF are never extracted
The system MUST NOT read GPS or other location-identifying EXIF tags.

#### Scenario: Location tags are ignored
- **WHEN** a frame contains GPS latitude/longitude EXIF tags
- **THEN** the loader does not populate any location fields in [`ExifMeta`] or downstream contracts
