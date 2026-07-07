## ADDED Requirements

### Requirement: Frame loader supports common photo formats
The system SHALL decode PNG, JPEG, BMP, and TIFF images into an 8-bit-normalized grayscale [`FrameImage`] suitable for detection.

#### Scenario: JPEG frame is loaded with dimensions preserved
- **WHEN** a valid JPEG file path is provided
- **THEN** the loader returns width, height, normalized grayscale pixels, and a source name stem

### Requirement: EXIF metadata seeds solve hints
The system SHALL extract focal-length and timestamp EXIF fields when present and expose them for FOV and epoch hints.

#### Scenario: 35 mm equivalent focal length becomes FOV prior
- **WHEN** `FocalLengthIn35mmFilm` is present and in a rectilinear sanity range
- **THEN** the frame exposes a horizontal FOV prior in degrees derived from the equivalent focal length

### Requirement: GPS and location EXIF are never extracted
The system MUST NOT read GPS or other location-identifying EXIF tags.

#### Scenario: Location tags are ignored
- **WHEN** a frame contains GPS latitude/longitude EXIF tags
- **THEN** the loader does not populate any location fields in [`ExifMeta`] or downstream contracts
