## ADDED Requirements

### Requirement: Magnitude-to-intensity mapping
The simulator SHALL convert stellar magnitude values into image intensity values using a documented deterministic mapping function that preserves relative brightness ordering for rendered stars.

#### Scenario: Brighter stars map to higher intensity
- **WHEN** two visible stars are rendered with different magnitudes where star A is brighter than star B
- **THEN** star A is rendered with equal or higher peak intensity than star B under identical frame conditions

### Requirement: Baseline PSF rendering
The simulator SHALL apply a baseline point spread function when rasterizing visible stars so that each star is represented as a compact distribution rather than a single hard pixel.

#### Scenario: Star footprint spans local neighborhood
- **WHEN** a visible star is rendered at non-saturated intensity
- **THEN** its rendered signal covers a local pixel neighborhood consistent with the configured PSF model

### Requirement: Dynamic-range constraints
The simulator SHALL enforce configured dynamic-range and clipping behavior to avoid physically impossible output values and to preserve deterministic rendering behavior.

#### Scenario: Saturating stars are clipped consistently
- **WHEN** the mapped intensity of a star exceeds the configured sensor upper bound
- **THEN** output pixel values are clipped to the configured maximum in a deterministic manner

### Requirement: Baseline degradations
The simulator SHALL support baseline degradations for shot noise, read noise, mild blur, and JPEG compression artifacts as configurable steps in frame synthesis.

#### Scenario: Degradations are applied in generated frame
- **WHEN** dataset generation is run with degradations enabled
- **THEN** each output frame reflects configured noise/blur/compression effects and stores the applied settings in metadata
