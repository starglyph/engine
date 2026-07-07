# Solver engine database

## Purpose

Lifecycle management for tetra3 pattern databases: bootstrap (blind solve) and on-demand dense-band databases built from the HYG catalog with disk caching and configurable magnitude depth.

## Requirements

### Requirement: Bootstrap database supports blind solving
The system SHALL build or load a broad multiscale tetra3 bootstrap database (approximately 10–70° FOV) from the HYG catalog for lost-in-space matching.

#### Scenario: Bootstrap database is cached on disk
- **WHEN** a bootstrap database is generated for the first time
- **THEN** the resulting `.bin` file is written to the configured cache directory with a stable versioned file name

#### Scenario: Cached bootstrap is reused
- **WHEN** a bootstrap database file already exists in the cache directory
- **THEN** subsequent engine instances load it without regenerating

### Requirement: Dense-band database is generated on demand
The system SHALL build a higher-density tetra3 database tuned to a known FOV band when the bootstrap database is too sparse for reliable matching.

#### Scenario: Dense band is keyed by rounded FOV center
- **WHEN** a FOV hint or blind retry center is resolved
- **THEN** the dense-band cache key rounds the center to a whole degree so nearby hints share one database

#### Scenario: Dense band is skipped for very narrow fields
- **WHEN** the dense-band center is below the configured minimum
- **THEN** dense-band generation is skipped because the mag-limited catalog cannot populate the field

### Requirement: Database magnitude depth is configurable
The system SHALL allow overriding the faintest catalog magnitude included in generated databases via environment variable without clobbering databases built at other depths.

#### Scenario: Mag limit env produces distinct cache files
- **WHEN** `STARGLYPH_DB_MAG_LIMIT` is set to a non-default value
- **THEN** generated cache file names include a magnitude tag (e.g. `mag70`) distinct from the default `mag65` set

### Requirement: Concurrent generation is single-flighted
The system MUST serialize concurrent database generation for the same cache directory so parallel HTTP workers do not rebuild identical multi-hundred-megabyte files.

#### Scenario: Parallel misses share one build
- **WHEN** multiple engines miss the same cache file simultaneously
- **THEN** only one generation runs and others wait for the result
