# Simulator dataset generation

## Purpose

Phase 1 dataset v1: one-command generation of train/val/test splits, seeded reproducibility, per-frame metadata, and visible-star truth for downstream evaluation.

## Requirements

### Requirement: One-command dataset build
The simulator tooling SHALL provide a single command entry point that generates dataset v1 outputs without requiring manual per-split orchestration.

#### Scenario: Dataset build command creates all splits
- **WHEN** the dataset generation command is invoked with a valid configuration
- **THEN** train, validation, and test splits are created in the configured output root

### Requirement: Seeded reproducibility
Dataset generation SHALL be reproducible: identical inputs and seed MUST produce byte-equivalent metadata and deterministic frame content for all splits.

#### Scenario: Re-running with same seed reproduces dataset
- **WHEN** the generation command is executed multiple times with identical config and seed
- **THEN** produced artifacts are equivalent according to deterministic reproducibility checks

### Requirement: Frame metadata contract
For each generated frame, the dataset SHALL include machine-readable metadata covering frame identifier, split membership, camera parameters, generation settings, and image artifact linkage.

#### Scenario: Metadata references frame and generation context
- **WHEN** a frame is present in any split
- **THEN** its metadata can be used to recover how the frame was generated and where the corresponding image is stored

### Requirement: Visible-star truth export
Each frame SHALL include truth data listing visible stars and their projected image-space locations for downstream evaluation.

#### Scenario: Truth file contains visible-star annotations
- **WHEN** a frame is generated
- **THEN** associated truth data contains star identifiers and projected coordinates for all stars marked visible by projection logic
