pub mod camera;
pub mod catalog;
pub mod config;
pub mod dataset;
pub mod degradations;
pub mod projection;
pub mod rendering;
pub mod rng;
pub mod sampling;

use anyhow::Result;
use config::DatasetConfig;
use dataset::DatasetManifest;

pub fn generate_dataset(config: &DatasetConfig) -> Result<DatasetManifest> {
    dataset::generate_dataset(config)
}

pub fn validate_reproducibility(config: &DatasetConfig) -> Result<dataset::ReproducibilityReport> {
    dataset::validate_reproducibility(config)
}
