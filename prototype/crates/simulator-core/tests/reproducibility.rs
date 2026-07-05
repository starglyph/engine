use anyhow::Result;
use tempfile::tempdir;

use simulator_core::config::{CameraConfig, DatasetConfig, SplitConfig};

#[test]
fn same_seed_generation_is_equivalent() -> Result<()> {
    let tmp = tempdir()?;
    let mut config = DatasetConfig::default();
    config.output_root = tmp.path().join("dataset-v1");
    config.splits = SplitConfig {
        train_frames: 2,
        val_frames: 1,
        test_frames: 1,
    };
    config.camera = CameraConfig {
        width_px: 320,
        height_px: 240,
        fov_deg: 62.0,
    };

    let report = simulator_core::validate_reproducibility(&config)?;
    assert!(report.equivalent);
    Ok(())
}
