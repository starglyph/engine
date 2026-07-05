use anyhow::Result;
use tempfile::tempdir;

use simulator_core::config::{CameraConfig, DatasetConfig, SplitConfig};

#[test]
fn dataset_builder_writes_split_artifacts() -> Result<()> {
    let tmp = tempdir()?;
    let mut config = DatasetConfig::default();
    config.output_root = tmp.path().join("dataset-v1");
    config.splits = SplitConfig {
        train_frames: 1,
        val_frames: 1,
        test_frames: 1,
    };
    config.camera = CameraConfig {
        width_px: 320,
        height_px: 240,
        fov_deg: 62.0,
    };

    let manifest = simulator_core::generate_dataset(&config)?;
    assert_eq!(manifest.frames.len(), 3);
    assert!(config.output_root.join("manifest.json").exists());
    for split in ["train", "val", "test"] {
        let split_dir = config.output_root.join(split);
        assert!(split_dir.exists(), "split directory missing: {split}");
    }
    Ok(())
}
