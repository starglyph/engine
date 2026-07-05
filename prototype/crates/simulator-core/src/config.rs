use std::path::PathBuf;

use serde::{Deserialize, Serialize};

pub const SCHEMA_VERSION: &str = "starglyph.synthetic.v1";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetConfig {
    pub seed: u64,
    pub output_root: PathBuf,
    pub splits: SplitConfig,
    pub catalog: CatalogConfig,
    pub camera: CameraConfig,
    pub camera_sampling: CameraSamplingConfig,
    pub render: RenderConfig,
    pub degradations: DegradationConfig,
}

impl Default for DatasetConfig {
    fn default() -> Self {
        Self {
            seed: 42,
            output_root: PathBuf::from("artifacts/simulator/dataset-v1"),
            splits: SplitConfig::default(),
            catalog: CatalogConfig::default(),
            camera: CameraConfig::default(),
            camera_sampling: CameraSamplingConfig::default(),
            render: RenderConfig::default(),
            degradations: DegradationConfig::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CatalogConfig {
    pub csv_path: Option<PathBuf>,
    pub name: String,
    pub subset: String,
    pub license: String,
}

impl Default for CatalogConfig {
    fn default() -> Self {
        Self {
            csv_path: None,
            name: "baseline-built-in".to_string(),
            subset: "bright-stars-sample".to_string(),
            license: "internal-demo".to_string(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SplitConfig {
    pub train_frames: usize,
    pub val_frames: usize,
    pub test_frames: usize,
}

impl SplitConfig {
    #[must_use]
    pub fn total_frames(&self) -> usize {
        self.train_frames + self.val_frames + self.test_frames
    }
}

impl Default for SplitConfig {
    fn default() -> Self {
        Self {
            train_frames: 100,
            val_frames: 20,
            test_frames: 20,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CameraConfig {
    pub width_px: u32,
    pub height_px: u32,
    pub fov_deg: f32,
}

impl Default for CameraConfig {
    fn default() -> Self {
        Self {
            width_px: 4032,
            height_px: 3024,
            fov_deg: 62.0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CameraSamplingConfig {
    pub ra_range_deg: (f32, f32),
    pub dec_range_deg: (f32, f32),
    pub roll_range_deg: (f32, f32),
}

impl Default for CameraSamplingConfig {
    fn default() -> Self {
        Self {
            ra_range_deg: (0.0, 360.0),
            dec_range_deg: (-45.0, 45.0),
            roll_range_deg: (-20.0, 20.0),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RenderConfig {
    pub psf_sigma_px: f32,
    pub dynamic_range_max: u16,
    pub reference_magnitude: f32,
    pub reference_intensity: f32,
    pub background_level: f32,
}

impl Default for RenderConfig {
    fn default() -> Self {
        Self {
            psf_sigma_px: 1.2,
            dynamic_range_max: 4095,
            reference_magnitude: 0.0,
            reference_intensity: 1800.0,
            background_level: 0.02,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DegradationConfig {
    pub shot_noise: bool,
    pub read_noise: bool,
    pub shot_noise_scale: f32,
    pub read_noise_sigma: f32,
    pub blur_sigma_px: f32,
    pub jpeg_quality: u8,
}

impl Default for DegradationConfig {
    fn default() -> Self {
        Self {
            shot_noise: true,
            read_noise: true,
            shot_noise_scale: 0.35,
            read_noise_sigma: 2.5,
            blur_sigma_px: 0.6,
            jpeg_quality: 95,
        }
    }
}
