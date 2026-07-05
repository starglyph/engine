use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};

use crate::camera::{CameraExtrinsics, CameraIntrinsics};
use crate::catalog::{load_catalog, Star};
use crate::config::{DatasetConfig, SCHEMA_VERSION};
use crate::degradations::{apply_baseline_degradations, AppliedDegradations};
use crate::projection::{project_catalog, ProjectedStar};
use crate::rendering::render_stars;
use crate::rng::{directory_digest, SeedDeriver};
use crate::sampling::sample_camera_pose;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DatasetSplit {
    Train,
    Val,
    Test,
}

impl DatasetSplit {
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Train => "train",
            Self::Val => "val",
            Self::Test => "test",
        }
    }
}

#[derive(Debug, Clone)]
pub struct PlannedFrame {
    pub split: DatasetSplit,
    pub split_frame_index: usize,
}

#[derive(Debug, Clone)]
pub struct DatasetPlan {
    pub frames: Vec<PlannedFrame>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetManifest {
    pub schema_version: String,
    pub seed: u64,
    pub output_root: PathBuf,
    pub splits: SplitCounts,
    pub frames: Vec<FrameArtifactRecord>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SplitCounts {
    pub train: usize,
    pub val: usize,
    pub test: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrameArtifactRecord {
    pub frame_id: String,
    pub split: String,
    pub image_path: PathBuf,
    pub meta_path: PathBuf,
    pub truth_path: PathBuf,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrameMetadata {
    pub schema_version: String,
    pub frame_id: String,
    pub split: String,
    pub timestamp_utc: String,
    pub frame_seed: u64,
    pub image_path: PathBuf,
    pub camera: FrameCameraMetadata,
    pub pose: CameraExtrinsics,
    pub render: crate::config::RenderConfig,
    pub catalog: FrameCatalogMetadata,
    pub degradations: AppliedDegradations,
    pub generator: FrameGeneratorMetadata,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrameCameraMetadata {
    pub width_px: u32,
    pub height_px: u32,
    pub fov_deg: f32,
    pub intrinsics: CameraIntrinsics,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrameGeneratorMetadata {
    pub split_frame_index: usize,
    pub total_split_frames: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrameCatalogMetadata {
    pub name: String,
    pub subset: String,
    pub license: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReproducibilityReport {
    pub first_digest: String,
    pub second_digest: String,
    pub equivalent: bool,
}

#[derive(Debug, thiserror::Error)]
pub enum DatasetError {
    #[error("split '{split}' requires at least one frame")]
    EmptySplit { split: &'static str },
    #[error("seed reproducibility check failed: {first_digest} != {second_digest}")]
    ReproducibilityMismatch {
        first_digest: String,
        second_digest: String,
    },
}

pub fn plan_from_config(config: &DatasetConfig) -> DatasetPlan {
    let mut frames = Vec::with_capacity(config.splits.total_frames());
    for split in [DatasetSplit::Train, DatasetSplit::Val, DatasetSplit::Test] {
        let count = split_frame_count(config, split);
        for split_frame_index in 0..count {
            frames.push(PlannedFrame {
                split,
                split_frame_index,
            });
        }
    }
    DatasetPlan { frames }
}

pub fn generate_dataset(config: &DatasetConfig) -> Result<DatasetManifest> {
    validate_splits(config)?;
    if config.output_root.exists() {
        fs::remove_dir_all(&config.output_root).with_context(|| {
            format!(
                "failed to remove existing output directory '{}'",
                config.output_root.display()
            )
        })?;
    }
    fs::create_dir_all(&config.output_root).with_context(|| {
        format!(
            "failed to create dataset output root '{}'",
            config.output_root.display()
        )
    })?;

    let catalog = load_catalog(&config.catalog)?;
    let seed_deriver = SeedDeriver::new(config.seed);
    let plan = plan_from_config(config);
    let mut records = Vec::with_capacity(plan.frames.len());
    for frame in plan.frames {
        records.push(generate_frame(config, &catalog, &seed_deriver, &frame)?);
    }

    let manifest = DatasetManifest {
        schema_version: SCHEMA_VERSION.to_string(),
        seed: config.seed,
        output_root: config.output_root.clone(),
        splits: SplitCounts {
            train: config.splits.train_frames,
            val: config.splits.val_frames,
            test: config.splits.test_frames,
        },
        frames: records,
    };
    let manifest_path = config.output_root.join("manifest.json");
    fs::write(&manifest_path, serde_json::to_vec_pretty(&manifest)?)
        .with_context(|| format!("failed to write manifest '{}'", manifest_path.display()))?;

    Ok(manifest)
}

pub fn validate_reproducibility(config: &DatasetConfig) -> Result<ReproducibilityReport> {
    let base_tmp = std::env::temp_dir().join(format!("starglyph-repro-{}", config.seed));
    let run_root = base_tmp.join("run");

    if base_tmp.exists() {
        fs::remove_dir_all(&base_tmp)?;
    }
    fs::create_dir_all(&base_tmp)?;

    let mut first_config = config.clone();
    first_config.output_root = run_root.clone();
    generate_dataset(&first_config)?;
    let first_digest = directory_digest(&run_root)?;

    generate_dataset(&first_config)?;
    let second_digest = directory_digest(&run_root)?;
    let equivalent = first_digest == second_digest;
    let report = ReproducibilityReport {
        first_digest,
        second_digest,
        equivalent,
    };

    fs::remove_dir_all(&base_tmp)?;
    if !report.equivalent {
        return Err(DatasetError::ReproducibilityMismatch {
            first_digest: report.first_digest,
            second_digest: report.second_digest,
        }
        .into());
    }
    Ok(report)
}

fn validate_splits(config: &DatasetConfig) -> Result<()> {
    for split in [DatasetSplit::Train, DatasetSplit::Val, DatasetSplit::Test] {
        if split_frame_count(config, split) == 0 {
            return Err(DatasetError::EmptySplit {
                split: split.as_str(),
            }
            .into());
        }
    }
    Ok(())
}

fn generate_frame(
    config: &DatasetConfig,
    catalog: &[Star],
    seed_deriver: &SeedDeriver,
    frame: &PlannedFrame,
) -> Result<FrameArtifactRecord> {
    let frame_seed = seed_deriver.frame_seed(frame.split, frame.split_frame_index);
    let mut frame_rng = seed_deriver.frame_rng(frame.split, frame.split_frame_index);

    let intrinsics = CameraIntrinsics::from_camera_config(&config.camera);
    let pose = sample_camera_pose(&config.camera_sampling, &mut frame_rng);
    let projected = project_catalog(catalog, &intrinsics, &pose);
    let rendered = render_stars(
        config.camera.width_px,
        config.camera.height_px,
        &projected,
        &config.render,
    );
    let (degraded, applied_degradations) =
        apply_baseline_degradations(rendered, &config.degradations, &mut frame_rng)?;

    let frame_id = format!("frame-{:06}", frame.split_frame_index + 1);
    let frame_dir = config
        .output_root
        .join(frame.split.as_str())
        .join(frame_id.clone());
    fs::create_dir_all(&frame_dir)
        .with_context(|| format!("failed to create frame dir '{}'", frame_dir.display()))?;

    let image_path = frame_dir.join("image.png");
    degraded
        .to_gray_image()
        .save_with_format(&image_path, image::ImageFormat::Png)
        .with_context(|| format!("failed to save image '{}'", image_path.display()))?;
    let truth_path = frame_dir.join("truth-stars.csv");
    write_truth_csv(&truth_path, &projected)?;

    let meta_path = frame_dir.join("meta.json");
    let metadata = build_frame_metadata(
        config,
        frame,
        &frame_id,
        frame_seed,
        &image_path,
        &intrinsics,
        &pose,
        applied_degradations,
    );
    fs::write(&meta_path, serde_json::to_vec_pretty(&metadata)?)
        .with_context(|| format!("failed to write metadata '{}'", meta_path.display()))?;

    Ok(FrameArtifactRecord {
        frame_id,
        split: frame.split.as_str().to_string(),
        image_path: relative_to_root(&config.output_root, &image_path),
        meta_path: relative_to_root(&config.output_root, &meta_path),
        truth_path: relative_to_root(&config.output_root, &truth_path),
    })
}

fn build_frame_metadata(
    config: &DatasetConfig,
    frame: &PlannedFrame,
    frame_id: &str,
    frame_seed: u64,
    image_path: &Path,
    intrinsics: &CameraIntrinsics,
    pose: &CameraExtrinsics,
    degradations: AppliedDegradations,
) -> FrameMetadata {
    let base_time: DateTime<Utc> = DateTime::UNIX_EPOCH + Duration::seconds(1_700_000_000);
    let offset_seconds = (frame.split_frame_index as i64)
        + match frame.split {
            DatasetSplit::Train => 0,
            DatasetSplit::Val => 1_000,
            DatasetSplit::Test => 2_000,
        };
    let timestamp = base_time + Duration::seconds(offset_seconds);

    FrameMetadata {
        schema_version: SCHEMA_VERSION.to_string(),
        frame_id: frame_id.to_string(),
        split: frame.split.as_str().to_string(),
        timestamp_utc: timestamp.to_rfc3339(),
        frame_seed,
        image_path: relative_to_root(&config.output_root, image_path),
        camera: FrameCameraMetadata {
            width_px: config.camera.width_px,
            height_px: config.camera.height_px,
            fov_deg: config.camera.fov_deg,
            intrinsics: intrinsics.clone(),
        },
        pose: pose.clone(),
        render: config.render.clone(),
        catalog: FrameCatalogMetadata {
            name: config.catalog.name.clone(),
            subset: config.catalog.subset.clone(),
            license: config.catalog.license.clone(),
        },
        degradations,
        generator: FrameGeneratorMetadata {
            split_frame_index: frame.split_frame_index,
            total_split_frames: split_frame_count(config, frame.split),
        },
    }
}

fn write_truth_csv(path: &Path, projected: &[ProjectedStar]) -> Result<()> {
    let mut csv =
        String::from("star_id,ra_deg,dec_deg,mag_v,x_px,y_px,flux_rel,is_in_frame,is_occluded\n");
    for star in projected.iter().filter(|star| star.is_visible()) {
        let flux_rel = 10.0_f32.powf(-0.4 * star.mag_v);
        csv.push_str(&format!(
            "{},{:.6},{:.6},{:.3},{:.3},{:.3},{:.6},1,0\n",
            star.id, star.ra_deg, star.dec_deg, star.mag_v, star.x_px, star.y_px, flux_rel
        ));
    }
    fs::write(path, csv).with_context(|| format!("failed to write truth csv '{}'", path.display()))
}

#[must_use]
fn split_frame_count(config: &DatasetConfig, split: DatasetSplit) -> usize {
    match split {
        DatasetSplit::Train => config.splits.train_frames,
        DatasetSplit::Val => config.splits.val_frames,
        DatasetSplit::Test => config.splits.test_frames,
    }
}

#[must_use]
fn relative_to_root(root: &Path, absolute_path: &Path) -> PathBuf {
    absolute_path
        .strip_prefix(root)
        .map(Path::to_path_buf)
        .unwrap_or_else(|_| absolute_path.to_path_buf())
}
