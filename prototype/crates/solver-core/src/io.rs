use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use image::GrayImage;
use simulator_core::catalog::Star;
use simulator_core::dataset::{DatasetManifest, FrameArtifactRecord, FrameMetadata};

#[derive(Debug, Clone)]
pub struct TruthStar {
    pub star_id: String,
    pub x_px: f32,
    pub y_px: f32,
}

#[derive(Debug, Clone)]
pub struct FrameInput {
    pub frame_id: String,
    pub split: String,
    pub image: GrayImage,
    pub metadata: FrameMetadata,
    pub truth_stars: Vec<TruthStar>,
    pub source_image_path: PathBuf,
}

pub fn load_manifest(dataset_root: &Path) -> Result<DatasetManifest> {
    let path = dataset_root.join("manifest.json");
    let bytes = fs::read(&path).with_context(|| format!("failed to read '{}'", path.display()))?;
    serde_json::from_slice::<DatasetManifest>(&bytes)
        .with_context(|| format!("failed to parse manifest '{}'", path.display()))
}

pub fn load_frame_input(dataset_root: &Path, frame: &FrameArtifactRecord) -> Result<FrameInput> {
    let image_path = dataset_root.join(&frame.image_path);
    let meta_path = dataset_root.join(&frame.meta_path);
    let truth_path = dataset_root.join(&frame.truth_path);

    let image = image::open(&image_path)
        .with_context(|| format!("failed to open frame image '{}'", image_path.display()))?
        .to_luma8();
    let metadata = serde_json::from_slice::<FrameMetadata>(
        &fs::read(&meta_path)
            .with_context(|| format!("failed to read '{}'", meta_path.display()))?,
    )
    .with_context(|| format!("failed to parse metadata '{}'", meta_path.display()))?;
    let truth_stars = load_truth_stars(&truth_path)?;

    Ok(FrameInput {
        frame_id: frame.frame_id.clone(),
        split: frame.split.clone(),
        image,
        metadata,
        truth_stars,
        source_image_path: image_path,
    })
}

pub fn load_truth_stars(path: &Path) -> Result<Vec<TruthStar>> {
    let mut reader = csv::ReaderBuilder::new()
        .has_headers(true)
        .from_path(path)
        .with_context(|| format!("failed to open truth csv '{}'", path.display()))?;
    let mut stars = Vec::new();
    for (row_idx, row) in reader.records().enumerate() {
        let row = row.with_context(|| {
            format!(
                "failed to read truth csv row {} from '{}'",
                row_idx + 2,
                path.display()
            )
        })?;
        let star_id = row.get(0).unwrap_or_default().to_string();
        let x_px = row
            .get(4)
            .unwrap_or_default()
            .parse::<f32>()
            .with_context(|| {
                format!(
                    "invalid x_px at row {} in '{}'",
                    row_idx + 2,
                    path.display()
                )
            })?;
        let y_px = row
            .get(5)
            .unwrap_or_default()
            .parse::<f32>()
            .with_context(|| {
                format!(
                    "invalid y_px at row {} in '{}'",
                    row_idx + 2,
                    path.display()
                )
            })?;
        stars.push(TruthStar {
            star_id,
            x_px,
            y_px,
        });
    }
    Ok(stars)
}

pub fn catalog_with_vectors(stars: &[Star]) -> Vec<CatalogEntry> {
    stars
        .iter()
        .map(|star| CatalogEntry {
            star_id: star.id.clone(),
            world_vector: simulator_core::camera::spherical_to_cartesian(star.ra_deg, star.dec_deg),
            mag_v: star.mag_v,
        })
        .collect()
}

#[derive(Debug, Clone)]
pub struct CatalogEntry {
    pub star_id: String,
    pub world_vector: [f32; 3],
    pub mag_v: f32,
}
