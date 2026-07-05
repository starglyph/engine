use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use image::{GrayImage, Rgb, RgbImage};
use simulator_core::camera::CameraIntrinsics;
use simulator_core::catalog::Star;
use simulator_core::projection::project_star;

use crate::config::OverlayConfig;
use crate::contracts::{DetectionStageResult, PoseStageResult, StarCorrespondence};
use crate::pose::estimated_pose_to_extrinsics;

const CONSTELLATION_LINES: &[(&str, &str)] = &[
    ("betelgeuse", "rigel"),
    ("betelgeuse", "procyon"),
    ("rigel", "aldebaran"),
    ("vega", "deneb"),
    ("deneb", "altair"),
    ("antares", "arcturus"),
    ("pollux", "procyon"),
    ("sirius", "procyon"),
];

#[derive(Debug, Clone, Default)]
pub struct RenderedLayers {
    pub overlay_path: Option<PathBuf>,
    pub detection_layer_path: Option<PathBuf>,
    pub correspondence_layer_path: Option<PathBuf>,
}

pub fn render_debug_layers(
    image: &GrayImage,
    detection: &DetectionStageResult,
    pose: &PoseStageResult,
    correspondences: &[StarCorrespondence],
    intrinsics: &CameraIntrinsics,
    catalog: &[Star],
    config: &OverlayConfig,
    output_dir: &Path,
) -> Result<RenderedLayers> {
    std::fs::create_dir_all(output_dir)
        .with_context(|| format!("failed to create '{}'", output_dir.display()))?;

    let mut layers = RenderedLayers::default();
    layers.detection_layer_path = Some(render_detection_layer(
        image,
        detection,
        config,
        &output_dir.join("detections.png"),
    )?);
    layers.correspondence_layer_path = Some(render_correspondence_layer(
        image,
        correspondences,
        pose,
        intrinsics,
        catalog,
        &output_dir.join("correspondences.png"),
    )?);
    layers.overlay_path =
        render_constellation_overlay(image, pose, intrinsics, catalog, output_dir)?;
    Ok(layers)
}

fn render_detection_layer(
    image: &GrayImage,
    detection: &DetectionStageResult,
    config: &OverlayConfig,
    path: &Path,
) -> Result<PathBuf> {
    let mut rgb = gray_to_rgb(image);
    for candidate in &detection.candidates {
        draw_circle(
            &mut rgb,
            candidate.x_px.round() as i32,
            candidate.y_px.round() as i32,
            config.detection_marker_radius_px as i32,
            Rgb([255, 220, 0]),
        );
    }
    rgb.save(path)
        .with_context(|| format!("failed to save '{}'", path.display()))?;
    Ok(path.to_path_buf())
}

fn render_correspondence_layer(
    image: &GrayImage,
    correspondences: &[StarCorrespondence],
    pose: &PoseStageResult,
    intrinsics: &CameraIntrinsics,
    catalog: &[Star],
    path: &Path,
) -> Result<PathBuf> {
    let mut rgb = gray_to_rgb(image);
    let catalog_map = catalog
        .iter()
        .map(|star| (star.id.as_str(), star))
        .collect::<HashMap<_, _>>();
    let Some(estimated_pose) = &pose.estimated_pose else {
        rgb.save(path)
            .with_context(|| format!("failed to save '{}'", path.display()))?;
        return Ok(path.to_path_buf());
    };
    let extrinsics = estimated_pose_to_extrinsics(estimated_pose);
    let inlier_ids = pose
        .inliers
        .iter()
        .map(|item| item.detection_index)
        .collect::<std::collections::HashSet<_>>();

    for correspondence in correspondences {
        let Some(star) = catalog_map.get(correspondence.star_id.as_str()) else {
            continue;
        };
        let projected = project_star(star, intrinsics, &extrinsics);
        if !projected.is_visible() {
            continue;
        }
        let color = if inlier_ids.contains(&correspondence.detection_index) {
            Rgb([0, 255, 0])
        } else {
            Rgb([255, 64, 64])
        };
        draw_line(
            &mut rgb,
            correspondence.image_point_px[0].round() as i32,
            correspondence.image_point_px[1].round() as i32,
            projected.x_px.round() as i32,
            projected.y_px.round() as i32,
            color,
        );
    }
    rgb.save(path)
        .with_context(|| format!("failed to save '{}'", path.display()))?;
    Ok(path.to_path_buf())
}

fn render_constellation_overlay(
    image: &GrayImage,
    pose: &PoseStageResult,
    intrinsics: &CameraIntrinsics,
    catalog: &[Star],
    output_dir: &Path,
) -> Result<Option<PathBuf>> {
    let Some(estimated_pose) = &pose.estimated_pose else {
        return Ok(None);
    };
    let path = output_dir.join("overlay.png");
    let mut rgb = gray_to_rgb(image);
    let catalog_map = catalog
        .iter()
        .map(|star| (star.id.as_str(), star))
        .collect::<HashMap<_, _>>();
    let extrinsics = estimated_pose_to_extrinsics(estimated_pose);
    for (left_id, right_id) in CONSTELLATION_LINES {
        let (Some(left), Some(right)) = (catalog_map.get(left_id), catalog_map.get(right_id))
        else {
            continue;
        };
        let left_proj = project_star(left, intrinsics, &extrinsics);
        let right_proj = project_star(right, intrinsics, &extrinsics);
        if !(left_proj.is_visible() && right_proj.is_visible()) {
            continue;
        }
        draw_line(
            &mut rgb,
            left_proj.x_px.round() as i32,
            left_proj.y_px.round() as i32,
            right_proj.x_px.round() as i32,
            right_proj.y_px.round() as i32,
            Rgb([64, 255, 255]),
        );
    }
    rgb.save(&path)
        .with_context(|| format!("failed to save '{}'", path.display()))?;
    Ok(Some(path))
}

fn gray_to_rgb(image: &GrayImage) -> RgbImage {
    let mut rgb = RgbImage::new(image.width(), image.height());
    for (x, y, px) in image.enumerate_pixels() {
        rgb.put_pixel(x, y, Rgb([px.0[0], px.0[0], px.0[0]]));
    }
    rgb
}

fn draw_circle(image: &mut RgbImage, cx: i32, cy: i32, radius: i32, color: Rgb<u8>) {
    for dy in -radius..=radius {
        for dx in -radius..=radius {
            if dx * dx + dy * dy > radius * radius {
                continue;
            }
            put_pixel_safe(image, cx + dx, cy + dy, color);
        }
    }
}

fn draw_line(image: &mut RgbImage, x0: i32, y0: i32, x1: i32, y1: i32, color: Rgb<u8>) {
    let mut x0_mut = x0;
    let mut y0_mut = y0;
    let dx = (x1 - x0_mut).abs();
    let sx = if x0_mut < x1 { 1 } else { -1 };
    let dy = -(y1 - y0_mut).abs();
    let sy = if y0_mut < y1 { 1 } else { -1 };
    let mut err = dx + dy;
    loop {
        put_pixel_safe(image, x0_mut, y0_mut, color);
        if x0_mut == x1 && y0_mut == y1 {
            break;
        }
        let e2 = 2 * err;
        if e2 >= dy {
            if x0_mut == x1 {
                break;
            }
            err += dy;
            x0_mut += sx;
        }
        if e2 <= dx {
            if y0_mut == y1 {
                break;
            }
            err += dx;
            y0_mut += sy;
        }
    }
}

fn put_pixel_safe(image: &mut RgbImage, x: i32, y: i32, color: Rgb<u8>) {
    if x < 0 || y < 0 || x >= image.width() as i32 || y >= image.height() as i32 {
        return;
    }
    image.put_pixel(x as u32, y as u32, color);
}
