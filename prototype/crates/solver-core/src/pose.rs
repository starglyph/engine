use nalgebra::{Matrix3, Vector3};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use simulator_core::camera::{CameraExtrinsics, CameraIntrinsics};

use crate::config::PoseConfig;
use crate::contracts::{
    EstimatedPose, PoseDiagnostics, PoseResidual, PoseStageResult, StarCorrespondence,
};
use crate::matching::pixel_to_camera_vector;

pub fn estimate_pose(
    correspondences: &[StarCorrespondence],
    intrinsics: &CameraIntrinsics,
    config: &PoseConfig,
    seed: u64,
) -> PoseStageResult {
    if correspondences.len() < config.min_inliers {
        return PoseStageResult {
            estimated_pose: None,
            diagnostics: PoseDiagnostics {
                ransac_iterations: 0,
                inlier_count: 0,
                outlier_count: correspondences.len(),
                rms_error_px: None,
                failure_reason: Some("not enough correspondences for robust pose".to_string()),
            },
            inliers: Vec::new(),
            outliers: correspondences
                .iter()
                .map(|corr| PoseResidual {
                    star_id: corr.star_id.clone(),
                    detection_index: corr.detection_index,
                    reprojection_error_px: f32::INFINITY,
                })
                .collect(),
        };
    }

    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut best_inlier_indices: Vec<usize> = Vec::new();
    let mut best_rms = f32::MAX;

    for _ in 0..config.ransac_iterations {
        let sample_indices = unique_sample(correspondences.len(), 3, &mut rng);
        let sample = sample_indices
            .iter()
            .map(|idx| &correspondences[*idx])
            .collect::<Vec<_>>();
        let Some(rotation) = fit_rotation(sample, intrinsics) else {
            continue;
        };
        let residuals = compute_residuals(correspondences, intrinsics, &rotation);
        let inlier_indices = residuals
            .iter()
            .enumerate()
            .filter_map(|(idx, residual)| {
                if residual.reprojection_error_px <= config.inlier_threshold_px {
                    Some(idx)
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();

        if inlier_indices.is_empty() {
            continue;
        }
        let rms = rms_from_indices(&residuals, &inlier_indices);
        if inlier_indices.len() > best_inlier_indices.len()
            || (inlier_indices.len() == best_inlier_indices.len() && rms < best_rms)
        {
            best_inlier_indices = inlier_indices;
            best_rms = rms;
        }
    }

    if best_inlier_indices.len() < config.min_inliers {
        return PoseStageResult {
            estimated_pose: None,
            diagnostics: PoseDiagnostics {
                ransac_iterations: config.ransac_iterations,
                inlier_count: best_inlier_indices.len(),
                outlier_count: correspondences
                    .len()
                    .saturating_sub(best_inlier_indices.len()),
                rms_error_px: None,
                failure_reason: Some("robust fit could not find stable inlier model".to_string()),
            },
            inliers: Vec::new(),
            outliers: correspondences
                .iter()
                .map(|corr| PoseResidual {
                    star_id: corr.star_id.clone(),
                    detection_index: corr.detection_index,
                    reprojection_error_px: f32::INFINITY,
                })
                .collect(),
        };
    }

    let inlier_corrs = best_inlier_indices
        .iter()
        .map(|idx| &correspondences[*idx])
        .collect::<Vec<_>>();
    let Some(rotation) = fit_rotation(inlier_corrs, intrinsics) else {
        return PoseStageResult {
            estimated_pose: None,
            diagnostics: PoseDiagnostics {
                ransac_iterations: config.ransac_iterations,
                inlier_count: 0,
                outlier_count: correspondences.len(),
                rms_error_px: None,
                failure_reason: Some("failed to fit pose from inlier correspondences".to_string()),
            },
            inliers: Vec::new(),
            outliers: correspondences
                .iter()
                .map(|corr| PoseResidual {
                    star_id: corr.star_id.clone(),
                    detection_index: corr.detection_index,
                    reprojection_error_px: f32::INFINITY,
                })
                .collect(),
        };
    };

    let residuals = compute_residuals(correspondences, intrinsics, &rotation);
    let inlier_set = best_inlier_indices
        .into_iter()
        .collect::<std::collections::HashSet<_>>();
    let mut inliers = Vec::new();
    let mut outliers = Vec::new();
    for (idx, residual) in residuals.into_iter().enumerate() {
        if inlier_set.contains(&idx) {
            inliers.push(residual);
        } else {
            outliers.push(residual);
        }
    }

    let pose = rotation_to_pose(&rotation);
    let rms_error_px = if inliers.is_empty() {
        None
    } else {
        Some(
            (inliers
                .iter()
                .map(|r| r.reprojection_error_px * r.reprojection_error_px)
                .sum::<f32>()
                / inliers.len() as f32)
                .sqrt(),
        )
    };

    PoseStageResult {
        estimated_pose: Some(pose),
        diagnostics: PoseDiagnostics {
            ransac_iterations: config.ransac_iterations,
            inlier_count: inliers.len(),
            outlier_count: outliers.len(),
            rms_error_px,
            failure_reason: None,
        },
        inliers,
        outliers,
    }
}

fn unique_sample(count: usize, sample_size: usize, rng: &mut impl Rng) -> Vec<usize> {
    let mut picked = std::collections::BTreeSet::new();
    while picked.len() < sample_size && picked.len() < count {
        picked.insert(rng.random_range(0..count));
    }
    picked.into_iter().collect()
}

fn fit_rotation(
    correspondences: Vec<&StarCorrespondence>,
    intrinsics: &CameraIntrinsics,
) -> Option<Matrix3<f32>> {
    if correspondences.len() < 2 {
        return None;
    }
    let mut covariance = Matrix3::<f32>::zeros();
    for correspondence in correspondences {
        let camera = pixel_to_camera_vector(
            correspondence.image_point_px[0],
            correspondence.image_point_px[1],
            intrinsics,
        );
        let cam_vec = Vector3::new(camera[0], camera[1], camera[2]);
        let world_vec = Vector3::new(
            correspondence.catalog_direction[0],
            correspondence.catalog_direction[1],
            correspondence.catalog_direction[2],
        );
        covariance += cam_vec * world_vec.transpose();
    }
    let svd = covariance.svd(true, true);
    let u = svd.u?;
    let v_t = svd.v_t?;
    let mut correction = Matrix3::<f32>::identity();
    if (u * v_t).determinant() < 0.0 {
        correction[(2, 2)] = -1.0;
    }
    Some(u * correction * v_t)
}

fn compute_residuals(
    correspondences: &[StarCorrespondence],
    intrinsics: &CameraIntrinsics,
    rotation: &Matrix3<f32>,
) -> Vec<PoseResidual> {
    correspondences
        .iter()
        .map(|correspondence| {
            let projected =
                project_catalog_vector(correspondence.catalog_direction, intrinsics, rotation);
            let dx = projected[0] - correspondence.image_point_px[0];
            let dy = projected[1] - correspondence.image_point_px[1];
            PoseResidual {
                star_id: correspondence.star_id.clone(),
                detection_index: correspondence.detection_index,
                reprojection_error_px: (dx * dx + dy * dy).sqrt(),
            }
        })
        .collect()
}

fn rms_from_indices(residuals: &[PoseResidual], indices: &[usize]) -> f32 {
    (indices
        .iter()
        .map(|idx| residuals[*idx].reprojection_error_px.powi(2))
        .sum::<f32>()
        / indices.len() as f32)
        .sqrt()
}

fn project_catalog_vector(
    world_vector: [f32; 3],
    intrinsics: &CameraIntrinsics,
    rotation: &Matrix3<f32>,
) -> [f32; 2] {
    let world = Vector3::new(world_vector[0], world_vector[1], world_vector[2]);
    let cam = rotation * world;
    let z = cam[2].max(1e-6);
    let x = intrinsics.fx * (cam[0] / z) + intrinsics.cx;
    let y = intrinsics.cy - intrinsics.fy * (cam[1] / z);
    [x, y]
}

fn rotation_to_pose(rotation: &Matrix3<f32>) -> EstimatedPose {
    let forward = [rotation[(2, 0)], rotation[(2, 1)], rotation[(2, 2)]];
    let right = [rotation[(0, 0)], rotation[(0, 1)], rotation[(0, 2)]];
    let ra_deg = forward[1].atan2(forward[0]).to_degrees().rem_euclid(360.0);
    let dec_deg = forward[2].clamp(-1.0, 1.0).asin().to_degrees();

    let world_up = if forward[2].abs() > 0.99 {
        [0.0, 1.0, 0.0]
    } else {
        [0.0, 0.0, 1.0]
    };
    let canonical_right = normalize(cross(forward, world_up));
    let canonical_up = normalize(cross(canonical_right, forward));
    let cos_roll = dot(right, canonical_right).clamp(-1.0, 1.0);
    let sin_roll = dot(right, canonical_up);
    let roll_deg = sin_roll.atan2(cos_roll).to_degrees();

    EstimatedPose {
        ra_deg,
        dec_deg,
        roll_deg,
    }
}

pub fn estimated_pose_to_extrinsics(pose: &EstimatedPose) -> CameraExtrinsics {
    CameraExtrinsics {
        ra_deg: pose.ra_deg,
        dec_deg: pose.dec_deg,
        roll_deg: pose.roll_deg,
        position: [0.0, 0.0, 0.0],
    }
}

fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn normalize(v: [f32; 3]) -> [f32; 3] {
    let norm = dot(v, v).sqrt().max(1e-6);
    [v[0] / norm, v[1] / norm, v[2] / norm]
}

#[cfg(test)]
mod tests {
    use super::*;
    use simulator_core::camera::{spherical_to_cartesian, CameraIntrinsics};
    use simulator_core::config::CameraConfig;

    fn intrinsics() -> CameraIntrinsics {
        CameraIntrinsics::from_camera_config(&CameraConfig {
            width_px: 1024,
            height_px: 768,
            fov_deg: 62.0,
        })
    }

    #[test]
    fn robust_pose_rejects_unstable_correspondences() {
        let correspondences = vec![
            StarCorrespondence {
                detection_index: 0,
                star_id: "a".to_string(),
                image_point_px: [100.0, 100.0],
                catalog_direction: spherical_to_cartesian(10.0, 10.0),
                similarity: 0.9,
            },
            StarCorrespondence {
                detection_index: 1,
                star_id: "b".to_string(),
                image_point_px: [200.0, 120.0],
                catalog_direction: spherical_to_cartesian(130.0, -20.0),
                similarity: 0.8,
            },
            StarCorrespondence {
                detection_index: 2,
                star_id: "c".to_string(),
                image_point_px: [800.0, 700.0],
                catalog_direction: spherical_to_cartesian(250.0, 30.0),
                similarity: 0.8,
            },
        ];
        let cfg = PoseConfig {
            min_inliers: 4,
            ..PoseConfig::default()
        };
        let pose = estimate_pose(&correspondences, &intrinsics(), &cfg, 42);
        assert!(pose.estimated_pose.is_none());
        assert!(pose.diagnostics.failure_reason.is_some());
    }
}
