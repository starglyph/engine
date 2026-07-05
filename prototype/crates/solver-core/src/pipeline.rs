use std::path::Path;
use std::time::Instant;

use anyhow::Result;
use simulator_core::catalog::Star;

use crate::config::SolverConfig;
use crate::contracts::{
    DebugArtifacts, PerFrameReport, PoseStageResult, SolveStatus, SolverDiagnostics, SolverResult,
    StageTimingsMs,
};
use crate::detection::{detect_stars, evaluate_detection_metrics};
use crate::io::FrameInput;
use crate::matching::match_catalog_hypotheses;
use crate::overlay::render_debug_layers;
use crate::pose::estimate_pose;

pub fn solve_frame(frame: &FrameInput, catalog: &[Star], config: &SolverConfig) -> SolverResult {
    solve_frame_internal(frame, catalog, config, None).0
}

pub fn solve_frame_with_outputs(
    frame: &FrameInput,
    catalog: &[Star],
    config: &SolverConfig,
    output_dir: &Path,
) -> Result<SolverResult> {
    Ok(solve_frame_internal(frame, catalog, config, Some(output_dir)).0)
}

pub fn solve_frame_and_report(
    frame: &FrameInput,
    catalog: &[Star],
    config: &SolverConfig,
    output_dir: Option<&Path>,
) -> Result<(SolverResult, PerFrameReport)> {
    let (result, metrics) = solve_frame_internal(frame, catalog, config, output_dir);
    let axis_angle_error_deg = result
        .pose
        .estimated_pose
        .as_ref()
        .map(|pose| axis_angle_error_deg(&frame.metadata.pose, pose.ra_deg, pose.dec_deg));
    let roll_error_deg = result.pose.estimated_pose.as_ref().map(|pose| {
        shortest_angular_difference_deg(frame.metadata.pose.roll_deg, pose.roll_deg).abs()
    });
    let report = PerFrameReport {
        frame_id: frame.frame_id.clone(),
        split: frame.split.clone(),
        status: result.status,
        confidence: result.confidence,
        ambiguous: result.ambiguous,
        axis_angle_error_deg,
        roll_error_deg,
        detection: result.detection.metrics.clone(),
        timings_ms: result.diagnostics.stage_timings_ms.clone(),
        rejection_reason: result.rejection_reason.clone(),
        diagnostics_path: metrics,
    };
    Ok((result, report))
}

fn solve_frame_internal(
    frame: &FrameInput,
    catalog: &[Star],
    config: &SolverConfig,
    output_dir: Option<&Path>,
) -> (SolverResult, std::path::PathBuf) {
    let total_start = Instant::now();

    let detect_start = Instant::now();
    let mut detection = detect_stars(&frame.image, &config.detection);
    detection.metrics = Some(evaluate_detection_metrics(
        &detection.candidates,
        &frame.truth_stars,
        config.detection.truth_match_tolerance_px,
    ));
    let detect_elapsed = detect_start.elapsed().as_secs_f64() * 1000.0;

    let match_start = Instant::now();
    let catalog_entries = crate::io::catalog_with_vectors(catalog);
    let matching = match_catalog_hypotheses(
        &detection.candidates,
        &catalog_entries,
        &frame.metadata.camera.intrinsics,
        &config.matching,
    );
    let match_elapsed = match_start.elapsed().as_secs_f64() * 1000.0;

    let pose_start = Instant::now();
    let pose =
        if matching.accepted_correspondences.is_empty() {
            PoseStageResult {
                estimated_pose: None,
                diagnostics: crate::contracts::PoseDiagnostics {
                    ransac_iterations: 0,
                    inlier_count: 0,
                    outlier_count: 0,
                    rms_error_px: None,
                    failure_reason: Some(matching.no_accept_reason.clone().unwrap_or_else(|| {
                        "matching did not produce accepted hypothesis".to_string()
                    })),
                },
                inliers: Vec::new(),
                outliers: Vec::new(),
            }
        } else {
            estimate_pose(
                &matching.accepted_correspondences,
                &frame.metadata.camera.intrinsics,
                &config.pose,
                frame.metadata.frame_seed,
            )
        };
    let pose_elapsed = pose_start.elapsed().as_secs_f64() * 1000.0;

    let overlay_start = Instant::now();
    let mut debug_artifacts = DebugArtifacts::default();
    let diagnostic_path = if let Some(path) = output_dir {
        match render_debug_layers(
            &frame.image,
            &detection,
            &pose,
            &matching.accepted_correspondences,
            &frame.metadata.camera.intrinsics,
            catalog,
            &config.overlay,
            path,
        ) {
            Ok(layers) => {
                debug_artifacts.overlay_path = layers.overlay_path;
                debug_artifacts.detection_layer_path = layers.detection_layer_path;
                debug_artifacts.correspondence_layer_path = layers.correspondence_layer_path;
            }
            Err(_) => {
                // Keep solver deterministic and non-failing when debug rendering cannot be emitted.
            }
        }
        path.join("result.json")
    } else {
        std::path::PathBuf::from("result.json")
    };
    let overlay_elapsed = overlay_start.elapsed().as_secs_f64() * 1000.0;

    let accepted = pose.estimated_pose.is_some();
    let confidence = matching
        .accepted_hypothesis_index
        .and_then(|idx| {
            matching
                .ranked_hypotheses
                .get(idx)
                .map(|hyp| hyp.confidence)
        })
        .unwrap_or_else(|| {
            matching
                .ranked_hypotheses
                .first()
                .map_or(0.0, |hyp| hyp.confidence)
        });
    let accepted_correspondence_count = matching.accepted_correspondences.len();
    let rejection_reason = if accepted {
        None
    } else {
        pose.diagnostics
            .failure_reason
            .clone()
            .or_else(|| matching.no_accept_reason.clone())
    };
    let total_elapsed = total_start.elapsed().as_secs_f64() * 1000.0;

    let result = SolverResult {
        frame_id: frame.frame_id.clone(),
        status: if accepted {
            SolveStatus::Accepted
        } else {
            SolveStatus::Rejected
        },
        confidence,
        ambiguous: matching.ambiguity,
        accepted,
        rejection_reason,
        detection,
        matching,
        pose,
        diagnostics: SolverDiagnostics {
            stage_timings_ms: StageTimingsMs {
                detect: detect_elapsed,
                match_candidates: match_elapsed,
                solve_pose: pose_elapsed,
                render_overlay: overlay_elapsed,
                total: total_elapsed,
            },
            accepted_correspondence_count,
        },
        debug_artifacts,
    };
    (result, diagnostic_path)
}

fn axis_angle_error_deg(
    true_pose: &simulator_core::camera::CameraExtrinsics,
    pred_ra_deg: f32,
    pred_dec_deg: f32,
) -> f32 {
    let true_forward =
        simulator_core::camera::spherical_to_cartesian(true_pose.ra_deg, true_pose.dec_deg);
    let pred_forward = simulator_core::camera::spherical_to_cartesian(pred_ra_deg, pred_dec_deg);
    let dot = (true_forward[0] * pred_forward[0]
        + true_forward[1] * pred_forward[1]
        + true_forward[2] * pred_forward[2])
        .clamp(-1.0, 1.0);
    dot.acos().to_degrees()
}

fn shortest_angular_difference_deg(lhs: f32, rhs: f32) -> f32 {
    (lhs - rhs + 180.0).rem_euclid(360.0) - 180.0
}
