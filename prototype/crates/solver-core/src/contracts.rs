use std::path::PathBuf;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SolveStatus {
    Accepted,
    Rejected,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SolverResult {
    pub frame_id: String,
    pub status: SolveStatus,
    pub confidence: f32,
    pub ambiguous: bool,
    pub accepted: bool,
    pub rejection_reason: Option<String>,
    pub detection: DetectionStageResult,
    pub matching: MatchingStageResult,
    pub pose: PoseStageResult,
    pub diagnostics: SolverDiagnostics,
    pub debug_artifacts: DebugArtifacts,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SolverDiagnostics {
    pub stage_timings_ms: StageTimingsMs,
    pub accepted_correspondence_count: usize,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct StageTimingsMs {
    pub detect: f64,
    pub match_candidates: f64,
    pub solve_pose: f64,
    pub render_overlay: f64,
    pub total: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectionStageResult {
    pub candidates: Vec<DetectionCandidate>,
    pub metrics: Option<DetectionMetrics>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DetectionCandidate {
    pub x_px: f32,
    pub y_px: f32,
    pub intensity: f32,
    pub rank: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectionMetrics {
    pub tp: usize,
    pub fp: usize,
    pub fn_: usize,
    pub precision: f32,
    pub recall: f32,
    pub tolerance_px: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MatchingStageResult {
    pub ranked_hypotheses: Vec<MatchHypothesis>,
    pub accepted_hypothesis_index: Option<usize>,
    pub accepted_correspondences: Vec<StarCorrespondence>,
    pub ambiguity: bool,
    pub no_accept_reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MatchHypothesis {
    pub id: String,
    pub confidence: f32,
    pub score: f32,
    pub correspondences: Vec<StarCorrespondence>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StarCorrespondence {
    pub detection_index: usize,
    pub star_id: String,
    pub image_point_px: [f32; 2],
    pub catalog_direction: [f32; 3],
    pub similarity: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PoseStageResult {
    pub estimated_pose: Option<EstimatedPose>,
    pub diagnostics: PoseDiagnostics,
    pub inliers: Vec<PoseResidual>,
    pub outliers: Vec<PoseResidual>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EstimatedPose {
    pub ra_deg: f32,
    pub dec_deg: f32,
    pub roll_deg: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PoseDiagnostics {
    pub ransac_iterations: usize,
    pub inlier_count: usize,
    pub outlier_count: usize,
    pub rms_error_px: Option<f32>,
    pub failure_reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PoseResidual {
    pub star_id: String,
    pub detection_index: usize,
    pub reprojection_error_px: f32,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DebugArtifacts {
    pub overlay_path: Option<PathBuf>,
    pub detection_layer_path: Option<PathBuf>,
    pub correspondence_layer_path: Option<PathBuf>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerFrameReport {
    pub frame_id: String,
    pub split: String,
    pub status: SolveStatus,
    pub confidence: f32,
    pub ambiguous: bool,
    pub axis_angle_error_deg: Option<f32>,
    pub roll_error_deg: Option<f32>,
    pub detection: Option<DetectionMetrics>,
    pub timings_ms: StageTimingsMs,
    pub rejection_reason: Option<String>,
    pub diagnostics_path: PathBuf,
}
