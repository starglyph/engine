//! Serializable DTOs shared between GUI and CLI.

use serde::{Deserialize, Serialize};

/// Overall solve outcome.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SolveStatus {
    Solved,
    Failed,
}

/// Failure details when [`SolveStatus::Failed`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SolveFailure {
    pub code: String,
    pub message: String,
}

/// Solved plate pose.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SolvePose {
    pub ra_deg: f64,
    pub dec_deg: f64,
    pub roll_deg: f64,
}

/// Field-of-view and focal length.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SolveFov {
    pub fov_x_deg: f64,
    pub fov_y_deg: f64,
    pub focal_px: f64,
}

/// Match quality metrics.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SolveQuality {
    pub n_detections: u32,
    pub n_inliers: u32,
    pub rms_px: f64,
    pub log_odds: f64,
    pub confidence: f64,
}

/// Per-stage timing in milliseconds.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SolveTimingMs {
    pub detect: u64,
    pub solve: u64,
    pub total: u64,
}

/// A single star detection.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SolveDetection {
    pub x: f64,
    pub y: f64,
    pub flux: f64,
    pub snr: f64,
    pub inlier: bool,
}

/// One constellation polyline overlay.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OverlayConstellation {
    pub abbr: String,
    pub name: String,
    pub lines: Vec<Vec<[f64; 2]>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub label_xy: Option<[f64; 2]>,
}

/// A labeled star overlay point.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OverlayStar {
    pub x: f64,
    pub y: f64,
    pub mag: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
    pub hip: u32,
}

/// A planet overlay point.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OverlayPlanet {
    pub x: f64,
    pub y: f64,
    pub name: String,
    pub mag: f64,
    pub approx: bool,
}

/// RA/Dec grid segment.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OverlayGridLine {
    pub kind: String,
    pub value_deg: f64,
    pub points: Vec<[f64; 2]>,
}

/// Full overlay geometry for a solved frame.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SolveOverlay {
    pub constellations: Vec<OverlayConstellation>,
    pub stars: Vec<OverlayStar>,
    pub planets: Vec<OverlayPlanet>,
    pub grid: Vec<OverlayGridLine>,
}

/// Complete solve report returned to the GUI/CLI.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SolveReport {
    pub status: SolveStatus,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub failure: Option<SolveFailure>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pose: Option<SolvePose>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fov: Option<SolveFov>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub quality: Option<SolveQuality>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timing_ms: Option<SolveTimingMs>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub detections: Vec<SolveDetection>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub overlay: Option<SolveOverlay>,
}

impl SolveReport {
    /// Build a failed report with the given machine-readable code and message.
    pub fn failed(code: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            status: SolveStatus::Failed,
            failure: Some(SolveFailure {
                code: code.into(),
                message: message.into(),
            }),
            pose: None,
            fov: None,
            quality: None,
            timing_ms: None,
            detections: Vec::new(),
            overlay: None,
        }
    }
}
