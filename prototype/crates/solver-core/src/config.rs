use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SolverConfig {
    pub detection: DetectionConfig,
    pub matching: MatchingConfig,
    pub pose: PoseConfig,
    pub overlay: OverlayConfig,
    pub benchmark: BenchmarkRunConfig,
}

impl Default for SolverConfig {
    fn default() -> Self {
        Self {
            detection: DetectionConfig::default(),
            matching: MatchingConfig::default(),
            pose: PoseConfig::default(),
            overlay: OverlayConfig::default(),
            benchmark: BenchmarkRunConfig::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectionConfig {
    pub min_peak_value: u8,
    pub non_max_radius_px: u32,
    pub max_candidates: usize,
    pub truth_match_tolerance_px: f32,
}

impl Default for DetectionConfig {
    fn default() -> Self {
        Self {
            min_peak_value: 24,
            non_max_radius_px: 4,
            max_candidates: 64,
            truth_match_tolerance_px: 2.5,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MatchingConfig {
    pub top_k_detections: usize,
    pub descriptor_neighbors: usize,
    pub descriptor_tolerance_deg: f32,
    pub absolute_accept_threshold: f32,
    pub ambiguity_margin: f32,
    pub max_ranked_hypotheses: usize,
}

impl Default for MatchingConfig {
    fn default() -> Self {
        Self {
            top_k_detections: 12,
            descriptor_neighbors: 5,
            descriptor_tolerance_deg: 1.8,
            absolute_accept_threshold: 0.65,
            ambiguity_margin: 0.10,
            max_ranked_hypotheses: 5,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PoseConfig {
    pub ransac_iterations: usize,
    pub inlier_threshold_px: f32,
    pub min_inliers: usize,
}

impl Default for PoseConfig {
    fn default() -> Self {
        Self {
            ransac_iterations: 96,
            inlier_threshold_px: 3.0,
            min_inliers: 4,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OverlayConfig {
    pub detection_marker_radius_px: u32,
}

impl Default for OverlayConfig {
    fn default() -> Self {
        Self {
            detection_marker_radius_px: 3,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkRunConfig {
    pub worst_case_count: usize,
    pub reproducibility_tolerance: f32,
    pub run_reproducibility_check: bool,
}

impl Default for BenchmarkRunConfig {
    fn default() -> Self {
        Self {
            worst_case_count: 10,
            reproducibility_tolerance: 0.01,
            run_reproducibility_check: true,
        }
    }
}
