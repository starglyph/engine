use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use starglyph_core::catalog::Catalog;
use starglyph_core::constellations::ConstellationSet;
use starglyph_core::geom::CameraSolution;
use starglyph_core::image_input::FrameImage;

/// Lazily loaded catalog and constellation data shared across solves.
#[derive(Debug)]
pub struct SolverAssets {
    pub catalog: Catalog,
    pub cons: ConstellationSet,
}

/// Camera pose from the last successful solve, for fast overlay recomputation.
#[derive(Debug, Clone)]
pub struct SolvedFrame {
    pub frame_id: u32,
    pub camera: CameraSolution,
    pub include_grid: bool,
}

/// In-memory store for loaded frame images and solver state.
#[derive(Debug)]
pub struct AppState {
    pub next_id: u32,
    pub frames: HashMap<u32, LoadedFrame>,
    pub solver_assets: Option<Arc<SolverAssets>>,
    pub fov_hint_deg: Option<f32>,
    pub utc_offset_hours: f64,
    pub last_solved: Option<SolvedFrame>,
}

impl Default for AppState {
    fn default() -> Self {
        Self {
            next_id: 1,
            frames: HashMap::new(),
            solver_assets: None,
            fov_hint_deg: None,
            utc_offset_hours: 0.0,
            last_solved: None,
        }
    }
}

/// A frame loaded from disk, keyed by id in [`AppState::frames`].
#[derive(Debug, Clone)]
pub struct LoadedFrame {
    pub frame: FrameImage,
    #[allow(dead_code)]
    pub path: PathBuf,
}
