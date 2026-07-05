use std::io::Cursor;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use image::{ImageBuffer, Luma};
use serde::Serialize;
use starglyph_core::catalog::Catalog;
use starglyph_core::constellations::ConstellationSet;
use starglyph_core::contracts::{SolveOverlay, SolveReport, SolveStatus};
use starglyph_core::image_input::{FrameImage, FrameTimestamp};
use starglyph_core::overlay::{build_overlay, OverlayOptions};
use starglyph_core::solve::{solve_frame_with_engine, SolveOptions, SolveStage};
use tauri::ipc::{Channel, Response};
use tauri::{AppHandle, Manager, State};
use tauri_plugin_dialog::DialogExt;

use crate::paths::{parse_startup_args, resolve_data_paths, StartupRequest};
use crate::state::{AppState, LoadedFrame, SolvedFrame, SolverAssets};

/// Metadata returned after loading an image.
#[derive(Debug, Clone, Serialize)]
pub struct ImageMeta {
    pub id: u32,
    pub file_name: String,
    pub width: u32,
    pub height: u32,
    pub timestamp: Option<String>,
    pub exposure_label: Option<String>,
}

/// Progress event streamed during solve.
#[derive(Debug, Clone, Serialize)]
pub struct ProgressEvent {
    pub stage: String,
    pub detail: Option<String>,
}

/// One third-party data attribution entry.
#[derive(Debug, Clone, Serialize)]
pub struct AttributionItem {
    pub name: String,
    pub license: String,
    pub url: String,
}

/// Attribution bundle for the status bar.
#[derive(Debug, Clone, Serialize)]
pub struct AttributionInfo {
    pub items: Vec<AttributionItem>,
}

/// Overlay geometry returned by [`recompute_overlay`].
pub type OverlayData = SolveOverlay;

#[tauri::command]
pub async fn pick_image(app: AppHandle) -> Result<Option<String>, String> {
    let app_handle = app.clone();
    tauri::async_runtime::spawn_blocking(move || pick_image_blocking(&app_handle))
        .await
        .map_err(|error| error.to_string())?
}

fn pick_image_blocking(app: &AppHandle) -> Result<Option<String>, String> {
    let picked = app
        .dialog()
        .file()
        .add_filter("Изображения", &["bmp", "png", "jpg", "jpeg"])
        .blocking_pick_file();

    match picked {
        None => Ok(None),
        Some(file_path) => {
            let path = file_path.into_path().map_err(|error| error.to_string())?;
            path_to_string(path)
        }
    }
}

#[tauri::command]
pub fn load_image(
    path: String,
    state: State<'_, Arc<Mutex<AppState>>>,
) -> Result<ImageMeta, String> {
    let path_buf = PathBuf::from(&path);
    let frame = FrameImage::load(&path_buf).map_err(|error| error.to_string())?;

    let file_name = path_buf
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or(&path)
        .to_string();

    let timestamp = frame.timestamp_from_name().map(format_timestamp);
    let exposure_label = parse_exposure_label(&frame.source_name);

    let mut guard = state
        .lock()
        .map_err(|_| "application state is poisoned".to_string())?;

    let id = guard.next_id;
    guard.next_id = guard.next_id.saturating_add(1);
    guard.last_solved = None;
    guard.frames.insert(
        id,
        LoadedFrame {
            frame,
            path: path_buf,
        },
    );

    let loaded = guard.frames.get(&id).expect("frame inserted above");
    Ok(ImageMeta {
        id,
        file_name,
        width: loaded.frame.width,
        height: loaded.frame.height,
        timestamp,
        exposure_label,
    })
}

#[tauri::command]
pub fn image_png(
    id: u32,
    view: String,
    state: State<'_, Arc<Mutex<AppState>>>,
) -> Result<Response, String> {
    let guard = state
        .lock()
        .map_err(|_| "application state is poisoned".to_string())?;
    let loaded = guard
        .frames
        .get(&id)
        .ok_or_else(|| format!("unknown image id {id}"))?;

    let png_bytes = encode_png(&loaded.frame, &view)?;
    Ok(Response::new(png_bytes))
}

#[tauri::command]
pub async fn solve_image(
    id: u32,
    utc_offset_hours: f64,
    on_progress: Channel<ProgressEvent>,
    app: AppHandle,
    state: State<'_, Arc<Mutex<AppState>>>,
) -> Result<SolveReport, String> {
    let frame = {
        let guard = state
            .lock()
            .map_err(|_| "application state is poisoned".to_string())?;
        let loaded = guard
            .frames
            .get(&id)
            .ok_or_else(|| format!("unknown image id {id}"))?;
        loaded.frame.clone()
    };

    let cache_dir = app
        .path()
        .app_cache_dir()
        .map_err(|error| error.to_string())?
        .join("index");

    let resource_dir = app.path().resource_dir().ok();
    let state = state.inner().clone();

    tauri::async_runtime::spawn_blocking(move || {
        solve_image_blocking(
            id,
            &frame,
            utc_offset_hours,
            &on_progress,
            state,
            cache_dir,
            resource_dir,
        )
    })
    .await
    .map_err(|error| error.to_string())?
}

fn solve_image_blocking(
    frame_id: u32,
    frame: &FrameImage,
    utc_offset_hours: f64,
    on_progress: &Channel<ProgressEvent>,
    state: Arc<Mutex<AppState>>,
    cache_dir: PathBuf,
    resource_dir: Option<PathBuf>,
) -> Result<SolveReport, String> {
    let emit = |stage: &str| {
        let _ = on_progress.send(ProgressEvent {
            stage: stage.to_string(),
            detail: None,
        });
    };

    {
        let mut guard = state
            .lock()
            .map_err(|_| "application state is poisoned".to_string())?;
        guard.last_solved = None;
        guard.utc_offset_hours = utc_offset_hours;
    }

    let assets = {
        let mut guard = state
            .lock()
            .map_err(|_| "application state is poisoned".to_string())?;

        if let Some(assets) = guard.solver_assets.clone() {
            assets
        } else {
            emit("load_assets");
            let paths = resolve_data_paths(resource_dir.as_deref())?;
            let catalog = Catalog::load(&paths.catalog).map_err(|error| error.to_string())?;
            let cons =
                ConstellationSet::load(&paths.constellation_lines, &paths.constellation_names)
                    .map_err(|error| error.to_string())?;
            let assets = Arc::new(SolverAssets { catalog, cons });
            guard.solver_assets = Some(assets.clone());
            assets
        }
    };

    let (fov_hint_deg, _) = {
        let guard = state
            .lock()
            .map_err(|_| "application state is poisoned".to_string())?;
        (guard.fov_hint_deg, guard.utc_offset_hours)
    };

    let timestamp = frame.timestamp_from_name();
    let jd = timestamp.map(|t| t.to_jd_utc(utc_offset_hours));
    let epoch_years = jd
        .map(starglyph_core::ephem::epoch_years)
        .or_else(|| timestamp.map(|t| t.to_epoch_years()));

    let opts = SolveOptions {
        fov_hint_deg,
        cache_dir,
        epoch_years,
        utc_offset_hours,
        include_grid: true,
        ..SolveOptions::default()
    };

    let mut engine = starglyph_core::engine::Engine::default();
    let (report, extras) = {
        let assets_ref = assets.as_ref();
        solve_frame_with_engine(
            frame,
            &assets_ref.catalog,
            &assets_ref.cons,
            &mut engine,
            &opts,
            &mut |stage| emit(stage_to_string(stage)),
        )
    };

    if report.status == SolveStatus::Solved {
        let mut guard = state
            .lock()
            .map_err(|_| "application state is poisoned".to_string())?;
        if let Some(fov) = report.fov.as_ref() {
            guard.fov_hint_deg = Some(fov.fov_x_deg as f32);
        }
        if let Some(camera) = extras.camera {
            guard.last_solved = Some(SolvedFrame {
                frame_id,
                camera,
                include_grid: true,
            });
        }
    }

    Ok(report)
}

fn stage_to_string(stage: SolveStage) -> &'static str {
    match stage {
        SolveStage::Detect => "detect",
        SolveStage::LoadIndex => "load_index",
        SolveStage::Match => "match",
        SolveStage::Verify => "verify",
        SolveStage::Refine => "refine",
        SolveStage::Overlay => "overlay",
    }
}

#[tauri::command]
pub fn recompute_overlay(
    id: u32,
    utc_offset_hours: f64,
    state: State<'_, Arc<Mutex<AppState>>>,
) -> Result<OverlayData, String> {
    let mut guard = state
        .lock()
        .map_err(|_| "application state is poisoned".to_string())?;

    guard.utc_offset_hours = utc_offset_hours;

    let solved = guard
        .last_solved
        .as_ref()
        .ok_or_else(|| "no solved frame in state".to_string())?;
    if solved.frame_id != id {
        return Err(format!("frame {id} is not the last solved frame"));
    }

    let loaded = guard
        .frames
        .get(&id)
        .ok_or_else(|| format!("unknown image id {id}"))?;
    let timestamp = loaded.frame.timestamp_from_name();
    let jd = timestamp.map(|t| t.to_jd_utc(utc_offset_hours));
    let epoch_years = jd
        .map(starglyph_core::ephem::epoch_years)
        .or_else(|| timestamp.map(|t| t.to_epoch_years()));

    let assets = guard
        .solver_assets
        .clone()
        .ok_or_else(|| "solver assets not loaded".to_string())?;

    let overlay = build_overlay(
        &solved.camera,
        &assets.catalog,
        &assets.cons,
        &OverlayOptions {
            epoch_years,
            jd_utc: jd,
            include_grid: solved.include_grid,
            ..OverlayOptions::default()
        },
    );

    Ok(overlay)
}

#[tauri::command]
pub fn startup_request() -> Option<StartupRequest> {
    parse_startup_args(std::env::args().skip(1))
}

#[tauri::command]
pub fn data_attribution() -> AttributionInfo {
    AttributionInfo {
        items: vec![
            AttributionItem {
                name: "HYG v4.2".to_string(),
                license: "CC BY-SA 4.0".to_string(),
                url: "https://codeberg.org/astronexus/hyg".to_string(),
            },
            AttributionItem {
                name: "d3-celestial".to_string(),
                license: "BSD-3-Clause".to_string(),
                url: "https://github.com/ofrohn/d3-celestial".to_string(),
            },
        ],
    }
}

fn path_to_string(path: PathBuf) -> Result<Option<String>, String> {
    path.into_os_string()
        .into_string()
        .map_err(|_| "invalid filesystem path".to_string())
        .map(Some)
}

fn format_timestamp(timestamp: FrameTimestamp) -> String {
    format!(
        "{:04}-{:02}-{:02} {:02}:{:02}:{:02}",
        timestamp.year,
        timestamp.month,
        timestamp.day,
        timestamp.hour,
        timestamp.minute,
        timestamp.second
    )
}

fn parse_exposure_label(stem: &str) -> Option<String> {
    if let Some(rest) = stem.split("Exp-").nth(1) {
        let token = rest.split('_').next().unwrap_or(rest);
        if !token.is_empty() {
            return Some(token.to_string());
        }
    }

    stem.split('_')
        .rfind(|part| is_exposure_token(part))
        .map(|part| (*part).to_string())
}

fn is_exposure_token(token: &str) -> bool {
    let (digits, suffix) = token
        .char_indices()
        .find(|(_, ch)| !ch.is_ascii_digit())
        .map_or((token, ""), |(index, _)| token.split_at(index));

    !digits.is_empty() && matches!(suffix, "ms" | "s" | "m")
}

fn encode_png(frame: &FrameImage, view: &str) -> Result<Vec<u8>, String> {
    let pixels = match view {
        "raw" => frame
            .gray
            .iter()
            .map(|value| (value.clamp(0.0, 1.0) * 255.0).round() as u8)
            .collect(),
        "stretched" => stretch_to_u8(&frame.gray),
        other => return Err(format!("unknown view mode '{other}'")),
    };

    let image = ImageBuffer::<Luma<u8>, Vec<u8>>::from_raw(frame.width, frame.height, pixels)
        .ok_or_else(|| "failed to build image buffer".to_string())?;

    let mut buffer = Vec::new();
    image
        .write_to(&mut Cursor::new(&mut buffer), image::ImageFormat::Png)
        .map_err(|error| error.to_string())?;
    Ok(buffer)
}

fn stretch_to_u8(gray: &[f32]) -> Vec<u8> {
    let mut sample: Vec<f32> = gray.iter().step_by(7).copied().collect();
    if sample.is_empty() {
        return Vec::new();
    }
    sample.sort_by(|left, right| left.partial_cmp(right).unwrap_or(std::cmp::Ordering::Equal));

    let lo = percentile(&sample, 10.0);
    let hi = percentile(&sample, 99.95);
    let span = hi - lo;

    if span <= 1e-6 {
        return gray
            .iter()
            .map(|value| (value.clamp(0.0, 1.0) * 255.0).round() as u8)
            .collect();
    }

    gray.iter()
        .map(|value| (((value - lo) / span).clamp(0.0, 1.0) * 255.0).round() as u8)
        .collect()
}

fn percentile(sorted: &[f32], percent: f64) -> f32 {
    if sorted.is_empty() {
        return 0.0;
    }
    if sorted.len() == 1 {
        return sorted[0];
    }

    let index = (percent / 100.0) * (sorted.len() - 1) as f64;
    let lower = index.floor() as usize;
    let upper = index.ceil() as usize;
    if lower == upper {
        return sorted[lower];
    }

    let fraction = (index - lower as f64) as f32;
    sorted[lower] * (1.0 - fraction) + sorted[upper] * fraction
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_exp_prefix_exposure() {
        assert_eq!(
            parse_exposure_label("2011-09-20_23-49-51-296_Gain-128_Exp-20m"),
            Some("20m".to_string())
        );
    }

    #[test]
    fn parses_trailing_exposure_token() {
        assert_eq!(parse_exposure_label("g128_40ms_1s"), Some("1s".to_string()));
    }

    #[test]
    fn returns_none_when_no_exposure() {
        assert_eq!(parse_exposure_label("frame_without_exposure"), None);
    }
}
