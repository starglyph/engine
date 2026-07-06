use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs;
use std::path::Path;

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use starglyph_core::contracts::SolveStatus;
use starglyph_core::engine::{DbKind, Engine};
use starglyph_core::eval::{
    compare_pose, ground_truth_pose, median, percentile, GroundTruthPose, PoseErrors,
    WcsCalibration,
};
use starglyph_core::image_input::FrameImage;
use starglyph_core::solve::{solve_frame_with_engine, SolveOptions, SolveStage};

use crate::{default_cache_dir, load_catalog_and_cons, print_engine_progress};

// ---------------------------------------------------------------------------
// Manifest
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
struct ManifestEntry {
    id: String,
    file: String,
    track: String,
    #[serde(default)]
    solve_status: Option<String>,
    wcs: Option<String>,
    #[serde(default)]
    width: Option<u32>,
    #[serde(default)]
    height: Option<u32>,
}

fn parse_manifest(text: &str) -> Result<Vec<ManifestEntry>> {
    serde_json::from_str(text).context("failed to parse manifest JSON")
}

fn parse_tracks(s: &str) -> Result<HashSet<String>> {
    let tracks: HashSet<String> = s
        .split(',')
        .map(str::trim)
        .filter(|t| !t.is_empty())
        .map(str::to_owned)
        .collect();
    if tracks.is_empty() {
        bail!("--tracks must list at least one track (solver, scene, stress)");
    }
    for t in &tracks {
        if t != "solver" && t != "scene" && t != "stress" {
            bail!("unknown track '{t}', expected solver, scene, or stress");
        }
    }
    Ok(tracks)
}

fn parse_ids(s: &str) -> Vec<String> {
    s.split(',')
        .map(str::trim)
        .filter(|id| !id.is_empty())
        .map(str::to_owned)
        .collect()
}

fn select_entries<'a>(
    manifest: &'a [ManifestEntry],
    tracks: &HashSet<String>,
    ids: Option<&[String]>,
) -> Result<Vec<&'a ManifestEntry>> {
    let mut selected: Vec<&ManifestEntry> = manifest
        .iter()
        .filter(|e| tracks.contains(&e.track))
        .collect();
    if let Some(want) = ids {
        let by_id: HashMap<&str, &ManifestEntry> =
            selected.iter().map(|e| (e.id.as_str(), *e)).collect();
        let mut out = Vec::with_capacity(want.len());
        for id in want {
            match by_id.get(id.as_str()) {
                Some(e) => out.push(*e),
                None => bail!("manifest has no entry with id '{id}' in selected tracks"),
            }
        }
        selected = out;
    }
    Ok(selected)
}

// ---------------------------------------------------------------------------
// Per-frame output
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum FrameStatus {
    Solved,
    Failed,
    MissingImage,
    LoadError,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct GtOutput {
    ra_deg: f64,
    dec_deg: f64,
    roll_deg: f64,
    fov_x_deg: Option<f64>,
    parity_physical: bool,
}

#[derive(Debug, Clone)]
pub(crate) struct FrameRecord {
    pub id: String,
    pub file: String,
    pub track: String,
    pub status: FrameStatus,
    pub failure: Option<starglyph_core::contracts::SolveFailure>,
    pub pose: Option<starglyph_core::contracts::SolvePose>,
    pub fov: Option<starglyph_core::contracts::SolveFov>,
    pub quality: Option<starglyph_core::contracts::SolveQuality>,
    pub timing_ms: Option<starglyph_core::contracts::SolveTimingMs>,
    pub gt: Option<GroundTruthPose>,
    pub gt_error: Option<String>,
    pub errors: Option<PoseErrors>,
}

impl FrameRecord {
    fn to_output(&self) -> PerFrameOutput {
        PerFrameOutput {
            id: self.id.clone(),
            file: self.file.clone(),
            track: self.track.clone(),
            status: self.status.clone(),
            failure: self.failure.clone(),
            pose: self.pose.clone(),
            fov: self.fov.clone(),
            quality: self.quality.clone(),
            timing_ms: self.timing_ms.clone(),
            gt: self.gt.as_ref().map(|g| GtOutput {
                ra_deg: g.ra_deg,
                dec_deg: g.dec_deg,
                roll_deg: g.roll_deg,
                fov_x_deg: g.fov_x_deg,
                parity_physical: g.parity_physical,
            }),
            gt_error: self.gt_error.clone(),
            errors: self.errors.clone(),
        }
    }
}

#[derive(Debug, Clone, Serialize)]
struct PerFrameOutput {
    id: String,
    file: String,
    track: String,
    status: FrameStatus,
    #[serde(skip_serializing_if = "Option::is_none")]
    failure: Option<starglyph_core::contracts::SolveFailure>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pose: Option<starglyph_core::contracts::SolvePose>,
    #[serde(skip_serializing_if = "Option::is_none")]
    fov: Option<starglyph_core::contracts::SolveFov>,
    #[serde(skip_serializing_if = "Option::is_none")]
    quality: Option<starglyph_core::contracts::SolveQuality>,
    #[serde(skip_serializing_if = "Option::is_none")]
    timing_ms: Option<starglyph_core::contracts::SolveTimingMs>,
    #[serde(skip_serializing_if = "Option::is_none")]
    gt: Option<GtOutput>,
    #[serde(skip_serializing_if = "Option::is_none")]
    gt_error: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    errors: Option<PoseErrors>,
}

// ---------------------------------------------------------------------------
// Summary
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct StatBlock {
    pub median: f64,
    pub p95: f64,
    pub max: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PoseErrorStats {
    pub n_gt: usize,
    pub n_compared: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub axis_angle_deg: Option<StatBlock>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub roll_error_deg: Option<StatBlock>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fov_error_rel: Option<StatBlock>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TrackStats {
    pub n: usize,
    pub attempted: usize,
    pub solved: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub solve_rate: Option<f64>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub failures_by_code: BTreeMap<String, usize>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub missing: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub load_errors: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SceneTrackStats {
    pub n: usize,
    pub attempted: usize,
    pub solved: usize,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub unexpected_solves: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct StressTrackStats {
    pub n: usize,
    pub attempted: usize,
    pub solved: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct WorstCase {
    pub id: String,
    pub axis_angle_deg: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DatasetInfo {
    pub manifest: String,
    pub tracks: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ids_filter: Option<Vec<String>>,
    pub n_selected: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ConfigInfo {
    pub fov_hint_deg: Option<f32>,
    pub catalog: String,
    pub blind: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TimingStats {
    pub detect: StatBlock,
    pub solve: StatBlock,
    pub total: StatBlock,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Summary {
    pub schema_version: u32,
    pub generated_by: String,
    pub dataset: DatasetInfo,
    pub config: ConfigInfo,
    pub solver_track: TrackStats,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub scene_track: Option<SceneTrackStats>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stress_track: Option<StressTrackStats>,
    pub pose_errors: PoseErrorStats,
    pub timing_ms: TimingStats,
    pub worst_cases: Vec<WorstCase>,
}

fn stat_block(values: &[f64]) -> StatBlock {
    StatBlock {
        median: median(values).unwrap_or(0.0),
        p95: percentile(values, 95.0).unwrap_or(0.0),
        max: values.iter().copied().fold(0.0_f64, f64::max),
    }
}

fn optional_stat_block(values: &[f64]) -> Option<StatBlock> {
    if values.is_empty() {
        None
    } else {
        Some(stat_block(values))
    }
}

pub(crate) fn build_summary(
    manifest_path: &str,
    tracks: &[String],
    ids_filter: Option<Vec<String>>,
    catalog_path: &str,
    fov_hint_deg: Option<f32>,
    frames: &[FrameRecord],
) -> Summary {
    let solver_frames: Vec<&FrameRecord> = frames.iter().filter(|f| f.track == "solver").collect();
    let scene_frames: Vec<&FrameRecord> = frames.iter().filter(|f| f.track == "scene").collect();
    let stress_frames: Vec<&FrameRecord> = frames.iter().filter(|f| f.track == "stress").collect();

    let solver_track = build_solver_track(&solver_frames);
    let scene_track = if tracks.iter().any(|t| t == "scene") {
        Some(build_scene_track(&scene_frames))
    } else {
        None
    };
    let stress_track = if tracks.iter().any(|t| t == "stress") {
        Some(build_stress_track(&stress_frames))
    } else {
        None
    };

    let pose_errors = build_pose_errors(frames);
    let timing_ms = build_timing_stats(frames);
    let worst_cases = build_worst_cases(frames);

    Summary {
        schema_version: 1,
        generated_by: format!("starglyph-cli {}", env!("CARGO_PKG_VERSION")),
        dataset: DatasetInfo {
            manifest: manifest_path.to_owned(),
            tracks: tracks.to_vec(),
            ids_filter,
            n_selected: frames.len(),
        },
        config: ConfigInfo {
            fov_hint_deg,
            catalog: catalog_path.to_owned(),
            blind: fov_hint_deg.is_none(),
        },
        solver_track,
        scene_track,
        stress_track,
        pose_errors,
        timing_ms,
        worst_cases,
    }
}

fn is_attempted(status: &FrameStatus) -> bool {
    !matches!(status, FrameStatus::MissingImage | FrameStatus::LoadError)
}

fn is_solved(status: &FrameStatus) -> bool {
    matches!(status, FrameStatus::Solved)
}

fn build_solver_track(frames: &[&FrameRecord]) -> TrackStats {
    let missing: Vec<String> = frames
        .iter()
        .filter(|f| matches!(f.status, FrameStatus::MissingImage))
        .map(|f| f.id.clone())
        .collect();
    let load_errors: Vec<String> = frames
        .iter()
        .filter(|f| matches!(f.status, FrameStatus::LoadError))
        .map(|f| f.id.clone())
        .collect();
    let attempted_frames: Vec<&&FrameRecord> =
        frames.iter().filter(|f| is_attempted(&f.status)).collect();
    let solved = attempted_frames
        .iter()
        .filter(|f| is_solved(&f.status))
        .count();
    let attempted = attempted_frames.len();
    let solve_rate = if attempted > 0 {
        Some(solved as f64 / attempted as f64)
    } else {
        None
    };
    let mut failures_by_code: BTreeMap<String, usize> = BTreeMap::new();
    for f in attempted_frames {
        if matches!(f.status, FrameStatus::Failed) {
            if let Some(ref failure) = f.failure {
                *failures_by_code.entry(failure.code.clone()).or_insert(0) += 1;
            }
        }
    }
    TrackStats {
        n: frames.len(),
        attempted,
        solved,
        solve_rate,
        failures_by_code,
        missing,
        load_errors,
    }
}

fn build_scene_track(frames: &[&FrameRecord]) -> SceneTrackStats {
    let attempted = frames.iter().filter(|f| is_attempted(&f.status)).count();
    let solved = frames.iter().filter(|f| is_solved(&f.status)).count();
    let unexpected_solves: Vec<String> = frames
        .iter()
        .filter(|f| is_solved(&f.status))
        .map(|f| f.id.clone())
        .collect();
    SceneTrackStats {
        n: frames.len(),
        attempted,
        solved,
        unexpected_solves,
    }
}

fn build_stress_track(frames: &[&FrameRecord]) -> StressTrackStats {
    StressTrackStats {
        n: frames.len(),
        attempted: frames.iter().filter(|f| is_attempted(&f.status)).count(),
        solved: frames.iter().filter(|f| is_solved(&f.status)).count(),
    }
}

pub(crate) fn build_pose_errors(frames: &[FrameRecord]) -> PoseErrorStats {
    let n_gt = frames.iter().filter(|f| f.gt.is_some()).count();
    let compared: Vec<&FrameRecord> = frames
        .iter()
        .filter(|f| matches!(f.status, FrameStatus::Solved) && f.errors.is_some())
        .collect();
    let n_compared = compared.len();

    if n_compared == 0 {
        return PoseErrorStats {
            n_gt,
            n_compared: 0,
            axis_angle_deg: None,
            roll_error_deg: None,
            fov_error_rel: None,
        };
    }

    let axis: Vec<f64> = compared
        .iter()
        .filter_map(|f| f.errors.as_ref().map(|e| e.axis_angle_deg))
        .collect();
    let roll: Vec<f64> = compared
        .iter()
        .filter_map(|f| f.errors.as_ref().and_then(|e| e.roll_error_deg))
        .collect();
    let fov: Vec<f64> = compared
        .iter()
        .filter_map(|f| f.errors.as_ref().and_then(|e| e.fov_error_rel))
        .collect();

    PoseErrorStats {
        n_gt,
        n_compared,
        axis_angle_deg: optional_stat_block(&axis),
        roll_error_deg: optional_stat_block(&roll),
        fov_error_rel: optional_stat_block(&fov),
    }
}

pub(crate) fn build_timing_stats(frames: &[FrameRecord]) -> TimingStats {
    let attempted: Vec<&FrameRecord> = frames.iter().filter(|f| is_attempted(&f.status)).collect();
    let detect: Vec<f64> = attempted
        .iter()
        .filter_map(|f| f.timing_ms.as_ref().map(|t| t.detect as f64))
        .collect();
    let solve: Vec<f64> = attempted
        .iter()
        .filter_map(|f| f.timing_ms.as_ref().map(|t| t.solve as f64))
        .collect();
    let total: Vec<f64> = attempted
        .iter()
        .filter_map(|f| f.timing_ms.as_ref().map(|t| t.total as f64))
        .collect();
    TimingStats {
        detect: stat_block(&detect),
        solve: stat_block(&solve),
        total: stat_block(&total),
    }
}

fn build_worst_cases(frames: &[FrameRecord]) -> Vec<WorstCase> {
    let mut compared: Vec<(&FrameRecord, f64)> = frames
        .iter()
        .filter_map(|f| f.errors.as_ref().map(|e| (f, e.axis_angle_deg)))
        .collect();
    compared.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    compared
        .into_iter()
        .take(10)
        .map(|(f, axis)| WorstCase {
            id: f.id.clone(),
            axis_angle_deg: axis,
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Gate
// ---------------------------------------------------------------------------

pub fn gate_violations(
    current: &Summary,
    baseline: &Summary,
    max_axis_regress_pct: f64,
) -> Vec<String> {
    let mut out = Vec::new();

    if let (Some(cur), Some(base)) = (
        current.solver_track.solve_rate,
        baseline.solver_track.solve_rate,
    ) {
        if cur < base {
            out.push(format!(
                "solver_track.solve_rate regressed: {cur:.4} < baseline {base:.4}"
            ));
        }
    }

    match (
        current.pose_errors.axis_angle_deg.as_ref(),
        baseline.pose_errors.axis_angle_deg.as_ref(),
    ) {
        (Some(cur), Some(base)) if base.p95 > f64::EPSILON => {
            let threshold = base.p95 * (1.0 + max_axis_regress_pct / 100.0);
            if cur.p95 > threshold {
                let pct = (cur.p95 - base.p95) / base.p95 * 100.0;
                out.push(format!(
                    "pose_errors.axis_angle_deg.p95 regressed: {:.4} vs baseline {:.4} (+{pct:.1}% > {max_axis_regress_pct}%)",
                    cur.p95, base.p95
                ));
            }
        }
        _ => {}
    }

    out
}

// ---------------------------------------------------------------------------
// Run
// ---------------------------------------------------------------------------

pub struct EvalArgs<'a> {
    pub manifest: &'a Path,
    pub out_dir: &'a Path,
    pub tracks: &'a str,
    pub ids: Option<&'a str>,
    pub catalog_path: Option<&'a Path>,
    pub lines_path: Option<&'a Path>,
    pub names_path: Option<&'a Path>,
    pub cache_dir: Option<&'a Path>,
    pub fov_hint: Option<f32>,
    pub baseline: Option<&'a Path>,
    pub max_axis_p95_regress_pct: f64,
    pub allow_missing: bool,
}

pub enum EvalOutcome {
    Success,
    GateFailed,
}

pub fn run_eval(args: EvalArgs<'_>) -> Result<EvalOutcome> {
    let manifest_text = fs::read_to_string(args.manifest)
        .with_context(|| format!("failed to read manifest '{}'", args.manifest.display()))?;
    let manifest = parse_manifest(&manifest_text)?;
    let tracks_set = parse_tracks(args.tracks)?;
    let ids_list: Option<Vec<String>> = args.ids.map(parse_ids);
    let ids_ref = ids_list.as_deref();
    let entries = select_entries(&manifest, &tracks_set, ids_ref)?;

    let manifest_dir = args
        .manifest
        .parent()
        .context("manifest path has no parent directory")?;

    let missing: Vec<String> = entries
        .iter()
        .filter(|e| !manifest_dir.join(&e.file).is_file())
        .map(|e| e.id.clone())
        .collect();
    if !missing.is_empty() && !args.allow_missing {
        bail!("missing image files for ids: {}", missing.join(", "));
    }

    let catalog_file = args
        .catalog_path
        .map(Path::to_path_buf)
        .unwrap_or_else(|| crate::data_root().join("catalogs/hyg_v3.csv"));
    let catalog_display = catalog_file.display().to_string();

    let (catalog, cons) =
        load_catalog_and_cons(args.catalog_path, args.lines_path, args.names_path)?;
    let cache_dir = args
        .cache_dir
        .map(Path::to_path_buf)
        .unwrap_or_else(default_cache_dir);

    fs::create_dir_all(args.out_dir)
        .with_context(|| format!("failed to create out dir '{}'", args.out_dir.display()))?;
    let per_frame_dir = args.out_dir.join("per-frame");
    fs::create_dir_all(&per_frame_dir)
        .with_context(|| format!("failed to create '{}'", per_frame_dir.display()))?;
    let worst_dir = args.out_dir.join("worst-cases");
    fs::create_dir_all(&worst_dir)
        .with_context(|| format!("failed to create '{}'", worst_dir.display()))?;

    let mut engine = Engine::default();
    engine
        .ensure_kind(
            &catalog,
            DbKind::Bootstrap,
            &cache_dir,
            &mut print_engine_progress,
        )
        .context("failed to prepare bootstrap database")?;

    let mut frames: Vec<FrameRecord> = Vec::new();
    for entry in entries {
        let image_path = manifest_dir.join(&entry.file);
        if !image_path.is_file() {
            let rec = FrameRecord {
                id: entry.id.clone(),
                file: entry.file.clone(),
                track: entry.track.clone(),
                status: FrameStatus::MissingImage,
                failure: None,
                pose: None,
                fov: None,
                quality: None,
                timing_ms: None,
                gt: None,
                gt_error: None,
                errors: None,
            };
            eprintln!("[eval] {}: missing_image", entry.id);
            frames.push(rec);
            continue;
        }

        let frame = match FrameImage::load(&image_path) {
            Ok(f) => f,
            Err(e) => {
                eprintln!("[eval] {}: load_error ({e})", entry.id);
                frames.push(FrameRecord {
                    id: entry.id.clone(),
                    file: entry.file.clone(),
                    track: entry.track.clone(),
                    status: FrameStatus::LoadError,
                    failure: None,
                    pose: None,
                    fov: None,
                    quality: None,
                    timing_ms: None,
                    gt: None,
                    gt_error: None,
                    errors: None,
                });
                continue;
            }
        };

        let (mut gt, mut gt_error) = (None, None);
        if let Some(ref wcs_rel) = entry.wcs {
            let wcs_path = manifest_dir.join(wcs_rel);
            match fs::read_to_string(&wcs_path) {
                Ok(text) => match WcsCalibration::from_json_str(&text) {
                    Ok(calib) => {
                        gt = Some(ground_truth_pose(&calib, frame.width, frame.height));
                    }
                    Err(e) => {
                        gt_error = Some(e.to_string());
                    }
                },
                Err(e) => {
                    gt_error = Some(format!("failed to read WCS '{}': {e}", wcs_path.display()));
                }
            }
        }

        let timestamp = frame.timestamp_from_name();
        let epoch_years = timestamp.map(|t| t.to_epoch_years());
        let opts = SolveOptions {
            fov_hint_deg: args.fov_hint,
            attitude_hint: None,
            cache_dir: cache_dir.clone(),
            allow_dense_band: true,
            epoch_years,
            utc_offset_hours: 0.0,
            include_grid: false,
        };
        let mut stage = |_s: SolveStage| {};
        let (report, _extras) =
            solve_frame_with_engine(&frame, &catalog, &cons, &mut engine, &opts, &mut stage);

        let ms = report.timing_ms.as_ref().map(|t| t.total).unwrap_or(0);
        let status = if report.status == SolveStatus::Solved {
            FrameStatus::Solved
        } else {
            FrameStatus::Failed
        };
        eprintln!(
            "[eval] {}: {} ({} ms)",
            entry.id,
            if report.status == SolveStatus::Solved {
                "solved"
            } else {
                "failed"
            },
            ms
        );

        let errors = if report.status == SolveStatus::Solved {
            gt.as_ref()
                .map(|g| compare_pose(report.pose.as_ref().unwrap(), report.fov.as_ref(), g))
        } else {
            None
        };

        frames.push(FrameRecord {
            id: entry.id.clone(),
            file: entry.file.clone(),
            track: entry.track.clone(),
            status,
            failure: report.failure.clone(),
            pose: report.pose.clone(),
            fov: report.fov.clone(),
            quality: report.quality.clone(),
            timing_ms: report.timing_ms.clone(),
            gt,
            gt_error,
            errors,
        });
    }

    let tracks_vec: Vec<String> = {
        let mut v: Vec<String> = tracks_set.into_iter().collect();
        v.sort();
        v
    };
    let summary = build_summary(
        &args.manifest.display().to_string(),
        &tracks_vec,
        ids_list,
        &catalog_display,
        args.fov_hint,
        &frames,
    );

    for rec in &frames {
        let path = per_frame_dir.join(format!("{}.json", rec.id));
        let json = serde_json::to_string_pretty(&rec.to_output())?;
        fs::write(&path, json).with_context(|| format!("failed to write '{}'", path.display()))?;
    }

    let worst: Vec<(usize, &FrameRecord)> = {
        let mut ranked: Vec<(usize, &FrameRecord, f64)> = frames
            .iter()
            .enumerate()
            .filter_map(|(i, f)| f.errors.as_ref().map(|e| (i, f, e.axis_angle_deg)))
            .collect();
        ranked.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
        ranked
            .into_iter()
            .take(10)
            .map(|(i, f, _)| (i, f))
            .collect()
    };
    for (rank, rec) in worst.iter().enumerate() {
        let name = format!("{:02}_{}.json", rank + 1, rec.1.id);
        let path = worst_dir.join(name);
        let json = serde_json::to_string_pretty(&rec.1.to_output())?;
        fs::write(&path, json).with_context(|| format!("failed to write '{}'", path.display()))?;
    }

    let summary_path = args.out_dir.join("summary.json");
    fs::write(&summary_path, serde_json::to_string_pretty(&summary)?)
        .with_context(|| format!("failed to write '{}'", summary_path.display()))?;

    if let Some(baseline_path) = args.baseline {
        let baseline_text = fs::read_to_string(baseline_path)
            .with_context(|| format!("failed to read baseline '{}'", baseline_path.display()))?;
        let baseline: Summary = serde_json::from_str(&baseline_text)
            .with_context(|| format!("failed to parse baseline '{}'", baseline_path.display()))?;
        let violations = gate_violations(&summary, &baseline, args.max_axis_p95_regress_pct);
        if violations.is_empty() {
            eprintln!("GATE PASS");
            return Ok(EvalOutcome::Success);
        }
        eprintln!("GATE FAIL: {}", violations.join("; "));
        return Ok(EvalOutcome::GateFailed);
    }

    Ok(EvalOutcome::Success)
}

#[cfg(test)]
mod tests {
    use super::*;
    use starglyph_core::contracts::{SolveFailure, SolvePose, SolveQuality, SolveTimingMs};

    const SAMPLE_MANIFEST: &str = r#"[
      {
        "id": "a_solver",
        "file": "images/a.tiff",
        "track": "solver",
        "solve_status": "solved",
        "wcs": "ground-truth/a.wcs.json",
        "width": 100,
        "height": 80,
        "extra_unknown": true
      },
      {
        "id": "b_scene",
        "file": "images/b.jpg",
        "track": "scene",
        "wcs": null,
        "width": 200,
        "height": 150
      },
      {
        "id": "c_stress",
        "file": "images-stress-tier-b/c.tiff",
        "track": "stress",
        "width": 50,
        "height": 50
      }
    ]"#;

    #[test]
    fn manifest_parsing_and_filtering() {
        let manifest = parse_manifest(SAMPLE_MANIFEST).expect("parse");
        assert_eq!(manifest.len(), 3);

        let solver_only = HashSet::from(["solver".to_owned()]);
        let sel = select_entries(&manifest, &solver_only, None).expect("select");
        assert_eq!(sel.len(), 1);
        assert_eq!(sel[0].id, "a_solver");

        let all_tracks =
            HashSet::from(["solver".to_owned(), "scene".to_owned(), "stress".to_owned()]);
        let sel = select_entries(&manifest, &all_tracks, None).expect("select");
        assert_eq!(sel.len(), 3);

        let ids = vec!["a_solver".to_owned(), "c_stress".to_owned()];
        let sel = select_entries(&manifest, &all_tracks, Some(&ids)).expect("ids");
        assert_eq!(sel.len(), 2);
        assert_eq!(sel[0].id, "a_solver");
        assert_eq!(sel[1].id, "c_stress");

        let bad = vec!["missing".to_owned()];
        assert!(select_entries(&manifest, &solver_only, Some(&bad)).is_err());
    }

    fn sample_summary(solve_rate: f64, axis_p95: Option<f64>) -> Summary {
        Summary {
            schema_version: 1,
            generated_by: "test".to_owned(),
            dataset: DatasetInfo {
                manifest: "m.json".to_owned(),
                tracks: vec!["solver".to_owned()],
                ids_filter: None,
                n_selected: 1,
            },
            config: ConfigInfo {
                fov_hint_deg: None,
                catalog: "cat.csv".to_owned(),
                blind: true,
            },
            solver_track: TrackStats {
                n: 8,
                attempted: 8,
                solved: (solve_rate * 8.0).round() as usize,
                solve_rate: Some(solve_rate),
                failures_by_code: BTreeMap::new(),
                missing: vec![],
                load_errors: vec![],
            },
            scene_track: None,
            stress_track: None,
            pose_errors: PoseErrorStats {
                n_gt: 7,
                n_compared: if axis_p95.is_some() { 3 } else { 0 },
                axis_angle_deg: axis_p95.map(|p95| StatBlock {
                    median: 0.1,
                    p95,
                    max: p95 + 0.1,
                }),
                roll_error_deg: None,
                fov_error_rel: None,
            },
            timing_ms: TimingStats {
                detect: StatBlock {
                    median: 1.0,
                    p95: 2.0,
                    max: 3.0,
                },
                solve: StatBlock {
                    median: 10.0,
                    p95: 20.0,
                    max: 30.0,
                },
                total: StatBlock {
                    median: 11.0,
                    p95: 22.0,
                    max: 33.0,
                },
            },
            worst_cases: vec![],
        }
    }

    #[test]
    fn gate_violations_cases() {
        let baseline = sample_summary(0.5, Some(0.4));
        let pass = sample_summary(0.5, Some(0.436));
        assert!(gate_violations(&pass, &baseline, 10.0).is_empty());

        let fail_rate = sample_summary(0.375, Some(0.4));
        let v = gate_violations(&fail_rate, &baseline, 10.0);
        assert!(v.iter().any(|s| s.contains("solve_rate")));

        let fail_axis = sample_summary(0.5, Some(0.4404));
        let v = gate_violations(&fail_axis, &baseline, 10.0);
        assert!(v.iter().any(|s| s.contains("axis_angle_deg.p95")));

        let pass_axis = sample_summary(0.5, Some(0.436));
        assert!(gate_violations(&pass_axis, &baseline, 10.0).is_empty());

        let null_current = sample_summary(0.5, None);
        assert!(gate_violations(&null_current, &baseline, 10.0).is_empty());

        let null_baseline = sample_summary(0.5, None);
        let with_axis = sample_summary(0.5, Some(1.0));
        assert!(gate_violations(&with_axis, &null_baseline, 10.0).is_empty());
    }

    #[test]
    fn summary_round_trip() {
        let s = sample_summary(0.375, Some(0.4));
        let json = serde_json::to_string(&s).expect("serialize");
        let back: Summary = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back.schema_version, s.schema_version);
        assert_eq!(back.solver_track.solve_rate, s.solver_track.solve_rate);
        assert_eq!(
            back.pose_errors.axis_angle_deg.as_ref().map(|b| b.p95),
            s.pose_errors.axis_angle_deg.as_ref().map(|b| b.p95)
        );
        assert_eq!(back.timing_ms.total.median, s.timing_ms.total.median);
    }

    #[test]
    fn aggregation_timing_vs_pose_subset() {
        let mk = |id: &str, status: FrameStatus, axis: Option<f64>, total_ms: u64| FrameRecord {
            id: id.to_owned(),
            file: format!("{id}.tiff"),
            track: "solver".to_owned(),
            status,
            failure: if matches!(status, FrameStatus::Failed) {
                Some(SolveFailure {
                    code: "no_match".to_owned(),
                    message: "x".to_owned(),
                })
            } else {
                None
            },
            pose: if matches!(status, FrameStatus::Solved) {
                Some(SolvePose {
                    ra_deg: 1.0,
                    dec_deg: 2.0,
                    roll_deg: 3.0,
                })
            } else {
                None
            },
            fov: None,
            quality: Some(SolveQuality {
                n_detections: 10,
                n_inliers: 5,
                rms_px: 1.0,
                log_odds: 2.0,
                confidence: 0.9,
            }),
            timing_ms: Some(SolveTimingMs {
                detect: total_ms / 3,
                solve: total_ms / 3,
                total: total_ms,
            }),
            gt: if axis.is_some() {
                Some(GroundTruthPose {
                    ra_deg: 1.0,
                    dec_deg: 2.0,
                    roll_deg: 3.0,
                    fov_x_deg: Some(10.0),
                    parity_physical: true,
                })
            } else {
                None
            },
            gt_error: None,
            errors: axis.map(|a| PoseErrors {
                axis_angle_deg: a,
                roll_error_deg: Some(0.1),
                fov_error_rel: Some(0.01),
            }),
        };

        let frames = vec![
            mk("f1", FrameStatus::Solved, Some(0.5), 100),
            mk("f2", FrameStatus::Solved, Some(1.5), 200),
            mk("f3", FrameStatus::Failed, None, 300),
        ];

        let pose = build_pose_errors(&frames);
        assert_eq!(pose.n_compared, 2);
        assert!((pose.axis_angle_deg.unwrap().median - 1.0).abs() < 1e-9);

        let timing = build_timing_stats(&frames);
        assert!((timing.total.median - 200.0).abs() < 1e-9);
        assert!((timing.total.max - 300.0).abs() < 1e-9);
    }
}
