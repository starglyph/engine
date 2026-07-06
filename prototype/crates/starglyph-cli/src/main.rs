use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use image::Rgb;
use serde::Serialize;
use starglyph_core::catalog::Catalog;
use starglyph_core::constellations::ConstellationSet;
use starglyph_core::contracts::{SolveOverlay, SolveReport, SolveStatus};
use starglyph_core::detect::{detect_stars, DetectConfig, DetectStats};
use starglyph_core::engine::{DbKind, Engine, EngineProgress};
use starglyph_core::geom::CameraSolution;
use starglyph_core::image_input::FrameImage;
use starglyph_core::overlay::{build_overlay, OverlayOptions};
use starglyph_core::render;
use starglyph_core::solve::{
    solve_frame_with_engine, SolveOptions, SolveStage, DEFAULT_BLIND_FOV_DEG,
};

mod eval_cmd;

#[derive(Debug, Parser)]
#[command(name = "starglyph")]
#[command(about = "Starglyph recognition data inspection CLI")]
struct Args {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Inspect a frame image and print basic intensity statistics.
    Inspect { image: PathBuf },
    /// Print HYG catalog summary statistics.
    CatalogInfo {
        #[arg(long)]
        catalog: PathBuf,
    },
    /// Print constellation geometry summary statistics.
    ConstellationsInfo {
        #[arg(long)]
        lines: PathBuf,
        #[arg(long)]
        names: PathBuf,
    },
    /// Detect stars in a frame image.
    Detect {
        image: PathBuf,
        #[arg(long)]
        debug_png: Option<PathBuf>,
        #[arg(long)]
        json_pretty: bool,
    },
    /// Render constellation/star overlay for a known pose.
    Overlay {
        image: PathBuf,
        #[arg(long)]
        ra: f64,
        #[arg(long)]
        dec: f64,
        #[arg(long)]
        roll: f64,
        #[arg(long)]
        focal_px: f64,
        #[arg(long, default_value_t = 0.0)]
        k1: f64,
        #[arg(long)]
        grid: bool,
        #[arg(long)]
        out: Option<PathBuf>,
        #[arg(long)]
        json: Option<PathBuf>,
        #[arg(long)]
        catalog: Option<PathBuf>,
        #[arg(long)]
        lines: Option<PathBuf>,
        #[arg(long)]
        names: Option<PathBuf>,
        /// Observer UTC offset in hours (positive east of Greenwich).
        #[arg(long, default_value_t = 0.0)]
        utc_offset: f64,
    },
    /// Blind-solve a single frame (detect → match → verify → refine).
    Solve {
        image: PathBuf,
        #[arg(long)]
        catalog: Option<PathBuf>,
        #[arg(long)]
        lines: Option<PathBuf>,
        #[arg(long)]
        names: Option<PathBuf>,
        #[arg(long)]
        cache_dir: Option<PathBuf>,
        #[arg(long)]
        fov_hint: Option<f32>,
        /// Attitude hint quaternion "w,x,y,z" (ICRS→camera), for tracking-mode tests.
        #[arg(long)]
        attitude_hint: Option<String>,
        #[arg(long)]
        grid: bool,
        #[arg(long)]
        json: Option<PathBuf>,
        #[arg(long)]
        overlay_png: Option<PathBuf>,
        /// Observer UTC offset in hours (positive east of Greenwich).
        #[arg(long, default_value_t = 0.0)]
        utc_offset: f64,
        /// Ignore EXIF metadata (no FOV prior / epoch fallback): pure-blind solve.
        #[arg(long)]
        no_exif: bool,
    },
    /// Evaluate the blind solver on a sky-samples manifest and write a report directory.
    Eval {
        #[arg(long)]
        manifest: PathBuf,
        #[arg(long)]
        out_dir: PathBuf,
        /// Comma-separated subset of solver, scene, stress (default: solver).
        #[arg(long, default_value = "solver")]
        tracks: String,
        /// Explicit frame subset (within selected tracks); all listed ids must exist.
        #[arg(long)]
        ids: Option<String>,
        #[arg(long)]
        catalog: Option<PathBuf>,
        #[arg(long)]
        lines: Option<PathBuf>,
        #[arg(long)]
        names: Option<PathBuf>,
        #[arg(long)]
        cache_dir: Option<PathBuf>,
        /// Passthrough to SolveOptions (default: fully blind).
        #[arg(long)]
        fov_hint: Option<f32>,
        /// Regression gate vs a previous summary.json (timing is not gated).
        #[arg(long)]
        baseline: Option<PathBuf>,
        #[arg(long, default_value_t = 10.0)]
        max_axis_p95_regress_pct: f64,
        /// Missing image files become status=missing_image instead of a hard error.
        #[arg(long)]
        allow_missing: bool,
        /// Ignore EXIF metadata (no per-frame FOV prior / epoch fallback).
        #[arg(long)]
        no_exif: bool,
    },
    /// Blind-solve every frame in a directory, using pass-2 hints for failures.
    BatchSolve {
        /// Directory of frames, or a single image file.
        input: PathBuf,
        #[arg(long)]
        out_dir: PathBuf,
        #[arg(long)]
        catalog: Option<PathBuf>,
        #[arg(long)]
        lines: Option<PathBuf>,
        #[arg(long)]
        names: Option<PathBuf>,
        #[arg(long)]
        cache_dir: Option<PathBuf>,
        #[arg(long)]
        fov_hint: Option<f32>,
        #[arg(long)]
        grid: bool,
        /// Ignore EXIF metadata (no per-frame FOV prior / epoch fallback).
        #[arg(long)]
        no_exif: bool,
    },
}

#[derive(Debug, Serialize)]
struct InspectOutput {
    file: String,
    width: u32,
    height: u32,
    min: u8,
    median: u8,
    p99: u8,
    max: u8,
}

#[derive(Debug, Serialize)]
struct CatalogInfoOutput {
    stars: usize,
    brighter_than: BrighterThanCounts,
}

#[derive(Debug, Serialize)]
struct BrighterThanCounts {
    #[serde(rename = "3.0")]
    mag_3_0: usize,
    #[serde(rename = "4.5")]
    mag_4_5: usize,
    #[serde(rename = "6.0")]
    mag_6_0: usize,
    #[serde(rename = "6.5")]
    mag_6_5: usize,
}

#[derive(Debug, Serialize)]
struct ConstellationsInfoOutput {
    constellations: usize,
    polylines: usize,
    vertices: usize,
}

#[derive(Debug, Serialize)]
struct DetectOutput {
    file: String,
    stats: DetectStatsJson,
    detections: Vec<DetectDetectionJson>,
}

#[derive(Debug, Serialize)]
struct DetectStatsJson {
    sigma: f32,
    background_median: f32,
    candidates: u32,
    rejected_border: u32,
    rejected_small: u32,
    rejected_large: u32,
    rejected_elongated: u32,
    rejected_faint: u32,
    rejected_diffuse: u32,
    accepted: u32,
}

#[derive(Debug, Serialize)]
struct DetectDetectionJson {
    x: f64,
    y: f64,
    flux: f32,
    snr: f32,
    area: u32,
    rank: u32,
}

#[derive(Debug, Serialize)]
struct OverlaySummary {
    constellations: usize,
    polylines: usize,
    stars: usize,
    planets: usize,
    grid: usize,
    fov_x_deg: f64,
    fov_y_deg: f64,
}

impl From<DetectStats> for DetectStatsJson {
    fn from(s: DetectStats) -> Self {
        Self {
            sigma: s.sigma,
            background_median: s.background_median,
            candidates: s.candidates,
            rejected_border: s.rejected_border,
            rejected_small: s.rejected_small,
            rejected_large: s.rejected_large,
            rejected_elongated: s.rejected_elongated,
            rejected_faint: s.rejected_faint,
            rejected_diffuse: s.rejected_diffuse,
            accepted: s.accepted,
        }
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    match args.command {
        Command::Inspect { image } => {
            let frame = FrameImage::load(&image)
                .with_context(|| format!("failed to load image '{}'", image.display()))?;
            let output = inspect_output(&image, &frame)?;
            println!("{}", serde_json::to_string(&output)?);
        }
        Command::CatalogInfo { catalog } => {
            let loaded = Catalog::load(&catalog)
                .with_context(|| format!("failed to load catalog '{}'", catalog.display()))?;
            let output = CatalogInfoOutput {
                stars: loaded.stars().len(),
                brighter_than: BrighterThanCounts {
                    mag_3_0: loaded.brighter_than(3.0).len(),
                    mag_4_5: loaded.brighter_than(4.5).len(),
                    mag_6_0: loaded.brighter_than(6.0).len(),
                    mag_6_5: loaded.brighter_than(6.5).len(),
                },
            };
            println!("{}", serde_json::to_string(&output)?);
        }
        Command::ConstellationsInfo { lines, names } => {
            let set = ConstellationSet::load(&lines, &names).with_context(|| {
                format!(
                    "failed to load constellations from '{}' and '{}'",
                    lines.display(),
                    names.display()
                )
            })?;
            let output = ConstellationsInfoOutput {
                constellations: set.constellations().len(),
                polylines: set.polyline_count(),
                vertices: set.vertex_count(),
            };
            println!("{}", serde_json::to_string(&output)?);
        }
        Command::Detect {
            image,
            debug_png,
            json_pretty,
        } => {
            let frame = FrameImage::load(&image)
                .with_context(|| format!("failed to load image '{}'", image.display()))?;
            let result = detect_stars(&frame, &DetectConfig::default());

            if let Some(png_path) = debug_png {
                write_debug_png(&frame, &result.detections, &png_path)?;
            }

            let output = DetectOutput {
                file: image
                    .file_name()
                    .and_then(|name| name.to_str())
                    .unwrap_or_default()
                    .to_string(),
                stats: result.stats.into(),
                detections: result
                    .detections
                    .iter()
                    .map(|d| DetectDetectionJson {
                        x: (d.x * 100.0).round() / 100.0,
                        y: (d.y * 100.0).round() / 100.0,
                        flux: d.flux,
                        snr: d.snr,
                        area: d.area,
                        rank: d.rank,
                    })
                    .collect(),
            };

            if json_pretty {
                println!("{}", serde_json::to_string_pretty(&output)?);
            } else {
                println!("{}", serde_json::to_string(&output)?);
            }
        }
        Command::Overlay {
            image,
            ra,
            dec,
            roll,
            focal_px,
            k1,
            grid,
            out,
            json,
            catalog,
            lines,
            names,
            utc_offset,
        } => {
            run_overlay(OverlayRun {
                image_path: &image,
                ra,
                dec,
                roll,
                focal_px,
                k1,
                grid,
                out_path: out.as_deref(),
                json_path: json.as_deref(),
                catalog_path: catalog.as_deref(),
                lines_path: lines.as_deref(),
                names_path: names.as_deref(),
                utc_offset,
            })?;
        }
        Command::Solve {
            image,
            catalog,
            lines,
            names,
            cache_dir,
            fov_hint,
            attitude_hint,
            grid,
            json,
            overlay_png,
            utc_offset,
            no_exif,
        } => {
            let attitude = attitude_hint.as_deref().map(parse_quat).transpose()?;
            run_solve(SolveRun {
                image_path: &image,
                catalog_path: catalog.as_deref(),
                lines_path: lines.as_deref(),
                names_path: names.as_deref(),
                cache_dir: cache_dir.as_deref(),
                fov_hint,
                attitude_hint: attitude,
                grid,
                json_path: json.as_deref(),
                overlay_png: overlay_png.as_deref(),
                utc_offset,
                no_exif,
            })?;
        }
        Command::BatchSolve {
            input,
            out_dir,
            catalog,
            lines,
            names,
            cache_dir,
            fov_hint,
            grid,
            no_exif,
        } => {
            run_batch_solve(BatchRun {
                input: &input,
                out_dir: &out_dir,
                catalog_path: catalog.as_deref(),
                lines_path: lines.as_deref(),
                names_path: names.as_deref(),
                cache_dir: cache_dir.as_deref(),
                fov_hint,
                grid,
                no_exif,
            })?;
        }
        Command::Eval {
            manifest,
            out_dir,
            tracks,
            ids,
            catalog,
            lines,
            names,
            cache_dir,
            fov_hint,
            baseline,
            max_axis_p95_regress_pct,
            allow_missing,
            no_exif,
        } => {
            match eval_cmd::run_eval(eval_cmd::EvalArgs {
                manifest: &manifest,
                out_dir: &out_dir,
                tracks: &tracks,
                ids: ids.as_deref(),
                catalog_path: catalog.as_deref(),
                lines_path: lines.as_deref(),
                names_path: names.as_deref(),
                cache_dir: cache_dir.as_deref(),
                fov_hint,
                baseline: baseline.as_deref(),
                max_axis_p95_regress_pct,
                allow_missing,
                no_exif,
            }) {
                Ok(eval_cmd::EvalOutcome::Success) => {}
                Ok(eval_cmd::EvalOutcome::GateFailed) => std::process::exit(2),
                Err(e) => return Err(e),
            }
        }
    }
    Ok(())
}

pub(crate) fn data_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../data")
}

struct OverlayRun<'a> {
    image_path: &'a Path,
    ra: f64,
    dec: f64,
    roll: f64,
    focal_px: f64,
    k1: f64,
    grid: bool,
    out_path: Option<&'a Path>,
    json_path: Option<&'a Path>,
    catalog_path: Option<&'a Path>,
    lines_path: Option<&'a Path>,
    names_path: Option<&'a Path>,
    utc_offset: f64,
}

fn run_overlay(args: OverlayRun<'_>) -> Result<()> {
    let frame = FrameImage::load(args.image_path)
        .with_context(|| format!("failed to load image '{}'", args.image_path.display()))?;

    let pose = CameraSolution {
        ra_deg: args.ra,
        dec_deg: args.dec,
        roll_deg: args.roll,
        focal_px: args.focal_px,
        k1: args.k1,
        width: frame.width,
        height: frame.height,
    };

    let catalog_file = args
        .catalog_path
        .map(Path::to_path_buf)
        .unwrap_or_else(|| data_root().join("catalogs/hyg_v3.csv"));
    let lines_file = args
        .lines_path
        .map(Path::to_path_buf)
        .unwrap_or_else(|| data_root().join("celestial/constellations.lines.json"));
    let names_file = args
        .names_path
        .map(Path::to_path_buf)
        .unwrap_or_else(|| data_root().join("celestial/constellations.json"));

    let catalog = Catalog::load(&catalog_file)
        .with_context(|| format!("failed to load catalog '{}'", catalog_file.display()))?;
    let cons = ConstellationSet::load(&lines_file, &names_file).with_context(|| {
        format!(
            "failed to load constellations from '{}' and '{}'",
            lines_file.display(),
            names_file.display()
        )
    })?;

    let timestamp = frame.timestamp_from_name();
    let jd = timestamp.map(|ts| ts.to_jd_utc(args.utc_offset));
    let epoch_years = jd
        .map(starglyph_core::ephem::epoch_years)
        .or_else(|| timestamp.map(|ts| ts.to_epoch_years()));
    let opts = OverlayOptions {
        epoch_years,
        jd_utc: jd,
        include_grid: args.grid,
        ..OverlayOptions::default()
    };
    let overlay = build_overlay(&pose, &catalog, &cons, &opts);

    if let Some(path) = args.json_path {
        let json = serde_json::to_string_pretty(&overlay)?;
        fs::write(path, json)
            .with_context(|| format!("failed to write overlay JSON '{}'", path.display()))?;
    }

    if let Some(path) = args.out_path {
        write_overlay_png(&frame, &overlay, path)?;
    }

    let summary = overlay_summary(&overlay, &pose);
    println!("{}", serde_json::to_string(&summary)?);
    Ok(())
}

fn overlay_summary(overlay: &SolveOverlay, pose: &CameraSolution) -> OverlaySummary {
    let polylines: usize = overlay.constellations.iter().map(|c| c.lines.len()).sum();
    OverlaySummary {
        constellations: overlay.constellations.len(),
        polylines,
        stars: overlay.stars.len(),
        planets: overlay.planets.len(),
        grid: overlay.grid.len(),
        fov_x_deg: (pose.fov_x_deg() * 1000.0).round() / 1000.0,
        fov_y_deg: (pose.fov_y_deg() * 1000.0).round() / 1000.0,
    }
}

fn write_overlay_png(frame: &FrameImage, overlay: &SolveOverlay, path: &Path) -> Result<()> {
    let mut img = render::stretched_base(frame);
    render::draw_overlay(&mut img, overlay);
    img.save(path)
        .with_context(|| format!("failed to write overlay PNG '{}'", path.display()))?;
    Ok(())
}

fn inspect_output(path: &Path, frame: &FrameImage) -> Result<InspectOutput> {
    let scaled = frame
        .gray
        .iter()
        .map(|value| (value * 255.0).round() as u32);
    let min = scaled.clone().min().context("image has no pixels")? as u8;
    let max = scaled.max().context("image has no pixels")? as u8;

    let mut values: Vec<u8> = frame
        .gray
        .iter()
        .map(|value| (value * 255.0).round() as u8)
        .collect();
    values.sort_unstable();
    let median = values[values.len() / 2];
    let p99_index = ((values.len() as f64) * 0.99).floor() as usize;
    let p99 = values[p99_index.min(values.len() - 1)];

    Ok(InspectOutput {
        file: path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or_default()
            .to_string(),
        width: frame.width,
        height: frame.height,
        min,
        median,
        p99,
        max,
    })
}

fn write_debug_png(
    frame: &FrameImage,
    detections: &[starglyph_core::detect::Detection],
    path: &Path,
) -> Result<()> {
    let mut img = render::stretched_base(frame);
    for det in detections {
        render::draw_circle_outline(&mut img, det.x, det.y, 6.0, Rgb([0, 255, 0]));
    }

    img.save(path)
        .with_context(|| format!("failed to write debug PNG '{}'", path.display()))?;
    Ok(())
}

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

pub(crate) fn default_cache_dir() -> PathBuf {
    workspace_root().join("artifacts/cache")
}

pub(crate) fn load_catalog_and_cons(
    catalog_path: Option<&Path>,
    lines_path: Option<&Path>,
    names_path: Option<&Path>,
) -> Result<(Catalog, ConstellationSet)> {
    let catalog_file = catalog_path
        .map(Path::to_path_buf)
        .unwrap_or_else(|| data_root().join("catalogs/hyg_v3.csv"));
    let lines_file = lines_path
        .map(Path::to_path_buf)
        .unwrap_or_else(|| data_root().join("celestial/constellations.lines.json"));
    let names_file = names_path
        .map(Path::to_path_buf)
        .unwrap_or_else(|| data_root().join("celestial/constellations.json"));
    let catalog = Catalog::load(&catalog_file)
        .with_context(|| format!("failed to load catalog '{}'", catalog_file.display()))?;
    let cons = ConstellationSet::load(&lines_file, &names_file).with_context(|| {
        format!(
            "failed to load constellations from '{}' and '{}'",
            lines_file.display(),
            names_file.display()
        )
    })?;
    Ok((catalog, cons))
}

/// One-line solve summary printed to stdout / stored in `summary.json`.
#[derive(Debug, Serialize)]
struct SolveLine {
    file: String,
    status: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    ra: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    dec: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    roll: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    fov_x: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    inliers: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    rms_px: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    log_odds: Option<f64>,
    ms: u64,
}

fn solve_line(file: String, report: &SolveReport) -> SolveLine {
    let solved = report.status == SolveStatus::Solved;
    SolveLine {
        file,
        status: if solved { "solved" } else { "failed" }.to_string(),
        ra: report.pose.as_ref().map(|p| round3(p.ra_deg)),
        dec: report.pose.as_ref().map(|p| round3(p.dec_deg)),
        roll: report.pose.as_ref().map(|p| round3(p.roll_deg)),
        fov_x: report.fov.as_ref().map(|f| round3(f.fov_x_deg)),
        inliers: report.quality.as_ref().map(|q| q.n_inliers),
        rms_px: report.quality.as_ref().map(|q| q.rms_px),
        log_odds: report.quality.as_ref().map(|q| q.log_odds),
        ms: report.timing_ms.as_ref().map(|t| t.total).unwrap_or(0),
    }
}

fn round3(v: f64) -> f64 {
    (v * 1000.0).round() / 1000.0
}

pub(crate) fn print_engine_progress(p: EngineProgress) {
    match p {
        EngineProgress::Loading { kind, path } => {
            eprintln!("[db] loading {kind:?} from {}", path.display());
        }
        EngineProgress::Generating { kind, star_count } => {
            eprintln!(
                "[db] generating {kind:?} from {star_count} stars (first run may take up to ~1 min)..."
            );
        }
        EngineProgress::Saving { kind, path } => {
            eprintln!("[db] saving {kind:?} to {}", path.display());
        }
        EngineProgress::Ready {
            kind,
            catalog_stars,
            patterns,
            bytes,
        } => {
            eprintln!(
                "[db] {kind:?} ready: {catalog_stars} stars, {patterns} patterns, {:.1} MB",
                bytes as f64 / 1e6
            );
        }
    }
}

fn dense_band_kind(fov_deg: f32) -> DbKind {
    DbKind::dense_for_center(fov_deg)
}

struct SolveRun<'a> {
    image_path: &'a Path,
    catalog_path: Option<&'a Path>,
    lines_path: Option<&'a Path>,
    names_path: Option<&'a Path>,
    cache_dir: Option<&'a Path>,
    fov_hint: Option<f32>,
    attitude_hint: Option<[f64; 4]>,
    grid: bool,
    json_path: Option<&'a Path>,
    overlay_png: Option<&'a Path>,
    utc_offset: f64,
    no_exif: bool,
}

fn parse_quat(s: &str) -> Result<[f64; 4]> {
    let parts: Vec<f64> = s
        .split(',')
        .map(|v| v.trim().parse::<f64>())
        .collect::<Result<_, _>>()
        .with_context(|| format!("invalid quaternion '{s}', expected 'w,x,y,z'"))?;
    let arr: [f64; 4] = parts
        .try_into()
        .map_err(|_| anyhow::anyhow!("quaternion must have exactly 4 components: '{s}'"))?;
    Ok(arr)
}

fn run_solve(args: SolveRun<'_>) -> Result<()> {
    let frame = FrameImage::load(args.image_path)
        .with_context(|| format!("failed to load image '{}'", args.image_path.display()))?;
    let (catalog, cons) =
        load_catalog_and_cons(args.catalog_path, args.lines_path, args.names_path)?;
    let cache_dir = args
        .cache_dir
        .map(Path::to_path_buf)
        .unwrap_or_else(default_cache_dir);
    let timestamp = frame.timestamp_from_name();
    let jd = timestamp.map(|t| t.to_jd_utc(args.utc_offset));
    let epoch = jd
        .map(starglyph_core::ephem::epoch_years)
        .or_else(|| timestamp.map(|t| t.to_epoch_years()));

    let opts = SolveOptions {
        fov_hint_deg: args.fov_hint,
        attitude_hint: args.attitude_hint,
        cache_dir,
        allow_dense_band: true,
        epoch_years: epoch,
        utc_offset_hours: args.utc_offset,
        include_grid: args.grid,
        allow_exif_hints: !args.no_exif,
    };

    let mut engine = Engine::default();
    engine
        .ensure_kind(
            &catalog,
            DbKind::Bootstrap,
            &opts.cache_dir,
            &mut print_engine_progress,
        )
        .context("failed to prepare bootstrap database")?;

    let mut stage = |s: SolveStage| eprintln!("[solve] {s:?}");
    let (report, extras) =
        solve_frame_with_engine(&frame, &catalog, &cons, &mut engine, &opts, &mut stage);
    if let Some(q) = extras.attitude_quat {
        eprintln!(
            "[solve] attitude_quat = {},{},{},{}",
            q[0], q[1], q[2], q[3]
        );
    }

    let file = file_name(args.image_path);
    println!("{}", serde_json::to_string(&solve_line(file, &report))?);

    if let Some(path) = args.json_path {
        fs::write(path, serde_json::to_string_pretty(&report)?)
            .with_context(|| format!("failed to write report JSON '{}'", path.display()))?;
    }
    if let Some(path) = args.overlay_png {
        if report.status == SolveStatus::Solved {
            render_report_overlay_png(&frame, &report, path)?;
        } else {
            eprintln!("[solve] not writing overlay PNG: frame did not solve");
        }
    }
    Ok(())
}

struct BatchRun<'a> {
    input: &'a Path,
    out_dir: &'a Path,
    catalog_path: Option<&'a Path>,
    lines_path: Option<&'a Path>,
    names_path: Option<&'a Path>,
    cache_dir: Option<&'a Path>,
    fov_hint: Option<f32>,
    grid: bool,
    no_exif: bool,
}

struct FrameState {
    path: PathBuf,
    name: String,
    epoch: Option<f64>,
    report: SolveReport,
    quat: Option<[f64; 4]>,
    fov_x: Option<f64>,
}

impl FrameState {
    fn solved(&self) -> bool {
        self.report.status == SolveStatus::Solved
    }
}

#[derive(Debug, Serialize)]
struct BatchAggregate {
    solved: usize,
    total: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    fov_median: Option<f64>,
}

#[derive(Debug, Serialize)]
struct BatchSummary {
    frames: Vec<SolveLine>,
    aggregate: BatchAggregate,
}

fn run_batch_solve(args: BatchRun<'_>) -> Result<()> {
    let (catalog, cons) =
        load_catalog_and_cons(args.catalog_path, args.lines_path, args.names_path)?;
    let cache_dir = args
        .cache_dir
        .map(Path::to_path_buf)
        .unwrap_or_else(default_cache_dir);
    fs::create_dir_all(args.out_dir)
        .with_context(|| format!("failed to create out dir '{}'", args.out_dir.display()))?;

    let files = collect_image_files(args.input)?;
    if files.is_empty() {
        anyhow::bail!("no image files found under '{}'", args.input.display());
    }
    eprintln!("[batch] {} frame(s)", files.len());

    let mut engine = Engine::default();
    engine
        .ensure_kind(
            &catalog,
            DbKind::Bootstrap,
            &cache_dir,
            &mut print_engine_progress,
        )
        .context("failed to prepare bootstrap database")?;
    // The narrow-field dense band is the workhorse for this camera; build it up
    // front (visible progress) so pass 1 can solve blind.
    let blind_center = args.fov_hint.unwrap_or(DEFAULT_BLIND_FOV_DEG);
    let _ = engine.ensure_kind(
        &catalog,
        dense_band_kind(blind_center),
        &cache_dir,
        &mut print_engine_progress,
    );

    // ── Pass 1: independent solves (no cross-frame hints) ─────────────────────
    let mut states: Vec<FrameState> = Vec::new();
    for path in &files {
        let frame = match FrameImage::load(path) {
            Ok(f) => f,
            Err(e) => {
                eprintln!("[batch] skip {}: {e}", path.display());
                continue;
            }
        };
        let epoch = frame.timestamp_from_name().map(|t| t.to_epoch_years());
        let opts = SolveOptions {
            fov_hint_deg: args.fov_hint,
            attitude_hint: None,
            cache_dir: cache_dir.clone(),
            allow_dense_band: true,
            epoch_years: epoch,
            utc_offset_hours: 0.0,
            include_grid: args.grid,
            allow_exif_hints: !args.no_exif,
        };
        let mut stage = |_s: SolveStage| {};
        let (report, extras) =
            solve_frame_with_engine(&frame, &catalog, &cons, &mut engine, &opts, &mut stage);
        let name = file_name(path);
        eprintln!(
            "[batch] pass1 {name}: {}",
            serde_json::to_string(&solve_line(name.clone(), &report)).unwrap_or_default()
        );
        states.push(FrameState {
            path: path.clone(),
            name,
            epoch,
            report,
            quat: extras.attitude_quat,
            fov_x: extras.fov_x_deg,
        });
    }

    let fov_median = median_fov(&states);
    eprintln!(
        "[batch] pass1 solved {}/{}, fov_median={:?}",
        states.iter().filter(|s| s.solved()).count(),
        states.len(),
        fov_median.map(round3)
    );

    // ── Pass 2: hint the failures with median FOV + closest solved attitude ────
    let median_f32 = fov_median.map(|m| m as f32).unwrap_or(blind_center);
    for i in 0..states.len() {
        if states[i].solved() {
            continue;
        }
        let hint = closest_solved_quat(&states, i);
        resolve_failed_frame(
            &mut states,
            i,
            &catalog,
            &cons,
            &mut engine,
            &cache_dir,
            median_f32,
            hint,
            args.grid,
            !args.no_exif,
        );
    }

    // ── Write overlays + summary ──────────────────────────────────────────────
    for state in &states {
        if !state.solved() {
            continue;
        }
        if let Ok(frame) = FrameImage::load(&state.path) {
            let png = args.out_dir.join(format!("{}.png", state.name));
            if let Err(e) = render_report_overlay_png(&frame, &state.report, &png) {
                eprintln!("[batch] overlay failed for {}: {e}", state.name);
            }
        }
    }

    let final_median = median_fov(&states);
    let solved = states.iter().filter(|s| s.solved()).count();
    let summary = BatchSummary {
        frames: states
            .iter()
            .map(|s| solve_line(s.name.clone(), &s.report))
            .collect(),
        aggregate: BatchAggregate {
            solved,
            total: states.len(),
            fov_median: final_median.map(round3),
        },
    };
    let summary_path = args.out_dir.join("summary.json");
    fs::write(&summary_path, serde_json::to_string_pretty(&summary)?)
        .with_context(|| format!("failed to write '{}'", summary_path.display()))?;

    println!(
        "{}",
        serde_json::to_string(&summary.aggregate).unwrap_or_default()
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn resolve_failed_frame(
    states: &mut [FrameState],
    i: usize,
    catalog: &Catalog,
    cons: &ConstellationSet,
    engine: &mut Engine,
    cache_dir: &Path,
    fov_hint: f32,
    attitude_hint: Option<[f64; 4]>,
    grid: bool,
    allow_exif_hints: bool,
) {
    let Ok(frame) = FrameImage::load(&states[i].path) else {
        return;
    };
    if std::env::var_os("STARGLYPH_SOLVE_DEBUG").is_some() {
        eprintln!(
            "[batch] pass2 trying {} (fov_hint={fov_hint:.1} attitude={})",
            states[i].name,
            attitude_hint.is_some()
        );
    }
    let opts = SolveOptions {
        fov_hint_deg: Some(fov_hint),
        attitude_hint,
        cache_dir: cache_dir.to_path_buf(),
        allow_dense_band: true,
        epoch_years: states[i].epoch,
        utc_offset_hours: 0.0,
        include_grid: grid,
        allow_exif_hints,
    };
    let mut stage = |_s: SolveStage| {};
    let (report, extras) =
        solve_frame_with_engine(&frame, catalog, cons, engine, &opts, &mut stage);
    if report.status == SolveStatus::Solved {
        eprintln!(
            "[batch] pass2 {}: {}",
            states[i].name,
            serde_json::to_string(&solve_line(states[i].name.clone(), &report)).unwrap_or_default()
        );
        states[i].report = report;
        states[i].quat = extras.attitude_quat;
        states[i].fov_x = extras.fov_x_deg;
    }
}

fn median_fov(states: &[FrameState]) -> Option<f64> {
    let mut fovs: Vec<f64> = states
        .iter()
        .filter(|s| s.solved())
        .filter_map(|s| {
            s.fov_x
                .or_else(|| s.report.fov.as_ref().map(|f| f.fov_x_deg))
        })
        .collect();
    if fovs.is_empty() {
        return None;
    }
    fovs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    Some(fovs[fovs.len() / 2])
}

/// Attitude quaternion of the solved frame temporally closest to `target`.
fn closest_solved_quat(states: &[FrameState], target: usize) -> Option<[f64; 4]> {
    let target_epoch = states[target].epoch;
    let mut best: Option<(f64, [f64; 4])> = None;
    let mut any_solved: Option<[f64; 4]> = None;
    for (j, s) in states.iter().enumerate() {
        if j == target || !s.solved() {
            continue;
        }
        let Some(q) = s.quat else { continue };
        any_solved.get_or_insert(q);
        if let (Some(te), Some(se)) = (target_epoch, s.epoch) {
            let d = (te - se).abs();
            if best.map(|(bd, _)| d < bd).unwrap_or(true) {
                best = Some((d, q));
            }
        }
    }
    best.map(|(_, q)| q).or(any_solved)
}

fn collect_image_files(input: &Path) -> Result<Vec<PathBuf>> {
    if input.is_file() {
        return Ok(vec![input.to_path_buf()]);
    }
    let mut files: Vec<PathBuf> = fs::read_dir(input)
        .with_context(|| format!("failed to read directory '{}'", input.display()))?
        .filter_map(|entry| entry.ok().map(|e| e.path()))
        .filter(|p| p.is_file() && is_image_ext(p))
        .collect();
    files.sort();
    Ok(files)
}

fn is_image_ext(path: &Path) -> bool {
    matches!(
        path.extension()
            .and_then(|e| e.to_str())
            .map(|e| e.to_ascii_lowercase())
            .as_deref(),
        Some("bmp" | "png" | "jpg" | "jpeg" | "tiff" | "tif")
    )
}

fn file_name(path: &Path) -> String {
    path.file_name()
        .and_then(|n| n.to_str())
        .unwrap_or_default()
        .to_string()
}

/// Render a solved frame: overlay geometry plus detection markers
/// (green = matched inlier, gray = unmatched).
fn render_report_overlay_png(frame: &FrameImage, report: &SolveReport, path: &Path) -> Result<()> {
    render::render_report(frame, report)
        .save(path)
        .with_context(|| format!("failed to write overlay PNG '{}'", path.display()))?;
    Ok(())
}

#[cfg(test)]
mod image_ext_tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn is_image_ext_recognizes_supported_formats() {
        assert!(is_image_ext(Path::new("foo.tiff")));
        assert!(is_image_ext(Path::new("foo.TIF")));
        assert!(is_image_ext(Path::new("foo.png")));
        assert!(!is_image_ext(Path::new("foo.txt")));
    }
}

#[cfg(test)]
mod overlay_integration_tests {
    use super::*;

    #[test]
    fn overlay_cli_cygnus_region() {
        let image = data_root().join("input/g128_40ms_1s.bmp");
        if !image.exists() {
            return;
        }
        let out_png = std::env::temp_dir().join("starglyph_overlay_test.png");
        let out_json = std::env::temp_dir().join("starglyph_overlay_test.json");

        run_overlay(OverlayRun {
            image_path: &image,
            ra: 310.0,
            dec: 45.0,
            roll: 0.0,
            focal_px: 800.0,
            k1: 0.0,
            grid: true,
            out_path: Some(out_png.as_path()),
            json_path: Some(out_json.as_path()),
            catalog_path: None,
            lines_path: None,
            names_path: None,
            utc_offset: 0.0,
        })
        .expect("overlay command");

        let png = image::open(&out_png).expect("open png");
        assert_eq!(png.width(), 740);
        assert_eq!(png.height(), 576);

        let json_text = fs::read_to_string(&out_json).expect("read json");
        let overlay: SolveOverlay = serde_json::from_str(&json_text).expect("parse overlay");
        assert!(
            !overlay.constellations.is_empty(),
            "expected at least one constellation in Cygnus region"
        );

        let _ = fs::remove_file(out_png);
        let _ = fs::remove_file(out_json);
    }
}
