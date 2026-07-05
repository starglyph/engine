use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use simulator_core::catalog::load_catalog;
use simulator_core::config::CatalogConfig;

use crate::config::SolverConfig;
use crate::contracts::{PerFrameReport, SolveStatus};
use crate::io::{load_frame_input, load_manifest};
use crate::pipeline::solve_frame_and_report;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkConfig {
    pub dataset_root: PathBuf,
    pub output_root: PathBuf,
    pub split_filter: Option<Vec<String>>,
    pub solver: SolverConfig,
    pub catalog: CatalogConfig,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            dataset_root: PathBuf::from("artifacts/simulator/dataset-v1"),
            output_root: PathBuf::from("artifacts/recognizer/run-latest"),
            split_filter: None,
            solver: SolverConfig::default(),
            catalog: CatalogConfig::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkReport {
    pub schema_version: String,
    pub dataset_root: PathBuf,
    pub output_root: PathBuf,
    pub frames_processed: usize,
    pub accepted_solves: usize,
    pub accepted_solve_rate: f32,
    pub detection_precision: f32,
    pub detection_recall: f32,
    pub pose_axis_angle_deg: DistributionSummary,
    pub pose_roll_error_deg: DistributionSummary,
    pub stage_timings_ms: StageTimingSummary,
    pub worst_cases: Vec<WorstCaseEntry>,
    pub reproducibility: Option<ReproducibilityCheck>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StageTimingSummary {
    pub detect_ms: DistributionSummary,
    pub match_ms: DistributionSummary,
    pub solve_pose_ms: DistributionSummary,
    pub overlay_ms: DistributionSummary,
    pub total_ms: DistributionSummary,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributionSummary {
    pub median: f32,
    pub p95: f32,
    pub max: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorstCaseEntry {
    pub frame_id: String,
    pub split: String,
    pub status: SolveStatus,
    pub axis_angle_error_deg: Option<f32>,
    pub confidence: f32,
    pub diagnostics_path: PathBuf,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReproducibilityCheck {
    pub passed: bool,
    pub metric_drift_fraction: f32,
    pub ranking_consistent: bool,
}

pub fn run_benchmark(config: &BenchmarkConfig) -> Result<BenchmarkReport> {
    let catalog = load_catalog(&config.catalog)?;
    let manifest = load_manifest(&config.dataset_root)?;
    fs::create_dir_all(&config.output_root)
        .with_context(|| format!("failed to create '{}'", config.output_root.display()))?;
    fs::write(
        config.output_root.join("config.json"),
        serde_json::to_vec_pretty(config)?,
    )
    .with_context(|| {
        format!(
            "failed to write '{}'",
            config.output_root.join("config.json").display()
        )
    })?;

    let mut per_frame_reports = Vec::new();
    for frame in manifest
        .frames
        .iter()
        .filter(|record| frame_matches_filter(record, &config.split_filter))
    {
        let input = load_frame_input(&config.dataset_root, frame)?;
        let frame_dir = config.output_root.join("per-frame").join(&frame.frame_id);
        fs::create_dir_all(&frame_dir)
            .with_context(|| format!("failed to create '{}'", frame_dir.display()))?;
        let (result, report) =
            solve_frame_and_report(&input, &catalog, &config.solver, Some(&frame_dir))?;
        fs::write(
            frame_dir.join("result.json"),
            serde_json::to_vec_pretty(&result)?,
        )
        .with_context(|| {
            format!(
                "failed to write '{}'",
                frame_dir.join("result.json").display()
            )
        })?;
        per_frame_reports.push(report);
    }

    let mut benchmark = aggregate_report(config, &manifest.schema_version, &per_frame_reports);
    export_worst_cases(
        &config.output_root,
        &per_frame_reports,
        config.solver.benchmark.worst_case_count,
        &mut benchmark.worst_cases,
    )?;

    if config.solver.benchmark.run_reproducibility_check {
        let repeat_report = rerun_for_reproducibility(config, &catalog)?;
        benchmark.reproducibility = Some(compare_reports(
            &benchmark,
            &repeat_report,
            config.solver.benchmark.reproducibility_tolerance,
        ));
    }

    fs::write(
        config.output_root.join("summary.json"),
        serde_json::to_vec_pretty(&benchmark)?,
    )
    .with_context(|| {
        format!(
            "failed to write '{}'",
            config.output_root.join("summary.json").display()
        )
    })?;
    Ok(benchmark)
}

fn rerun_for_reproducibility(
    config: &BenchmarkConfig,
    catalog: &[simulator_core::catalog::Star],
) -> Result<BenchmarkReport> {
    let manifest = load_manifest(&config.dataset_root)?;
    let mut reports = Vec::new();
    for frame in manifest
        .frames
        .iter()
        .filter(|record| frame_matches_filter(record, &config.split_filter))
    {
        let input = load_frame_input(&config.dataset_root, frame)?;
        let (_, report) = solve_frame_and_report(&input, catalog, &config.solver, None)?;
        reports.push(report);
    }
    let mut aggregate = aggregate_report(config, &manifest.schema_version, &reports);
    aggregate.worst_cases = select_worst_cases(&reports, config.solver.benchmark.worst_case_count);
    Ok(aggregate)
}

fn compare_reports(
    baseline: &BenchmarkReport,
    repeated: &BenchmarkReport,
    tolerance: f32,
) -> ReproducibilityCheck {
    let metrics = [
        relative_diff(baseline.detection_precision, repeated.detection_precision),
        relative_diff(baseline.detection_recall, repeated.detection_recall),
        relative_diff(
            baseline.pose_axis_angle_deg.median,
            repeated.pose_axis_angle_deg.median,
        ),
        relative_diff(
            baseline.pose_axis_angle_deg.p95,
            repeated.pose_axis_angle_deg.p95,
        ),
    ];
    let metric_drift_fraction = metrics.into_iter().fold(0.0_f32, f32::max);
    let ranking_consistent = baseline
        .worst_cases
        .iter()
        .map(|entry| &entry.frame_id)
        .collect::<Vec<_>>()
        == repeated
            .worst_cases
            .iter()
            .map(|entry| &entry.frame_id)
            .collect::<Vec<_>>();
    ReproducibilityCheck {
        passed: metric_drift_fraction <= tolerance && ranking_consistent,
        metric_drift_fraction,
        ranking_consistent,
    }
}

fn relative_diff(lhs: f32, rhs: f32) -> f32 {
    let denom = lhs.abs().max(1e-6);
    (lhs - rhs).abs() / denom
}

fn export_worst_cases(
    output_root: &Path,
    reports: &[PerFrameReport],
    count: usize,
    destination: &mut Vec<WorstCaseEntry>,
) -> Result<()> {
    let worst = select_worst_cases(reports, count);
    let worst_dir = output_root.join("worst-cases");
    fs::create_dir_all(&worst_dir)
        .with_context(|| format!("failed to create '{}'", worst_dir.display()))?;
    destination.clear();
    for frame in &worst {
        let src_dir = output_root.join("per-frame").join(&frame.frame_id);
        let dst_dir = worst_dir.join(&frame.frame_id);
        fs::create_dir_all(&dst_dir)
            .with_context(|| format!("failed to create '{}'", dst_dir.display()))?;
        for file in [
            "result.json",
            "overlay.png",
            "detections.png",
            "correspondences.png",
        ] {
            let src = src_dir.join(file);
            if src.exists() {
                let dst = dst_dir.join(file);
                fs::copy(&src, &dst).with_context(|| {
                    format!("failed to copy '{}' to '{}'", src.display(), dst.display())
                })?;
            }
        }
        destination.push(WorstCaseEntry {
            diagnostics_path: dst_dir.join("result.json"),
            ..frame.clone()
        });
    }
    Ok(())
}

fn select_worst_cases(reports: &[PerFrameReport], count: usize) -> Vec<WorstCaseEntry> {
    let mut ranked = reports.to_vec();
    ranked.sort_by(|a, b| {
        worst_case_score(b)
            .total_cmp(&worst_case_score(a))
            .then_with(|| a.frame_id.cmp(&b.frame_id))
    });
    ranked
        .into_iter()
        .take(count)
        .map(|frame| WorstCaseEntry {
            frame_id: frame.frame_id,
            split: frame.split,
            status: frame.status,
            axis_angle_error_deg: frame.axis_angle_error_deg,
            confidence: frame.confidence,
            diagnostics_path: frame.diagnostics_path,
        })
        .collect()
}

fn worst_case_score(report: &PerFrameReport) -> f32 {
    report
        .axis_angle_error_deg
        .unwrap_or(if report.status == SolveStatus::Rejected {
            180.0
        } else {
            0.0
        })
}

fn frame_matches_filter(
    frame: &simulator_core::dataset::FrameArtifactRecord,
    filter: &Option<Vec<String>>,
) -> bool {
    filter
        .as_ref()
        .map(|items| items.iter().any(|split| split == &frame.split))
        .unwrap_or(true)
}

fn aggregate_report(
    config: &BenchmarkConfig,
    schema_version: &str,
    reports: &[PerFrameReport],
) -> BenchmarkReport {
    let frames_processed = reports.len();
    let accepted_solves = reports
        .iter()
        .filter(|report| report.status == SolveStatus::Accepted)
        .count();
    let accepted_solve_rate = if frames_processed == 0 {
        0.0
    } else {
        accepted_solves as f32 / frames_processed as f32
    };
    let (tp, fp, fn_) = sum_detection_counts(reports);
    let detection_precision = if tp + fp == 0 {
        0.0
    } else {
        tp as f32 / (tp + fp) as f32
    };
    let detection_recall = if tp + fn_ == 0 {
        0.0
    } else {
        tp as f32 / (tp + fn_) as f32
    };
    let axis = reports
        .iter()
        .filter_map(|report| report.axis_angle_error_deg)
        .collect::<Vec<_>>();
    let roll = reports
        .iter()
        .filter_map(|report| report.roll_error_deg)
        .collect::<Vec<_>>();

    let detect_ms = reports
        .iter()
        .map(|r| r.timings_ms.detect as f32)
        .collect::<Vec<_>>();
    let match_ms = reports
        .iter()
        .map(|r| r.timings_ms.match_candidates as f32)
        .collect::<Vec<_>>();
    let solve_pose_ms = reports
        .iter()
        .map(|r| r.timings_ms.solve_pose as f32)
        .collect::<Vec<_>>();
    let overlay_ms = reports
        .iter()
        .map(|r| r.timings_ms.render_overlay as f32)
        .collect::<Vec<_>>();
    let total_ms = reports
        .iter()
        .map(|r| r.timings_ms.total as f32)
        .collect::<Vec<_>>();

    BenchmarkReport {
        schema_version: schema_version.to_string(),
        dataset_root: config.dataset_root.clone(),
        output_root: config.output_root.clone(),
        frames_processed,
        accepted_solves,
        accepted_solve_rate,
        detection_precision,
        detection_recall,
        pose_axis_angle_deg: summarize_distribution(axis),
        pose_roll_error_deg: summarize_distribution(roll),
        stage_timings_ms: StageTimingSummary {
            detect_ms: summarize_distribution(detect_ms),
            match_ms: summarize_distribution(match_ms),
            solve_pose_ms: summarize_distribution(solve_pose_ms),
            overlay_ms: summarize_distribution(overlay_ms),
            total_ms: summarize_distribution(total_ms),
        },
        worst_cases: reports
            .iter()
            .map(|frame| WorstCaseEntry {
                frame_id: frame.frame_id.clone(),
                split: frame.split.clone(),
                status: frame.status,
                axis_angle_error_deg: frame.axis_angle_error_deg,
                confidence: frame.confidence,
                diagnostics_path: frame.diagnostics_path.clone(),
            })
            .collect(),
        reproducibility: None,
    }
}

fn sum_detection_counts(reports: &[PerFrameReport]) -> (usize, usize, usize) {
    reports
        .iter()
        .filter_map(|report| report.detection.as_ref())
        .fold((0, 0, 0), |(tp_acc, fp_acc, fn_acc), metric| {
            (tp_acc + metric.tp, fp_acc + metric.fp, fn_acc + metric.fn_)
        })
}

fn summarize_distribution(mut values: Vec<f32>) -> DistributionSummary {
    if values.is_empty() {
        return DistributionSummary {
            median: 0.0,
            p95: 0.0,
            max: 0.0,
        };
    }
    values.sort_by(f32::total_cmp);
    DistributionSummary {
        median: percentile(&values, 0.5),
        p95: percentile(&values, 0.95),
        max: *values.last().unwrap_or(&0.0),
    }
}

fn percentile(values: &[f32], p: f32) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    let idx = ((values.len() - 1) as f32 * p).round() as usize;
    values[idx.min(values.len() - 1)]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn summarizes_distribution_consistently() {
        let summary = summarize_distribution(vec![1.0, 4.0, 2.0, 3.0, 10.0]);
        assert_eq!(summary.median, 3.0);
        assert_eq!(summary.max, 10.0);
        assert!(summary.p95 >= summary.median);
    }

    #[test]
    fn compares_reports_with_tolerance() {
        let baseline = BenchmarkReport {
            schema_version: "v".to_string(),
            dataset_root: PathBuf::new(),
            output_root: PathBuf::new(),
            frames_processed: 1,
            accepted_solves: 1,
            accepted_solve_rate: 1.0,
            detection_precision: 0.9,
            detection_recall: 0.8,
            pose_axis_angle_deg: DistributionSummary {
                median: 1.0,
                p95: 2.0,
                max: 2.5,
            },
            pose_roll_error_deg: DistributionSummary {
                median: 0.5,
                p95: 1.0,
                max: 2.0,
            },
            stage_timings_ms: StageTimingSummary {
                detect_ms: DistributionSummary {
                    median: 1.0,
                    p95: 2.0,
                    max: 3.0,
                },
                match_ms: DistributionSummary {
                    median: 1.0,
                    p95: 2.0,
                    max: 3.0,
                },
                solve_pose_ms: DistributionSummary {
                    median: 1.0,
                    p95: 2.0,
                    max: 3.0,
                },
                overlay_ms: DistributionSummary {
                    median: 1.0,
                    p95: 2.0,
                    max: 3.0,
                },
                total_ms: DistributionSummary {
                    median: 4.0,
                    p95: 6.0,
                    max: 9.0,
                },
            },
            worst_cases: vec![WorstCaseEntry {
                frame_id: "frame-1".to_string(),
                split: "test".to_string(),
                status: SolveStatus::Accepted,
                axis_angle_error_deg: Some(1.0),
                confidence: 0.8,
                diagnostics_path: PathBuf::from("a"),
            }],
            reproducibility: None,
        };
        let mut repeated = baseline.clone();
        repeated.detection_precision = 0.91;
        let check = compare_reports(&baseline, &repeated, 0.02);
        assert!(check.passed);
    }
}
