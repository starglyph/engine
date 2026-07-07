//! Structured per-request solve telemetry (Stage 0 · D1).
//!
//! Every `/solve` request appends exactly one JSON line to an append-only
//! log. The record is deliberately anonymous: no user identity, no client
//! address, no raw EXIF dump (GPS is never parsed at all — see
//! `image_input`). User-level attribution and aggregation belong to closed
//! wrappers on the other side of the HTTP boundary, not to this service.

use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use serde::Serialize;
use starglyph_core::contracts::{SolveReport, SolveStatus};

/// Bumped only on incompatible record changes; new fields are just added.
pub const SCHEMA_VERSION: u32 = 1;

/// One line of the telemetry log.
#[derive(Debug, Serialize)]
pub struct SolveRecord {
    pub schema: u32,
    /// RFC3339 UTC with millisecond precision.
    pub ts: String,
    /// `solved`/`failed` — the solve ran; `rejected` — the request never
    /// reached a solve (bad input, not ready, busy, timeout).
    pub outcome: &'static str,
    pub http_status: u16,
    /// Error-envelope `code` for rejected requests.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reject_code: Option<&'static str>,
    /// `SolveReport.failure.code` for failed solves.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub failure_code: Option<String>,
    /// Upload file-name stem (never a path) — the only client-chosen text kept.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub image: Option<ImageInfo>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub exif: Option<ExifInfo>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hints: Option<HintsInfo>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub n_detections: Option<u32>,
    /// Present only for solved frames.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<ResultInfo>,
    pub timing: TimingInfo,
}

#[derive(Debug, Serialize)]
pub struct ImageInfo {
    pub width: u32,
    pub height: u32,
}

/// What EXIF contributed, without echoing the metadata itself.
#[derive(Debug, Serialize)]
pub struct ExifInfo {
    pub present: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fov_prior_deg: Option<f64>,
    pub has_timestamp: bool,
}

#[derive(Debug, Serialize)]
pub struct HintsInfo {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fov_hint_deg: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub epoch: Option<f64>,
    pub utc_offset_hours: f64,
    pub no_exif: bool,
    pub grid: bool,
    pub overlay: &'static str,
}

#[derive(Debug, Serialize)]
pub struct ResultInfo {
    pub ra_deg: f64,
    pub dec_deg: f64,
    pub roll_deg: f64,
    pub fov_x_deg: f64,
    pub fov_y_deg: f64,
    pub n_inliers: u32,
    pub rms_px: f64,
    pub log_odds: f64,
    pub confidence: f64,
}

#[derive(Debug, Serialize)]
pub struct TimingInfo {
    /// Wall time inside the handler (queue wait + solve + render + encode).
    pub wall_ms: u64,
    /// Time spent waiting for a free engine (0 when unknown/rejected earlier).
    pub queue_ms: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detect_ms: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub solve_ms: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total_ms: Option<u64>,
}

/// Request facts the handler learns as it advances; consumed into a record.
#[derive(Debug, Default)]
pub struct RequestProbe {
    pub source: Option<String>,
    pub image: Option<ImageInfo>,
    pub exif: Option<ExifInfo>,
    pub hints: Option<HintsInfo>,
    pub queue: Option<Duration>,
}

impl SolveRecord {
    /// Record for a request that never reached a solve.
    pub fn rejected(
        probe: RequestProbe,
        code: &'static str,
        http_status: u16,
        wall: Duration,
    ) -> Self {
        Self {
            schema: SCHEMA_VERSION,
            ts: rfc3339_utc_ms(SystemTime::now()),
            outcome: "rejected",
            http_status,
            reject_code: Some(code),
            failure_code: None,
            source: probe.source,
            image: probe.image,
            exif: probe.exif,
            hints: probe.hints,
            n_detections: None,
            result: None,
            timing: TimingInfo {
                wall_ms: wall.as_millis() as u64,
                queue_ms: probe.queue.map_or(0, |d| d.as_millis() as u64),
                detect_ms: None,
                solve_ms: None,
                total_ms: None,
            },
        }
    }

    /// Record for a request whose solve actually ran (solved or failed).
    pub fn completed(
        probe: RequestProbe,
        report: &SolveReport,
        http_status: u16,
        wall: Duration,
    ) -> Self {
        let solved = report.status == SolveStatus::Solved;
        let result = if solved {
            match (&report.pose, &report.fov, &report.quality) {
                (Some(pose), Some(fov), Some(q)) => Some(ResultInfo {
                    ra_deg: pose.ra_deg,
                    dec_deg: pose.dec_deg,
                    roll_deg: pose.roll_deg,
                    fov_x_deg: fov.fov_x_deg,
                    fov_y_deg: fov.fov_y_deg,
                    n_inliers: q.n_inliers,
                    rms_px: q.rms_px,
                    log_odds: q.log_odds,
                    confidence: q.confidence,
                }),
                _ => None,
            }
        } else {
            None
        };
        Self {
            schema: SCHEMA_VERSION,
            ts: rfc3339_utc_ms(SystemTime::now()),
            outcome: if solved { "solved" } else { "failed" },
            http_status,
            reject_code: None,
            failure_code: report.failure.as_ref().map(|f| f.code.clone()),
            source: probe.source,
            image: probe.image,
            exif: probe.exif,
            hints: probe.hints,
            n_detections: report.quality.as_ref().map(|q| q.n_detections).or_else(|| {
                (!report.detections.is_empty()).then_some(report.detections.len() as u32)
            }),
            result,
            timing: TimingInfo {
                wall_ms: wall.as_millis() as u64,
                queue_ms: probe.queue.map_or(0, |d| d.as_millis() as u64),
                detect_ms: report.timing_ms.as_ref().map(|t| t.detect),
                solve_ms: report.timing_ms.as_ref().map(|t| t.solve),
                total_ms: report.timing_ms.as_ref().map(|t| t.total),
            },
        }
    }
}

/// Append-only JSONL sink. Writes are best-effort: telemetry must never fail
/// or slow a request beyond one small `write(2)` to the page cache.
pub struct TelemetrySink {
    path: PathBuf,
    file: Mutex<File>,
    write_error_logged: AtomicBool,
}

impl TelemetrySink {
    pub fn open(path: &Path) -> std::io::Result<Self> {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)?;
            }
        }
        let file = OpenOptions::new().create(true).append(true).open(path)?;
        Ok(Self {
            path: path.to_path_buf(),
            file: Mutex::new(file),
            write_error_logged: AtomicBool::new(false),
        })
    }

    pub fn append(&self, record: &SolveRecord) {
        let mut line = match serde_json::to_string(record) {
            Ok(line) => line,
            Err(e) => {
                self.warn_once(&e);
                return;
            }
        };
        line.push('\n');
        let mut file = self
            .file
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        // One write per record: O_APPEND keeps concurrent lines intact.
        if let Err(e) = file.write_all(line.as_bytes()) {
            self.warn_once(&e);
        }
    }

    fn warn_once(&self, err: &dyn std::fmt::Display) {
        if !self.write_error_logged.swap(true, Ordering::Relaxed) {
            eprintln!(
                "[serve] telemetry write to '{}' failed (further errors suppressed): {err}",
                self.path.display()
            );
        }
    }
}

/// RFC3339 UTC (`2026-07-06T18:20:00.123Z`) without a date-time dependency.
pub fn rfc3339_utc_ms(t: SystemTime) -> String {
    let d = t.duration_since(UNIX_EPOCH).unwrap_or_default();
    let secs = d.as_secs();
    let (y, m, day) = civil_from_days((secs / 86_400) as i64);
    let sod = secs % 86_400;
    format!(
        "{y:04}-{m:02}-{day:02}T{:02}:{:02}:{:02}.{:03}Z",
        sod / 3600,
        (sod % 3600) / 60,
        sod % 60,
        d.subsec_millis()
    )
}

/// Days since 1970-01-01 → (year, month, day). Howard Hinnant's
/// `civil_from_days`, exact for the proleptic Gregorian calendar.
fn civil_from_days(z: i64) -> (i64, u64, u64) {
    let z = z + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097) as u64;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    (if m <= 2 { y + 1 } else { y }, m, d)
}

#[cfg(test)]
mod tests {
    use super::*;
    use starglyph_core::contracts::{SolveFov, SolvePose, SolveQuality, SolveTimingMs};

    fn at(secs: u64, millis: u32) -> SystemTime {
        UNIX_EPOCH + Duration::from_secs(secs) + Duration::from_millis(millis as u64)
    }

    #[test]
    fn rfc3339_formats_known_instants() {
        assert_eq!(rfc3339_utc_ms(at(0, 0)), "1970-01-01T00:00:00.000Z");
        assert_eq!(rfc3339_utc_ms(at(86_399, 999)), "1970-01-01T23:59:59.999Z");
        // 2000-02-29 (leap day) and the day after.
        assert_eq!(
            rfc3339_utc_ms(at(951_782_400, 0)),
            "2000-02-29T00:00:00.000Z"
        );
        assert_eq!(
            rfc3339_utc_ms(at(951_868_800, 1)),
            "2000-03-01T00:00:00.001Z"
        );
    }

    #[test]
    fn solved_record_serializes_result_and_omits_reject_fields() {
        let report = SolveReport {
            status: SolveStatus::Solved,
            failure: None,
            pose: Some(SolvePose {
                ra_deg: 10.0,
                dec_deg: -5.0,
                roll_deg: 90.0,
            }),
            fov: Some(SolveFov {
                fov_x_deg: 22.0,
                fov_y_deg: 16.5,
                focal_px: 1900.0,
            }),
            quality: Some(SolveQuality {
                n_detections: 40,
                n_inliers: 25,
                rms_px: 0.4,
                log_odds: 80.0,
                confidence: 1.0,
            }),
            timing_ms: Some(SolveTimingMs {
                detect: 30,
                solve: 100,
                total: 130,
            }),
            detections: Vec::new(),
            overlay: None,
        };
        let probe = RequestProbe {
            source: Some("frame".into()),
            image: Some(ImageInfo {
                width: 740,
                height: 576,
            }),
            exif: Some(ExifInfo {
                present: false,
                fov_prior_deg: None,
                has_timestamp: false,
            }),
            hints: Some(HintsInfo {
                fov_hint_deg: None,
                epoch: None,
                utc_offset_hours: 0.0,
                no_exif: false,
                grid: false,
                overlay: "none",
            }),
            queue: Some(Duration::from_millis(3)),
        };
        let record = SolveRecord::completed(probe, &report, 200, Duration::from_millis(180));
        let json = serde_json::to_value(&record).unwrap();
        assert_eq!(json["outcome"], "solved");
        assert_eq!(json["http_status"], 200);
        assert_eq!(json["n_detections"], 40);
        assert_eq!(json["result"]["ra_deg"], 10.0);
        assert_eq!(json["timing"]["queue_ms"], 3);
        assert_eq!(json["timing"]["total_ms"], 130);
        assert!(json.get("reject_code").is_none());
        assert!(json.get("failure_code").is_none());
    }

    #[test]
    fn failed_record_keeps_failure_code_and_detection_count() {
        let mut report = SolveReport::failed("no_confident_match", "no candidate survived");
        report.timing_ms = Some(SolveTimingMs {
            detect: 25,
            solve: 900,
            total: 925,
        });
        let record = SolveRecord::completed(
            RequestProbe::default(),
            &report,
            200,
            Duration::from_millis(930),
        );
        let json = serde_json::to_value(&record).unwrap();
        assert_eq!(json["outcome"], "failed");
        assert_eq!(json["failure_code"], "no_confident_match");
        assert!(json.get("result").is_none());
    }

    #[test]
    fn sink_appends_one_line_per_record() {
        let path = std::env::temp_dir().join(format!(
            "starglyph-telemetry-unit-{}.jsonl",
            std::process::id()
        ));
        let _ = std::fs::remove_file(&path);
        let sink = TelemetrySink::open(&path).unwrap();
        for code in ["missing_image", "busy"] {
            sink.append(&SolveRecord::rejected(
                RequestProbe::default(),
                code,
                400,
                Duration::from_millis(1),
            ));
        }
        let contents = std::fs::read_to_string(&path).unwrap();
        let lines: Vec<_> = contents.lines().collect();
        assert_eq!(lines.len(), 2);
        for line in &lines {
            let value: serde_json::Value = serde_json::from_str(line).unwrap();
            assert_eq!(value["outcome"], "rejected");
            assert_eq!(value["schema"], SCHEMA_VERSION as i64);
        }
        let _ = std::fs::remove_file(&path);
    }
}
