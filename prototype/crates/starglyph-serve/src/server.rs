//! Axum router and handlers: `POST /solve` (multipart image → `SolveReport`
//! JSON, optionally with a rendered overlay PNG), `GET /healthz` (liveness),
//! `GET /readyz` (readiness — bootstrap database warmed).

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::extract::multipart::MultipartError;
use axum::extract::{DefaultBodyLimit, Multipart, Query, State};
use axum::http::{header, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use base64::Engine as _;
use serde::{Deserialize, Serialize};
use starglyph_core::catalog::Catalog;
use starglyph_core::constellations::ConstellationSet;
use starglyph_core::contracts::{SolveReport, SolveStatus};
use starglyph_core::engine::Engine;
use starglyph_core::image_input::FrameImage;
use starglyph_core::render;
use starglyph_core::solve::{solve_frame_with_engine, SolveOptions};

use crate::pool::EnginePool;
use crate::telemetry::{ExifInfo, HintsInfo, ImageInfo, RequestProbe, SolveRecord, TelemetrySink};

const USAGE: &str = "\
starglyph-serve: headless plate solver

POST /solve    multipart/form-data:
               image      the photo (required; PNG/JPEG/BMP/TIFF)
               fov_hint   horizontal FOV prior, degrees (optional)
               epoch      observation epoch, fractional years (optional)
               utc_offset observer UTC offset, hours (optional)
               no_exif    'true' to ignore EXIF-derived hints (optional)
               grid       'true' to include the RA/Dec grid overlay (optional)
               overlay    'png' or 'inline' (also as ?overlay=...)
GET  /healthz  liveness
GET  /readyz   readiness (200 once the bootstrap database is warmed)
";

/// Shared state handed to every request handler.
#[derive(Clone)]
pub struct AppState {
    pub catalog: Arc<Catalog>,
    pub cons: Arc<ConstellationSet>,
    pub pool: EnginePool,
    pub ready: Arc<AtomicBool>,
    pub cache_dir: PathBuf,
    /// Max wait for a free engine before answering 503.
    pub queue_timeout: Duration,
    /// Max wall time for one solve before answering 504 (the engine returns
    /// to the pool whenever the abandoned solve actually finishes).
    pub solve_timeout: Duration,
    /// Anonymous per-request telemetry (Stage 0 · D1); `None` disables.
    pub telemetry: Option<Arc<TelemetrySink>>,
}

pub fn router(state: AppState, max_body_bytes: usize) -> Router {
    Router::new()
        .route("/", get(|| async { USAGE }))
        .route("/healthz", get(|| async { "ok" }))
        .route("/readyz", get(readyz))
        .route("/solve", post(solve))
        .layer(DefaultBodyLimit::max(max_body_bytes))
        .with_state(state)
}

async fn readyz(State(state): State<AppState>) -> Response {
    if state.ready.load(Ordering::Acquire) {
        (StatusCode::OK, "ready").into_response()
    } else {
        (
            StatusCode::SERVICE_UNAVAILABLE,
            "warming: bootstrap database not ready",
        )
            .into_response()
    }
}

// ── Error envelope ────────────────────────────────────────────────────────────

struct ApiError {
    status: StatusCode,
    code: &'static str,
    message: String,
}

impl ApiError {
    fn bad_request(code: &'static str, message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            code,
            message: message.into(),
        }
    }

    fn internal(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            code: "internal",
            message: message.into(),
        }
    }

    fn not_ready() -> Self {
        Self {
            status: StatusCode::SERVICE_UNAVAILABLE,
            code: "not_ready",
            message: "bootstrap database is still warming; poll /readyz".to_string(),
        }
    }

    fn busy(waited: Duration) -> Self {
        Self {
            status: StatusCode::SERVICE_UNAVAILABLE,
            code: "busy",
            message: format!(
                "no engine became free within {}s; retry later",
                waited.as_secs()
            ),
        }
    }

    fn solve_timeout(limit: Duration) -> Self {
        Self {
            status: StatusCode::GATEWAY_TIMEOUT,
            code: "solve_timeout",
            message: format!("solve exceeded {}s", limit.as_secs()),
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let body = serde_json::json!({
            "error": { "code": self.code, "message": self.message }
        });
        (self.status, Json(body)).into_response()
    }
}

fn multipart_err(e: MultipartError) -> ApiError {
    ApiError {
        status: e.status(),
        code: "bad_multipart",
        message: e.body_text(),
    }
}

// ── Request parsing ───────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OverlayMode {
    None,
    Png,
    Inline,
}

fn parse_overlay(value: &str) -> Result<OverlayMode, ApiError> {
    match value.trim().to_ascii_lowercase().as_str() {
        "" | "none" => Ok(OverlayMode::None),
        "png" => Ok(OverlayMode::Png),
        "inline" => Ok(OverlayMode::Inline),
        other => Err(ApiError::bad_request(
            "bad_overlay",
            format!("overlay must be 'png', 'inline' or 'none', got '{other}'"),
        )),
    }
}

fn parse_bool(name: &str, value: &str) -> Result<bool, ApiError> {
    match value.trim().to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "on" => Ok(true),
        "" | "0" | "false" | "no" | "off" => Ok(false),
        other => Err(ApiError::bad_request(
            "bad_bool",
            format!("field '{name}' must be a boolean, got '{other}'"),
        )),
    }
}

fn parse_ranged_f64(
    name: &str,
    value: &str,
    range: std::ops::RangeInclusive<f64>,
) -> Result<f64, ApiError> {
    let parsed: f64 = value.trim().parse().map_err(|_| {
        ApiError::bad_request(
            "bad_number",
            format!("field '{name}' must be a number, got '{value}'"),
        )
    })?;
    if !parsed.is_finite() || !range.contains(&parsed) {
        return Err(ApiError::bad_request(
            "out_of_range",
            format!(
                "field '{name}' must be within [{}, {}], got {parsed}",
                range.start(),
                range.end()
            ),
        ));
    }
    Ok(parsed)
}

#[derive(Default)]
struct SolveForm {
    image: Option<(String, Vec<u8>)>,
    fov_hint: Option<f32>,
    epoch: Option<f64>,
    utc_offset: f64,
    no_exif: bool,
    grid: bool,
    overlay: Option<OverlayMode>,
}

async fn parse_solve_form(mut multipart: Multipart) -> Result<SolveForm, ApiError> {
    let mut form = SolveForm::default();
    while let Some(field) = multipart.next_field().await.map_err(multipart_err)? {
        let name = field.name().unwrap_or_default().to_string();
        match name.as_str() {
            "image" => {
                let source_name = field
                    .file_name()
                    .and_then(|f| Path::new(f).file_stem())
                    .and_then(|s| s.to_str())
                    .unwrap_or("upload")
                    .to_string();
                let bytes = field.bytes().await.map_err(multipart_err)?;
                form.image = Some((source_name, bytes.to_vec()));
            }
            "fov_hint" => {
                let text = field.text().await.map_err(multipart_err)?;
                form.fov_hint = Some(parse_ranged_f64("fov_hint", &text, 0.5..=120.0)? as f32);
            }
            "epoch" => {
                let text = field.text().await.map_err(multipart_err)?;
                form.epoch = Some(parse_ranged_f64("epoch", &text, 1800.0..=2200.0)?);
            }
            "utc_offset" => {
                let text = field.text().await.map_err(multipart_err)?;
                form.utc_offset = parse_ranged_f64("utc_offset", &text, -14.0..=14.0)?;
            }
            "no_exif" => {
                let text = field.text().await.map_err(multipart_err)?;
                form.no_exif = parse_bool("no_exif", &text)?;
            }
            "grid" => {
                let text = field.text().await.map_err(multipart_err)?;
                form.grid = parse_bool("grid", &text)?;
            }
            "overlay" => {
                let text = field.text().await.map_err(multipart_err)?;
                form.overlay = Some(parse_overlay(&text)?);
            }
            _ => {
                // Drain and ignore unknown fields (still surfacing body-limit errors).
                field.bytes().await.map_err(multipart_err)?;
            }
        }
    }
    Ok(form)
}

// ── POST /solve ───────────────────────────────────────────────────────────────

#[derive(Deserialize)]
struct SolveQuery {
    overlay: Option<String>,
}

/// JSON shape for `overlay=inline`: the report plus the rendered PNG.
#[derive(Serialize)]
struct InlineSolveResponse {
    report: SolveReport,
    #[serde(skip_serializing_if = "Option::is_none")]
    overlay_png_base64: Option<String>,
}

/// Successful `/solve` outcome: the solve ran (whatever its verdict).
struct SolveOutcome {
    report: SolveReport,
    overlay_png: Option<Vec<u8>>,
    overlay_mode: OverlayMode,
}

impl SolveOutcome {
    /// The HTTP status this outcome will answer with.
    fn http_status(&self) -> StatusCode {
        match (self.overlay_mode, &self.overlay_png) {
            // The solve ran but there is nothing to draw.
            (OverlayMode::Png, None) => StatusCode::UNPROCESSABLE_ENTITY,
            _ => StatusCode::OK,
        }
    }
}

/// Telemetry wrapper: exactly one record per request, whatever the path out.
async fn solve(
    State(state): State<AppState>,
    Query(query): Query<SolveQuery>,
    multipart: Multipart,
) -> Response {
    let wall = Instant::now();
    let mut probe = RequestProbe::default();
    let result = solve_inner(&state, query, multipart, &mut probe).await;
    if let Some(sink) = &state.telemetry {
        // One small append to the page cache; comparable to the eprintln
        // logging this handler already does, so written inline.
        let record = match &result {
            Ok(outcome) => SolveRecord::completed(
                probe,
                &outcome.report,
                outcome.http_status().as_u16(),
                wall.elapsed(),
            ),
            Err(e) => SolveRecord::rejected(probe, e.code, e.status.as_u16(), wall.elapsed()),
        };
        sink.append(&record);
    }

    let outcome = match result {
        Ok(outcome) => outcome,
        Err(e) => return e.into_response(),
    };
    let status = outcome.http_status();
    match outcome.overlay_mode {
        OverlayMode::None => Json(outcome.report).into_response(),
        OverlayMode::Png => match outcome.overlay_png {
            Some(png) => ([(header::CONTENT_TYPE, "image/png")], png).into_response(),
            // 422: hand the report back so the caller sees why (see docs).
            None => (status, Json(outcome.report)).into_response(),
        },
        OverlayMode::Inline => Json(InlineSolveResponse {
            overlay_png_base64: outcome
                .overlay_png
                .map(|png| base64::engine::general_purpose::STANDARD.encode(png)),
            report: outcome.report,
        })
        .into_response(),
    }
}

async fn solve_inner(
    state: &AppState,
    query: SolveQuery,
    multipart: Multipart,
    probe: &mut RequestProbe,
) -> Result<SolveOutcome, ApiError> {
    if !state.ready.load(Ordering::Acquire) {
        return Err(ApiError::not_ready());
    }

    let form = parse_solve_form(multipart).await?;
    let overlay_mode = match form.overlay {
        Some(mode) => mode,
        None => query
            .overlay
            .as_deref()
            .map(parse_overlay)
            .transpose()?
            .unwrap_or(OverlayMode::None),
    };
    probe.hints = Some(HintsInfo {
        fov_hint_deg: form.fov_hint,
        epoch: form.epoch,
        utc_offset_hours: form.utc_offset,
        no_exif: form.no_exif,
        grid: form.grid,
        overlay: match overlay_mode {
            OverlayMode::None => "none",
            OverlayMode::Png => "png",
            OverlayMode::Inline => "inline",
        },
    });
    let (source_name, image_bytes) = form.image.ok_or_else(|| {
        ApiError::bad_request("missing_image", "multipart field 'image' is required")
    })?;
    probe.source = Some(source_name.clone());

    // Decode off the async runtime. No engine is needed yet, so undecodable
    // uploads are rejected without consuming a pool slot.
    let frame =
        tokio::task::spawn_blocking(move || FrameImage::from_bytes(&image_bytes, &source_name))
            .await
            .map_err(|e| ApiError::internal(format!("decode task died: {e}")))?
            .map_err(|e| ApiError::bad_request("bad_image", format!("cannot decode image: {e}")))?;
    probe.image = Some(ImageInfo {
        width: frame.width,
        height: frame.height,
    });
    probe.exif = Some(ExifInfo {
        present: frame.exif.is_some(),
        fov_prior_deg: frame
            .exif
            .as_ref()
            .and_then(|e| e.fov_x_deg(frame.width, frame.height)),
        has_timestamp: frame
            .exif
            .as_ref()
            .is_some_and(|e| e.datetime_original.is_some()),
    });

    let opts = SolveOptions {
        fov_hint_deg: form.fov_hint,
        attitude_hint: None,
        cache_dir: state.cache_dir.clone(),
        allow_dense_band: true,
        epoch_years: form.epoch,
        utc_offset_hours: form.utc_offset,
        include_grid: form.grid,
        allow_exif_hints: !form.no_exif,
    };

    let queue_started = Instant::now();
    let (permit, engine) = tokio::time::timeout(state.queue_timeout, state.pool.checkout())
        .await
        .map_err(|_| ApiError::busy(state.queue_timeout))?;
    probe.queue = Some(queue_started.elapsed());

    let catalog = Arc::clone(&state.catalog);
    let cons = Arc::clone(&state.cons);
    let name = frame.source_name.clone();
    let solve_task = tokio::task::spawn_blocking(move || {
        let mut engine = engine;
        let started = Instant::now();
        let (report, _extras) =
            solve_frame_with_engine(&frame, &catalog, &cons, &mut engine, &opts, &mut |_| {});
        let overlay_png = (overlay_mode != OverlayMode::None
            && report.status == SolveStatus::Solved)
            .then(|| render::encode_png(&render::render_report(&frame, &report)))
            .transpose()
            .unwrap_or_else(|e| {
                eprintln!("[serve] overlay PNG encode failed: {e}");
                None
            });
        (engine, report, overlay_png, started.elapsed())
    });

    // Detached reclaim: the engine and permit go back to the pool even when
    // this handler is cancelled (client disconnect) or gives up below.
    let (result_tx, result_rx) = tokio::sync::oneshot::channel();
    let pool = state.pool.clone();
    tokio::spawn(async move {
        match solve_task.await {
            Ok((engine, report, overlay_png, elapsed)) => {
                pool.checkin(permit, engine);
                let _ = result_tx.send((report, overlay_png, elapsed));
            }
            Err(join_error) => {
                eprintln!("[serve] solve task died: {join_error}");
                // The replacement engine reloads databases lazily from the disk cache.
                pool.checkin(permit, Engine::default());
                // result_tx dropped → the handler answers with an internal error.
            }
        }
    });

    let (report, overlay_png, elapsed) =
        match tokio::time::timeout(state.solve_timeout, result_rx).await {
            Err(_elapsed) => return Err(ApiError::solve_timeout(state.solve_timeout)),
            Ok(Err(_sender_dropped)) => {
                return Err(ApiError::internal("solve task died; see server log"))
            }
            Ok(Ok(outcome)) => outcome,
        };

    let status = match report.status {
        SolveStatus::Solved => "solved",
        SolveStatus::Failed => "failed",
    };
    eprintln!("[serve] {name}: {status} in {} ms", elapsed.as_millis());

    Ok(SolveOutcome {
        report,
        overlay_png,
        overlay_mode,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::Request;
    use http_body_util::BodyExt;
    use std::sync::LazyLock;
    use tower::ServiceExt;

    static CATALOG_AND_CONS: LazyLock<(Arc<Catalog>, Arc<ConstellationSet>)> =
        LazyLock::new(|| {
            let data = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../data");
            // The committed catalog (hyg_v3.csv is a local fetch artifact, absent in CI).
            let catalog = Catalog::load(&data.join("catalogs/hyg_v42.csv.gz")).expect("catalog");
            let cons = ConstellationSet::load(
                &data.join("celestial/constellations.lines.json"),
                &data.join("celestial/constellations.json"),
            )
            .expect("constellations");
            (Arc::new(catalog), Arc::new(cons))
        });

    fn test_state(ready: bool) -> AppState {
        let (catalog, cons) = CATALOG_AND_CONS.clone();
        AppState {
            catalog,
            cons,
            pool: EnginePool::new(),
            ready: Arc::new(AtomicBool::new(ready)),
            cache_dir: PathBuf::from("unused-in-tests"),
            queue_timeout: Duration::from_millis(200),
            solve_timeout: Duration::from_secs(5),
            telemetry: None,
        }
    }

    /// Fresh telemetry sink in the OS temp dir; the caller removes the file.
    fn test_telemetry(name: &str) -> (PathBuf, Arc<TelemetrySink>) {
        let path = std::env::temp_dir().join(format!(
            "starglyph-serve-telemetry-{}-{name}.jsonl",
            std::process::id()
        ));
        let _ = std::fs::remove_file(&path);
        let sink = Arc::new(TelemetrySink::open(&path).expect("open telemetry sink"));
        (path, sink)
    }

    fn read_single_record(path: &Path) -> serde_json::Value {
        let contents = std::fs::read_to_string(path).expect("read telemetry log");
        let mut lines = contents.lines();
        let line = lines.next().expect("one telemetry record");
        assert!(lines.next().is_none(), "expected exactly one record");
        serde_json::from_str(line).expect("valid JSON record")
    }

    const BOUNDARY: &str = "starglyph-test-boundary";

    fn multipart_request(uri: &str, parts: &[(&str, Option<&str>, &[u8])]) -> Request<Body> {
        let mut body = Vec::new();
        for (name, filename, bytes) in parts {
            body.extend_from_slice(format!("--{BOUNDARY}\r\n").as_bytes());
            let disposition = match filename {
                Some(f) => {
                    format!(
                        "Content-Disposition: form-data; name=\"{name}\"; filename=\"{f}\"\r\n\r\n"
                    )
                }
                None => format!("Content-Disposition: form-data; name=\"{name}\"\r\n\r\n"),
            };
            body.extend_from_slice(disposition.as_bytes());
            body.extend_from_slice(bytes);
            body.extend_from_slice(b"\r\n");
        }
        body.extend_from_slice(format!("--{BOUNDARY}--\r\n").as_bytes());
        Request::builder()
            .method("POST")
            .uri(uri)
            .header(
                header::CONTENT_TYPE,
                format!("multipart/form-data; boundary={BOUNDARY}"),
            )
            .body(Body::from(body))
            .expect("request")
    }

    async fn body_json(response: Response) -> serde_json::Value {
        let bytes = response
            .into_body()
            .collect()
            .await
            .expect("collect body")
            .to_bytes();
        serde_json::from_slice(&bytes).expect("json body")
    }

    fn tiny_png() -> Vec<u8> {
        let img = image::GrayImage::from_pixel(8, 8, image::Luma([12u8]));
        let mut bytes = Vec::new();
        image::DynamicImage::ImageLuma8(img)
            .write_to(
                &mut std::io::Cursor::new(&mut bytes),
                image::ImageFormat::Png,
            )
            .expect("encode png");
        bytes
    }

    #[tokio::test]
    async fn healthz_is_ok_and_root_shows_usage() {
        let app = router(test_state(false), 1024);
        let response = app
            .clone()
            .oneshot(Request::get("/healthz").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);

        let response = app
            .oneshot(Request::get("/").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn readyz_gates_on_the_flag() {
        let app = router(test_state(false), 1024);
        let response = app
            .oneshot(Request::get("/readyz").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);

        let app = router(test_state(true), 1024);
        let response = app
            .oneshot(Request::get("/readyz").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn solve_is_rejected_before_ready() {
        let app = router(test_state(false), 1 << 20);
        let request = multipart_request("/solve", &[("image", Some("x.png"), &tiny_png())]);
        let response = app.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(body_json(response).await["error"]["code"], "not_ready");
    }

    #[tokio::test]
    async fn solve_requires_the_image_field() {
        let app = router(test_state(true), 1 << 20);
        let request = multipart_request("/solve", &[("fov_hint", None, b"22.0")]);
        let response = app.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        assert_eq!(body_json(response).await["error"]["code"], "missing_image");
    }

    #[tokio::test]
    async fn solve_rejects_undecodable_images() {
        let app = router(test_state(true), 1 << 20);
        let request = multipart_request("/solve", &[("image", Some("junk.png"), b"not an image")]);
        let response = app.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        assert_eq!(body_json(response).await["error"]["code"], "bad_image");
    }

    #[tokio::test]
    async fn solve_validates_numeric_fields() {
        let app = router(test_state(true), 1 << 20);
        let request = multipart_request("/solve", &[("fov_hint", None, b"garbage")]);
        let response = app.clone().oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        assert_eq!(body_json(response).await["error"]["code"], "bad_number");

        let request = multipart_request("/solve", &[("fov_hint", None, b"200")]);
        let response = app.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        assert_eq!(body_json(response).await["error"]["code"], "out_of_range");
    }

    #[tokio::test]
    async fn solve_rejects_unknown_overlay_modes() {
        let app = router(test_state(true), 1 << 20);
        let request = multipart_request(
            "/solve?overlay=jpeg",
            &[("image", Some("x.png"), &tiny_png())],
        );
        let response = app.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        assert_eq!(body_json(response).await["error"]["code"], "bad_overlay");
    }

    #[tokio::test]
    async fn oversized_uploads_yield_413() {
        let app = router(test_state(true), 1024);
        let request = multipart_request("/solve", &[("image", Some("big.png"), &[0u8; 4096])]);
        let response = app.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::PAYLOAD_TOO_LARGE);
    }

    #[tokio::test]
    async fn telemetry_logs_rejected_requests_with_parsed_hints() {
        let (path, sink) = test_telemetry("rejected");
        let mut state = test_state(true);
        state.telemetry = Some(sink);
        let app = router(state, 1 << 20);
        // Hints parse fine, but the image field is missing.
        let request = multipart_request("/solve", &[("fov_hint", None, b"22.0")]);
        let response = app.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);

        let record = read_single_record(&path);
        assert_eq!(record["outcome"], "rejected");
        assert_eq!(record["reject_code"], "missing_image");
        assert_eq!(record["http_status"], 400);
        assert_eq!(record["hints"]["fov_hint_deg"], 22.0);
        assert_eq!(record["hints"]["overlay"], "none");
        assert!(record.get("source").is_none(), "no image field was seen");
        let _ = std::fs::remove_file(&path);
    }

    #[tokio::test]
    async fn telemetry_captures_image_facts_up_to_a_busy_reject() {
        let (path, sink) = test_telemetry("busy");
        let mut state = test_state(true); // empty pool → checkout must time out
        state.telemetry = Some(sink);
        let app = router(state, 1 << 20);
        let request = multipart_request(
            "/solve?overlay=png",
            &[("image", Some("night.png"), &tiny_png())],
        );
        let response = app.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);

        let record = read_single_record(&path);
        assert_eq!(record["outcome"], "rejected");
        assert_eq!(record["reject_code"], "busy");
        assert_eq!(record["http_status"], 503);
        assert_eq!(record["source"], "night");
        assert_eq!(record["image"]["width"], 8);
        assert_eq!(record["image"]["height"], 8);
        assert_eq!(record["exif"]["present"], false);
        assert_eq!(record["hints"]["overlay"], "png");
        // The 200 ms queue timeout dominates the handler wall time.
        assert!(record["timing"]["wall_ms"].as_u64().unwrap() >= 150);
        let _ = std::fs::remove_file(&path);
    }

    #[tokio::test]
    async fn solve_answers_busy_when_no_engine_frees_up() {
        // ready=true but the pool is empty: checkout can never succeed, so the
        // queue timeout (200 ms in tests) must surface as 503 "busy".
        let app = router(test_state(true), 1 << 20);
        let request = multipart_request("/solve", &[("image", Some("x.png"), &tiny_png())]);
        let response = app.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(body_json(response).await["error"]["code"], "busy");
    }
}
