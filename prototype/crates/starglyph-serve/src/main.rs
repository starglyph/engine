//! starglyph-serve: headless HTTP plate-solve service (Stage 0 · C1).
//!
//! Thin HTTP front over `solve_frame_with_engine`. The bootstrap pattern
//! database is warmed into a pool of engines at startup and reused across
//! requests — a solve never rebuilds databases the disk cache already holds.
//! See `docs/serve.md` for the API, configuration and concurrency model.

use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use anyhow::{ensure, Context, Result};
use clap::Parser;
use starglyph_core::catalog::Catalog;
use starglyph_core::constellations::ConstellationSet;
use starglyph_core::engine::{DbKind, Engine, EngineProgress};

mod pool;
mod server;
mod telemetry;

use pool::EnginePool;
use server::AppState;
use telemetry::TelemetrySink;

/// Dense-band centers below this cannot be populated by the mag ≤ 6.5
/// catalog (mirrors the solver's own skip threshold).
const MIN_PREWARM_CENTER_DEG: f32 = 8.0;
const MAX_PREWARM_CENTER_DEG: f32 = 95.0;

#[derive(Debug, Parser)]
#[command(
    name = "starglyph-serve",
    about = "Headless HTTP plate solver: POST a sky photo to /solve, get a SolveReport JSON (+ overlay PNG)"
)]
struct Config {
    /// Socket address to listen on.
    #[arg(long, default_value = "127.0.0.1:8080", env = "STARGLYPH_SERVE_ADDR")]
    addr: SocketAddr,
    /// HYG catalog CSV (defaults to the repo copy).
    #[arg(long, env = "STARGLYPH_SERVE_CATALOG")]
    catalog: Option<PathBuf>,
    /// Constellation line geometry JSON (defaults to the repo copy).
    #[arg(long, env = "STARGLYPH_SERVE_LINES")]
    lines: Option<PathBuf>,
    /// Constellation names JSON (defaults to the repo copy).
    #[arg(long, env = "STARGLYPH_SERVE_NAMES")]
    names: Option<PathBuf>,
    /// Directory for cached tetra3 databases; persists across restarts.
    #[arg(long, env = "STARGLYPH_SERVE_CACHE_DIR")]
    cache_dir: Option<PathBuf>,
    /// Warmed engines to pool == max concurrent solves.
    #[arg(long, default_value_t = 2, env = "STARGLYPH_SERVE_POOL_SIZE")]
    pool_size: usize,
    /// Maximum upload size in MiB.
    #[arg(long, default_value_t = 32, env = "STARGLYPH_SERVE_MAX_BODY_MIB")]
    max_body_mib: usize,
    /// Per-request solve timeout in seconds (504 past it).
    #[arg(long, default_value_t = 120, env = "STARGLYPH_SERVE_SOLVE_TIMEOUT_S")]
    solve_timeout_s: u64,
    /// Max wait for a free engine in seconds (503 past it).
    #[arg(long, default_value_t = 30, env = "STARGLYPH_SERVE_QUEUE_TIMEOUT_S")]
    queue_timeout_s: u64,
    /// Dense-band centers (degrees, comma-separated) to pre-generate into the
    /// disk cache after startup; the default covers the blind band ladder.
    /// Empty string disables prewarming.
    #[arg(
        long,
        default_value = "22,40,65",
        env = "STARGLYPH_SERVE_PREWARM_DENSE"
    )]
    prewarm_dense: String,
    /// Append-only JSONL solve-telemetry log: one anonymous record per /solve
    /// request (no user identity). Defaults to artifacts/telemetry/ in the
    /// repo; an empty string disables telemetry.
    #[arg(long, env = "STARGLYPH_SERVE_TELEMETRY_LOG")]
    telemetry_log: Option<PathBuf>,
}

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn data_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../data")
}

/// Committed-first catalog candidates (mirrors the desktop resolver): the repo
/// ships `hyg_v42.csv.gz`; `hyg_v3.csv` is a local `make fetch-catalog`
/// artifact, so a fresh clone must not default to it.
const CATALOG_CANDIDATES: &[&str] = &[
    "hyg_v42.csv.gz",
    "hyg_v42.csv",
    "hyg_v3.csv.gz",
    "hyg_v3.csv",
];

fn default_catalog_path() -> Result<PathBuf> {
    let dir = data_root().join("catalogs");
    CATALOG_CANDIDATES
        .iter()
        .map(|name| dir.join(name))
        .find(|path| path.exists())
        .with_context(|| {
            format!(
                "no HYG catalog found in '{}' (tried {}); pass --catalog",
                dir.display(),
                CATALOG_CANDIDATES.join(", ")
            )
        })
}

#[tokio::main]
async fn main() -> Result<()> {
    let cfg = Config::parse();
    ensure!(cfg.pool_size >= 1, "--pool-size must be at least 1");
    ensure!(cfg.max_body_mib >= 1, "--max-body-mib must be at least 1");
    ensure!(
        cfg.solve_timeout_s >= 1 && cfg.queue_timeout_s >= 1,
        "timeouts must be at least 1 second"
    );
    let prewarm = parse_prewarm(&cfg.prewarm_dense)?;

    let catalog_path = match cfg.catalog {
        Some(path) => path,
        None => default_catalog_path()?,
    };
    let lines_path = cfg
        .lines
        .unwrap_or_else(|| data_root().join("celestial/constellations.lines.json"));
    let names_path = cfg
        .names
        .unwrap_or_else(|| data_root().join("celestial/constellations.json"));
    let cache_dir = cfg
        .cache_dir
        .unwrap_or_else(|| workspace_root().join("artifacts/cache"));
    let telemetry_path = match cfg.telemetry_log {
        Some(path) if path.as_os_str().is_empty() => None,
        Some(path) => Some(path),
        None => Some(workspace_root().join("artifacts/telemetry/solve-log.jsonl")),
    };
    let telemetry = match telemetry_path {
        Some(path) => {
            let sink = TelemetrySink::open(&path)
                .with_context(|| format!("failed to open telemetry log '{}'", path.display()))?;
            eprintln!("[serve] telemetry → '{}'", path.display());
            Some(Arc::new(sink))
        }
        None => {
            eprintln!("[serve] telemetry disabled");
            None
        }
    };

    eprintln!("[serve] loading catalog '{}'", catalog_path.display());
    let (catalog, cons) = tokio::task::spawn_blocking(move || -> Result<_> {
        let catalog = Catalog::load(&catalog_path)
            .with_context(|| format!("failed to load catalog '{}'", catalog_path.display()))?;
        let cons = ConstellationSet::load(&lines_path, &names_path).with_context(|| {
            format!(
                "failed to load constellations from '{}' and '{}'",
                lines_path.display(),
                names_path.display()
            )
        })?;
        Ok((Arc::new(catalog), Arc::new(cons)))
    })
    .await
    .context("catalog load task died")??;

    let state = AppState {
        catalog: Arc::clone(&catalog),
        cons,
        pool: EnginePool::new(),
        ready: Arc::new(AtomicBool::new(false)),
        cache_dir: cache_dir.clone(),
        queue_timeout: Duration::from_secs(cfg.queue_timeout_s),
        solve_timeout: Duration::from_secs(cfg.solve_timeout_s),
        telemetry,
    };

    // Warm in the background so the listener is up immediately: /readyz turns
    // green once every pooled engine holds the bootstrap database (first ever
    // run generates it — about a minute; afterwards it loads from the cache in
    // seconds). Dense-band prewarm then continues into the disk cache only;
    // pooled engines pick bands up lazily on demand.
    {
        let pool = state.pool.clone();
        let ready = Arc::clone(&state.ready);
        let catalog = Arc::clone(&catalog);
        let cache_dir = cache_dir.clone();
        let pool_size = cfg.pool_size;
        tokio::task::spawn_blocking(move || {
            warmup(&catalog, pool_size, &cache_dir, &pool, &ready, &prewarm);
        });
    }

    let listener = tokio::net::TcpListener::bind(cfg.addr)
        .await
        .with_context(|| format!("failed to bind {}", cfg.addr))?;
    eprintln!(
        "[serve] listening on {} (pool={}, max-body={} MiB, solve-timeout={}s, cache '{}')",
        listener.local_addr()?,
        cfg.pool_size,
        cfg.max_body_mib,
        cfg.solve_timeout_s,
        cache_dir.display(),
    );

    axum::serve(
        listener,
        server::router(state, cfg.max_body_mib * 1024 * 1024),
    )
    .with_graceful_shutdown(shutdown_signal())
    .await
    .context("server error")?;
    Ok(())
}

/// Bootstrap every pooled engine, flip readiness, then prewarm dense bands.
/// A failed bootstrap is fatal: the service could never become ready.
fn warmup(
    catalog: &Catalog,
    pool_size: usize,
    cache_dir: &Path,
    pool: &EnginePool,
    ready: &AtomicBool,
    prewarm: &[f32],
) {
    for slot in 1..=pool_size {
        match Engine::ensure(
            catalog,
            DbKind::Bootstrap,
            cache_dir,
            &mut log_engine_progress,
        ) {
            Ok(engine) => {
                pool.install(engine);
                eprintln!("[serve] engine {slot}/{pool_size} warmed");
            }
            Err(e) => {
                eprintln!("[serve] FATAL: bootstrap warmup failed: {e}");
                std::process::exit(1);
            }
        }
    }
    ready.store(true, Ordering::Release);
    eprintln!("[serve] ready: {pool_size} engine(s) warmed");

    for &center in prewarm {
        // The throwaway engine frees the band's RAM right away; only the disk
        // cache matters here.
        if let Err(e) = Engine::ensure(
            catalog,
            DbKind::dense_for_center(center),
            cache_dir,
            &mut log_engine_progress,
        ) {
            eprintln!("[serve] dense prewarm {center}° failed: {e}");
        }
    }
    if !prewarm.is_empty() {
        eprintln!("[serve] dense prewarm complete");
    }
}

fn log_engine_progress(progress: EngineProgress) {
    match progress {
        EngineProgress::Loading { kind, path } => {
            eprintln!("[engine] loading {kind:?} from '{}'", path.display());
        }
        EngineProgress::Generating { kind, star_count } => {
            eprintln!(
                "[engine] generating {kind:?} from {star_count} stars (no cache; may take a while)"
            );
        }
        EngineProgress::Saving { kind, path } => {
            eprintln!("[engine] saving {kind:?} to '{}'", path.display());
        }
        EngineProgress::Ready {
            kind,
            catalog_stars,
            patterns,
            bytes,
        } => {
            eprintln!(
                "[engine] {kind:?} ready: {catalog_stars} stars, {patterns} patterns, {} MiB",
                bytes / (1024 * 1024),
            );
        }
    }
}

fn parse_prewarm(list: &str) -> Result<Vec<f32>> {
    list.split(',')
        .map(str::trim)
        .filter(|token| !token.is_empty())
        .map(|token| {
            let center: f32 = token
                .parse()
                .with_context(|| format!("invalid dense prewarm center '{token}'"))?;
            ensure!(
                (MIN_PREWARM_CENTER_DEG..=MAX_PREWARM_CENTER_DEG).contains(&center),
                "dense prewarm center {center}° out of range [{MIN_PREWARM_CENTER_DEG}, {MAX_PREWARM_CENTER_DEG}]"
            );
            Ok(center)
        })
        .collect()
}

async fn shutdown_signal() {
    let ctrl_c = async {
        tokio::signal::ctrl_c()
            .await
            .expect("failed to install ctrl-c handler");
    };
    #[cfg(unix)]
    let terminate = async {
        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("failed to install SIGTERM handler")
            .recv()
            .await;
    };
    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        () = ctrl_c => {},
        () = terminate => {},
    }
    eprintln!("[serve] shutdown signal received");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prewarm_list_parses_and_validates() {
        assert_eq!(parse_prewarm("22,40,65").unwrap(), vec![22.0, 40.0, 65.0]);
        assert_eq!(parse_prewarm(" 22 , 40 ").unwrap(), vec![22.0, 40.0]);
        assert!(parse_prewarm("").unwrap().is_empty());
        assert!(parse_prewarm("  ").unwrap().is_empty());
        assert!(parse_prewarm("abc").is_err());
        assert!(parse_prewarm("5").is_err(), "below the catalog floor");
        assert!(parse_prewarm("120").is_err(), "beyond rectilinear range");
    }
}
