//! Solver database lifecycle: build or load tetra3 pattern databases from the
//! HYG catalog, caching the generated `.bin` files on disk.
//!
//! Two database kinds are used by the solve pipeline:
//! - [`DbKind::Bootstrap`]: a broad multiscale database (10–70°) used for blind
//!   lost-in-space solving. Cheap to build (~a few seconds, ~24 MB).
//! - [`DbKind::DenseBand`]: a narrow multiscale band with a much higher pattern
//!   density, generated on demand when a frame's FOV is known but the bootstrap
//!   database is too sparse to match. Expensive to build the first time.

use std::path::{Path, PathBuf};
use std::sync::Mutex;

use tetra3::{GenerateDatabaseConfig, SolverDatabase, Star as T3Star};

use crate::catalog::Catalog;

/// Process-wide single-flight lock for database *generation*. Engines pooled
/// by the HTTP service share one disk cache; without this, N engines missing
/// the same cache file would generate the same multi-hundred-MB database N
/// times in parallel (generation is already rayon-parallel inside, so
/// serializing concurrent builds also avoids oversubscribing the CPU).
/// Loading an existing cache file stays lock-free.
static GENERATION_LOCK: Mutex<()> = Mutex::new(());

/// Faintest magnitude included in every generated database (default).
pub const DB_MAG_LIMIT: f32 = 6.5;
/// Env override for [`DB_MAG_LIMIT`] — the B4 sweep knob. The effective value
/// lands in the cache file names (`mag65`/`mag70`/…), so databases built at
/// different depths coexist in one cache directory instead of clobbering the
/// default set.
const MAG_LIMIT_ENV: &str = "STARGLYPH_DB_MAG_LIMIT";
/// Patterns per lattice field for the dense band.
const DENSE_PATTERNS_PER_FIELD: u32 = 1200;
/// Verification (and pattern-pool) stars per FOV for the dense band. Raising this
/// from 30 to 45 pulls the fainter stars the sparse Cassiopeia fields need into
/// the asterism pool — the decisive lever for indexing their patterns.
const DENSE_VERIFICATION_STARS_PER_FOV: u32 = 45;
/// Edge-ratio match tolerance for the dense band (matches the spike's dense DB).
const DENSE_PATTERN_MAX_ERROR: f32 = 0.006;
/// Version tag embedded in cache file names (tetra3 on-disk format version).
const DB_VERSION_TAG: &str = "v0.8.0";
/// Proper-motion default year; catalog positions are treated as already at epoch.
const DEFAULT_PM_YEAR: f64 = 2000.0;

/// Which pattern database to build or load.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DbKind {
    /// Broad multiscale database (10–70°, mag ≤ [`db_mag_limit`], patterns/field 50).
    Bootstrap,
    /// Narrow multiscale band tuned to a known FOV (mag ≤ [`db_mag_limit`],
    /// patterns/field 1200).
    DenseBand { min_fov_deg: f32, max_fov_deg: f32 },
}

impl DbKind {
    /// Dense band covering a FOV `center` (degrees): spans `[0.75·c, 1.35·c]`
    /// with the center rounded to a whole degree. Rounding matches the cache
    /// [`key`](DbKind::key) granularity, so nearby FOV hints (e.g. a blind 22.0
    /// and a batch median 22.32) resolve to a single shared band rather than two
    /// near-identical multi-hundred-MB databases. The band top is capped at 90°:
    /// past that the rectilinear projection tetra3 patterns assume degenerates
    /// (tan → ∞ at the hemisphere edge), and no realistic non-fisheye camera
    /// reaches it.
    #[must_use]
    pub fn dense_for_center(center_deg: f32) -> DbKind {
        let c = center_deg.round();
        let max_fov_deg = ((c * 1.35).ceil()).min(90.0);
        DbKind::DenseBand {
            min_fov_deg: (c * 0.75).floor().min(max_fov_deg - 1.0),
            max_fov_deg,
        }
    }

    /// Stable, filesystem-safe parameter fragment for the cache file name.
    fn params_tag(&self) -> String {
        match *self {
            DbKind::Bootstrap => "bootstrap-10-70".to_string(),
            DbKind::DenseBand {
                min_fov_deg,
                max_fov_deg,
            } => format!(
                "dense-{}-{}",
                min_fov_deg.round() as i32,
                max_fov_deg.round() as i32
            ),
        }
    }

    /// Cache file name: `tetra3-{params}-{mag}-{version}.bin` (e.g. `mag65`).
    fn cache_file_name(&self) -> String {
        format!(
            "tetra3-{}-{}-{DB_VERSION_TAG}.bin",
            self.params_tag(),
            mag_tag(db_mag_limit())
        )
    }

    /// tetra3 generation config for this kind.
    fn generate_config(&self) -> GenerateDatabaseConfig {
        let (min_fov, max_fov, ppf, vpf, pmax_err) = match *self {
            DbKind::Bootstrap => (10.0, 70.0, 50, 30, 0.005),
            DbKind::DenseBand {
                min_fov_deg,
                max_fov_deg,
            } => (
                min_fov_deg,
                max_fov_deg,
                DENSE_PATTERNS_PER_FIELD,
                DENSE_VERIFICATION_STARS_PER_FOV,
                DENSE_PATTERN_MAX_ERROR,
            ),
        };
        GenerateDatabaseConfig {
            max_fov_deg: max_fov,
            min_fov_deg: Some(min_fov),
            star_max_magnitude: Some(db_mag_limit()),
            pattern_max_error: pmax_err,
            lattice_field_oversampling: 100,
            patterns_per_lattice_field: ppf,
            verification_stars_per_fov: vpf,
            multiscale_step: 1.5,
            epoch_proper_motion_year: None,
            catalog_nside: 16,
        }
    }

    /// Equality key used to reuse an already-loaded database of the same kind.
    fn key(&self) -> (u8, i32, i32) {
        match *self {
            DbKind::Bootstrap => (0, 0, 0),
            DbKind::DenseBand {
                min_fov_deg,
                max_fov_deg,
            } => (1, min_fov_deg.round() as i32, max_fov_deg.round() as i32),
        }
    }
}

/// Progress events emitted while a database is prepared.
#[derive(Debug, Clone)]
pub enum EngineProgress {
    /// A cached database file was found and is being loaded.
    Loading { kind: DbKind, path: PathBuf },
    /// No cache present; generating from the catalog (may be slow).
    Generating { kind: DbKind, star_count: usize },
    /// The freshly generated database is being written to the cache.
    Saving { kind: DbKind, path: PathBuf },
    /// The database is ready for solving.
    Ready {
        kind: DbKind,
        catalog_stars: usize,
        patterns: u32,
        bytes: u64,
    },
}

/// Errors from database preparation.
#[derive(Debug, thiserror::Error)]
pub enum EngineError {
    #[error("failed to create cache directory '{path}': {source}")]
    CreateCacheDir {
        path: String,
        source: std::io::Error,
    },
    #[error("failed to load solver database '{path}': {source}")]
    Load { path: String, source: tetra3::Error },
    #[error("failed to save solver database '{path}': {source}")]
    Save { path: String, source: tetra3::Error },
    #[error("failed to publish solver database '{path}': {source}")]
    Publish {
        path: String,
        source: std::io::Error,
    },
    #[error("cache path '{path}' is not valid UTF-8 (tetra3 requires a string path)")]
    NonUtf8Path { path: String },
}

/// Holds one or more loaded tetra3 solver databases, keyed by [`DbKind`].
///
/// Reusing a single `Engine` across many frames avoids re-deserializing the
/// (potentially large) `.bin` files on every solve.
#[derive(Debug, Default)]
pub struct Engine {
    dbs: Vec<(DbKind, SolverDatabase)>,
}

impl Engine {
    /// Build or load the database for `kind`, returning an `Engine` that holds it.
    ///
    /// If a cache file exists it is loaded; otherwise the database is generated
    /// from `catalog` (stars up to [`db_mag_limit`]) and written to `cache_dir`.
    pub fn ensure(
        catalog: &Catalog,
        kind: DbKind,
        cache_dir: &Path,
        progress: &mut dyn FnMut(EngineProgress),
    ) -> Result<Engine, EngineError> {
        let mut engine = Engine::default();
        engine.ensure_kind(catalog, kind, cache_dir, progress)?;
        Ok(engine)
    }

    /// Load or build `kind` into this engine if not already present, then return it.
    pub fn ensure_kind(
        &mut self,
        catalog: &Catalog,
        kind: DbKind,
        cache_dir: &Path,
        progress: &mut dyn FnMut(EngineProgress),
    ) -> Result<&SolverDatabase, EngineError> {
        if let Some(pos) = self.dbs.iter().position(|(k, _)| k.key() == kind.key()) {
            return Ok(&self.dbs[pos].1);
        }

        std::fs::create_dir_all(cache_dir).map_err(|source| EngineError::CreateCacheDir {
            path: cache_dir.display().to_string(),
            source,
        })?;
        let cache_path = cache_dir.join(kind.cache_file_name());

        let db = if cache_path.exists() {
            load_db(&cache_path, kind, progress)?
        } else {
            // Single-flight: only one thread generates at a time; the file is
            // published atomically (tmp + rename), so the lock-free `exists`
            // fast path above never observes a partially written database.
            let _flight = GENERATION_LOCK
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            if cache_path.exists() {
                // Another thread built it while we waited for the lock.
                load_db(&cache_path, kind, progress)?
            } else {
                let stars = build_star_list(catalog);
                progress(EngineProgress::Generating {
                    kind,
                    star_count: stars.len(),
                });
                let cfg = kind.generate_config();
                let db = SolverDatabase::generate_from_star_list(stars, &cfg, DEFAULT_PM_YEAR);
                progress(EngineProgress::Saving {
                    kind,
                    path: cache_path.clone(),
                });
                let tmp_path = cache_dir.join(format!("{}.tmp", kind.cache_file_name()));
                let tmp_str = path_to_str(&tmp_path)?;
                if let Err(source) = db.save_to_file(tmp_str) {
                    let _ = std::fs::remove_file(&tmp_path);
                    return Err(EngineError::Save {
                        path: tmp_path.display().to_string(),
                        source,
                    });
                }
                std::fs::rename(&tmp_path, &cache_path).map_err(|source| {
                    let _ = std::fs::remove_file(&tmp_path);
                    EngineError::Publish {
                        path: cache_path.display().to_string(),
                        source,
                    }
                })?;
                db
            }
        };

        let bytes = std::fs::metadata(&cache_path).map(|m| m.len()).unwrap_or(0);
        progress(EngineProgress::Ready {
            kind,
            catalog_stars: db.star_catalog.len(),
            patterns: db.props.num_patterns,
            bytes,
        });

        self.dbs.push((kind, db));
        Ok(&self.dbs.last().expect("just pushed").1)
    }

    /// Borrow an already-loaded database of the given kind, if present.
    #[must_use]
    pub fn get(&self, kind: DbKind) -> Option<&SolverDatabase> {
        self.dbs
            .iter()
            .find(|(k, _)| k.key() == kind.key())
            .map(|(_, db)| db)
    }
}

/// Effective database depth: [`MAG_LIMIT_ENV`] (validated) or [`DB_MAG_LIMIT`].
/// Read once per process — sweeps run one eval process per magnitude.
pub fn db_mag_limit() -> f32 {
    static LIMIT: std::sync::OnceLock<f32> = std::sync::OnceLock::new();
    *LIMIT.get_or_init(|| match std::env::var(MAG_LIMIT_ENV) {
        Ok(raw) => parse_mag_limit(&raw).unwrap_or_else(|| {
            eprintln!(
                "[engine] ignoring invalid {MAG_LIMIT_ENV}='{raw}' \
                 (want a number in [5.0, 9.0]); using {DB_MAG_LIMIT}"
            );
            DB_MAG_LIMIT
        }),
        Err(_) => DB_MAG_LIMIT,
    })
}

/// Parse and validate a magnitude limit. Below 5 the bootstrap range starves;
/// past 9 HYG completeness drops off and databases balloon for no gain.
fn parse_mag_limit(raw: &str) -> Option<f32> {
    let mag: f32 = raw.trim().parse().ok()?;
    (5.0..=9.0).contains(&mag).then_some(mag)
}

/// `6.5 → "mag65"`: tenth-of-magnitude resolution keeps names filesystem-safe.
fn mag_tag(mag: f32) -> String {
    format!("mag{}", (mag * 10.0).round() as i32)
}

/// Convert catalog stars (mag ≤ [`db_mag_limit`], finite) into tetra3 `Star`s.
fn build_star_list(catalog: &Catalog) -> Vec<T3Star> {
    let mag_limit = db_mag_limit();
    catalog
        .stars()
        .iter()
        .filter(|s| s.mag.is_finite() && s.mag <= mag_limit)
        .map(|s| T3Star {
            id: i64::from(s.id),
            ra_rad: s.ra_deg.to_radians() as f32,
            dec_rad: s.dec_deg.to_radians() as f32,
            mag: s.mag,
        })
        .collect()
}

/// Load a cached database file, emitting the `Loading` progress event.
fn load_db(
    cache_path: &Path,
    kind: DbKind,
    progress: &mut dyn FnMut(EngineProgress),
) -> Result<SolverDatabase, EngineError> {
    progress(EngineProgress::Loading {
        kind,
        path: cache_path.to_path_buf(),
    });
    SolverDatabase::load_from_file(path_to_str(cache_path)?).map_err(|source| EngineError::Load {
        path: cache_path.display().to_string(),
        source,
    })
}

fn path_to_str(path: &Path) -> Result<&str, EngineError> {
    path.to_str().ok_or_else(|| EngineError::NonUtf8Path {
        path: path.display().to_string(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cache_file_names_are_stable_and_distinct() {
        assert_eq!(
            DbKind::Bootstrap.cache_file_name(),
            "tetra3-bootstrap-10-70-mag65-v0.8.0.bin"
        );
        let dense = DbKind::DenseBand {
            min_fov_deg: 16.65,
            max_fov_deg: 29.97,
        };
        assert_eq!(
            dense.cache_file_name(),
            "tetra3-dense-17-30-mag65-v0.8.0.bin"
        );
    }

    #[test]
    fn dense_band_key_quantizes_nearby_fovs_together() {
        let a = DbKind::DenseBand {
            min_fov_deg: 16.6,
            max_fov_deg: 30.1,
        };
        let b = DbKind::DenseBand {
            min_fov_deg: 16.55,
            max_fov_deg: 29.95,
        };
        assert_eq!(a.key(), b.key());
        assert_ne!(a.key(), DbKind::Bootstrap.key());
    }

    #[test]
    fn dense_band_caps_at_90_degrees() {
        let DbKind::DenseBand {
            min_fov_deg,
            max_fov_deg,
        } = DbKind::dense_for_center(95.0)
        else {
            panic!("expected dense band");
        };
        assert_eq!(max_fov_deg, 90.0);
        assert!(min_fov_deg < max_fov_deg);

        // Phone wide-angle default band stays under the cap.
        let DbKind::DenseBand {
            min_fov_deg,
            max_fov_deg,
        } = DbKind::dense_for_center(65.0)
        else {
            panic!("expected dense band");
        };
        assert_eq!((min_fov_deg, max_fov_deg), (48.0, 88.0));
    }

    #[test]
    fn generate_config_matches_kind() {
        let boot = DbKind::Bootstrap.generate_config();
        assert_eq!(boot.patterns_per_lattice_field, 50);
        assert_eq!(boot.min_fov_deg, Some(10.0));
        assert_eq!(boot.max_fov_deg, 70.0);
        assert_eq!(boot.star_max_magnitude, Some(6.5));

        let dense = DbKind::DenseBand {
            min_fov_deg: 16.0,
            max_fov_deg: 30.0,
        }
        .generate_config();
        assert_eq!(dense.patterns_per_lattice_field, 1200);
        assert_eq!(dense.min_fov_deg, Some(16.0));
    }

    #[test]
    fn mag_tags_are_filesystem_safe_and_distinct() {
        assert_eq!(mag_tag(6.5), "mag65");
        assert_eq!(mag_tag(7.0), "mag70");
        assert_eq!(mag_tag(7.5), "mag75");
    }

    #[test]
    fn mag_limit_parsing_validates_the_sane_band() {
        assert_eq!(parse_mag_limit("7.0"), Some(7.0));
        assert_eq!(parse_mag_limit(" 6.5 "), Some(6.5));
        assert_eq!(parse_mag_limit("4.9"), None, "starves the bootstrap range");
        assert_eq!(parse_mag_limit("9.5"), None, "beyond HYG completeness");
        assert_eq!(parse_mag_limit("abc"), None);
        assert_eq!(parse_mag_limit(""), None);
    }
}
