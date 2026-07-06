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

use tetra3::{GenerateDatabaseConfig, SolverDatabase, Star as T3Star};

use crate::catalog::Catalog;

/// Faintest magnitude included in every generated database.
pub const DB_MAG_LIMIT: f32 = 6.5;
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
    /// Broad multiscale database (10–70°, mag 6.5, patterns/field 50).
    Bootstrap,
    /// Narrow multiscale band tuned to a known FOV (mag 6.5, patterns/field 1200).
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

    /// Cache file name: `tetra3-{params}-mag65-{version}.bin`.
    fn cache_file_name(&self) -> String {
        format!("tetra3-{}-mag65-{DB_VERSION_TAG}.bin", self.params_tag())
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
            star_max_magnitude: Some(DB_MAG_LIMIT),
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
    /// from `catalog` (stars up to magnitude 6.5) and written to `cache_dir`.
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
        let cache_str = path_to_str(&cache_path)?;

        let db = if cache_path.exists() {
            progress(EngineProgress::Loading {
                kind,
                path: cache_path.clone(),
            });
            SolverDatabase::load_from_file(cache_str).map_err(|source| EngineError::Load {
                path: cache_path.display().to_string(),
                source,
            })?
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
            db.save_to_file(cache_str)
                .map_err(|source| EngineError::Save {
                    path: cache_path.display().to_string(),
                    source,
                })?;
            db
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

/// Convert catalog stars (mag ≤ 6.5, finite) into tetra3 `Star`s.
fn build_star_list(catalog: &Catalog) -> Vec<T3Star> {
    catalog
        .stars()
        .iter()
        .filter(|s| s.mag.is_finite() && s.mag <= DB_MAG_LIMIT)
        .map(|s| T3Star {
            id: i64::from(s.id),
            ra_rad: s.ra_deg.to_radians() as f32,
            dec_rad: s.dec_deg.to_radians() as f32,
            mag: s.mag,
        })
        .collect()
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
}
