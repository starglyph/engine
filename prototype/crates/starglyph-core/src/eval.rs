//! Evaluation harness: compare a solved camera pose against astrometry.net WCS
//! ground truth.
//!
//! This module parses astrometry.net-style `.wcs.json` calibration sidecars, converts
//! them into a ground-truth camera pose expressed in *this repo's* camera convention
//! (see [`crate::geom`]), and computes pose-error metrics. It is a pure library; the
//! `starglyph eval` subcommand is wired on top of it in a later slice.
//!
//! # The subtle bit: astrometry `orientation` + `parity` -> our `roll`
//!
//! ## astrometry.net's definition (the thing we invert)
//!
//! nova.astrometry.net computes the reported calibration from the TAN-WCS `CD` matrix.
//! The `CD` matrix maps pixel offsets `(dx, dy)` (from the reference pixel) to the
//! intermediate world coordinates `(xi, eta)` in **degrees**, where `xi` points **East**
//! (increasing RA on the tangent plane) and `eta` points **North** (increasing Dec):
//!
//! ```text
//! [ xi  ]   [ cd11  cd12 ] [ dx ]
//! [ eta ] = [ cd21  cd22 ] [ dy ]
//! ```
//!
//! Its exact formulas are (from astrometry.net `net/wcs.py`, which is what produced our
//! sidecars; identical to the C `sip.c`/`tan_get_orientation`):
//!
//! ```text
//! det      = cd11*cd22 - cd12*cd21
//! parity   = +1 if det >= 0 else -1
//! pixscale = 3600 * sqrt(|det|)                 // arcsec/pixel
//! T        = parity*cd11 + cd22
//! A        = parity*cd21 - cd12
//! orient   = -degrees(atan2(A, T))              // degrees
//! ```
//!
//! Geometrically, for a pure rotation+scale `CD` this makes `orientation` the position
//! angle **East of North** of the `CD` frame's `+y` pixel axis (positive rotating the
//! up-vector from North toward East), and `parity` the handedness of the pixel->sky map
//! (a direct photograph vs a mirror-flipped one).
//!
//! ## Inverting it
//!
//! Given `(pixscale, orientation = theta, parity = p)` we reconstruct a canonical
//! similarity/reflection `CD` (dropping the positive scale, which does not affect
//! directions):
//!
//! ```text
//! CD = [ p*cos(theta)   sin(theta) ]
//!      [ -p*sin(theta)  cos(theta) ]
//! ```
//!
//! This is the unique similarity (for `p = +1`) / single-axis reflection (for `p = -1`)
//! that reproduces astrometry's `get_orientation`, `get_parity` and `get_pixscale`
//! exactly (checked in unit tests). From it we read the world directions of image `+x`
//! and image `+y` (`dir_x = cd11*East + cd21*North`, `dir_y = cd12*East + cd22*North`)
//! and assemble our camera basis, then extract the roll with [`crate::geom`]'s
//! `rotation_to_pose` machinery.
//!
//! ## Our camera convention (from [`crate::geom`])
//!
//! * World = equatorial J2000 unit sphere.
//! * `pose_to_rotation(ra, dec, roll)` builds rows `[right; up; forward]`; pre-roll
//!   `right = -East` (West) and `up = +North`.
//! * `project`: image `+x = camera right`, image `+y = -camera up` (image `+y` points
//!   **DOWN**).
//!
//! Consequences, both derived (not folklore) and locked by tests:
//! * A non-mirrored camera always yields `det(CD) > 0`, i.e. **parity `+1`** in the
//!   displayed (top-left origin, `+y` down) pixel frame. All "normal photo" sidecars in
//!   `data/samples/sky-samples/ground-truth/` report `parity = +1` (only the one
//!   processed/mirrored frame reports `-1`), which is what pins astrometry's reported
//!   frame to our `+y`-down frame.
//! * Camera `roll` equals the position angle East-of-North of the camera **up** vector;
//!   astrometry's `orientation` is the PA of image `+y` = camera **down**. Hence, under
//!   this frame identification, `roll = orientation + 180` (mod 360). This `+180` is a
//!   geometric consequence and is *not* the tunable constant below - it emerges from the
//!   pipeline via `rotation_to_pose`.
//!
//! ## The reported frame depends on how nova ingested the file -> [`WcsRowConvention`]
//!
//! Whether astrometry's *reported* `orientation` lives in our `+y`-down frame or in a
//! vertically flipped one turns out to depend on the **source container** the frame was
//! uploaded as. nova.astrometry.net converts non-FITS uploads internally, and the row
//! order of that conversion differs between the pipeline used for 16-bit TIFFs and the
//! one used for consumer JPEGs (the committed sidecars have no TIFF `Orientation` tags,
//! so the flip is nova's, not the files'). A vertical flip maps the position angle
//! `theta -> 180 - theta`, i.e. `roll = orientation + 180` becomes `roll = -orientation`.
//!
//! **EMPIRICALLY CALIBRATED (2026-07-05/06)** against inlier-certified poses of the live
//! blind solver (a pose backed by 16-34 verified star matches at sub-pixel rms *is* the
//! image-frame truth up to arcminutes):
//!
//! | frame (source)                  | astrometry `orientation` | solver roll | rule |
//! |---------------------------------|--------------------------|-------------|------|
//! | `tetra3_alt60` (16-bit TIFF)    | −148.444°                | 30.954°     | `θ+180` (err 0.60°) |
//! | `tetra3_alt40` (16-bit TIFF)    | −152.308°                | ≈ `θ+180`   | `θ+180` |
//! | `wm_constellation_orion` (JPEG) | 37.985°                  | −37.806°    | `−θ` (err 0.18°) |
//! | `flickr_orion_rahn` (JPEG)      | 21.724°                  | −22.217°    | `−θ` (err 0.49°) |
//! | `flickr_cygnus_fermion` (JPEG)  | 68.443°                  | −71.250°    | `−θ` (err 2.81°) |
//!
//! For the TIFF frames the flipped rule would produce ~180° errors and vice versa — the
//! two families are mutually exclusive and internally consistent. The per-convention
//! residual mapping is isolated in [`OrientationCalibration`]; [`WcsRowConvention`]
//! selects it from the ground-truth source container. Residual sub-3° differences are
//! genuine astrometry-vs-starglyph pose differences (SIP distortion vs our k1-only
//! model), recorded honestly by the harness rather than calibrated away. The unit tests
//! lock **self consistency** (extract-then-reconstruct round trips), the **definitional
//! direction** of `orientation` (East of North), parity handling, and the algebraic
//! `-θ` identity of the flipped branch.

use nalgebra::{Matrix3, RowVector3};
use serde::{Deserialize, Serialize};

use crate::contracts::{SolveFov, SolvePose};
use crate::geom::{angular_sep, radec_to_unit, rotation_to_pose};

/// Residual mapping from astrometry.net's reported `orientation` to the internal
/// orientation that our reconstruction consumes: `internal = sign * reported + offset`.
///
/// See the module docs; selected per ground-truth source by [`WcsRowConvention`].
#[derive(Debug, Clone, Copy)]
struct OrientationCalibration {
    sign: f64,
    offset_deg: f64,
}

/// Pixel-row convention of the astrometry solve that produced a WCS sidecar,
/// relative to our top-down (`+y` down) pixel frame. See the module docs for the
/// empirical calibration table.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WcsRowConvention {
    /// Rows reached astrometry in our top-down order (observed for the dataset's
    /// 16-bit TIFF frames): identity calibration, `roll = orientation + 180`.
    TopDown,
    /// Rows were vertically flipped during ingestion (observed for consumer JPEG
    /// frames): the reported angle describes the mirrored frame, and in our frame
    /// `roll = -orientation` (equivalently `sign = -1`, `offset = -180`).
    Flipped,
}

impl WcsRowConvention {
    /// Convention for a ground-truth sidecar, keyed by the source frame's container.
    ///
    /// 16-bit TIFFs went through nova's FITS-like path (no flip); everything else
    /// (JPEG in the committed dataset) went through the flipping image path. If a
    /// future GT frame violates this mapping, its roll error will sit near 180° —
    /// exactly the signal to extend this function, not to tune thresholds.
    #[must_use]
    pub fn for_image_path(path: &std::path::Path) -> WcsRowConvention {
        match path
            .extension()
            .and_then(|e| e.to_str())
            .map(|e| e.to_ascii_lowercase())
            .as_deref()
        {
            Some("tif" | "tiff") => WcsRowConvention::TopDown,
            _ => WcsRowConvention::Flipped,
        }
    }

    fn calibration(self) -> OrientationCalibration {
        match self {
            WcsRowConvention::TopDown => OrientationCalibration {
                sign: 1.0,
                offset_deg: 0.0,
            },
            WcsRowConvention::Flipped => OrientationCalibration {
                sign: -1.0,
                offset_deg: -180.0,
            },
        }
    }
}

/// Errors from parsing or validating a WCS calibration sidecar.
#[derive(Debug, thiserror::Error)]
pub enum EvalError {
    /// The input was not valid JSON, or the calibration object was missing required
    /// fields (e.g. `pixscale`) or had the wrong shape.
    #[error("failed to parse WCS JSON: {0}")]
    Json(#[from] serde_json::Error),
    /// The wrapper object had an explicit `null` `calibration`.
    #[error("WCS JSON has a null `calibration`")]
    MissingCalibration,
    /// A calibration field was present but out of range (e.g. non-positive pixscale).
    #[error("invalid calibration value: {0}")]
    InvalidValue(String),
}

/// A parsed astrometry.net calibration (fields renamed to explicit units).
///
/// Accepts both the nova API wrapper (`{ "calibration": { .. } }`) and a bare
/// calibration object via [`WcsCalibration::from_json_str`]. Unknown fields are ignored.
#[derive(Debug, Clone, Deserialize)]
pub struct WcsCalibration {
    /// Reference RA in degrees (JSON `ra`).
    #[serde(rename = "ra")]
    pub ra_deg: f64,
    /// Reference Dec in degrees (JSON `dec`).
    #[serde(rename = "dec")]
    pub dec_deg: f64,
    /// Plate scale in arcsec/pixel (JSON `pixscale`).
    #[serde(rename = "pixscale")]
    pub pixscale_arcsec: f64,
    /// Field orientation in degrees, E of N of image up (JSON `orientation`).
    #[serde(rename = "orientation")]
    pub orientation_deg: f64,
    /// Handedness of the pixel->sky map (JSON `parity`, `+1.0` / `-1.0`).
    pub parity: f64,
    /// Field width in arcsec, if present (JSON `width_arcsec`).
    #[serde(default)]
    pub width_arcsec: Option<f64>,
    /// Field height in arcsec, if present (JSON `height_arcsec`).
    #[serde(default)]
    pub height_arcsec: Option<f64>,
    /// Field radius in degrees, if present (JSON `radius`).
    #[serde(default, rename = "radius")]
    pub radius_deg: Option<f64>,
}

impl WcsCalibration {
    /// Parse a calibration from a JSON string, accepting either the nova API wrapper
    /// object (with a `calibration` key) or a bare calibration object.
    ///
    /// # Errors
    ///
    /// Returns [`EvalError`] if the input is not valid JSON, the `calibration` key is
    /// present but `null`, a required field (e.g. `pixscale`) is missing, or a value is
    /// out of range.
    pub fn from_json_str(s: &str) -> Result<Self, EvalError> {
        let value: serde_json::Value = serde_json::from_str(s)?;
        let calib_value = match value.get("calibration") {
            Some(serde_json::Value::Null) => return Err(EvalError::MissingCalibration),
            Some(inner) => inner.clone(),
            None => value,
        };
        let calib: WcsCalibration = serde_json::from_value(calib_value)?;
        calib.validate()?;
        Ok(calib)
    }

    fn validate(&self) -> Result<(), EvalError> {
        if !self.pixscale_arcsec.is_finite() || self.pixscale_arcsec <= 0.0 {
            return Err(EvalError::InvalidValue(format!(
                "pixscale must be finite and > 0, got {}",
                self.pixscale_arcsec
            )));
        }
        if !(self.ra_deg.is_finite()
            && self.dec_deg.is_finite()
            && self.orientation_deg.is_finite())
        {
            return Err(EvalError::InvalidValue(
                "ra/dec/orientation must be finite".to_owned(),
            ));
        }
        Ok(())
    }
}

/// Ground-truth camera pose derived from a [`WcsCalibration`], in the [`crate::geom`]
/// convention under the primary hypothesis (see module docs).
#[derive(Debug, Clone)]
pub struct GroundTruthPose {
    /// Boresight RA in degrees.
    pub ra_deg: f64,
    /// Boresight Dec in degrees.
    pub dec_deg: f64,
    /// Camera roll in degrees, in [`crate::geom`]'s convention, decoded per the
    /// sidecar's [`WcsRowConvention`].
    pub roll_deg: f64,
    /// Horizontal FoV in degrees: `width_arcsec/3600` when present, else
    /// `pixscale_arcsec * width_px / 3600`.
    pub fov_x_deg: Option<f64>,
    /// `true` when the calibration handedness is reproducible by a physical (non-mirror)
    /// camera. When `false`, [`compare_pose`] does not compare roll.
    pub parity_physical: bool,
}

/// Convert a WCS calibration into a ground-truth pose in our camera convention.
///
/// `width_px`/`height_px` are the pixel dimensions of the frame the calibration
/// describes (needed for the field-of-view fallback); `convention` says which
/// pixel-row order the astrometry solve saw (see [`WcsRowConvention`]).
#[must_use]
pub fn ground_truth_pose(
    calib: &WcsCalibration,
    width_px: u32,
    height_px: u32,
    convention: WcsRowConvention,
) -> GroundTruthPose {
    let parity_sign = if calib.parity >= 0.0 { 1.0 } else { -1.0 };
    let parity_physical = parity_sign > 0.0;

    let internal_orientation_deg =
        reported_orientation_to_internal(calib.orientation_deg, convention.calibration());
    let roll_deg = roll_from_orientation(
        calib.ra_deg,
        calib.dec_deg,
        internal_orientation_deg,
        parity_sign,
    );

    let fov_x_deg = Some(match calib.width_arcsec {
        Some(w) => w / 3600.0,
        None => calib.pixscale_arcsec * f64::from(width_px) / 3600.0,
    });
    // `height_px` is part of the S3-facing signature; it feeds a future vertical FoV and
    // keeps the call site symmetric. Not needed for the current outputs.
    let _ = height_px;

    GroundTruthPose {
        ra_deg: calib.ra_deg,
        dec_deg: calib.dec_deg,
        roll_deg: normalize_angle_deg(roll_deg),
        fov_x_deg,
        parity_physical,
    }
}

/// Pose-error metrics between a predicted [`SolvePose`] and a [`GroundTruthPose`].
#[derive(Debug, Clone, Serialize)]
pub struct PoseErrors {
    /// Great-circle angle between predicted and ground-truth boresights, in degrees.
    pub axis_angle_deg: f64,
    /// Wrapped absolute roll error in degrees; `None` when `!parity_physical`.
    pub roll_error_deg: Option<f64>,
    /// Relative horizontal FoV error `|pred - gt| / gt`; `None` when either is unknown.
    pub fov_error_rel: Option<f64>,
}

/// Compare a predicted pose (and optional FoV) against ground truth.
#[must_use]
pub fn compare_pose(
    pred: &SolvePose,
    pred_fov: Option<&SolveFov>,
    gt: &GroundTruthPose,
) -> PoseErrors {
    let axis_angle_deg = angular_sep(
        radec_to_unit(pred.ra_deg, pred.dec_deg),
        radec_to_unit(gt.ra_deg, gt.dec_deg),
    );
    let roll_error_deg = if gt.parity_physical {
        Some(normalize_angle_deg(pred.roll_deg - gt.roll_deg).abs())
    } else {
        None
    };
    let fov_error_rel = match (pred_fov, gt.fov_x_deg) {
        (Some(fov), Some(gt_fov)) if gt_fov.abs() > f64::EPSILON => {
            Some((fov.fov_x_deg - gt_fov).abs() / gt_fov)
        }
        _ => None,
    };
    PoseErrors {
        axis_angle_deg,
        roll_error_deg,
        fov_error_rel,
    }
}

/// Wrap an angle in degrees to the half-open interval `[-180, 180)`.
#[must_use]
pub fn normalize_angle_deg(a: f64) -> f64 {
    let mut x = a % 360.0;
    if x < -180.0 {
        x += 360.0;
    } else if x >= 180.0 {
        x -= 360.0;
    }
    x
}

/// Median of `values`, or `None` when empty. Even counts average the two middle values.
#[must_use]
pub fn median(values: &[f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let mut v = values.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = v.len();
    let mid = n / 2;
    if n % 2 == 1 {
        Some(v[mid])
    } else {
        Some((v[mid - 1] + v[mid]) / 2.0)
    }
}

/// Linear-interpolated percentile of `values` (`p` in `[0, 100]`), or `None` when empty.
#[must_use]
pub fn percentile(values: &[f64], p: f64) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let mut v = values.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = v.len();
    if n == 1 {
        return Some(v[0]);
    }
    let rank = p.clamp(0.0, 100.0) / 100.0 * (n as f64 - 1.0);
    let lo = rank.floor() as usize;
    let hi = rank.ceil() as usize;
    if lo == hi {
        return Some(v[lo]);
    }
    let frac = rank - lo as f64;
    Some(v[lo] + frac * (v[hi] - v[lo]))
}

// ---------------------------------------------------------------------------
// Internal geometry
// ---------------------------------------------------------------------------

/// Apply a convention's calibration to map a reported orientation to the internal one.
fn reported_orientation_to_internal(reported_deg: f64, cal: OrientationCalibration) -> f64 {
    cal.sign * reported_deg + cal.offset_deg
}

/// Reconstruct the roll from an internal orientation, via the transparent
/// CD -> world-directions -> camera-basis -> `rotation_to_pose` pipeline.
fn roll_from_orientation(ra_deg: f64, dec_deg: f64, orientation_deg: f64, parity_sign: f64) -> f64 {
    let (sin_t, cos_t) = orientation_deg.to_radians().sin_cos();
    // Canonical CD (positive scale dropped) that inverts astrometry's
    // get_orientation/get_parity: rows are (East, North), columns are (dx, dy).
    let cd11 = parity_sign * cos_t;
    let cd12 = sin_t;
    let cd21 = -parity_sign * sin_t;
    let cd22 = cos_t;

    let east = east_hat(ra_deg);
    let north = north_hat(ra_deg, dec_deg);
    let forward = radec_to_unit(ra_deg, dec_deg);

    // World directions of image +x (dx=1, dy=0) and image +y (dx=0, dy=1).
    let dir_x = lin_comb(cd11, east, cd21, north);
    let dir_y = lin_comb(cd12, east, cd22, north);

    let right = unit(dir_x);
    // Our image +y points DOWN, so camera up = -(image +y).
    let up = unit(neg(dir_y));

    let rot = Matrix3::from_rows(&[
        RowVector3::new(right[0], right[1], right[2]),
        RowVector3::new(up[0], up[1], up[2]),
        RowVector3::new(forward[0], forward[1], forward[2]),
    ]);
    let (_, _, roll_deg) = rotation_to_pose(&rot);
    roll_deg
}

/// Unit East vector (increasing RA) at the given RA, in equatorial J2000 coordinates.
fn east_hat(ra_deg: f64) -> [f64; 3] {
    let a = ra_deg.to_radians();
    [-a.sin(), a.cos(), 0.0]
}

/// Unit North vector (increasing Dec) at the given RA/Dec, in equatorial J2000
/// coordinates.
fn north_hat(ra_deg: f64, dec_deg: f64) -> [f64; 3] {
    let a = ra_deg.to_radians();
    let d = dec_deg.to_radians();
    [-d.sin() * a.cos(), -d.sin() * a.sin(), d.cos()]
}

fn lin_comb(a: f64, u: [f64; 3], b: f64, v: [f64; 3]) -> [f64; 3] {
    [
        a * u[0] + b * v[0],
        a * u[1] + b * v[1],
        a * u[2] + b * v[2],
    ]
}

fn neg(v: [f64; 3]) -> [f64; 3] {
    [-v[0], -v[1], -v[2]]
}

fn unit(v: [f64; 3]) -> [f64; 3] {
    let n = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if n <= f64::EPSILON {
        [0.0, 0.0, 0.0]
    } else {
        [v[0] / n, v[1] / n, v[2] / n]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geom::{principal_point, unit_to_radec, unproject, CameraSolution};

    /// astrometry.net's `get_orientation`/`get_parity` applied to a `CD` matrix
    /// (`cd[row][col]`, row 0 = East, row 1 = North). Returns `(orientation_deg,
    /// parity_sign)`. This is the definitional forward formula that the library inverts.
    fn get_orientation_of_cd(cd: [[f64; 2]; 2]) -> (f64, f64) {
        let det = cd[0][0] * cd[1][1] - cd[0][1] * cd[1][0];
        let parity = if det >= 0.0 { 1.0 } else { -1.0 };
        let t = parity * cd[0][0] + cd[1][1];
        let a = parity * cd[1][0] - cd[0][1];
        let orient = -(a.atan2(t)).to_degrees();
        (orient, parity)
    }

    /// Inverse of [`reported_orientation_to_internal`] for the top-down convention;
    /// used by the test-only extractor so the round trip is independent of the
    /// calibration values.
    fn internal_orientation_to_reported(internal_deg: f64) -> f64 {
        let cal = WcsRowConvention::TopDown.calibration();
        (internal_deg - cal.offset_deg) / cal.sign
    }

    fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
        a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
    }

    fn focal_for_fov(width: u32, fov_x_deg: f64) -> f64 {
        f64::from(width) / (2.0 * (fov_x_deg.to_radians() / 2.0).tan())
    }

    /// Numerically extract the astrometry-style calibration FROM a camera's true
    /// pixel->sky mapping, using the definitional formulas. `mirror_x` flips the pixel
    /// x-axis (`x_px -> width - x_px`) to synthesize a mirror-parity mapping.
    fn extract_calibration(sol: &CameraSolution, mirror_x: bool) -> WcsCalibration {
        let rot = sol.rotation();
        let (f, k1, w, h) = (sol.focal_px, sol.k1, sol.width, sol.height);
        let (cx, cy) = principal_point(w, h);

        let sky = |x: f64, y: f64| -> [f64; 3] {
            let xx = if mirror_x { f64::from(w) - x } else { x };
            unproject(&rot, f, k1, w, h, xx, y)
        };

        let center = sky(cx, cy);
        let (ra, dec) = unit_to_radec(center);
        let east = east_hat(ra);
        let north = north_hat(ra, dec);

        // Exact gnomonic tangent-plane coordinates (radians) of a sky direction about
        // `center`: project onto the plane tangent at `center` (scale by 1/(d·center)).
        let tangent = |d: [f64; 3]| -> (f64, f64) {
            let denom = dot(d, center);
            (dot(d, east) / denom, dot(d, north) / denom)
        };

        let eps = 1.0_f64;
        // Central differences -> partial derivatives of (East, North) w.r.t. pixels.
        let (xp_e, xp_n) = tangent(sky(cx + eps, cy));
        let (xm_e, xm_n) = tangent(sky(cx - eps, cy));
        let (yp_e, yp_n) = tangent(sky(cx, cy + eps));
        let (ym_e, ym_n) = tangent(sky(cx, cy - eps));
        let d_east_dx = (xp_e - xm_e) / (2.0 * eps);
        let d_north_dx = (xp_n - xm_n) / (2.0 * eps);
        let d_east_dy = (yp_e - ym_e) / (2.0 * eps);
        let d_north_dy = (yp_n - ym_n) / (2.0 * eps);

        // CD in deg/pixel: rows = (East, North), columns = (x, y).
        let cd = [
            [d_east_dx.to_degrees(), d_east_dy.to_degrees()],
            [d_north_dx.to_degrees(), d_north_dy.to_degrees()],
        ];
        let det = cd[0][0] * cd[1][1] - cd[0][1] * cd[1][0];
        let pixscale = 3600.0 * det.abs().sqrt();
        let (orient_internal, parity) = get_orientation_of_cd(cd);
        let reported = internal_orientation_to_reported(orient_internal);

        WcsCalibration {
            ra_deg: ra,
            dec_deg: dec,
            pixscale_arcsec: pixscale,
            orientation_deg: reported,
            parity,
            width_arcsec: Some(pixscale * f64::from(w)),
            height_arcsec: Some(pixscale * f64::from(h)),
            radius_deg: None,
        }
    }

    /// Algebraic identity of the flipped branch: for any reported orientation θ
    /// and any boresight, the reconstructed roll equals `-θ` (mod 360). The
    /// top-down branch analogously yields `θ + 180`.
    #[test]
    fn flipped_convention_reconstructs_minus_orientation() {
        let thetas = [-170.0, -148.444, -30.0, 0.0, 21.724, 68.443, 179.0];
        for &(ra, dec) in &[
            (0.0, 0.0),
            (83.497, -2.61),
            (304.484, 34.584),
            (240.0, 70.0),
        ] {
            for &theta in &thetas {
                let calib = WcsCalibration {
                    ra_deg: ra,
                    dec_deg: dec,
                    pixscale_arcsec: 60.0,
                    orientation_deg: theta,
                    parity: 1.0,
                    width_arcsec: None,
                    height_arcsec: None,
                    radius_deg: None,
                };
                let flipped = ground_truth_pose(&calib, 1000, 800, WcsRowConvention::Flipped);
                let d_flip = normalize_angle_deg(flipped.roll_deg - (-theta)).abs();
                assert!(
                    d_flip < 1e-6,
                    "flipped: theta={theta} ra={ra} dec={dec} roll={} want {}",
                    flipped.roll_deg,
                    -theta
                );

                let topdown = ground_truth_pose(&calib, 1000, 800, WcsRowConvention::TopDown);
                let d_top = normalize_angle_deg(topdown.roll_deg - (theta + 180.0)).abs();
                assert!(
                    d_top < 1e-6,
                    "topdown: theta={theta} roll={} want {}",
                    topdown.roll_deg,
                    theta + 180.0
                );
            }
        }
    }

    /// Dataset-grounded check of the two conventions against inlier-certified
    /// solver rolls (see the module-docs calibration table).
    #[test]
    fn convention_matches_dataset_solver_rolls() {
        // tetra3_alt60 (16-bit TIFF, top-down): orientation -148.444 -> roll 31.556.
        let alt60 = WcsCalibration {
            ra_deg: 240.465,
            dec_deg: 28.94,
            pixscale_arcsec: 40.329,
            orientation_deg: -148.443_789_402_455,
            parity: 1.0,
            width_arcsec: Some(41_297.157_807),
            height_arcsec: Some(30_972.868_355),
            radius_deg: None,
        };
        let gt = ground_truth_pose(&alt60, 1024, 768, WcsRowConvention::TopDown);
        let solver_roll = 30.954_36;
        assert!(
            normalize_angle_deg(gt.roll_deg - solver_roll).abs() < 1.0,
            "alt60 gt roll {} vs solver {solver_roll}",
            gt.roll_deg
        );

        // wm_constellation_orion (JPEG, flipped): orientation 37.985 -> roll -37.985.
        let wm = WcsCalibration {
            ra_deg: 83.497,
            dec_deg: -2.61,
            pixscale_arcsec: 58.687,
            orientation_deg: 37.985,
            parity: 1.0,
            width_arcsec: Some(99_122.98),
            height_arcsec: Some(68_722.918),
            radius_deg: None,
        };
        let gt = ground_truth_pose(&wm, 1689, 1171, WcsRowConvention::Flipped);
        let solver_roll = -37.805_6;
        assert!(
            normalize_angle_deg(gt.roll_deg - solver_roll).abs() < 1.0,
            "wm_orion gt roll {} vs solver {solver_roll}",
            gt.roll_deg
        );
    }

    #[test]
    fn convention_from_path_extension() {
        use std::path::Path;
        assert_eq!(
            WcsRowConvention::for_image_path(Path::new("images/a.tiff")),
            WcsRowConvention::TopDown
        );
        assert_eq!(
            WcsRowConvention::for_image_path(Path::new("images/a.TIF")),
            WcsRowConvention::TopDown
        );
        assert_eq!(
            WcsRowConvention::for_image_path(Path::new("images/a.jpg")),
            WcsRowConvention::Flipped
        );
        assert_eq!(
            WcsRowConvention::for_image_path(Path::new("images/noext")),
            WcsRowConvention::Flipped
        );
    }

    #[test]
    fn round_trip_self_consistency() {
        let ras = [0.0, 90.0, 250.3];
        let decs = [-60.0, -2.6, 0.0, 45.0, 88.0];
        let rolls = [-170.0, -90.0, 0.0, 30.0, 90.0, 179.0];
        let (width, height) = (1024u32, 768u32);
        let focal = focal_for_fov(width, 11.4);

        for &ra in &ras {
            for &dec in &decs {
                for &roll in &rolls {
                    let sol = CameraSolution {
                        ra_deg: ra,
                        dec_deg: dec,
                        roll_deg: roll,
                        focal_px: focal,
                        k1: 0.0,
                        width,
                        height,
                    };
                    let calib = extract_calibration(&sol, false);
                    let gt = ground_truth_pose(&calib, width, height, WcsRowConvention::TopDown);

                    let sep =
                        angular_sep(radec_to_unit(ra, dec), radec_to_unit(gt.ra_deg, gt.dec_deg));
                    assert!(sep < 1e-3, "axis ra={ra} dec={dec} roll={roll} sep={sep}");

                    let droll = normalize_angle_deg(roll - gt.roll_deg).abs();
                    assert!(
                        droll < 1e-3,
                        "roll ra={ra} dec={dec} roll={roll} got={} d={droll}",
                        gt.roll_deg
                    );
                    assert!(gt.parity_physical);
                }
            }
        }
    }

    #[test]
    fn mirror_parity_flags_nonphysical() {
        let (width, height) = (1024u32, 768u32);
        let focal = focal_for_fov(width, 11.4);
        let sol = CameraSolution {
            ra_deg: 120.0,
            dec_deg: 15.0,
            roll_deg: 33.0,
            focal_px: focal,
            k1: 0.0,
            width,
            height,
        };

        let normal = extract_calibration(&sol, false);
        let mirrored = extract_calibration(&sol, true);
        assert!(normal.parity > 0.0, "normal parity {}", normal.parity);
        assert!(mirrored.parity < 0.0, "mirror parity {}", mirrored.parity);

        let gt = ground_truth_pose(&mirrored, width, height, WcsRowConvention::TopDown);
        assert!(!gt.parity_physical);

        let pred = SolvePose {
            ra_deg: 120.0,
            dec_deg: 15.0,
            roll_deg: 33.0,
        };
        let errs = compare_pose(&pred, None, &gt);
        assert!(errs.roll_error_deg.is_none());
        assert!(errs.axis_angle_deg < 1e-3, "axis {}", errs.axis_angle_deg);
    }

    #[test]
    fn compare_pose_basics() {
        let gt = GroundTruthPose {
            ra_deg: 10.0,
            dec_deg: 20.0,
            roll_deg: 5.0,
            fov_x_deg: Some(11.0),
            parity_physical: true,
        };
        let pred = SolvePose {
            ra_deg: 10.0,
            dec_deg: 20.0,
            roll_deg: 5.0,
        };
        let fov = SolveFov {
            fov_x_deg: 11.0,
            fov_y_deg: 8.0,
            focal_px: 1000.0,
        };
        let e = compare_pose(&pred, Some(&fov), &gt);
        assert!(e.axis_angle_deg < 1e-9);
        assert!(e.roll_error_deg.unwrap() < 1e-9);
        assert!(e.fov_error_rel.unwrap() < 1e-9);

        // Axis offset of exactly 2 degrees (Dec + 2).
        let pred_off = SolvePose {
            ra_deg: 10.0,
            dec_deg: 22.0,
            roll_deg: 5.0,
        };
        let e_off = compare_pose(&pred_off, None, &gt);
        assert!((e_off.axis_angle_deg - 2.0).abs() < 1e-9);
        assert!(e_off.fov_error_rel.is_none());

        // Roll wrap: 179 vs -179 -> 2.
        let gt_wrap = GroundTruthPose {
            ra_deg: 0.0,
            dec_deg: 0.0,
            roll_deg: -179.0,
            fov_x_deg: None,
            parity_physical: true,
        };
        let pred_wrap = SolvePose {
            ra_deg: 0.0,
            dec_deg: 0.0,
            roll_deg: 179.0,
        };
        let e_wrap = compare_pose(&pred_wrap, None, &gt_wrap);
        assert!((e_wrap.roll_error_deg.unwrap() - 2.0).abs() < 1e-9);

        // FoV 10 vs GT 11 -> 1/11.
        let fov10 = SolveFov {
            fov_x_deg: 10.0,
            fov_y_deg: 7.0,
            focal_px: 1.0,
        };
        let e_fov = compare_pose(&pred, Some(&fov10), &gt);
        assert!((e_fov.fov_error_rel.unwrap() - 1.0 / 11.0).abs() < 1e-9);
    }

    #[test]
    fn orientation_is_position_angle_east_of_north() {
        // Image +y axis pointing exactly North -> orientation 0, parity +1.
        let (o0, p0) = get_orientation_of_cd([[1.0, 0.0], [0.0, 1.0]]);
        assert!(o0.abs() < 1e-9, "north-up orientation {o0}");
        assert!(p0 > 0.0);

        // Image +y tilted 10 deg toward East -> orientation +10 (E of N, positive East).
        let a = 10.0_f64.to_radians();
        let (o10, _) = get_orientation_of_cd([[a.cos(), a.sin()], [-a.sin(), a.cos()]]);
        assert!((o10 - 10.0).abs() < 1e-9, "tilt orientation {o10}");

        // Negative determinant -> parity -1.
        let (_, pm) = get_orientation_of_cd([[-1.0, 0.0], [0.0, 1.0]]);
        assert!(pm < 0.0);
    }

    #[test]
    fn reconstruct_inverts_astrometry_orientation_formula() {
        // cd_from_orientation (as used by roll_from_orientation) is the exact inverse of
        // astrometry's get_orientation/get_parity, for both parities and a wide angle
        // sweep. This locks the definitional inversion independent of camera geometry.
        for &parity in &[1.0_f64, -1.0] {
            for &theta in &[-170.0_f64, -90.0, -33.3, 0.0, 12.7, 90.0, 179.0] {
                let (sin_t, cos_t) = theta.to_radians().sin_cos();
                let cd = [[parity * cos_t, sin_t], [-parity * sin_t, cos_t]];
                let (orient, p) = get_orientation_of_cd(cd);
                assert!(
                    normalize_angle_deg(orient - theta).abs() < 1e-9,
                    "theta={theta} parity={parity} orient={orient}"
                );
                assert!((p - parity).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn primary_hypothesis_roll_is_orientation_plus_180() {
        // Documents (and locks) the current primary hypothesis: with the default,
        // identity ORIENTATION_CALIBRATION a parity=+1 calibration maps to
        // roll = orientation + 180 (mod 360). S4 may flip the constant; update this lock
        // together with it.
        let (width, height) = (1024u32, 768u32);
        for &orient in &[-152.3, 0.0, 37.98, 179.0, -90.0] {
            let calib = WcsCalibration {
                ra_deg: 230.0,
                dec_deg: 11.0,
                pixscale_arcsec: 40.0,
                orientation_deg: orient,
                parity: 1.0,
                width_arcsec: None,
                height_arcsec: None,
                radius_deg: None,
            };
            let gt = ground_truth_pose(&calib, width, height, WcsRowConvention::TopDown);
            let expected = normalize_angle_deg(orient + 180.0);
            assert!(
                normalize_angle_deg(gt.roll_deg - expected).abs() < 1e-9,
                "orient={orient} roll={} expected={expected}",
                gt.roll_deg
            );
            assert!(gt.parity_physical);
        }
    }

    #[test]
    fn fov_from_width_arcsec_and_fallback() {
        let calib_with = WcsCalibration {
            ra_deg: 0.0,
            dec_deg: 0.0,
            pixscale_arcsec: 40.0,
            orientation_deg: 0.0,
            parity: 1.0,
            width_arcsec: Some(7200.0),
            height_arcsec: Some(3600.0),
            radius_deg: None,
        };
        let gt = ground_truth_pose(&calib_with, 1000, 500, WcsRowConvention::TopDown);
        assert!((gt.fov_x_deg.unwrap() - 2.0).abs() < 1e-12);

        let calib_without = WcsCalibration {
            width_arcsec: None,
            ..calib_with.clone()
        };
        // pixscale 40"/px * 1000 px / 3600 = 11.111... deg
        let gt2 = ground_truth_pose(&calib_without, 1000, 500, WcsRowConvention::TopDown);
        assert!((gt2.fov_x_deg.unwrap() - 40.0 * 1000.0 / 3600.0).abs() < 1e-12);
    }

    #[test]
    fn parse_wrapper_bare_and_real_file() {
        let bare = r#"{"ra":230.6,"dec":11.0,"pixscale":40.3,"orientation":-152.3,
            "parity":1.0,"radius":7.1,"width_arcsec":41296.8,"height_arcsec":30972.6}"#;
        let c = WcsCalibration::from_json_str(bare).expect("bare parses");
        assert!((c.ra_deg - 230.6).abs() < 1e-9);
        assert!((c.radius_deg.unwrap() - 7.1).abs() < 1e-9);
        assert!(c.parity > 0.0);

        let wrapper = format!(r#"{{"jobid":1,"calibration":{bare},"tags":["x"]}}"#);
        let c2 = WcsCalibration::from_json_str(&wrapper).expect("wrapper parses");
        assert!((c2.pixscale_arcsec - 40.3).abs() < 1e-9);

        let real_path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../data/samples/sky-samples/ground-truth/tetra3_alt40.wcs.json"
        );
        let text = std::fs::read_to_string(real_path).expect("read real sidecar");
        let c3 = WcsCalibration::from_json_str(&text).expect("real sidecar parses");
        assert!((c3.ra_deg - 230.664_120_786_817_22).abs() < 1e-9);
        assert!((c3.orientation_deg - (-152.308_338_723_914)).abs() < 1e-9);
        assert!(c3.parity > 0.0);
        assert!(c3.width_arcsec.is_some());

        // Missing pixscale -> error.
        assert!(
            WcsCalibration::from_json_str(r#"{"ra":1,"dec":2,"orientation":3,"parity":1}"#)
                .is_err()
        );
        // Garbage -> error.
        assert!(WcsCalibration::from_json_str("not json").is_err());
        // Non-positive pixscale -> error.
        assert!(WcsCalibration::from_json_str(
            r#"{"ra":1,"dec":2,"pixscale":0,"orientation":3,"parity":1}"#
        )
        .is_err());
        // Explicit null calibration -> error.
        assert!(WcsCalibration::from_json_str(r#"{"calibration":null}"#).is_err());
    }

    #[test]
    fn normalize_angle_edges() {
        assert!((normalize_angle_deg(180.0) - (-180.0)).abs() < 1e-12);
        assert!((normalize_angle_deg(-180.0) - (-180.0)).abs() < 1e-12);
        assert!((normalize_angle_deg(190.0) - (-170.0)).abs() < 1e-12);
        assert!((normalize_angle_deg(-190.0) - 170.0).abs() < 1e-12);
        assert!((normalize_angle_deg(360.0)).abs() < 1e-12);
        assert!((normalize_angle_deg(720.0 + 45.0) - 45.0).abs() < 1e-12);
        assert!((normalize_angle_deg(-360.0)).abs() < 1e-12);
    }

    #[test]
    fn median_edges() {
        assert!(median(&[]).is_none());
        assert!((median(&[5.0]).unwrap() - 5.0).abs() < 1e-12);
        assert!((median(&[3.0, 1.0, 2.0]).unwrap() - 2.0).abs() < 1e-12);
        assert!((median(&[1.0, 2.0, 3.0, 4.0]).unwrap() - 2.5).abs() < 1e-12);
    }

    #[test]
    fn percentile_edges() {
        assert!(percentile(&[], 50.0).is_none());
        assert!((percentile(&[7.0], 95.0).unwrap() - 7.0).abs() < 1e-12);
        assert!((percentile(&[10.0, 20.0, 30.0, 40.0], 50.0).unwrap() - 25.0).abs() < 1e-12);

        let v: Vec<f64> = (1..=10).map(f64::from).collect();
        assert!((percentile(&v, 95.0).unwrap() - 9.55).abs() < 1e-9);
        assert!((percentile(&v, 0.0).unwrap() - 1.0).abs() < 1e-12);
        assert!((percentile(&v, 100.0).unwrap() - 10.0).abs() < 1e-12);
    }
}
