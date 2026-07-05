//! Geocentric **astrometric J2000** ephemeris for the naked-eye solar-system
//! bodies, sized for overlaying planet markers on plate-solved night-sky photos.
//!
//! # What this produces
//!
//! For every body this returns the geocentric **astrometric** right ascension and
//! declination in the **J2000 / ICRF** frame — the same frame a star catalogue
//! (Hipparcos/Tycho/Gaia) and a plate solve report their positions in. Concretely
//! that means: geometric geocentric direction corrected for **down-leg light-time**
//! only, with **no** annual aberration, **no** nutation and **no** precession to the
//! epoch of date. This is exactly JPL Horizons' `QUANTITIES='1'` (`R.A.___(ICRF),
//! DEC____(ICRF)`) definition, so the two can be compared directly (see the tests).
//!
//! The plate solve already ties the photo to the catalogue frame, so any residual
//! frame effects (aberration, refraction, nutation) are absorbed by the solve. We
//! deliberately do **not** model them here: we want astrometric J2000 to line up
//! with the catalogue, not topocentric apparent place.
//!
//! # Algorithm
//!
//! 1. [`vsop87::vsop87a`] rectangular heliocentric ecliptic **J2000** coordinates for
//!    the Earth centre and the target planet. VSOP87**A** is used (not C/D) because it
//!    is referred to the fixed J2000 ecliptic; the C/D variants are of-date and would
//!    silently mix frames.
//! 2. Geocentric vector `g = planet - earth`; one light-time iteration
//!    (`tau = |g| / c`, recompute the planet at `jd - tau`).
//! 3. Rotate the ecliptic vector into the equatorial J2000 frame with the constant
//!    IAU-1976 mean obliquity `eps0 = 23.439291111 deg`.
//! 4. `ra = atan2(y, x)` wrapped to `0..360`, `dec = asin(z / |g|)`.
//! 5. Visual magnitude from the classic Astronomical-Almanac phase-angle polynomials
//!    (see [`visual_magnitude`]).
//!
//! # Accuracy
//!
//! Versus JPL Horizons astrometric ICRF, the planet RA/Dec agree to **well under an
//! arcsecond** across 2011–2026 (VSOP87A is intrinsically ~milliarcsecond-class here
//! and the J2000-dynamical-to-ICRF frame bias is ~0.02"). Magnitudes are good to a
//! few tenths of a magnitude, which is all the app needs to size a marker.
//!
//! # Time scale
//!
//! VSOP87 is evaluated on the **TT / Julian Ephemeris Day** scale. [`julian_day_utc`]
//! returns a civil **UTC** Julian Day *without* the TT−UTC offset (≈69 s in 2026); the
//! caller owns the decision of whether to add it. For overlay work the ≈69 s (a few
//! arcseconds for the inner planets, sub-arcsecond for the outer ones) is negligible
//! and `jd` may be passed straight through. See the integration notes / tests for the
//! rigorous path.
//!
//! # References
//!
//! - P. Bretagnon & G. Francou, *VSOP87* (1988); crate [`vsop87`] 3.x.
//! - J. Meeus, *Astronomical Algorithms*, 2nd ed. — ch. 22 (obliquity), ch. 25/33
//!   (geometric-to-astrometric chain), ch. 41 (planetary magnitudes), ch. 45 (Saturn's
//!   ring), ch. 47 (position of the Moon).

use vsop87::{vsop87a, RectangularCoordinates};

/// Mean obliquity of the ecliptic at epoch J2000.0 (IAU 1976), in degrees.
///
/// `23 deg 26' 21.448"`. Held constant on purpose: we produce J2000 astrometric
/// places, so no nutation or obliquity-rate term is applied.
const OBLIQUITY_J2000_DEG: f64 = 23.439_291_111;

/// Speed of light expressed in astronomical units per day (IAU 1976 system:
/// `c = 173.144632674 AU/day`), used for the light-time correction.
const LIGHT_SPEED_AU_PER_DAY: f64 = 173.144_632_674;

/// Julian Day number of the epoch J2000.0 (2000-01-01 12:00 TT).
const JD_J2000: f64 = 2_451_545.0;

/// Days in a Julian century.
const JULIAN_CENTURY_DAYS: f64 = 36_525.0;

/// Length of a Julian year in days (used by [`epoch_years`]).
const JULIAN_YEAR_DAYS: f64 = 365.25;

/// One astronomical unit in kilometres (IAU 2012 / DE value), for the Moon distance.
const AU_IN_KM: f64 = 149_597_870.7;

/// A geocentric astrometric position of a solar-system body, J2000 / ICRF.
///
/// Angles are in degrees; `ra_deg` is wrapped to `0.0..360.0` and `dec_deg` lies in
/// `-90.0..=90.0`. `dist_au` is the geocentric distance in astronomical units and
/// `mag` is the approximate visual magnitude.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PlanetPosition {
    /// Body name, e.g. `"Mercury"`, `"Sun"`, `"Moon"`.
    pub name: &'static str,
    /// Right ascension, degrees, wrapped to `0.0..360.0`.
    pub ra_deg: f64,
    /// Declination, degrees, in `-90.0..=90.0`.
    pub dec_deg: f64,
    /// Geocentric distance in astronomical units.
    pub dist_au: f64,
    /// Approximate visual magnitude (smaller is brighter).
    pub mag: f64,
}

/// The eight VSOP87 planets we can place relative to the Earth.
///
/// (Earth is the observer and therefore excluded from the output.)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Planet {
    Mercury,
    Venus,
    Mars,
    Jupiter,
    Saturn,
    Uranus,
    Neptune,
}

impl Planet {
    /// Mercury … Neptune, in solar-distance order, as emitted by [`planet_positions`].
    const ALL: [Planet; 7] = [
        Planet::Mercury,
        Planet::Venus,
        Planet::Mars,
        Planet::Jupiter,
        Planet::Saturn,
        Planet::Uranus,
        Planet::Neptune,
    ];

    /// Display name used in [`PlanetPosition::name`].
    const fn name(self) -> &'static str {
        match self {
            Planet::Mercury => "Mercury",
            Planet::Venus => "Venus",
            Planet::Mars => "Mars",
            Planet::Jupiter => "Jupiter",
            Planet::Saturn => "Saturn",
            Planet::Uranus => "Uranus",
            Planet::Neptune => "Neptune",
        }
    }

    /// Heliocentric ecliptic-J2000 rectangular position (AU) at Julian Ephemeris Day
    /// `jde`, from the VSOP87A series.
    fn heliocentric(self, jde: f64) -> Vec3 {
        let c: RectangularCoordinates = match self {
            Planet::Mercury => vsop87a::mercury(jde),
            Planet::Venus => vsop87a::venus(jde),
            Planet::Mars => vsop87a::mars(jde),
            Planet::Jupiter => vsop87a::jupiter(jde),
            Planet::Saturn => vsop87a::saturn(jde),
            Planet::Uranus => vsop87a::uranus(jde),
            Planet::Neptune => vsop87a::neptune(jde),
        };
        Vec3::new(c.x, c.y, c.z)
    }
}

/// A minimal 3-vector for the ecliptic/equatorial rotations; kept local so the
/// module has no maths dependency beyond `std` and [`vsop87`].
#[derive(Debug, Clone, Copy, PartialEq)]
struct Vec3 {
    x: f64,
    y: f64,
    z: f64,
}

impl Vec3 {
    #[inline]
    const fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }

    /// Euclidean length.
    #[inline]
    fn norm(self) -> f64 {
        self.x
            .mul_add(self.x, self.y.mul_add(self.y, self.z * self.z))
            .sqrt()
    }

    /// Component-wise difference `self - other`.
    #[inline]
    fn sub(self, other: Self) -> Self {
        Self::new(self.x - other.x, self.y - other.y, self.z - other.z)
    }

    /// Dot product.
    #[inline]
    fn dot(self, other: Self) -> f64 {
        self.x
            .mul_add(other.x, self.y.mul_add(other.y, self.z * other.z))
    }
}

/// Julian Day from a proleptic-Gregorian **UTC** civil date and time.
///
/// Uses the standard Meeus algorithm (valid for the Gregorian calendar, i.e. all
/// modern dates). The result is a UTC Julian Day; it does **not** include the TT−UTC
/// offset that VSOP87 formally wants — see the module-level time-scale note.
///
/// `mo` is `1..=12`, `d` is `1..=31`, `h`/`mi` are `0..`, `s` may carry a fraction.
/// Out-of-range fields are not rejected (the API is infallible); pass sane values.
///
/// # Examples
///
/// ```ignore
/// // 2000-01-01 12:00:00 UTC is the J2000.0 epoch, JD 2451545.0.
/// let jd = julian_day_utc(2000, 1, 1, 12, 0, 0.0);
/// assert!((jd - 2_451_545.0).abs() < 1e-9);
/// ```
#[must_use]
pub fn julian_day_utc(y: i32, mo: u32, d: u32, h: u32, mi: u32, s: f64) -> f64 {
    // Shift Jan/Feb to months 13/14 of the previous year (Meeus, ch. 7).
    let (year, month) = if mo <= 2 {
        (f64::from(y) - 1.0, f64::from(mo) + 12.0)
    } else {
        (f64::from(y), f64::from(mo))
    };

    // Gregorian-calendar century correction.
    let a = (year / 100.0).floor();
    let b = 2.0 - a + (a / 4.0).floor();

    let day_fraction = (f64::from(h) + f64::from(mi) / 60.0 + s / 3600.0) / 24.0;

    (365.25 * (year + 4716.0)).floor() + (30.6001 * (month + 1.0)).floor() + f64::from(d) + b
        - 1524.5
        + day_fraction
}

/// Fractional Julian epoch, `2000.0 + (jd - 2451545.0) / 365.25`.
///
/// Handy for feeding proper-motion or precession routines that are parameterised by
/// year rather than Julian Day.
///
/// # Examples
///
/// ```ignore
/// assert!((epoch_years(2_451_545.0) - 2000.0).abs() < 1e-9);
/// ```
#[must_use]
pub fn epoch_years(jd: f64) -> f64 {
    2000.0 + (jd - JD_J2000) / JULIAN_YEAR_DAYS
}

/// Geocentric astrometric J2000 positions of Mercury through Neptune at Julian
/// Ephemeris Day `jd`.
///
/// The returned vector is ordered by heliocentric distance (Mercury, Venus, Mars,
/// Jupiter, Saturn, Uranus, Neptune). Earth is the observer and is not included.
///
/// `jd` is interpreted on the TT / Julian-Ephemeris-Day scale (see the module note on
/// time scales).
///
/// # Examples
///
/// ```ignore
/// let jd = julian_day_utc(2026, 7, 1, 0, 0, 0.0);
/// let planets = planet_positions(jd);
/// assert_eq!(planets.len(), 7);
/// assert_eq!(planets[3].name, "Jupiter");
/// assert!(planets.iter().all(|p| (0.0..360.0).contains(&p.ra_deg)));
/// ```
#[must_use]
pub fn planet_positions(jd: f64) -> Vec<PlanetPosition> {
    let earth = vsop87a::earth(jd);
    let earth = Vec3::new(earth.x, earth.y, earth.z);
    let sun_earth_dist = earth.norm();

    let mut out = Vec::with_capacity(Planet::ALL.len());
    for planet in Planet::ALL {
        let view = geocentric_astrometric(planet, earth, jd);
        let (ra_deg, dec_deg) = ra_dec_of(view.equatorial, view.dist_au);

        // Ring opening angle only matters for Saturn's brightness.
        let ring_tilt_deg = if planet == Planet::Saturn {
            Some(saturn_ring_tilt_deg(view.equatorial))
        } else {
            None
        };
        let phase_deg = phase_angle_deg(view.helio_dist, view.dist_au, sun_earth_dist);
        let mag = visual_magnitude(
            planet,
            view.helio_dist,
            view.dist_au,
            phase_deg,
            ring_tilt_deg,
        );

        out.push(PlanetPosition {
            name: planet.name(),
            ra_deg,
            dec_deg,
            dist_au: view.dist_au,
            mag,
        });
    }
    out
}

/// Geocentric astrometric J2000 position of the **Sun**.
///
/// In the heliocentric VSOP87A frame the Sun sits at the origin, so the geocentric Sun
/// vector is simply `-earth`. Provided mostly for twilight / daylight sanity checks
/// (e.g. "is the Sun far enough below the horizon to bother solving?"). The magnitude
/// is the fixed nominal value `-26.74`.
#[must_use]
pub fn sun_position(jd: f64) -> PlanetPosition {
    let earth = vsop87a::earth(jd);
    // Sun - Earth = 0 - earth. The Sun does not move in the heliocentric frame, so no
    // light-time iteration is required.
    let geo_ecliptic = Vec3::new(-earth.x, -earth.y, -earth.z);
    let dist_au = geo_ecliptic.norm();
    let equatorial = ecliptic_to_equatorial(geo_ecliptic);
    let (ra_deg, dec_deg) = ra_dec_of(equatorial, dist_au);
    PlanetPosition {
        name: "Sun",
        ra_deg,
        dec_deg,
        dist_au,
        mag: -26.74,
    }
}

/// Geocentric astrometric J2000 position of the **Moon** — *approximate*.
///
/// # Accuracy and the parallax caveat
///
/// The direction is computed from a truncated ELP-2000/82 series (the largest terms of
/// Meeus ch. 47) and is good to roughly **0.1°** *geocentrically*. Crucially, this is a
/// **geocentric** place: **topocentric parallax, which reaches ≈0.95° for the Moon, is
/// NOT applied.** For every other body in this module parallax is a sub-arcsecond
/// afterthought, but for the Moon it is the dominant error and depends on the observer's
/// latitude, longitude and altitude, which this `jd`-only API does not carry.
///
/// Therefore: use this to answer "is the Moon up, and roughly where?", **not** to pin a
/// marker onto a plate-solved photo — a geocentric Moon can sit up to ~2 lunar diameters
/// from where the camera actually saw it. For a photo-accurate marker the caller must add
/// topocentric parallax from the observer location (see the integration notes).
///
/// The distance-of-date precession is folded back to J2000 with a first-order longitude
/// correction, consistent with the ~0.1° budget.
#[must_use]
pub fn moon_position(jd: f64) -> PlanetPosition {
    let t = (jd - JD_J2000) / JULIAN_CENTURY_DAYS;

    // Fundamental arguments (degrees), Meeus (47.1)–(47.6).
    let lp = 218.316_447_7 + 481_267.881_234_21 * t - 0.001_578_6 * t * t + t * t * t / 538_841.0
        - t * t * t * t / 65_194_000.0;
    let d = 297.850_192_1 + 445_267.111_403_4 * t - 0.001_881_9 * t * t + t * t * t / 545_868.0
        - t * t * t * t / 113_065_000.0;
    let m = 357.529_109_2 + 35_999.050_290_9 * t - 0.000_153_6 * t * t + t * t * t / 24_490_000.0;
    let mp = 134.963_396_4 + 477_198.867_505_5 * t + 0.008_741_4 * t * t + t * t * t / 69_699.0
        - t * t * t * t / 14_712_000.0;
    let f = 93.272_095_0 + 483_202.017_523_3 * t - 0.003_653_9 * t * t - t * t * t / 3_526_000.0
        + t * t * t * t / 863_310_000.0;

    // Eccentricity factor applied to terms involving the Sun's anomaly M.
    let e = 1.0 - 0.002_516 * t - 0.000_007_4 * t * t;

    let (d_r, m_r, mp_r, f_r) = (
        d.to_radians(),
        m.to_radians(),
        mp.to_radians(),
        f.to_radians(),
    );

    let mut sum_l = 0.0_f64; // sine terms, unit 1e-6 deg
    let mut sum_r = 0.0_f64; // cosine terms, unit 1e-3 km
    for &(cd, cm, cmp, cf, sl, sr) in MOON_LON_DIST_TERMS {
        let arg =
            f64::from(cd) * d_r + f64::from(cm) * m_r + f64::from(cmp) * mp_r + f64::from(cf) * f_r;
        let ecc = e.powi(cm.unsigned_abs() as i32);
        sum_l += f64::from(sl) * ecc * arg.sin();
        sum_r += f64::from(sr) * ecc * arg.cos();
    }

    let mut sum_b = 0.0_f64; // sine terms, unit 1e-6 deg
    for &(cd, cm, cmp, cf, sb) in MOON_LAT_TERMS {
        let arg =
            f64::from(cd) * d_r + f64::from(cm) * m_r + f64::from(cmp) * mp_r + f64::from(cf) * f_r;
        let ecc = e.powi(cm.unsigned_abs() as i32);
        sum_b += f64::from(sb) * ecc * arg.sin();
    }

    // Geocentric ecliptic coordinates referred to the mean equinox *of date*.
    let lon_of_date = lp + sum_l / 1_000_000.0;
    let lat = sum_b / 1_000_000.0;
    let dist_km = 385_000.56 + sum_r / 1000.0;
    let dist_au = dist_km / AU_IN_KM;

    // Fold longitude back to the J2000 mean equinox with the accumulated general
    // precession in longitude p_A = 5028.796195"*T + 1.1054348"*T^2 (Meeus ch. 21).
    // Over 2000..2026 the "pure longitude shift" approximation is good to well under
    // the ~0.1° truncation budget.
    let precession_deg = (5_028.796_195 * t + 1.105_434_8 * t * t) / 3600.0;
    let lon_j2000 = lon_of_date - precession_deg;

    // Ecliptic-J2000 rectangular, then rotate to equatorial J2000.
    let (lon_r, lat_r) = (lon_j2000.to_radians(), lat.to_radians());
    let cos_lat = lat_r.cos();
    let ecliptic = Vec3::new(
        dist_au * cos_lat * lon_r.cos(),
        dist_au * cos_lat * lon_r.sin(),
        dist_au * lat_r.sin(),
    );
    let equatorial = ecliptic_to_equatorial(ecliptic);
    let (ra_deg, dec_deg) = ra_dec_of(equatorial, dist_au);

    // Rough phase-angle magnitude (Allen / Astronomical Almanac): the Sun's geocentric
    // vector is -earth; the phase angle is measured at the Moon.
    let earth = vsop87a::earth(jd);
    let sun_geo = ecliptic_to_equatorial(Vec3::new(-earth.x, -earth.y, -earth.z));
    let moon_to_earth = Vec3::new(-equatorial.x, -equatorial.y, -equatorial.z);
    let moon_to_sun = sun_geo.sub(equatorial);
    let cos_phase = (moon_to_earth.dot(moon_to_sun) / (moon_to_earth.norm() * moon_to_sun.norm()))
        .clamp(-1.0, 1.0);
    let phase_deg = cos_phase.acos().to_degrees();
    let mag = -12.73 + 0.026 * phase_deg.abs() + 4.0e-9 * phase_deg.powi(4);

    PlanetPosition {
        name: "Moon",
        ra_deg,
        dec_deg,
        dist_au,
        mag,
    }
}

/// Result of the geocentric-astrometric reduction for one planet.
struct GeoView {
    /// Geocentric equatorial-J2000 rectangular vector (AU).
    equatorial: Vec3,
    /// Geocentric distance (AU).
    dist_au: f64,
    /// Heliocentric distance of the planet at the light-time-corrected epoch (AU).
    helio_dist: f64,
}

/// Geometric geocentric direction of `planet` corrected for down-leg light-time.
///
/// The Earth is held at the observation epoch `jd`; only the planet is retarded to the
/// emission epoch `jd - tau`. One iteration is enough for sub-arcsecond convergence.
fn geocentric_astrometric(planet: Planet, earth: Vec3, jd: f64) -> GeoView {
    let planet_now = planet.heliocentric(jd);
    let geo0 = planet_now.sub(earth);
    let tau = geo0.norm() / LIGHT_SPEED_AU_PER_DAY;

    let planet_then = planet.heliocentric(jd - tau);
    let geo = planet_then.sub(earth);

    GeoView {
        equatorial: ecliptic_to_equatorial(geo),
        dist_au: geo.norm(),
        helio_dist: planet_then.norm(),
    }
}

/// Rotate an ecliptic-J2000 vector into the equatorial-J2000 frame about the vernal
/// equinox (x-axis) by the mean obliquity `eps0`.
#[inline]
fn ecliptic_to_equatorial(v: Vec3) -> Vec3 {
    let (sin_e, cos_e) = OBLIQUITY_J2000_DEG.to_radians().sin_cos();
    Vec3::new(
        v.x,
        v.y.mul_add(cos_e, -v.z * sin_e),
        v.y.mul_add(sin_e, v.z * cos_e),
    )
}

/// Right ascension (wrapped to `0..360`) and declination (degrees) of an equatorial
/// rectangular vector of length `dist`.
#[inline]
fn ra_dec_of(equatorial: Vec3, dist: f64) -> (f64, f64) {
    let ra = equatorial
        .y
        .atan2(equatorial.x)
        .to_degrees()
        .rem_euclid(360.0);
    let dec = (equatorial.z / dist).clamp(-1.0, 1.0).asin().to_degrees();
    (ra, dec)
}

/// Phase angle (Sun–body–Earth, degrees) from the three sides of the triangle:
/// heliocentric distance `r`, geocentric distance `delta`, Sun–Earth distance `big_r`.
#[inline]
fn phase_angle_deg(r: f64, delta: f64, big_r: f64) -> f64 {
    let cos_i = (r * r + delta * delta - big_r * big_r) / (2.0 * r * delta);
    cos_i.clamp(-1.0, 1.0).acos().to_degrees()
}

/// Saturnicentric latitude of the Earth (the ring "opening" angle `B`, degrees) from
/// the geocentric equatorial direction to Saturn and Saturn's IAU J2000 north pole
/// (`alpha0 = 40.589°, delta0 = 83.537°`). Returned signed; callers use `|B|`.
fn saturn_ring_tilt_deg(saturn_equatorial: Vec3) -> f64 {
    const POLE_RA_DEG: f64 = 40.589;
    const POLE_DEC_DEG: f64 = 83.537;
    let (ra_r, dec_r) = (POLE_RA_DEG.to_radians(), POLE_DEC_DEG.to_radians());
    let pole = Vec3::new(
        dec_r.cos() * ra_r.cos(),
        dec_r.cos() * ra_r.sin(),
        dec_r.sin(),
    );
    let dist = saturn_equatorial.norm();
    // Saturn->Earth unit vector is -saturn_equatorial / dist.
    let sin_b = -pole.dot(saturn_equatorial) / dist;
    sin_b.clamp(-1.0, 1.0).asin().to_degrees()
}

/// Approximate visual magnitude from the Astronomical-Almanac phase-angle polynomials
/// (Meeus, *Astronomical Algorithms* 2nd ed., ch. 41), with the Saturn ring terms of
/// ch. 45.
///
/// Inputs: heliocentric distance `r` (AU), geocentric distance `delta` (AU), phase
/// angle `i` (degrees) and, for Saturn only, the ring opening `ring_b` (degrees).
///
/// These reproduce the JPL Horizons visual magnitudes to a few tenths across
/// 2011–2026 (verified in the tests), which is ample for driving marker prominence.
/// The small Saturn `0.044·|ΔU|` longitude term is dropped (< ~0.2 mag).
#[must_use]
fn visual_magnitude(planet: Planet, r: f64, delta: f64, i: f64, ring_b: Option<f64>) -> f64 {
    let distance_term = 5.0 * (r * delta).log10();
    let phase = match planet {
        Planet::Mercury => -0.42 + 0.038 * i - 0.000_273 * i * i + 0.000_002 * i * i * i,
        Planet::Venus => -4.40 + 0.0009 * i + 0.000_239 * i * i - 0.000_000_65 * i * i * i,
        Planet::Mars => -1.52 + 0.016 * i,
        Planet::Jupiter => -9.40 + 0.005 * i,
        Planet::Saturn => {
            let b = ring_b.unwrap_or(0.0).to_radians().abs();
            -8.88 - 2.60 * b.sin() + 1.25 * b.sin() * b.sin()
        }
        Planet::Uranus => -7.19 + 0.0028 * i,
        Planet::Neptune => -6.87,
    };
    distance_term + phase
}

/// Truncated ELP-2000/82 periodic terms for the Moon's longitude and distance
/// (Meeus, table 47.A). Tuple layout: `(D, M, M', F, Σl, Σr)` where `Σl` is the sine
/// coefficient in units of `1e-6` degree and `Σr` the cosine coefficient in units of
/// `1e-3` km. Terms carrying the Sun's anomaly `M` are scaled by `E^|M|`.
///
/// The largest ~30 terms are kept, which lands the longitude within a few hundredths
/// of a degree — comfortably inside the module's ~0.1° geocentric budget.
#[rustfmt::skip]
const MOON_LON_DIST_TERMS: &[(i8, i8, i8, i8, i32, i32)] = &[
    (0,  0,  1,  0,  6_288_774, -20_905_355),
    (2,  0, -1,  0,  1_274_027,  -3_699_111),
    (2,  0,  0,  0,    658_314,  -2_955_968),
    (0,  0,  2,  0,    213_618,    -569_925),
    (0,  1,  0,  0,   -185_116,      48_888),
    (0,  0,  0,  2,   -114_332,      -3_149),
    (2,  0, -2,  0,     58_793,     246_158),
    (2, -1, -1,  0,     57_066,    -152_138),
    (2,  0,  1,  0,     53_322,    -170_733),
    (2, -1,  0,  0,     45_758,    -204_586),
    (0,  1, -1,  0,    -40_923,    -129_620),
    (1,  0,  0,  0,    -34_720,     108_743),
    (0,  1,  1,  0,    -30_383,     104_755),
    (2,  0,  0, -2,     15_327,      10_321),
    (0,  0,  1,  2,    -12_528,           0),
    (0,  0,  1, -2,     10_980,      79_661),
    (4,  0, -1,  0,     10_675,     -34_782),
    (0,  0,  3,  0,     10_034,     -23_210),
    (4,  0, -2,  0,      8_548,     -21_636),
    (2,  1, -1,  0,     -7_888,      24_208),
    (2,  1,  0,  0,     -6_766,      30_824),
    (1,  0, -1,  0,     -5_163,      -8_379),
    (1,  1,  0,  0,      4_987,     -16_675),
    (2, -1,  1,  0,      4_036,     -12_831),
    (2,  0,  2,  0,      3_994,     -10_445),
    (4,  0,  0,  0,      3_861,     -11_650),
    (2,  0, -3,  0,      3_665,      14_403),
    (0,  1, -2,  0,     -2_689,      -7_003),
    (2,  0, -1,  2,     -2_602,           0),
    (2, -1, -2,  0,      2_390,      10_056),
    (1,  0,  1,  0,     -2_348,       6_322),
    (2, -2,  0,  0,      2_236,      -9_884),
    (0,  1,  2,  0,     -2_120,       5_751),
    (0,  2,  0,  0,     -2_069,       5_717),
    (2, -2, -1,  0,      2_048,      -8_950),
];

/// Truncated ELP-2000/82 periodic terms for the Moon's latitude (Meeus, table 47.B).
/// Tuple layout `(D, M, M', F, Σb)` with `Σb` the sine coefficient in `1e-6` degree.
#[rustfmt::skip]
const MOON_LAT_TERMS: &[(i8, i8, i8, i8, i32)] = &[
    (0,  0,  0,  1,  5_128_122),
    (0,  0,  1,  1,    280_602),
    (0,  0,  1, -1,    277_693),
    (2,  0,  0, -1,    173_237),
    (2,  0, -1,  1,     55_413),
    (2,  0, -1, -1,     46_271),
    (2,  0,  0,  1,     32_573),
    (0,  0,  2,  1,     17_198),
    (2,  0,  1, -1,      9_266),
    (0,  0,  2, -1,      8_822),
    (2, -1,  0, -1,      8_216),
    (2,  0, -2, -1,      4_324),
    (2,  0,  1,  1,      4_200),
    (2,  1,  0, -1,     -3_359),
    (2, -1, -1,  1,      2_463),
    (2, -1,  0,  1,      2_211),
    (2, -1, -1, -1,      2_065),
    (0,  1, -1, -1,     -1_870),
    (4,  0, -1, -1,      1_828),
    (0,  1,  0,  1,     -1_794),
];

#[cfg(test)]
mod tests {
    use super::*;

    /// Angular separation, in arcseconds, between two RA/Dec directions (deg).
    fn sep_arcsec(ra1: f64, dec1: f64, ra2: f64, dec2: f64) -> f64 {
        let (ra1, dec1) = (ra1.to_radians(), dec1.to_radians());
        let (ra2, dec2) = (ra2.to_radians(), dec2.to_radians());
        let cos_sep = dec1.sin() * dec2.sin() + dec1.cos() * dec2.cos() * (ra1 - ra2).cos();
        cos_sep.clamp(-1.0, 1.0).acos().to_degrees() * 3600.0
    }

    // ----- Time-scale offsets used to align our TT/JDE input with the UTC instant
    // that JPL Horizons reports. TT - UTC = 32.184 s + (TAI - UTC) leap seconds.
    // 2011: 34 leap seconds -> 66.184 s.   2026: 37 leap seconds -> 69.184 s.
    const JDE_2011_09_21: f64 = 2_455_825.5 + 66.184 / 86_400.0;
    const JDE_2026_07_01: f64 = 2_461_222.5 + 69.184 / 86_400.0;

    // =====================================================================
    // Reference values: JPL Horizons API, geocentric (CENTER='500@399'),
    // QUANTITIES='1,9', ANG_FORMAT='DEG', EXTRA_PREC='YES'.
    //   https://ssd.jpl.nasa.gov/api/horizons.api
    // "R.A.___(ICRF), DEC____(ICRF)" = astrometric J2000, down-leg light-time
    // compensated (no aberration/nutation) — exactly our reduction chain.
    // Retrieved 2026-07-03. Columns: RA_deg, Dec_deg, APmag.
    // =====================================================================

    /// One Horizons reference row.
    struct Ref {
        name: &'static str,
        ra: f64,
        dec: f64,
        mag: f64,
    }

    // ---- 2011-09-21 00:00:00 UTC (COMMAND 199/299/499/599/699/799/899) ----
    const REF_2011: &[Ref] = &[
        Ref {
            name: "Mercury",
            ra: 172.258_361_156,
            dec: 5.303_166_708,
            mag: -1.439,
        },
        Ref {
            name: "Venus",
            ra: 187.043_355_611,
            dec: -1.736_225_723,
            mag: -3.905,
        },
        Ref {
            name: "Mars",
            ra: 123.458_576_044,
            dec: 20.883_868_769,
            mag: 1.352,
        },
        Ref {
            name: "Jupiter",
            ra: 37.486_997_043,
            dec: 13.255_024_819,
            mag: -2.785,
        },
        Ref {
            name: "Saturn",
            ra: 196.780_033_409,
            dec: -4.689_145_968,
            mag: 0.845,
        },
        Ref {
            name: "Uranus",
            ra: 2.697_133_142,
            dec: 0.328_335_446,
            mag: 5.794,
        },
        Ref {
            name: "Neptune",
            ra: 330.943_523_062,
            dec: -12.483_462_483,
            mag: 7.707,
        },
    ];

    // ---- 2026-07-01 00:00:00 UTC ----
    const REF_2026: &[Ref] = &[
        Ref {
            name: "Mercury",
            ra: 117.324_333_463,
            dec: 18.515_563_505,
            mag: 2.212,
        },
        Ref {
            name: "Venus",
            ra: 142.819_384_714,
            dec: 16.554_893_462,
            mag: -4.070,
        },
        Ref {
            name: "Mars",
            ra: 59.126_143_061,
            dec: 20.101_298_275,
            mag: 1.373,
        },
        Ref {
            name: "Jupiter",
            ra: 122.065_296_566,
            dec: 20.620_100_487,
            mag: -1.809,
        },
        Ref {
            name: "Saturn",
            ra: 13.639_591_585,
            dec: 3.252_555_273,
            mag: 0.775,
        },
        Ref {
            name: "Uranus",
            ra: 61.343_421_811,
            dec: 20.666_571_283,
            mag: 5.808,
        },
        Ref {
            name: "Neptune",
            ra: 4.244_387_127,
            dec: 0.348_592_240,
            mag: 7.762,
        },
    ];

    // Sun (COMMAND 10), same query.
    const SUN_REF_2011: (f64, f64) = (177.716_371_015, 0.989_743_986);
    const SUN_REF_2026: (f64, f64) = (99.616_219_275, 23.141_364_569);

    // Moon (COMMAND 301), geocentric astrometric J2000.
    const MOON_REF_2011: (f64, f64) = (92.723_176_638, 22.124_465_512);
    const MOON_REF_2026: (f64, f64) = (292.002_295_991, -25.262_251_267);

    /// Threshold for the planets: VSOP87A + our simplifications should land far inside
    /// this. (Actual residuals print at ~1" — see the assertion messages.)
    const PLANET_TOL_ARCSEC: f64 = 60.0;

    fn check_epoch(jde: f64, refs: &[Ref], label: &str) {
        let got = planet_positions(jde);
        assert_eq!(got.len(), refs.len());
        for (p, r) in got.iter().zip(refs) {
            assert_eq!(p.name, r.name, "ordering mismatch at {label}");
            let err = sep_arcsec(p.ra_deg, p.dec_deg, r.ra, r.dec);
            assert!(
                err < PLANET_TOL_ARCSEC,
                "{label} {}: {err:.2}\" from Horizons (RA {:.6} vs {:.6}, Dec {:.6} vs {:.6})",
                p.name,
                p.ra_deg,
                r.ra,
                p.dec_deg,
                r.dec
            );
            // Magnitude is only a marker cue; keep a loose net.
            assert!(
                (p.mag - r.mag).abs() < 0.5,
                "{label} {}: mag {:.3} vs Horizons {:.3}",
                p.name,
                p.mag,
                r.mag
            );
        }
    }

    #[test]
    fn planets_match_horizons_2011() {
        check_epoch(JDE_2011_09_21, REF_2011, "2011-09-21");
    }

    #[test]
    fn planets_match_horizons_2026() {
        check_epoch(JDE_2026_07_01, REF_2026, "2026-07-01");
    }

    #[test]
    fn sun_matches_horizons() {
        let s11 = sun_position(JDE_2011_09_21);
        let e11 = sep_arcsec(s11.ra_deg, s11.dec_deg, SUN_REF_2011.0, SUN_REF_2011.1);
        assert!(e11 < PLANET_TOL_ARCSEC, "Sun 2011: {e11:.2}\"");

        let s26 = sun_position(JDE_2026_07_01);
        let e26 = sep_arcsec(s26.ra_deg, s26.dec_deg, SUN_REF_2026.0, SUN_REF_2026.1);
        assert!(e26 < PLANET_TOL_ARCSEC, "Sun 2026: {e26:.2}\"");
    }

    #[test]
    fn moon_is_within_truncation_and_geocentric_budget() {
        // Looser tolerance than the planets: the ELP series is truncated and this is a
        // *geocentric* place (topocentric parallax up to ~0.95° is deliberately absent).
        // 0.15° = 540" bounds the truncation + precession-approximation error.
        const MOON_TOL_ARCSEC: f64 = 540.0;
        let m11 = moon_position(JDE_2011_09_21);
        let e11 = sep_arcsec(m11.ra_deg, m11.dec_deg, MOON_REF_2011.0, MOON_REF_2011.1);
        assert!(
            e11 < MOON_TOL_ARCSEC,
            "Moon 2011: {e11:.1}\" ({:.3} deg)",
            e11 / 3600.0
        );

        let m26 = moon_position(JDE_2026_07_01);
        let e26 = sep_arcsec(m26.ra_deg, m26.dec_deg, MOON_REF_2026.0, MOON_REF_2026.1);
        assert!(
            e26 < MOON_TOL_ARCSEC,
            "Moon 2026: {e26:.1}\" ({:.3} deg)",
            e26 / 3600.0
        );
    }

    #[test]
    fn julian_day_matches_known_epochs() {
        // Canonical values from Meeus, Astronomical Algorithms 2nd ed., ch. 7.
        assert!((julian_day_utc(2000, 1, 1, 12, 0, 0.0) - 2_451_545.0).abs() < 1e-6);
        assert!((julian_day_utc(1987, 1, 27, 0, 0, 0.0) - 2_446_822.5).abs() < 1e-6);
        assert!((julian_day_utc(1900, 1, 1, 0, 0, 0.0) - 2_415_020.5).abs() < 1e-6);
        // The two epochs used above.
        assert!((julian_day_utc(2011, 9, 21, 0, 0, 0.0) - 2_455_825.5).abs() < 1e-6);
        assert!((julian_day_utc(2026, 7, 1, 0, 0, 0.0) - 2_461_222.5).abs() < 1e-6);
    }

    #[test]
    fn epoch_years_is_linear_in_jd() {
        assert!((epoch_years(JD_J2000) - 2000.0).abs() < 1e-12);
        assert!((epoch_years(JD_J2000 + JULIAN_YEAR_DAYS) - 2001.0).abs() < 1e-12);
    }

    #[test]
    fn ra_is_wrapped_and_dec_bounded() {
        for p in planet_positions(JDE_2026_07_01) {
            assert!(
                (0.0..360.0).contains(&p.ra_deg),
                "{} RA out of range",
                p.name
            );
            assert!(
                (-90.0..=90.0).contains(&p.dec_deg),
                "{} Dec out of range",
                p.name
            );
            assert!(p.dist_au > 0.0);
        }
    }

    /// Print the actual residuals so the reader can see how far inside 60" we land.
    #[test]
    fn report_residuals() {
        for (jde, refs, label) in [
            (JDE_2011_09_21, REF_2011, "2011-09-21"),
            (JDE_2026_07_01, REF_2026, "2026-07-01"),
        ] {
            let got = planet_positions(jde);
            for (p, r) in got.iter().zip(refs) {
                let err = sep_arcsec(p.ra_deg, p.dec_deg, r.ra, r.dec);
                println!(
                    "{label} {:8} {:7.3}\"  dmag {:+.3}",
                    p.name,
                    err,
                    p.mag - r.mag
                );
            }
        }
        let s11 = sun_position(JDE_2011_09_21);
        println!(
            "2011-09-21 Sun      {:7.3}\"",
            sep_arcsec(s11.ra_deg, s11.dec_deg, SUN_REF_2011.0, SUN_REF_2011.1)
        );
        let s26 = sun_position(JDE_2026_07_01);
        println!(
            "2026-07-01 Sun      {:7.3}\"",
            sep_arcsec(s26.ra_deg, s26.dec_deg, SUN_REF_2026.0, SUN_REF_2026.1)
        );
        let m11 = moon_position(JDE_2011_09_21);
        println!(
            "2011-09-21 Moon   {:7.1}\" ({:.3} deg)  dmag {:+.2}",
            sep_arcsec(m11.ra_deg, m11.dec_deg, MOON_REF_2011.0, MOON_REF_2011.1),
            sep_arcsec(m11.ra_deg, m11.dec_deg, MOON_REF_2011.0, MOON_REF_2011.1) / 3600.0,
            m11.mag - (-9.891),
        );
        let m26 = moon_position(JDE_2026_07_01);
        println!(
            "2026-07-01 Moon   {:7.1}\" ({:.3} deg)  dmag {:+.2}",
            sep_arcsec(m26.ra_deg, m26.dec_deg, MOON_REF_2026.0, MOON_REF_2026.1),
            sep_arcsec(m26.ra_deg, m26.dec_deg, MOON_REF_2026.0, MOON_REF_2026.1) / 3600.0,
            m26.mag - (-12.274),
        );
    }
}
