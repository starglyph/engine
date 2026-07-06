use std::path::Path;

use image::GenericImageView;

/// Timestamp parsed from a frame filename stem.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FrameTimestamp {
    pub year: i32,
    pub month: u32,
    pub day: u32,
    pub hour: u32,
    pub minute: u32,
    pub second: u32,
    pub millisecond: u32,
}

/// Camera metadata extracted from a frame's EXIF block, when present.
///
/// Only solve-relevant, non-identifying fields are read: the focal length pair
/// (FOV prior) and the acquisition timestamp (proper-motion epoch, planet
/// positions). GPS tags are deliberately never extracted — the engine has no
/// use for them yet and keeping location data out of the pipeline is part of
/// the PII-minimization stance for collected beta frames.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ExifMeta {
    /// Physical focal length in millimeters (`FocalLength`).
    pub focal_length_mm: Option<f64>,
    /// 35 mm-equivalent focal length in millimeters (`FocalLengthIn35mmFilm`).
    pub focal_length_35mm: Option<f64>,
    /// Digital zoom factor (`DigitalZoomRatio`), when meaningful (> 0).
    pub digital_zoom_ratio: Option<f64>,
    /// Local civil time the exposure started (`DateTimeOriginal`).
    pub datetime_original: Option<FrameTimestamp>,
}

/// Grayscale frame loaded from an image file.
#[derive(Debug, Clone)]
pub struct FrameImage {
    pub width: u32,
    pub height: u32,
    pub gray: Vec<f32>,
    pub source_name: String,
    /// EXIF metadata, if the source file carried a parseable EXIF block.
    pub exif: Option<ExifMeta>,
}

#[derive(Debug, thiserror::Error)]
pub enum ImageInputError {
    #[error("failed to read image file '{path}': {source}")]
    Read {
        path: String,
        source: std::io::Error,
    },
    #[error("failed to open image at '{path}': {source}")]
    Open {
        path: String,
        source: image::ImageError,
    },
    #[error("image at '{path}' is missing a file stem")]
    MissingStem { path: String },
}

impl FrameTimestamp {
    /// Fractional year (e.g. 2011.72 for mid-September 2011).
    #[must_use]
    pub fn to_epoch_years(&self) -> f64 {
        let day_of_year = day_of_year(self.year, self.month, self.day) as f64;
        let days_in_year = if is_leap_year(self.year) {
            366.0
        } else {
            365.0
        };
        let day_fraction = (f64::from(self.hour)
            + f64::from(self.minute) / 60.0
            + (f64::from(self.second) + f64::from(self.millisecond) / 1000.0) / 60.0)
            / 24.0;
        f64::from(self.year) + (day_of_year - 1.0 + day_fraction) / days_in_year
    }

    /// Julian Day (UTC) for this civil timestamp, given the observer's UTC offset in hours.
    ///
    /// `utc_offset_hours` is positive east of Greenwich (e.g. Moscow UTC+4 → `4.0`).
    /// The fields in [`FrameTimestamp`] are interpreted as local civil time; the offset
    /// is subtracted to obtain UTC before calling [`crate::ephem::julian_day_utc`].
    #[must_use]
    pub fn to_jd_utc(&self, utc_offset_hours: f64) -> f64 {
        let local_secs = f64::from(self.hour) * 3600.0
            + f64::from(self.minute) * 60.0
            + f64::from(self.second)
            + f64::from(self.millisecond) / 1000.0;
        let mut utc_secs = local_secs - utc_offset_hours * 3600.0;
        let mut year = self.year;
        let mut month = self.month;
        let mut day = self.day;

        while utc_secs < 0.0 {
            utc_secs += 86_400.0;
            let (y, m, d) = prev_day(year, month, day);
            year = y;
            month = m;
            day = d;
        }
        while utc_secs >= 86_400.0 {
            utc_secs -= 86_400.0;
            let (y, m, d) = next_day(year, month, day);
            year = y;
            month = m;
            day = d;
        }

        let hour = (utc_secs / 3600.0).floor() as u32;
        let rem = utc_secs - f64::from(hour) * 3600.0;
        let minute = (rem / 60.0).floor() as u32;
        let second = rem - f64::from(minute) * 60.0;

        crate::ephem::julian_day_utc(year, month, day, hour, minute, second)
    }
}

fn is_leap_year(year: i32) -> bool {
    (year % 4 == 0 && year % 100 != 0) || year % 400 == 0
}

fn day_of_year(year: i32, month: u32, day: u32) -> u32 {
    const DAYS_BEFORE: [u32; 13] = [0, 0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334];
    let mut doy = DAYS_BEFORE[month as usize] + day;
    if month > 2 && is_leap_year(year) {
        doy += 1;
    }
    doy
}

fn days_in_month(year: i32, month: u32) -> u32 {
    match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 => {
            if is_leap_year(year) {
                29
            } else {
                28
            }
        }
        _ => 30,
    }
}

fn prev_day(year: i32, month: u32, day: u32) -> (i32, u32, u32) {
    if day > 1 {
        return (year, month, day - 1);
    }
    if month > 1 {
        let pm = month - 1;
        return (year, pm, days_in_month(year, pm));
    }
    (year - 1, 12, 31)
}

fn next_day(year: i32, month: u32, day: u32) -> (i32, u32, u32) {
    let dim = days_in_month(year, month);
    if day < dim {
        return (year, month, day + 1);
    }
    if month < 12 {
        return (year, month + 1, 1);
    }
    (year + 1, 1, 1)
}

/// Diagonal of a full 35 mm film frame (36×24 mm) in millimeters.
///
/// Per CIPA DC-008, `FocalLengthIn35mmFilm` is the focal length that would give
/// the same *diagonal* angle of view on a 36×24 mm frame, so the diagonal is
/// the invariant to convert through.
const FILM_DIAGONAL_MM: f64 = 43.266_615_305_567_87;

/// Sanity window for an EXIF-derived horizontal FOV (degrees). Outside of it
/// the metadata is either corrupt or the lens is so extreme (fisheye) that the
/// rectilinear model the solver assumes no longer applies; such frames go down
/// the blind path instead of being steered by a wrong prior.
const EXIF_FOV_MIN_DEG: f64 = 3.0;
const EXIF_FOV_MAX_DEG: f64 = 95.0;

impl ExifMeta {
    /// Parse EXIF out of a raw image file (JPEG/TIFF container).
    ///
    /// Returns `None` when the file has no EXIF block or it cannot be parsed;
    /// metadata problems must never fail the pixel load.
    #[must_use]
    pub fn from_bytes(bytes: &[u8]) -> Option<ExifMeta> {
        let exif = exif::Reader::new()
            .read_from_container(&mut std::io::Cursor::new(bytes))
            .ok()?;
        let rational = |tag: exif::Tag| -> Option<f64> {
            match &exif.get_field(tag, exif::In::PRIMARY)?.value {
                exif::Value::Rational(v) => v.first().map(exif::Rational::to_f64),
                exif::Value::Short(v) => v.first().map(|&s| f64::from(s)),
                exif::Value::Long(v) => v.first().map(|&l| f64::from(l)),
                _ => None,
            }
        };
        let finite_pos = |v: f64| v.is_finite() && v > 0.0;

        Some(ExifMeta {
            focal_length_mm: rational(exif::Tag::FocalLength).filter(|&v| finite_pos(v)),
            focal_length_35mm: rational(exif::Tag::FocalLengthIn35mmFilm)
                .filter(|&v| finite_pos(v)),
            digital_zoom_ratio: rational(exif::Tag::DigitalZoomRatio).filter(|&v| finite_pos(v)),
            datetime_original: exif
                .get_field(exif::Tag::DateTimeOriginal, exif::In::PRIMARY)
                .and_then(|f| match &f.value {
                    exif::Value::Ascii(lines) => lines.first().cloned(),
                    _ => None,
                })
                .and_then(|ascii| exif::DateTime::from_ascii(&ascii).ok())
                .and_then(timestamp_from_exif_datetime),
        })
    }

    /// Horizontal FOV prior (degrees) derived from `FocalLengthIn35mmFilm`.
    ///
    /// The 35 mm-equivalent focal preserves the diagonal angle of view; the
    /// diagonal is split into the horizontal component through the pixel
    /// aspect ratio in linear tan space (rectilinear projection). Per CIPA the
    /// 35 mm equivalent does *not* include digital zoom, so a reported
    /// `DigitalZoomRatio > 1` scales the effective focal.
    #[must_use]
    pub fn fov_x_deg(&self, width: u32, height: u32) -> Option<f64> {
        let f35 = self.focal_length_35mm?;
        let zoom = self.digital_zoom_ratio.filter(|&z| z > 1.0).unwrap_or(1.0);
        let w = f64::from(width);
        let h = f64::from(height);
        let diag_px = (w * w + h * h).sqrt();
        if diag_px <= 0.0 {
            return None;
        }
        let tan_half_diag = FILM_DIAGONAL_MM / (2.0 * f35 * zoom);
        let tan_half_x = tan_half_diag * w / diag_px;
        let fov = 2.0 * tan_half_x.atan().to_degrees();
        (EXIF_FOV_MIN_DEG..=EXIF_FOV_MAX_DEG)
            .contains(&fov)
            .then_some(fov)
    }
}

/// Convert a parsed EXIF datetime into a [`FrameTimestamp`], rejecting fields
/// that are out of calendar range (some cameras write zeroed dates).
fn timestamp_from_exif_datetime(dt: exif::DateTime) -> Option<FrameTimestamp> {
    let (year, month, day) = (i32::from(dt.year), u32::from(dt.month), u32::from(dt.day));
    let (hour, minute, second) = (
        u32::from(dt.hour),
        u32::from(dt.minute),
        u32::from(dt.second),
    );
    if !(1..=12).contains(&month) || day < 1 || day > days_in_month(year, month) {
        return None;
    }
    if hour > 23 || minute > 59 || second > 59 || !(1900..=9999).contains(&year) {
        return None;
    }
    Some(FrameTimestamp {
        year,
        month,
        day,
        hour,
        minute,
        second,
        millisecond: dt.nanosecond.map_or(0, |ns| ns / 1_000_000),
    })
}

impl FrameImage {
    /// Load a frame image and convert pixels to row-major normalized luma values.
    ///
    /// The format is sniffed from the file contents. An EXIF block, when
    /// present and parseable, is captured into [`FrameImage::exif`]; EXIF
    /// problems never fail the load.
    pub fn load(path: &Path) -> Result<Self, ImageInputError> {
        let bytes = std::fs::read(path).map_err(|source| ImageInputError::Read {
            path: path.display().to_string(),
            source,
        })?;
        let image = image::load_from_memory(&bytes).map_err(|source| ImageInputError::Open {
            path: path.display().to_string(),
            source,
        })?;
        let exif = ExifMeta::from_bytes(&bytes);
        let (width, height) = image.dimensions();
        let gray = image
            .to_luma8()
            .into_raw()
            .into_iter()
            .map(|value| f32::from(value) / 255.0)
            .collect();

        let source_name = path
            .file_stem()
            .and_then(|stem| stem.to_str())
            .ok_or_else(|| ImageInputError::MissingStem {
                path: path.display().to_string(),
            })?
            .to_string();

        Ok(Self {
            width,
            height,
            gray,
            source_name,
            exif,
        })
    }

    /// Parse an acquisition timestamp encoded in `source_name`, if present.
    pub fn timestamp_from_name(&self) -> Option<FrameTimestamp> {
        parse_timestamp_from_stem(&self.source_name)
    }

    /// Acquisition timestamp: filename-encoded first (capture rigs put the
    /// authoritative time there), then EXIF `DateTimeOriginal`.
    #[must_use]
    pub fn acquisition_timestamp(&self) -> Option<FrameTimestamp> {
        self.timestamp_from_name()
            .or_else(|| self.exif.as_ref().and_then(|e| e.datetime_original))
    }

    /// Horizontal FOV prior (degrees) from EXIF, if derivable. See
    /// [`ExifMeta::fov_x_deg`].
    #[must_use]
    pub fn exif_fov_deg(&self) -> Option<f64> {
        self.exif
            .as_ref()
            .and_then(|e| e.fov_x_deg(self.width, self.height))
    }
}

fn parse_timestamp_from_stem(stem: &str) -> Option<FrameTimestamp> {
    if let Some(timestamp) = parse_prefix_timestamp(stem) {
        return Some(timestamp);
    }
    parse_cd_timestamp(stem)
}

fn parse_prefix_timestamp(stem: &str) -> Option<FrameTimestamp> {
    // YYYY-MM-DD_HH-MM-SS-mmm
    let year: i32 = stem.get(0..4)?.parse().ok()?;
    if stem.as_bytes().get(4)? != &b'-' {
        return None;
    }
    let month: u32 = stem.get(5..7)?.parse().ok()?;
    if stem.as_bytes().get(7)? != &b'-' {
        return None;
    }
    let day: u32 = stem.get(8..10)?.parse().ok()?;
    if stem.as_bytes().get(10)? != &b'_' {
        return None;
    }
    let hour: u32 = stem.get(11..13)?.parse().ok()?;
    if stem.as_bytes().get(13)? != &b'-' {
        return None;
    }
    let minute: u32 = stem.get(14..16)?.parse().ok()?;
    if stem.as_bytes().get(16)? != &b'-' {
        return None;
    }
    let second: u32 = stem.get(17..19)?.parse().ok()?;
    if stem.as_bytes().get(19)? != &b'-' {
        return None;
    }
    let millisecond: u32 = stem.get(20..23)?.parse().ok()?;

    Some(FrameTimestamp {
        year,
        month,
        day,
        hour,
        minute,
        second,
        millisecond,
    })
}

fn parse_cd_timestamp(stem: &str) -> Option<FrameTimestamp> {
    const PREFIX: &str = "CD_";
    if !stem.starts_with(PREFIX) {
        return None;
    }
    let rest = &stem[PREFIX.len()..];
    let year: i32 = rest.get(0..4)?.parse().ok()?;
    if rest.as_bytes().get(4)? != &b'-' {
        return None;
    }
    let month: u32 = rest.get(5..7)?.parse().ok()?;
    if rest.as_bytes().get(7)? != &b'-' {
        return None;
    }
    let day: u32 = rest.get(8..10)?.parse().ok()?;
    if rest.as_bytes().get(10)? != &b'_' {
        return None;
    }
    let hhmm = rest.get(11..15)?;
    let hour: u32 = hhmm.get(0..2)?.parse().ok()?;
    let minute: u32 = hhmm.get(2..4)?.parse().ok()?;

    Some(FrameTimestamp {
        year,
        month,
        day,
        hour,
        minute,
        second: 0,
        millisecond: 0,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_prefix_timestamp_pattern() {
        let frame = FrameImage {
            width: 1,
            height: 1,
            gray: vec![0.0],
            source_name: "2011-09-20_23-49-51-296_Gain-128_Exp-20m".to_string(),
            exif: None,
        };
        assert_eq!(
            frame.timestamp_from_name(),
            Some(FrameTimestamp {
                year: 2011,
                month: 9,
                day: 20,
                hour: 23,
                minute: 49,
                second: 51,
                millisecond: 296,
            })
        );
    }

    #[test]
    fn parses_cd_timestamp_pattern() {
        let frame = FrameImage {
            width: 1,
            height: 1,
            gray: vec![0.0],
            source_name: "CD_2011-09-19_0020".to_string(),
            exif: None,
        };
        assert_eq!(
            frame.timestamp_from_name(),
            Some(FrameTimestamp {
                year: 2011,
                month: 9,
                day: 19,
                hour: 0,
                minute: 20,
                second: 0,
                millisecond: 0,
            })
        );
    }

    #[test]
    fn returns_none_for_unrecognized_name() {
        let frame = FrameImage {
            width: 1,
            height: 1,
            gray: vec![0.0],
            source_name: "g128_40ms_1s".to_string(),
            exif: None,
        };
        assert_eq!(frame.timestamp_from_name(), None);
    }

    #[test]
    fn epoch_years_for_prefix_timestamp() {
        let frame = FrameImage {
            width: 1,
            height: 1,
            gray: vec![0.0],
            source_name: "2011-09-20_23-49-51-296_Gain-128_Exp-20m".to_string(),
            exif: None,
        };
        let ts = frame.timestamp_from_name().expect("timestamp");
        let epoch = ts.to_epoch_years();
        assert!(
            (2011.71..=2011.73).contains(&epoch),
            "unexpected epoch years: {epoch}"
        );
    }

    #[test]
    fn to_jd_utc_applies_offset() {
        let ts = FrameTimestamp {
            year: 2011,
            month: 9,
            day: 21,
            hour: 4,
            minute: 0,
            second: 0,
            millisecond: 0,
        };
        // Moscow UTC+4: local 04:00 → UTC 00:00 on the same calendar day.
        let jd = ts.to_jd_utc(4.0);
        let expect = crate::ephem::julian_day_utc(2011, 9, 21, 0, 0, 0.0);
        assert!((jd - expect).abs() < 1e-6, "jd={jd} expect={expect}");
    }

    /// Textbook horizontal FOV values: a 28 mm lens on a 36×24 frame covers
    /// 2·atan(18/28) ≈ 65.47° horizontally; in portrait orientation the short
    /// film side maps to the pixel width: 2·atan(12/28) ≈ 46.40°.
    #[test]
    fn fov_from_35mm_equivalent_matches_textbook_values() {
        let meta = ExifMeta {
            focal_length_35mm: Some(28.0),
            ..ExifMeta::default()
        };
        let landscape = meta.fov_x_deg(3000, 2000).expect("landscape fov");
        assert!((landscape - 65.47).abs() < 0.05, "landscape {landscape}");
        let portrait = meta.fov_x_deg(2000, 3000).expect("portrait fov");
        assert!((portrait - 46.40).abs() < 0.05, "portrait {portrait}");

        // Typical phone main camera: f35 = 26 mm, 4:3 sensor.
        let phone = ExifMeta {
            focal_length_35mm: Some(26.0),
            ..ExifMeta::default()
        };
        let fov = phone.fov_x_deg(4032, 3024).expect("phone fov");
        assert!((fov - 67.31).abs() < 0.05, "phone {fov}");
    }

    #[test]
    fn fov_applies_digital_zoom_and_rejects_nonsense() {
        let zoomed = ExifMeta {
            focal_length_35mm: Some(26.0),
            digital_zoom_ratio: Some(2.0),
            ..ExifMeta::default()
        };
        let base = ExifMeta {
            focal_length_35mm: Some(52.0),
            ..ExifMeta::default()
        };
        let a = zoomed.fov_x_deg(4032, 3024).expect("zoomed");
        let b = base.fov_x_deg(4032, 3024).expect("base");
        assert!((a - b).abs() < 1e-9, "zoom 2x must equal doubled focal");

        // Zoom ratios ≤ 1 are reported by many cameras and mean "no zoom".
        let unity = ExifMeta {
            focal_length_35mm: Some(26.0),
            digital_zoom_ratio: Some(1.0),
            ..ExifMeta::default()
        };
        assert_eq!(
            unity.fov_x_deg(4032, 3024),
            ExifMeta {
                focal_length_35mm: Some(26.0),
                ..ExifMeta::default()
            }
            .fov_x_deg(4032, 3024)
        );

        let none = ExifMeta::default();
        assert_eq!(none.fov_x_deg(4032, 3024), None);

        // Fisheye-range result (f35 = 6 mm → ~149° diagonal) is out of the
        // rectilinear sanity window and must not become a prior.
        let fisheye = ExifMeta {
            focal_length_35mm: Some(6.0),
            ..ExifMeta::default()
        };
        assert_eq!(fisheye.fov_x_deg(4000, 3000), None);
    }

    #[test]
    fn exif_datetime_conversion_validates_ranges() {
        let good = exif::DateTime::from_ascii(b"2023:08:15 21:04:33").expect("parse");
        let ts = timestamp_from_exif_datetime(good).expect("timestamp");
        assert_eq!(
            ts,
            FrameTimestamp {
                year: 2023,
                month: 8,
                day: 15,
                hour: 21,
                minute: 4,
                second: 33,
                millisecond: 0,
            }
        );

        // Zeroed dates (written by some cameras) must be rejected, not panic
        // later in day-of-year math.
        let zeroed = exif::DateTime::from_ascii(b"0000:00:00 00:00:00").expect("parse");
        assert_eq!(timestamp_from_exif_datetime(zeroed), None);
    }

    /// Minimal little-endian TIFF/EXIF blob: IFD0 → ExifIFD with
    /// FocalLengthIn35mmFilm, DigitalZoomRatio and DateTimeOriginal.
    fn exif_tiff_blob(f35: u16, zoom: (u32, u32), datetime: &[u8; 19]) -> Vec<u8> {
        let mut b: Vec<u8> = Vec::new();
        b.extend_from_slice(b"II\x2a\x00");
        b.extend_from_slice(&8u32.to_le_bytes()); // IFD0 offset

        // IFD0: single entry pointing at the Exif sub-IFD.
        let exif_ifd_offset = 8 + 2 + 12 + 4; // header→IFD0(1 entry)→next ptr
        b.extend_from_slice(&1u16.to_le_bytes());
        b.extend_from_slice(&0x8769u16.to_le_bytes()); // ExifIFDPointer
        b.extend_from_slice(&4u16.to_le_bytes()); // LONG
        b.extend_from_slice(&1u32.to_le_bytes());
        b.extend_from_slice(&(exif_ifd_offset as u32).to_le_bytes());
        b.extend_from_slice(&0u32.to_le_bytes()); // no next IFD

        // Exif IFD: 3 entries (ascending tag order), then external data.
        let n = 3u16;
        let data_start = exif_ifd_offset + 2 + 12 * usize::from(n) + 4;
        b.extend_from_slice(&n.to_le_bytes());

        // 0x9003 DateTimeOriginal, ASCII, 20 bytes (external).
        b.extend_from_slice(&0x9003u16.to_le_bytes());
        b.extend_from_slice(&2u16.to_le_bytes());
        b.extend_from_slice(&20u32.to_le_bytes());
        b.extend_from_slice(&(data_start as u32).to_le_bytes());

        // 0xA404 DigitalZoomRatio, RATIONAL, 1 (external, 8 bytes).
        b.extend_from_slice(&0xA404u16.to_le_bytes());
        b.extend_from_slice(&5u16.to_le_bytes());
        b.extend_from_slice(&1u32.to_le_bytes());
        b.extend_from_slice(&((data_start + 20) as u32).to_le_bytes());

        // 0xA405 FocalLengthIn35mmFilm, SHORT, 1 (inline).
        b.extend_from_slice(&0xA405u16.to_le_bytes());
        b.extend_from_slice(&3u16.to_le_bytes());
        b.extend_from_slice(&1u32.to_le_bytes());
        b.extend_from_slice(&f35.to_le_bytes());
        b.extend_from_slice(&[0u8, 0]);

        b.extend_from_slice(&0u32.to_le_bytes()); // no next IFD

        assert_eq!(b.len(), data_start);
        b.extend_from_slice(datetime);
        b.push(0); // ASCII NUL terminator → 20 bytes
        b.extend_from_slice(&zoom.0.to_le_bytes());
        b.extend_from_slice(&zoom.1.to_le_bytes());
        b
    }

    /// Wrap a TIFF/EXIF blob into an APP1 segment spliced right after the SOI
    /// marker of a real JPEG produced by the `image` encoder.
    fn jpeg_with_exif(width: u32, height: u32, tiff: &[u8]) -> Vec<u8> {
        use image::{DynamicImage, ImageBuffer, Luma};

        let buffer: ImageBuffer<Luma<u8>, Vec<u8>> =
            ImageBuffer::from_fn(width, height, |x, _| Luma([if x == 0 { 200 } else { 10 }]));
        let mut jpeg = Vec::new();
        DynamicImage::ImageLuma8(buffer)
            .write_to(
                &mut std::io::Cursor::new(&mut jpeg),
                image::ImageFormat::Jpeg,
            )
            .expect("encode jpeg");
        assert_eq!(&jpeg[..2], &[0xFF, 0xD8], "SOI expected");

        let payload_len = 2 + 6 + tiff.len(); // length field + "Exif\0\0" + TIFF
        let mut out = Vec::with_capacity(jpeg.len() + payload_len + 2);
        out.extend_from_slice(&jpeg[..2]);
        out.extend_from_slice(&[0xFF, 0xE1]);
        out.extend_from_slice(&(payload_len as u16).to_be_bytes());
        out.extend_from_slice(b"Exif\0\0");
        out.extend_from_slice(tiff);
        out.extend_from_slice(&jpeg[2..]);
        out
    }

    #[test]
    fn load_reads_exif_fov_and_timestamp_from_jpeg() {
        let tiff = exif_tiff_blob(26, (3, 2), b"2024:11:30 22:15:07");
        let bytes = jpeg_with_exif(64, 48, &tiff);

        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("phone_shot.jpg");
        std::fs::write(&path, &bytes).expect("write jpeg");

        let frame = FrameImage::load(&path).expect("load jpeg with exif");
        assert_eq!((frame.width, frame.height), (64, 48));
        let exif = frame.exif.clone().expect("exif parsed");
        assert_eq!(exif.focal_length_35mm, Some(26.0));
        assert_eq!(exif.digital_zoom_ratio, Some(1.5));
        assert_eq!(
            exif.datetime_original,
            Some(FrameTimestamp {
                year: 2024,
                month: 11,
                day: 30,
                hour: 22,
                minute: 15,
                second: 7,
                millisecond: 0,
            })
        );
        assert_eq!(frame.acquisition_timestamp(), exif.datetime_original);

        // f35_eff = 26·1.5 = 39 mm on a 4:3 frame.
        let fov = frame.exif_fov_deg().expect("fov");
        let expect = 2.0
            * ((FILM_DIAGONAL_MM / (2.0 * 39.0)) * 0.8)
                .atan()
                .to_degrees();
        assert!((fov - expect).abs() < 1e-9, "fov {fov} vs {expect}");
    }

    #[test]
    fn load_without_exif_yields_none_meta() {
        use image::{DynamicImage, ImageBuffer, Luma};

        let buffer: ImageBuffer<Luma<u8>, Vec<u8>> = ImageBuffer::new(8, 8);
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("plain.png");
        DynamicImage::ImageLuma8(buffer).save(&path).expect("save");

        let frame = FrameImage::load(&path).expect("load");
        assert_eq!(frame.exif, None);
        assert_eq!(frame.acquisition_timestamp(), None);
        assert_eq!(frame.exif_fov_deg(), None);
    }

    #[test]
    fn loads_16_bit_grayscale_tiff() {
        use image::{DynamicImage, ImageBuffer, Luma};

        let mut buffer: ImageBuffer<Luma<u16>, Vec<u16>> = ImageBuffer::new(4, 3);
        for (x, y, pixel) in buffer.enumerate_pixels_mut() {
            *pixel = if x == 0 && y == 0 {
                Luma([u16::MAX])
            } else {
                Luma([0])
            };
        }

        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("frame.tiff");
        DynamicImage::ImageLuma16(buffer)
            .save(&path)
            .expect("save 16-bit tiff");

        let frame = FrameImage::load(&path).expect("load 16-bit tiff");
        assert_eq!(frame.width, 4);
        assert_eq!(frame.height, 3);
        assert!(
            (frame.gray[0] - 1.0).abs() < 1e-3,
            "bright pixel: {}",
            frame.gray[0]
        );
        assert!(
            (frame.gray[1] - 0.0).abs() < 1e-3,
            "dark pixel: {}",
            frame.gray[1]
        );
    }

    #[test]
    fn loads_8_bit_rgb_tiff() {
        use image::{DynamicImage, ImageBuffer, Rgb};

        let buffer: ImageBuffer<Rgb<u8>, Vec<u8>> =
            ImageBuffer::from_fn(2, 2, |x, _| Rgb([if x == 0 { 255 } else { 0 }; 3]));

        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("frame.tiff");
        DynamicImage::ImageRgb8(buffer)
            .save(&path)
            .expect("save rgb tiff");

        let frame = FrameImage::load(&path).expect("load rgb tiff");
        assert_eq!(frame.width, 2);
        assert_eq!(frame.height, 2);
        assert_eq!(frame.gray.len(), 4);
        assert!(
            (frame.gray[0] - 1.0).abs() < 1e-3,
            "white pixel: {}",
            frame.gray[0]
        );
        assert!(
            (frame.gray[1] - 0.0).abs() < 1e-3,
            "black pixel: {}",
            frame.gray[1]
        );
    }
}
