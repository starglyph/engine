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

/// Grayscale frame loaded from an image file.
#[derive(Debug, Clone)]
pub struct FrameImage {
    pub width: u32,
    pub height: u32,
    pub gray: Vec<f32>,
    pub source_name: String,
}

#[derive(Debug, thiserror::Error)]
pub enum ImageInputError {
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

impl FrameImage {
    /// Load a frame image and convert pixels to row-major normalized luma values.
    pub fn load(path: &Path) -> Result<Self, ImageInputError> {
        let image = image::open(path).map_err(|source| ImageInputError::Open {
            path: path.display().to_string(),
            source,
        })?;
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
        })
    }

    /// Parse an acquisition timestamp encoded in `source_name`, if present.
    pub fn timestamp_from_name(&self) -> Option<FrameTimestamp> {
        parse_timestamp_from_stem(&self.source_name)
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
