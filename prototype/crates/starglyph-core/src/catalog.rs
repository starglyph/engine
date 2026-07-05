use std::f64::consts::PI;
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;

use csv::StringRecord;
use flate2::read::GzDecoder;

const RAD_TO_DEG: f64 = 180.0 / PI;

/// A star entry from the HYG catalog.
#[derive(Debug, Clone, PartialEq)]
pub struct Star {
    pub id: u32,
    pub hip: Option<u32>,
    pub proper: Option<String>,
    pub ra_deg: f64,
    pub dec_deg: f64,
    pub mag: f32,
    pub con: Option<String>,
    /// Proper motion in RA (μα·cosδ convention), radians per year.
    pub pmra_rad_yr: f64,
    /// Proper motion in Dec, radians per year.
    pub pmdec_rad_yr: f64,
}

impl Star {
    /// Unit direction on the celestial sphere at the given epoch (J2000 + years).
    #[must_use]
    pub fn unit_at_epoch(&self, epoch_years: f64) -> [f64; 3] {
        let dt = epoch_years - 2000.0;
        let alpha = self.ra_deg * (PI / 180.0);
        let delta = self.dec_deg * (PI / 180.0);
        let east = [-alpha.sin(), alpha.cos(), 0.0];
        let north = [
            -delta.sin() * alpha.cos(),
            -delta.sin() * alpha.sin(),
            delta.cos(),
        ];
        let u = radec_to_unit(self.ra_deg, self.dec_deg);
        let shifted = [
            u[0] + dt * (self.pmra_rad_yr * east[0] + self.pmdec_rad_yr * north[0]),
            u[1] + dt * (self.pmra_rad_yr * east[1] + self.pmdec_rad_yr * north[1]),
            u[2] + dt * (self.pmra_rad_yr * east[2] + self.pmdec_rad_yr * north[2]),
        ];
        normalize_unit(shifted)
    }
}

#[must_use]
fn radec_to_unit(ra_deg: f64, dec_deg: f64) -> [f64; 3] {
    let ra = ra_deg * (PI / 180.0);
    let dec = dec_deg * (PI / 180.0);
    let cos_dec = dec.cos();
    [cos_dec * ra.cos(), cos_dec * ra.sin(), dec.sin()]
}

#[must_use]
fn normalize_unit(v: [f64; 3]) -> [f64; 3] {
    let norm = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if norm <= f64::EPSILON {
        [0.0, 0.0, 0.0]
    } else {
        [v[0] / norm, v[1] / norm, v[2] / norm]
    }
}

/// In-memory HYG star catalog sorted by ascending magnitude.
#[derive(Debug, Clone)]
pub struct Catalog {
    stars: Vec<Star>,
}

#[derive(Debug, thiserror::Error)]
pub enum CatalogError {
    #[error("failed to open catalog at '{path}': {source}")]
    Open {
        path: String,
        source: std::io::Error,
    },
    #[error("failed to read catalog csv header from '{path}'")]
    ReadHeader { path: String },
    #[error("catalog csv at '{path}' is missing required column '{column}'")]
    MissingColumn { path: String, column: &'static str },
    #[error("catalog row {row} in '{path}' has invalid value in column '{column}': '{value}'")]
    InvalidField {
        path: String,
        row: usize,
        column: &'static str,
        value: String,
    },
}

impl Catalog {
    /// Load a HYG catalog from a plain `.csv` or gzip-compressed `.csv.gz` file.
    pub fn load(path: &Path) -> Result<Self, CatalogError> {
        let reader = open_reader(path)?;
        let mut csv_reader = csv::ReaderBuilder::new()
            .has_headers(true)
            .from_reader(reader);
        let headers = csv_reader
            .headers()
            .map_err(|_| CatalogError::ReadHeader {
                path: path.display().to_string(),
            })?
            .clone();
        let column_names = header_names(&headers);
        let idx_id = find_column(&column_names, "id").ok_or(CatalogError::MissingColumn {
            path: path.display().to_string(),
            column: "id",
        })?;
        let idx_hip = find_column(&column_names, "hip");
        let idx_proper = find_column(&column_names, "proper");
        let idx_ra = find_column(&column_names, "ra").ok_or(CatalogError::MissingColumn {
            path: path.display().to_string(),
            column: "ra",
        })?;
        let idx_dec = find_column(&column_names, "dec").ok_or(CatalogError::MissingColumn {
            path: path.display().to_string(),
            column: "dec",
        })?;
        let idx_mag = find_column(&column_names, "mag").ok_or(CatalogError::MissingColumn {
            path: path.display().to_string(),
            column: "mag",
        })?;
        let idx_rarad = find_column(&column_names, "rarad");
        let idx_decrad = find_column(&column_names, "decrad");
        let idx_con = find_column(&column_names, "con");
        let idx_pmrarad = find_column(&column_names, "pmrarad");
        let idx_pmdecrad = find_column(&column_names, "pmdecrad");

        let path_display = path.display().to_string();
        let mut stars = Vec::new();
        for (row_idx, record) in csv_reader.records().enumerate() {
            let record = record.map_err(|err| CatalogError::InvalidField {
                path: path_display.clone(),
                row: row_idx + 2,
                column: "record",
                value: err.to_string(),
            })?;
            let row_number = row_idx + 2;

            let id = parse_u32_field(&record, idx_id, &path_display, row_number, "id")?;
            if id == 0 {
                continue;
            }

            let mag = match parse_f32_optional(&record, idx_mag) {
                Some(value) => value,
                None => continue,
            };

            let (ra_deg, dec_deg) = match parse_coordinates(
                &record,
                idx_ra,
                idx_dec,
                idx_rarad,
                idx_decrad,
                &path_display,
                row_number,
            )? {
                Some(coords) => coords,
                None => continue,
            };

            let hip = idx_hip.and_then(|idx| parse_u32_optional(&record, idx));
            let proper = idx_proper.and_then(|idx| parse_optional_string(&record, idx));
            let con = idx_con.and_then(|idx| parse_optional_string(&record, idx));
            let pmra_rad_yr = idx_pmrarad
                .and_then(|idx| parse_f64_optional(&record, idx))
                .unwrap_or(0.0);
            let pmdec_rad_yr = idx_pmdecrad
                .and_then(|idx| parse_f64_optional(&record, idx))
                .unwrap_or(0.0);

            stars.push(Star {
                id,
                hip,
                proper,
                ra_deg,
                dec_deg,
                mag,
                con,
                pmra_rad_yr,
                pmdec_rad_yr,
            });
        }

        stars.sort_by(|left, right| {
            left.mag
                .partial_cmp(&right.mag)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(Self { stars })
    }

    /// All catalog stars in ascending magnitude order.
    pub fn stars(&self) -> &[Star] {
        &self.stars
    }

    /// Stars brighter than `mag` (strictly lower magnitude value).
    pub fn brighter_than(&self, mag: f32) -> &[Star] {
        let mut lo = 0;
        let mut hi = self.stars.len();
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            if self.stars[mid].mag < mag {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        &self.stars[..lo]
    }
}

fn open_reader(path: &Path) -> Result<BufReader<Box<dyn Read>>, CatalogError> {
    let file = File::open(path).map_err(|source| CatalogError::Open {
        path: path.display().to_string(),
        source,
    })?;
    let reader: Box<dyn Read> = if path
        .extension()
        .and_then(|ext| ext.to_str())
        .is_some_and(|ext| ext.eq_ignore_ascii_case("gz"))
    {
        Box::new(GzDecoder::new(file))
    } else {
        Box::new(file)
    };
    Ok(BufReader::new(reader))
}

fn header_names(headers: &StringRecord) -> Vec<String> {
    headers
        .iter()
        .map(|field| field.trim().to_ascii_lowercase())
        .collect()
}

fn find_column(names: &[String], column: &str) -> Option<usize> {
    names.iter().position(|name| name == column)
}

fn field_value(record: &StringRecord, index: usize) -> Option<&str> {
    record
        .get(index)
        .map(str::trim)
        .filter(|value| !value.is_empty())
}

fn parse_optional_string(record: &StringRecord, index: usize) -> Option<String> {
    field_value(record, index).map(str::to_string)
}

fn parse_u32_optional(record: &StringRecord, index: usize) -> Option<u32> {
    let value = field_value(record, index)?;
    value.parse().ok()
}

fn parse_f32_optional(record: &StringRecord, index: usize) -> Option<f32> {
    let value = field_value(record, index)?;
    value.parse().ok()
}

fn parse_u32_field(
    record: &StringRecord,
    index: usize,
    path: &str,
    row: usize,
    column: &'static str,
) -> Result<u32, CatalogError> {
    let value = field_value(record, index).ok_or_else(|| CatalogError::InvalidField {
        path: path.to_string(),
        row,
        column,
        value: String::new(),
    })?;
    value.parse().map_err(|_| CatalogError::InvalidField {
        path: path.to_string(),
        row,
        column,
        value: value.to_string(),
    })
}

fn parse_coordinates(
    record: &StringRecord,
    idx_ra: usize,
    idx_dec: usize,
    idx_rarad: Option<usize>,
    idx_decrad: Option<usize>,
    path: &str,
    row: usize,
) -> Result<Option<(f64, f64)>, CatalogError> {
    if let (Some(rarad_idx), Some(decrad_idx)) = (idx_rarad, idx_decrad) {
        if let (Some(ra_rad), Some(dec_rad)) = (
            parse_f64_optional(record, rarad_idx),
            parse_f64_optional(record, decrad_idx),
        ) {
            return Ok(Some((ra_rad * RAD_TO_DEG, dec_rad * RAD_TO_DEG)));
        }
    }

    let ra_hours = match parse_f64_optional(record, idx_ra) {
        Some(value) => value,
        None => return Ok(None),
    };
    let dec_deg = match parse_f64_optional(record, idx_dec) {
        Some(value) => value,
        None => return Ok(None),
    };

    if !ra_hours.is_finite() || !dec_deg.is_finite() {
        return Err(CatalogError::InvalidField {
            path: path.to_string(),
            row,
            column: "ra/dec",
            value: format!("{ra_hours},{dec_deg}"),
        });
    }

    Ok(Some((ra_hours * 15.0, dec_deg)))
}

fn parse_f64_optional(record: &StringRecord, index: usize) -> Option<f64> {
    let value = field_value(record, index)?;
    value.parse().ok()
}

#[cfg(test)]
mod tests {
    use std::io::Write;

    use super::*;
    use flate2::write::GzEncoder;
    use flate2::Compression;
    use tempfile::NamedTempFile;

    const HYG_HEADER: &str = "id,hip,hd,hr,gl,bf,proper,ra,dec,dist,pmra,pmdec,rv,mag,absmag,spect,ci,x,y,z,vx,vy,vz,rarad,decrad,pmrarad,pmdecrad,bayer,flam,con,comp,comp_primary,base,lum,var,var_min,var_max\n";

    fn hyg_row(fields: [&str; 37]) -> String {
        fields.join(",") + "\n"
    }

    fn write_catalog_csv(contents: &str) -> NamedTempFile {
        let mut file = NamedTempFile::new().expect("temp file");
        write!(file, "{HYG_HEADER}{contents}").expect("write catalog");
        file
    }

    fn write_catalog_gz(contents: &str) -> NamedTempFile {
        let mut file = NamedTempFile::with_suffix(".csv.gz").expect("temp file");
        let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
        write!(encoder, "{HYG_HEADER}{contents}").expect("write gzip catalog");
        let compressed = encoder.finish().expect("finish gzip");
        file.write_all(&compressed)
            .expect("write compressed catalog");
        file
    }

    #[test]
    fn parses_rarad_and_decrad_columns() {
        let file = write_catalog_csv(&hyg_row([
            "1", "1", "", "", "", "", "TestStar", "0.0", "0.0", "", "", "", "", "2.5", "", "", "",
            "", "", "", "", "", "", "1.0", "0.5", "", "", "", "", "CMa", "", "", "", "", "", "",
            "",
        ]));
        let catalog = Catalog::load(file.path()).expect("load catalog");
        assert_eq!(catalog.stars().len(), 1);
        let star = &catalog.stars()[0];
        assert!((star.ra_deg - 57.2957795).abs() < 1e-4);
        assert!((star.dec_deg - 28.6478898).abs() < 1e-4);
        assert_eq!(star.proper.as_deref(), Some("TestStar"));
        assert_eq!(star.con.as_deref(), Some("CMa"));
    }

    #[test]
    fn falls_back_to_ra_hours_and_dec_degrees() {
        let file = write_catalog_csv(&hyg_row([
            "2", "", "", "", "", "", "Fallback", "6.0", "-10.0", "", "", "", "", "3.0", "", "", "",
            "", "", "", "", "", "", "", "", "", "", "", "", "And", "", "", "", "", "", "", "",
        ]));
        let catalog = Catalog::load(file.path()).expect("load catalog");
        assert_eq!(catalog.stars().len(), 1);
        let star = &catalog.stars()[0];
        assert!((star.ra_deg - 90.0).abs() < 1e-6);
        assert!((star.dec_deg + 10.0).abs() < 1e-6);
    }

    #[test]
    fn skips_sun_and_unparsable_rows() {
        let file = write_catalog_csv(&format!(
            "{}{}{}{}{}",
            hyg_row([
                "0", "", "", "", "", "", "Sol", "0.0", "0.0", "", "", "", "", "-26.7", "", "", "",
                "", "", "", "", "", "", "0.0", "0.0", "", "", "", "", "", "", "", "", "", "", "",
                "",
            ]),
            hyg_row([
                "3", "", "", "", "", "", "", "bad", "1.0", "", "", "", "", "4.0", "", "", "", "",
                "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "",
            ]),
            hyg_row([
                "4", "", "", "", "", "", "", "6.0", "5.0", "", "", "", "", "", "", "", "", "", "",
                "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "",
            ]),
            hyg_row([
                "5", "", "", "", "", "", "Bright", "1.0", "1.0", "", "", "", "", "1.0", "", "", "",
                "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "",
            ]),
            hyg_row([
                "6", "", "", "", "", "", "Dim", "2.0", "2.0", "", "", "", "", "5.0", "", "", "",
                "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "",
            ]),
        ));
        let catalog = Catalog::load(file.path()).expect("load catalog");
        assert_eq!(catalog.stars().len(), 2);
        assert_eq!(catalog.stars()[0].mag, 1.0);
        assert_eq!(catalog.stars()[1].mag, 5.0);
        assert_eq!(catalog.brighter_than(3.0).len(), 1);
    }

    #[test]
    fn loads_gzip_catalog_by_extension() {
        let file = write_catalog_gz(&hyg_row([
            "7", "", "", "", "", "", "Gzip", "3.0", "4.0", "", "", "", "", "2.0", "", "", "", "",
            "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "",
        ]));
        let catalog = Catalog::load(file.path()).expect("load gzip catalog");
        assert_eq!(catalog.stars().len(), 1);
        assert_eq!(catalog.stars()[0].id, 7);
    }

    #[test]
    fn loads_real_hyg_catalog_when_present() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../data/catalogs/hyg_v3.csv");
        if !path.exists() {
            return;
        }
        let catalog = Catalog::load(&path).expect("load real catalog");
        assert!(catalog.stars().len() >= 100_000);
        let sirius = catalog
            .stars()
            .iter()
            .find(|star| star.proper.as_deref() == Some("Sirius") && star.mag < -1.0);
        assert!(sirius.is_some(), "expected Sirius with mag < -1.0");
    }

    #[test]
    fn barnard_star_proper_motion_shift() {
        use crate::geom::angular_sep;

        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../data/catalogs/hyg_v3.csv");
        if !path.exists() {
            return;
        }
        let catalog = Catalog::load(&path).expect("load real catalog");
        let barnard = catalog
            .stars()
            .iter()
            .find(|star| star.hip == Some(87_937))
            .expect("Barnard's Star (HIP 87937)");
        let j2000 = barnard.unit_at_epoch(2000.0);
        let epoch = barnard.unit_at_epoch(2011.72);
        let shift_deg = angular_sep(j2000, epoch);
        let shift_arcmin = shift_deg * 60.0;
        assert!(
            (1.8..=2.3).contains(&shift_arcmin),
            "Barnard PM shift {shift_arcmin:.3}′ outside [1.8, 2.3]′"
        );
    }
}
