use std::f32::consts::PI;
use std::path::Path;

use anyhow::{Context, Result};
use csv::StringRecord;
use serde::{Deserialize, Serialize};

use crate::config::CatalogConfig;

/// Upstream HYG CSV stores `ra` in decimal hours; `Star` uses degrees. When `rarad`/`decrad`
/// columns exist (full Astronomy Nexus export), use them to avoid unit ambiguity.
const RAD_TO_DEG: f32 = 180.0_f32 / PI;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Star {
    pub id: String,
    pub ra_deg: f32,
    pub dec_deg: f32,
    pub mag_v: f32,
}

#[derive(Debug, thiserror::Error)]
pub enum CatalogError {
    #[error("catalog csv was not found at '{path}'")]
    MissingCatalogFile { path: String },
    #[error("catalog csv is missing required column '{column}'")]
    MissingColumn { column: &'static str },
    #[error("catalog row {row} has invalid float value in column '{column}': '{value}'")]
    InvalidFloat {
        row: usize,
        column: &'static str,
        value: String,
    },
}

pub fn load_catalog(config: &CatalogConfig) -> Result<Vec<Star>> {
    if let Some(csv_path) = &config.csv_path {
        if !csv_path.exists() {
            return Err(CatalogError::MissingCatalogFile {
                path: csv_path.display().to_string(),
            }
            .into());
        }
        return load_hyg_csv(csv_path);
    }
    Ok(baseline_catalog())
}

fn load_hyg_csv(path: &Path) -> Result<Vec<Star>> {
    let mut reader = csv::ReaderBuilder::new()
        .has_headers(true)
        .from_path(path)
        .with_context(|| format!("failed to open catalog csv '{}'", path.display()))?;
    let headers = reader
        .headers()
        .context("failed to read catalog csv header")?
        .clone();
    let names = header_field_names(&headers);
    let idx_id = find_column_index(&names, &["id", "star_id", "hip"])?;
    let idx_hip = find_column_index(&names, &["hip"]).ok();
    let idx_ra = find_column_index(&names, &["ra", "ra_deg"])?;
    let idx_dec = find_column_index(&names, &["dec", "dec_deg"])?;
    let idx_mag = find_column_index(&names, &["mag", "mag_v"])?;
    let idx_rarad = find_column_index_optional(&names, &["rarad"]);
    let idx_decrad = find_column_index_optional(&names, &["decrad"]);

    let mut stars = Vec::new();
    for (row_idx, row) in reader.records().enumerate() {
        let row =
            row.with_context(|| format!("failed to read csv record at row {}", row_idx + 2))?;
        let row_number = row_idx + 2;
        let mut id = parse_string(&row, idx_id);
        if id.is_empty() {
            if let Some(ih) = idx_hip {
                id = parse_string(&row, ih);
            }
        }
        if id.is_empty() {
            continue;
        }
        let (ra_deg, dec_deg) = match (idx_rarad, idx_decrad) {
            (Some(ir), Some(idr)) => {
                let ra_rad = parse_f32(&row, ir, row_number, "rarad")?;
                let dec_rad = parse_f32(&row, idr, row_number, "decrad")?;
                (ra_rad * RAD_TO_DEG, dec_rad * RAD_TO_DEG)
            }
            (Some(ir), None) => {
                let ra_rad = parse_f32(&row, ir, row_number, "rarad")?;
                let dec_deg = parse_f32(&row, idx_dec, row_number, "dec")?;
                (ra_rad * RAD_TO_DEG, dec_deg)
            }
            (None, _) => {
                let ra_deg = parse_f32(&row, idx_ra, row_number, "ra")?;
                let dec_deg = parse_f32(&row, idx_dec, row_number, "dec")?;
                (ra_deg, dec_deg)
            }
        };
        let mag_v = parse_f32(&row, idx_mag, row_number, "mag")?;
        stars.push(Star {
            id,
            ra_deg,
            dec_deg,
            mag_v,
        });
    }
    Ok(stars)
}

fn header_field_names(headers: &StringRecord) -> Vec<String> {
    headers
        .iter()
        .map(|h| h.trim().trim_start_matches('\u{feff}').to_string())
        .collect()
}

fn find_column_index(headers: &[String], candidates: &[&'static str]) -> Result<usize> {
    find_column_index_optional(headers, candidates).ok_or_else(|| {
        CatalogError::MissingColumn {
            column: candidates[0],
        }
        .into()
    })
}

fn find_column_index_optional(headers: &[String], candidates: &[&'static str]) -> Option<usize> {
    for candidate in candidates {
        if let Some(idx) = headers
            .iter()
            .position(|column| column.eq_ignore_ascii_case(candidate))
        {
            return Some(idx);
        }
    }
    None
}

fn parse_string(row: &StringRecord, idx: usize) -> String {
    row.get(idx).map(str::trim).unwrap_or_default().to_string()
}

fn parse_f32(
    row: &StringRecord,
    idx: usize,
    row_number: usize,
    column: &'static str,
) -> Result<f32> {
    let value = row.get(idx).map(str::trim).unwrap_or_default();
    value.parse::<f32>().map_err(|_| {
        CatalogError::InvalidFloat {
            row: row_number,
            column,
            value: value.to_string(),
        }
        .into()
    })
}

#[must_use]
pub fn baseline_catalog() -> Vec<Star> {
    vec![
        Star {
            id: "sirius".to_string(),
            ra_deg: 101.287,
            dec_deg: -16.716,
            mag_v: -1.46,
        },
        Star {
            id: "canopus".to_string(),
            ra_deg: 95.987,
            dec_deg: -52.696,
            mag_v: -0.72,
        },
        Star {
            id: "arcturus".to_string(),
            ra_deg: 213.915,
            dec_deg: 19.182,
            mag_v: -0.05,
        },
        Star {
            id: "vega".to_string(),
            ra_deg: 279.234,
            dec_deg: 38.783,
            mag_v: 0.03,
        },
        Star {
            id: "capella".to_string(),
            ra_deg: 79.172,
            dec_deg: 45.997,
            mag_v: 0.08,
        },
        Star {
            id: "rigel".to_string(),
            ra_deg: 78.634,
            dec_deg: -8.201,
            mag_v: 0.12,
        },
        Star {
            id: "procyon".to_string(),
            ra_deg: 114.825,
            dec_deg: 5.225,
            mag_v: 0.38,
        },
        Star {
            id: "betelgeuse".to_string(),
            ra_deg: 88.793,
            dec_deg: 7.407,
            mag_v: 0.42,
        },
        Star {
            id: "achernar".to_string(),
            ra_deg: 24.429,
            dec_deg: -57.237,
            mag_v: 0.46,
        },
        Star {
            id: "hadar".to_string(),
            ra_deg: 210.955,
            dec_deg: -60.373,
            mag_v: 0.61,
        },
        Star {
            id: "altair".to_string(),
            ra_deg: 297.696,
            dec_deg: 8.868,
            mag_v: 0.77,
        },
        Star {
            id: "aldebaran".to_string(),
            ra_deg: 68.98,
            dec_deg: 16.509,
            mag_v: 0.86,
        },
        Star {
            id: "antares".to_string(),
            ra_deg: 247.351,
            dec_deg: -26.432,
            mag_v: 1.06,
        },
        Star {
            id: "pollux".to_string(),
            ra_deg: 113.65,
            dec_deg: 28.026,
            mag_v: 1.14,
        },
        Star {
            id: "fomalhaut".to_string(),
            ra_deg: 344.412,
            dec_deg: -29.622,
            mag_v: 1.16,
        },
        Star {
            id: "deneb".to_string(),
            ra_deg: 310.358,
            dec_deg: 45.281,
            mag_v: 1.25,
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn parses_hyg_like_csv_catalog() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let csv_path = dir.path().join("hyg_v3.csv");
        fs::write(
            &csv_path,
            "id,ra,dec,mag\nsirius,101.287,-16.716,-1.46\nvega,279.234,38.783,0.03\n",
        )?;

        let stars = load_catalog(&CatalogConfig {
            csv_path: Some(csv_path),
            name: "hyg-v3".to_string(),
            subset: "test".to_string(),
            license: "CC BY-SA 4.0".to_string(),
        })?;

        assert_eq!(stars.len(), 2);
        assert_eq!(stars[0].id, "sirius");
        assert_eq!(stars[1].id, "vega");
        Ok(())
    }

    #[test]
    fn parses_hyg_rarad_decrad_columns() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let csv_path = dir.path().join("hyg_v3.csv");
        fs::write(
            &csv_path,
            "id,ra,dec,mag,rarad,decrad\n\
             betel,5.919529,7.407063,0.42,1.5497291183713153,0.12927763169419373\n",
        )?;

        let stars = load_catalog(&CatalogConfig {
            csv_path: Some(csv_path),
            name: "hyg-v3".to_string(),
            subset: "test".to_string(),
            license: "CC BY-SA 4.0".to_string(),
        })?;

        assert_eq!(stars.len(), 1);
        assert!((stars[0].ra_deg - 88.792_94).abs() < 0.01);
        assert!((stars[0].dec_deg - 7.407_063).abs() < 0.001);
        Ok(())
    }
}
