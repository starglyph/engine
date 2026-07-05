use std::fs;
use std::path::Path;

use serde::Deserialize;

/// One constellation outline with IAU abbreviation and display name.
#[derive(Debug, Clone, PartialEq)]
pub struct Constellation {
    pub abbr: String,
    pub name: String,
    pub polylines: Vec<Vec<[f64; 2]>>,
}

/// Constellation stick figures loaded from GeoJSON feature collections.
#[derive(Debug, Clone)]
pub struct ConstellationSet {
    constellations: Vec<Constellation>,
}

#[derive(Debug, thiserror::Error)]
pub enum ConstellationError {
    #[error("failed to read constellation data at '{path}': {source}")]
    Read {
        path: String,
        source: std::io::Error,
    },
    #[error("failed to parse constellation json at '{path}': {source}")]
    Parse {
        path: String,
        source: serde_json::Error,
    },
    #[error("constellation feature at index {index} in '{path}' is missing an id")]
    MissingId { path: String, index: usize },
    #[error(
        "constellation feature '{abbr}' in '{path}' has unsupported geometry type '{geometry_type}'"
    )]
    UnsupportedGeometry {
        path: String,
        abbr: String,
        geometry_type: String,
    },
}

impl ConstellationSet {
    /// Load constellation line geometry and canonical names from paired GeoJSON files.
    pub fn load(lines_path: &Path, names_path: &Path) -> Result<Self, ConstellationError> {
        let lines_bytes = fs::read(lines_path).map_err(|source| ConstellationError::Read {
            path: lines_path.display().to_string(),
            source,
        })?;
        let names_bytes = fs::read(names_path).map_err(|source| ConstellationError::Read {
            path: names_path.display().to_string(),
            source,
        })?;

        let lines_collection: FeatureCollection =
            serde_json::from_slice(&lines_bytes).map_err(|source| ConstellationError::Parse {
                path: lines_path.display().to_string(),
                source,
            })?;
        let names_collection: NameFeatureCollection = serde_json::from_slice(&names_bytes)
            .map_err(|source| ConstellationError::Parse {
                path: names_path.display().to_string(),
                source,
            })?;

        let mut names_by_abbr = std::collections::HashMap::new();
        for feature in names_collection.features {
            if let Some(abbr) = feature.id {
                let name = feature
                    .properties
                    .and_then(|props| props.name)
                    .unwrap_or_else(|| abbr.clone());
                names_by_abbr.insert(abbr, name);
            }
        }

        let lines_path_display = lines_path.display().to_string();
        let mut constellations = Vec::with_capacity(lines_collection.features.len());
        for (index, feature) in lines_collection.features.into_iter().enumerate() {
            let abbr = feature.id.ok_or_else(|| ConstellationError::MissingId {
                path: lines_path_display.clone(),
                index,
            })?;
            let geometry =
                feature
                    .geometry
                    .ok_or_else(|| ConstellationError::UnsupportedGeometry {
                        path: lines_path_display.clone(),
                        abbr: abbr.clone(),
                        geometry_type: "missing".to_string(),
                    })?;
            if geometry.kind != "MultiLineString" {
                return Err(ConstellationError::UnsupportedGeometry {
                    path: lines_path_display.clone(),
                    abbr,
                    geometry_type: geometry.kind,
                });
            }

            let polylines = geometry
                .coordinates
                .into_iter()
                .filter_map(convert_polyline)
                .collect();

            let name = names_by_abbr
                .get(&abbr)
                .cloned()
                .unwrap_or_else(|| abbr.clone());

            constellations.push(Constellation {
                abbr,
                name,
                polylines,
            });
        }

        Ok(Self { constellations })
    }

    /// Loaded constellations in source file order.
    pub fn constellations(&self) -> &[Constellation] {
        &self.constellations
    }

    /// Total number of polylines across all constellations.
    pub fn polyline_count(&self) -> usize {
        self.constellations
            .iter()
            .map(|constellation| constellation.polylines.len())
            .sum()
    }

    /// Total number of polyline vertices across all constellations.
    pub fn vertex_count(&self) -> usize {
        self.constellations
            .iter()
            .flat_map(|constellation| constellation.polylines.iter())
            .map(|polyline| polyline.len())
            .sum()
    }
}

fn convert_polyline(line: Vec<[f64; 2]>) -> Option<Vec<[f64; 2]>> {
    if line.len() < 2 {
        return None;
    }
    let vertices = line
        .into_iter()
        .map(|[lon, lat]| [lon_to_ra_deg(lon), lat])
        .collect::<Vec<_>>();
    Some(vertices)
}

fn lon_to_ra_deg(lon: f64) -> f64 {
    if lon < 0.0 {
        lon + 360.0
    } else {
        lon
    }
}

#[derive(Debug, Deserialize)]
struct FeatureCollection {
    features: Vec<LineFeature>,
}

#[derive(Debug, Deserialize)]
struct NameFeatureCollection {
    features: Vec<NameFeature>,
}

#[derive(Debug, Deserialize)]
struct LineFeature {
    id: Option<String>,
    geometry: Option<Geometry>,
}

#[derive(Debug, Deserialize)]
struct NameFeature {
    id: Option<String>,
    properties: Option<FeatureProperties>,
}

#[derive(Debug, Deserialize)]
struct FeatureProperties {
    name: Option<String>,
}

#[derive(Debug, Deserialize)]
struct Geometry {
    #[serde(rename = "type")]
    kind: String,
    coordinates: Vec<Vec<[f64; 2]>>,
}

#[cfg(test)]
mod tests {
    use std::io::Write;

    use super::*;
    use tempfile::NamedTempFile;

    fn write_json(contents: &str) -> NamedTempFile {
        let mut file = NamedTempFile::new().expect("temp file");
        write!(file, "{contents}").expect("write json");
        file
    }

    #[test]
    fn converts_negative_longitude_to_ra_degrees() {
        let lines = write_json(
            r#"{"type":"FeatureCollection","features":[{"type":"Feature","id":"Tst","geometry":{"type":"MultiLineString","coordinates":[[[-10.0,5.0],[20.0,-3.0]]]}}]}"#,
        );
        let names = write_json(
            r#"{"type":"FeatureCollection","features":[{"type":"Feature","id":"Tst","properties":{"name":"Test"}}]}"#,
        );
        let set = ConstellationSet::load(lines.path(), names.path()).expect("load constellations");
        assert_eq!(set.constellations().len(), 1);
        let polyline = &set.constellations()[0].polylines[0];
        assert_eq!(polyline[0], [350.0, 5.0]);
        assert_eq!(polyline[1], [20.0, -3.0]);
    }

    #[test]
    fn filters_single_vertex_polylines() {
        let lines = write_json(
            r#"{"type":"FeatureCollection","features":[{"type":"Feature","id":"Tst","geometry":{"type":"MultiLineString","coordinates":[[[1.0,2.0]],[[3.0,4.0],[5.0,6.0]]]}}]}"#,
        );
        let names = write_json(
            r#"{"type":"FeatureCollection","features":[{"type":"Feature","id":"Tst","properties":{"name":"Test"}}]}"#,
        );
        let set = ConstellationSet::load(lines.path(), names.path()).expect("load constellations");
        assert_eq!(set.constellations()[0].polylines.len(), 1);
        assert_eq!(set.polyline_count(), 1);
        assert_eq!(set.vertex_count(), 2);
    }

    #[test]
    fn real_constellation_count_is_within_expected_range() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../data/celestial");
        let lines = root.join("constellations.lines.json");
        let names = root.join("constellations.json");
        if !lines.exists() || !names.exists() {
            return;
        }
        let set = ConstellationSet::load(&lines, &names).expect("load real constellations");
        let count = set.constellations().len();
        assert!(
            (85..=90).contains(&count),
            "unexpected constellation count: {count}"
        );
    }
}
