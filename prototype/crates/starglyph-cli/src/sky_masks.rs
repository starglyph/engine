//! Opt-in evaluation annotations, bound to the exact oriented input image.
use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use starglyph_core::sky_mask::SkyMask;
use std::{
    collections::{HashMap, HashSet},
    fs,
    path::Path,
};

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct MaskFile {
    schema_version: u32,
    coordinates: String,
    description: String,
    masks: Vec<Annotation>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Annotation {
    pub id: String,
    pub source_sha256: String,
    pub width: u32,
    pub height: u32,
    pub sky_polygon: Vec<[f64; 2]>,
    pub note: String,
}

impl Annotation {
    pub fn for_image(&self, sha256: &str, width: u32, height: u32) -> Result<SkyMask> {
        if self.source_sha256 != sha256 || self.width != width || self.height != height {
            bail!(
                "{}: sky mask SHA-256 or EXIF-oriented dimensions mismatch",
                self.id
            );
        }
        SkyMask::new(self.sky_polygon.clone()).map_err(anyhow::Error::msg)
    }
}

pub(crate) fn load(path: &Path, known_ids: &HashSet<&str>) -> Result<HashMap<String, Annotation>> {
    let text = fs::read_to_string(path).context("read sky masks")?;
    parse(&text, known_ids)
}

fn parse(text: &str, known_ids: &HashSet<&str>) -> Result<HashMap<String, Annotation>> {
    let file: MaskFile = serde_json::from_str(text).context("parse sky masks")?;
    if file.schema_version != 1
        || file.coordinates != "exif_oriented_normalized_image_edges"
        || file.description.is_empty()
        || file.masks.is_empty()
    {
        bail!("unsupported or empty sky mask file");
    }
    let mut masks = HashMap::new();
    for annotation in file.masks {
        if !known_ids.contains(annotation.id.as_str()) {
            bail!("unknown sky mask ID: {}", annotation.id);
        }
        if annotation.width == 0
            || annotation.height == 0
            || annotation.source_sha256.len() != 64
            || !annotation
                .source_sha256
                .bytes()
                .all(|c| c.is_ascii_hexdigit())
        {
            bail!("invalid sky mask identity: {}", annotation.id);
        }
        SkyMask::new(annotation.sky_polygon.clone()).map_err(anyhow::Error::msg)?;
        if masks.insert(annotation.id.clone(), annotation).is_some() {
            bail!("duplicate sky mask ID");
        }
    }
    Ok(masks)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn statistics_option_requires_masks() {
        use clap::Parser;
        assert!(crate::Args::try_parse_from([
            "starglyph",
            "eval",
            "--manifest",
            "m.json",
            "--out-dir",
            "out",
            "--sky-statistics"
        ])
        .is_err());
        assert!(crate::Args::try_parse_from([
            "starglyph",
            "eval",
            "--manifest",
            "m.json",
            "--out-dir",
            "out",
            "--sky-statistics",
            "--sky-masks",
            "mask.json"
        ])
        .is_ok());
    }

    #[test]
    fn annotation_requires_exact_hash_and_oriented_dimensions() {
        let annotation = Annotation {
            id: "portrait".into(),
            source_sha256: "a".repeat(64),
            width: 2252,
            height: 4000,
            sky_polygon: vec![[0., 0.], [1., 0.], [1., 1.]],
            note: String::new(),
        };
        assert!(annotation.for_image(&"a".repeat(64), 2252, 4000).is_ok());
        assert!(annotation.for_image(&"b".repeat(64), 2252, 4000).is_err());
        assert!(annotation.for_image(&"a".repeat(64), 4000, 2252).is_err());
    }
    #[test]
    fn file_rejects_unknown_duplicate_and_invalid_polygons() {
        let mut file = serde_json::json!({"schema_version": 1, "coordinates": "exif_oriented_normalized_image_edges", "description": "test", "masks": [{"id": "a", "source_sha256": "a".repeat(64), "width": 100, "height": 100, "sky_polygon": [[0,0],[1,0],[1,1]], "note": "test"}]});
        let ids = HashSet::from(["a"]);
        assert!(parse(&file.to_string(), &ids).is_ok());
        assert!(parse(&file.to_string(), &HashSet::new()).is_err());
        let entry = file["masks"][0].clone();
        file["masks"].as_array_mut().unwrap().push(entry);
        assert!(parse(&file.to_string(), &ids).is_err());
        file["masks"].as_array_mut().unwrap().pop();
        file["masks"][0]["sky_polygon"] = serde_json::json!([[0, 0], [2, 0], [1, 1]]);
        assert!(parse(&file.to_string(), &ids).is_err());
    }
}
