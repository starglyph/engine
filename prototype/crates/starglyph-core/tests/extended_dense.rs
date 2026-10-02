//! Regression on committed phone/mask fixtures, not an independent WCS reference.
use std::path::Path;

use starglyph_core::catalog::Catalog;
use starglyph_core::constellations::ConstellationSet;
use starglyph_core::contracts::SolveStatus;
use starglyph_core::engine::Engine;
use starglyph_core::image_input::FrameImage;
use starglyph_core::sky_mask::SkyMask;
use starglyph_core::solve::{solve_frame_with_engine_and_mask, SolveOptions};

#[test]
#[ignore = "repository fixtures and pattern databases; run by eval-smartphone-sky-fill-gate"]
fn recovers_masked_phone_beyond_original_deep_prefixes() -> Result<(), Box<dyn std::error::Error>> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../..");
    let frame = FrameImage::load(&root.join("data/input/smartphone/20260829_215950.jpg"))?;
    let annotations: serde_json::Value = serde_json::from_slice(&std::fs::read(
        root.join("data/input/smartphone/sky-masks.json"),
    )?)?;
    let annotation = annotations["masks"]
        .as_array()
        .expect("mask array")
        .iter()
        .find(|m| m["id"] == "20260829_215950")
        .expect("committed phone mask");
    assert_eq!(annotation["width"], frame.width);
    assert_eq!(annotation["height"], frame.height);
    let mask = SkyMask::new(serde_json::from_value(annotation["sky_polygon"].clone())?)
        .expect("valid committed polygon")
        .with_sky_fill();
    let catalog = Catalog::load(&root.join("data/catalogs/hyg_v42.csv.gz"))?;
    let constellations = ConstellationSet::load(
        &root.join("data/celestial/constellations.lines.json"),
        &root.join("data/celestial/constellations.json"),
    )?;
    let opts = SolveOptions {
        cache_dir: root.join("prototype/artifacts/cache"),
        ..SolveOptions::default()
    };
    let (report, _) = solve_frame_with_engine_and_mask(
        &frame,
        &catalog,
        &constellations,
        &mut Engine::default(),
        &opts,
        Some(&mask),
        &mut |_| {},
    );
    assert_eq!(report.status, SolveStatus::Solved);
    let quality = report.quality.expect("solved quality");
    assert!(quality.n_inliers >= 20, "{quality:?}");
    assert!(quality.rms_px <= 3.5, "{quality:?}");
    let pose = report.pose.expect("solved pose");
    assert!((pose.ra_deg - 289.82).abs() < 0.2, "{pose:?}");
    assert!((pose.dec_deg + 2.30).abs() < 0.2, "{pose:?}");
    Ok(())
}
