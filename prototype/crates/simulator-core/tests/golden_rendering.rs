use std::fs;
use std::io::Cursor;
use std::path::PathBuf;

use anyhow::{Context, Result};
use image::{DynamicImage, ImageFormat};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use serde::Deserialize;
use sha2::{Digest, Sha256};
use simulator_core::camera::{CameraExtrinsics, CameraIntrinsics};
use simulator_core::catalog::baseline_catalog;
use simulator_core::config::{CameraConfig, DegradationConfig, RenderConfig};
use simulator_core::degradations::apply_baseline_degradations;
use simulator_core::projection::project_catalog;
use simulator_core::rendering::render_stars;

#[derive(Debug, Deserialize)]
struct GoldenCase {
    name: String,
    seed: u64,
    degraded: bool,
    expected_sha256: String,
}

#[test]
fn rendering_golden_fixtures_match() -> Result<()> {
    let fixtures = load_cases()?;
    let camera = CameraConfig {
        width_px: 320,
        height_px: 240,
        fov_deg: 62.0,
    };
    let intrinsics = CameraIntrinsics::from_camera_config(&camera);
    let pose = CameraExtrinsics {
        ra_deg: 90.0,
        dec_deg: 5.0,
        roll_deg: 2.0,
        position: [0.0, 0.0, 0.0],
    };
    let catalog = baseline_catalog();
    let projected = project_catalog(&catalog, &intrinsics, &pose);
    let render_config = RenderConfig::default();

    for case in fixtures {
        let base = render_stars(
            camera.width_px,
            camera.height_px,
            &projected,
            &render_config,
        );
        let frame = if case.degraded {
            let mut rng = ChaCha8Rng::seed_from_u64(case.seed);
            let (degraded, _) =
                apply_baseline_degradations(base, &DegradationConfig::default(), &mut rng)?;
            degraded
        } else {
            base
        };

        let mut encoded_png = Vec::new();
        DynamicImage::ImageLuma8(frame.to_gray_image())
            .write_to(&mut Cursor::new(&mut encoded_png), ImageFormat::Png)
            .context("failed to encode frame as png")?;
        let actual_hash = format!("{:x}", Sha256::digest(encoded_png));
        assert_eq!(
            actual_hash, case.expected_sha256,
            "golden mismatch for case '{}' (update fixture if intentional)",
            case.name
        );
    }
    Ok(())
}

fn load_cases() -> Result<Vec<GoldenCase>> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("golden_cases.json");
    let content = fs::read_to_string(&path)
        .with_context(|| format!("failed to read golden fixture '{}'", path.display()))?;
    serde_json::from_str(&content).context("failed to parse golden fixture json")
}
