//! Fixed-candidate diagnostic using the existing iteration-6 stage observer.
use super::*;
use serde::Deserialize;
use serde_json::{json, Value};

#[derive(Deserialize)]
struct SavedCamera {
    ra_deg: f64,
    dec_deg: f64,
    roll_deg: f64,
    focal_px: f64,
    k1: f64,
    width: u32,
    height: u32,
}
#[derive(Deserialize)]
struct SavedMatch {
    hyg_ids: Vec<i64>,
    world: [f64; 3],
    xy: [f64; 2],
}
#[derive(Deserialize)]
struct SavedDetection {
    x: f64,
    y: f64,
    flux: f32,
    peak: f32,
    snr: f32,
    area: u32,
    elongation: f32,
    rank: u32,
}
#[derive(Deserialize)]
struct Case {
    name: String,
    matches: Vec<SavedMatch>,
    detections: Vec<SavedDetection>,
}
#[derive(Deserialize)]
struct Input {
    id: String,
    split: String,
    image: PathBuf,
    catalog: PathBuf,
    initial: SavedCamera,
    cases: Vec<Case>,
}

#[test]
#[ignore = "fixed development candidate: STARGLYPH_REMATCH_INPUT and OUTPUT"]
fn fixed_candidate() -> Result<(), Box<dyn std::error::Error>> {
    let input: Input =
        serde_json::from_slice(&std::fs::read(std::env::var("STARGLYPH_REMATCH_INPUT")?)?)?;
    assert_eq!(input.id, "wm_r_143159342");
    assert_eq!(input.split, "development");
    let original = FrameImage::load(&input.image)?;
    assert!(
        original.acquisition_timestamp().is_none(),
        "frozen epoch-free photograph required"
    );
    let catalog = Catalog::load(&input.catalog)?;
    let stars = VerifyStars::build(&catalog, None);
    let p = input.initial;
    let pose = CameraSolution {
        ra_deg: p.ra_deg,
        dec_deg: p.dec_deg,
        roll_deg: p.roll_deg,
        focal_px: p.focal_px,
        k1: p.k1,
        width: p.width,
        height: p.height,
    };
    let mut cases: Vec<Value> = Vec::new();
    for case in input.cases {
        let detections: Vec<_> = case
            .detections
            .iter()
            .map(|d| Detection {
                x: d.x,
                y: d.y,
                flux: d.flux,
                peak: d.peak,
                snr: d.snr,
                area: d.area,
                elongation: d.elongation,
                rank: d.rank,
            })
            .collect();
        let matches: Vec<_> = case
            .matches
            .iter()
            .map(|m| {
                assert!(!m.hyg_ids.is_empty(), "catalogue identity required");
                for id in &m.hyg_ids {
                    assert_eq!(stars.by_id.get(id), Some(&m.world), "vector round-trip");
                }
                Match {
                    world: m.world,
                    px: m.xy[0],
                    py: m.xy[1],
                }
            })
            .collect();
        let candidate = Candidate {
            label: "fixed-footprint",
            pose: pose.clone(),
            attitude_quat: None,
            verify: VerifyResult {
                hits: matches.len() as u32,
                predicted: 0,
                log_odds: 0.,
                rms_px: 0.,
                matches,
                matched_detections: Vec::new(),
            },
        };
        // No candidate selection or acceptance is simulated here. The observer
        // consumes only the fixed pose and initial matches from this Candidate.
        let started = Instant::now();
        let stages = super::geometry_replay::stages(&candidate, &detections, &stars, &original);
        cases.push(
            json!({"name":case.name,"n_detections":detections.len(),"stages":stages,
            "elapsed_ms":started.elapsed().as_secs_f64()*1000.}),
        );
    }
    std::fs::write(
        std::env::var("STARGLYPH_REMATCH_OUTPUT")?,
        serde_json::to_vec_pretty(&json!({
            "id":input.id,"split":input.split,"cases":cases,"acceptance_tested":false
        }))?,
    )?;
    Ok(())
}
