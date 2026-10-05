//! Paired geometric test, no search, verification selection or LM.
use super::*;
use serde::Deserialize;
use serde_json::{json, Value};

#[path = "roll_candidate_adapter.rs"]
mod adapter;
#[path = "centre_transfer_fixture.rs"]
mod fixture;
#[path = "roll_synthetic_fixture.rs"]
mod inputs;
#[path = "robust_roll.rs"]
mod robust_roll;

#[derive(Deserialize)]
struct Case {
    id: String,
    size: [u32; 2],
    fov_deg: f64,
    sky: [f64; 3],
    support: String,
    pairs: usize,
    sigma_px: f64,
    seed: u64,
    bad_count: usize,
    bad_angle_deg: f64,
}

fn metrics(pose: &CameraSolution, probes: &[fixture::Point], elapsed_us: f64) -> Value {
    let rot = pose.rotation();
    let errors: Vec<_> = probes
        .iter()
        .map(|p| {
            let (x, y) = geom::project(
                &rot,
                pose.focal_px,
                pose.k1,
                pose.width,
                pose.height,
                p.world,
            )
            .expect("known visible TAN probe");
            (x - p.xy[0] - 0.5).hypot(y - p.xy[1] - 0.5)
        })
        .collect();
    json!({"roll_deg":pose.roll_deg,"rms_px":(errors.iter().map(|x|x*x).sum::<f64>()/errors.len() as f64).sqrt(),
        "max_px":errors.into_iter().fold(0_f64,f64::max),"elapsed_us":elapsed_us})
}

#[test]
#[ignore = "explicit frozen synthetic experiment"]
fn export() -> Result<(), Box<dyn std::error::Error>> {
    let cases: Vec<Case> =
        serde_json::from_slice(&std::fs::read(std::env::var("STARGLYPH_ROLL_CASES")?)?)?;
    let mut rows = vec![];
    for (position, case) in cases.into_iter().enumerate() {
        let mut sol = fixture::solution(case.size, case.fov_deg, case.sky, 0.);
        sol.matched_catalog_ids = (0..case.pairs as i64).collect();
        sol.matched_centroid_indices = (0..case.pairs).collect();
        sol.num_matches = u32::try_from(case.pairs)?;
        let input = inputs::input(
            &sol,
            case.support == "upper",
            case.pairs,
            case.sigma_px,
            case.seed,
            case.bad_count,
            case.bad_angle_deg,
        );
        let detections: Vec<_> = input
            .detections
            .iter()
            .map(|p| Detection {
                x: p[0],
                y: p[1],
                flux: 1.,
                peak: 1.,
                snr: 100.,
                area: 1,
                elongation: 1.,
                rank: 0,
            })
            .collect();
        let stars = VerifyStars {
            list: vec![],
            by_id: input
                .worlds
                .iter()
                .enumerate()
                .map(|(i, w)| (i as i64, *w))
                .collect(),
        };
        let mut arms = serde_json::Map::new();
        // Alternate call order; these tiny descriptive timings are not a benchmark.
        let order = if position % 2 == 0 {
            ["control", "median"]
        } else {
            ["median", "control"]
        };
        for arm in order {
            let started = Instant::now();
            let pose = if arm == "control" {
                pose_from_solution(&sol, &detections, &stars)
            } else {
                adapter::median_pose_from_solution(&sol, &detections, &stars)
            }
            .expect("synthetic adapter");
            let elapsed_us = started.elapsed().as_secs_f64() * 1e6;
            arms.insert(arm.into(), metrics(&pose, &input.probes, elapsed_us));
        }
        rows.push(json!({"id":case.id,"corrupted_indices":input.corrupted_indices,"arms":arms}));
    }
    std::fs::write(
        std::env::var("STARGLYPH_ROLL_OUTPUT")?,
        serde_json::to_vec(&json!({"cases":rows}))?,
    )?;
    Ok(())
}
