//! Offline replay of frozen candidates. Omit pairs only from roll estimation;
//! the production verifier always receives the complete detection/catalog sets.
use super::*;
use serde::Deserialize;
use serde_json::{json, Value};

#[derive(Deserialize)]
struct Saved {
    solution: Solution,
    detections: Vec<[f64; 3]>,
    camera: Value,
    verification: Value,
    pairs: Vec<Value>,
}

fn camera(p: &CameraSolution) -> Value {
    json!({"ra_deg":p.ra_deg,"dec_deg":p.dec_deg,"roll_deg":p.roll_deg,
        "focal_px":p.focal_px,"k1":p.k1,"width":p.width,"height":p.height})
}

fn snapshot(sol: &Solution, saved: &Saved, dets: &[Detection], stars: &VerifyStars) -> Value {
    let started = Instant::now();
    let pose = pose_from_solution(sol, dets, stars).expect("frozen adapter input");
    let verify = match_predictions(&pose, stars, dets, VERIFY_RADIUS_PX);
    let elapsed_ms = started.elapsed().as_secs_f64() * 1000.;
    let rot = pose.rotation();
    let pairs: Vec<_> = saved
        .pairs
        .iter()
        .map(|p| {
            let id = p["catalog_id"].as_i64().expect("catalog id");
            let index = p["detection_index"].as_u64().expect("index") as usize;
            let world = stars.by_id[&id];
            let (x, y) =
                geom::project(&rot, pose.focal_px, pose.k1, pose.width, pose.height, world)
                    .expect("visible saved pair");
            json!({"catalog_id":id,"detection_index":index,"projected_xy":[x,y],
            "residual_px":[x-dets[index].x,y-dets[index].y],
            "error_px":(x-dets[index].x).hypot(y-dets[index].y)})
        })
        .collect();
    let matches: Vec<_> = verify
        .matches
        .iter()
        .zip(&verify.matched_detections)
        .map(|(m, i)| json!({"world":m.world,"xy":[m.px,m.py],"detection_index":i}))
        .collect();
    json!({"camera":camera(&pose),"adapter_pair_count":sol.matched_catalog_ids.len(),
        "detection_count":dets.len(),"catalog_count":stars.list.len(),"elapsed_ms":elapsed_ms,
        "verification":{"hits":verify.hits,"predicted":verify.predicted,
            "log_odds":verify.log_odds,"rms_px":verify.rms_px,"matches":matches},
        "counterfactual_gates_with_original_search_evidence":{
            "hard":sol.num_matches>=HARD_MIN_MATCHES && sol.prob<HARD_MAX_PROB && verify.hits>=VERIFY_MIN_HITS,
            "soft":verify.is_verified()},
        "original_pairs":pairs})
}

fn close(actual: f64, expected: f64) {
    assert!((actual - expected).abs() < 1e-10, "{actual} != {expected}");
}

#[test]
#[ignore = "explicit frozen offline diagnostic"]
fn export() -> Result<(), Box<dyn std::error::Error>> {
    let plan: Value =
        serde_json::from_slice(&std::fs::read(std::env::var("STARGLYPH_ROLL_PLAN")?)?)?;
    assert_eq!(plan["id"], "20260831_214709");
    assert_eq!(plan["holdout"], false);
    let frame = FrameImage::load(std::path::Path::new(plan["image"].as_str().expect("image")))?;
    // Match eval_cmd's explicit hint followed by solve_single_scale's fallback.
    let filename_epoch = frame.timestamp_from_name().map(|t| t.to_epoch_years());
    let timestamp = frame.acquisition_timestamp();
    let epoch =
        filename_epoch.or_else(|| timestamp.map(|t| crate::ephem::epoch_years(t.to_jd_utc(0.0))));
    let catalog = Catalog::load(std::path::Path::new(
        plan["catalog"].as_str().expect("catalog"),
    ))?;
    let stars = VerifyStars::build(&catalog, epoch);
    let mut arms = serde_json::Map::new();
    for arm in ["control", "centered"] {
        let saved: Saved = serde_json::from_value(plan["first_candidates"][arm].clone())?;
        assert_eq!(saved.detections.len(), 50);
        assert_eq!(saved.pairs.len(), 13);
        let dets: Vec<_> = saved
            .detections
            .iter()
            .enumerate()
            .map(|(rank, d)| Detection {
                x: d[0],
                y: d[1],
                flux: d[2] as f32,
                peak: 1.,
                snr: 100.,
                area: 1,
                elongation: 1.,
                rank: u32::try_from(rank).expect("50 frozen detections"),
            })
            .collect();
        for p in &saved.pairs {
            let id = p["catalog_id"].as_i64().expect("id");
            for (i, value) in stars.by_id[&id].iter().enumerate() {
                assert!((value - p["world"][i].as_f64().expect("world")).abs() < 1e-14);
            }
        }
        let baseline = snapshot(&saved.solution, &saved, &dets, &stars);
        for key in [
            "ra_deg", "dec_deg", "roll_deg", "focal_px", "k1", "width", "height",
        ] {
            close(
                baseline["camera"][key].as_f64().expect("camera"),
                saved.camera[key].as_f64().expect("camera"),
            );
        }
        for key in ["hits", "predicted", "log_odds", "rms_px"] {
            close(
                baseline["verification"][key].as_f64().expect("metric"),
                saved.verification[key].as_f64().expect("metric"),
            );
        }
        assert_eq!(
            baseline["verification"]["matches"],
            saved.verification["matches"]
        );
        let mut omissions = vec![];
        for i in 0..saved.solution.matched_catalog_ids.len() {
            // Deserializing a fresh copy preserves the frozen tetra3 search evidence.
            let mut sol: Solution =
                serde_json::from_value(plan["first_candidates"][arm]["solution"].clone())?;
            let id = sol.matched_catalog_ids.remove(i);
            let index = sol.matched_centroid_indices.remove(i);
            assert_eq!(sol.num_matches, saved.solution.num_matches);
            assert_eq!(sol.prob, saved.solution.prob);
            assert_eq!(sol.matched_catalog_ids.len(), 12);
            let mut row = snapshot(&sol, &saved, &dets, &stars);
            row["omitted_from_roll"] =
                json!({"pair_position":i,"catalog_id":id,"detection_index":index});
            row["delta_roll_deg"] = json!(
                row["camera"]["roll_deg"].as_f64().expect("roll")
                    - baseline["camera"]["roll_deg"].as_f64().expect("roll")
            );
            for key in ["ra_deg", "dec_deg", "focal_px", "k1", "width", "height"] {
                assert_eq!(row["camera"][key], baseline["camera"][key]);
            }
            omissions.push(row);
        }
        arms.insert(arm.into(), json!({"baseline_reproduced":true,"baseline":baseline,
            "original_search_evidence":{"num_matches":saved.solution.num_matches,"prob":saved.solution.prob},
            "omissions":omissions}));
    }
    let result = json!({"iteration":22,"id":"20260831_214709","holdout":false,"epoch_years":epoch,
        "status":"offline_diagnostic_only","new_searches":0,"refinements":0,
        "verification_radius_px":VERIFY_RADIUS_PX,"arms":arms});
    std::fs::write(
        std::env::var("STARGLYPH_ROLL_OUTPUT")?,
        serde_json::to_vec_pretty(&result)?,
    )?;
    Ok(())
}
