//! Diagnostic observer and fixed 2x2 replay. Generated run() uses the unchanged
//! production refinement block; acceptance is never replayed or relaxed.
use super::*;
use serde_json::{json, Value};

fn camera(p: &CameraSolution) -> Value {
    let r = p.rotation();
    let rows: Vec<Vec<f64>> = (0..3)
        .map(|i| (0..3).map(|j| r[(i, j)]).collect())
        .collect();
    json!({"ra_deg":p.ra_deg,"dec_deg":p.dec_deg,"roll_deg":p.roll_deg,
        "focal_px":p.focal_px,"k1":p.k1,"width":p.width,"height":p.height,
        "world_to_camera":rows})
}

fn matches(rows: &[Match]) -> Value {
    json!(rows
        .iter()
        .map(|m| json!({"world":m.world,"xy":[m.px,m.py]}))
        .collect::<Vec<_>>())
}

fn verification(v: &VerifyResult) -> Value {
    json!({"hits":v.hits,"predicted":v.predicted,"log_odds":v.log_odds,
        "rms_px":v.rms_px,"matches":matches(&v.matches),"detection_indices":v.matched_detections})
}

pub(super) fn input(
    c: &Candidate,
    stars: &VerifyStars,
    dets: &[Detection],
    original: (u32, u32),
    epoch: Option<f64>,
) {
    eprintln!(
        "REFINEMENT_INPUT {}",
        json!({"camera":camera(&c.pose),
        "verification":verification(&c.verify),"detections":dets,"catalog":stars.list,
        "original_size":original,"epoch_years":epoch,"label":c.label})
    );
}

// GENERATED_REFINEMENT_PATH

#[cfg(test)]
mod replay {
    use super::*;

    fn number(v: &Value, key: &str) -> f64 {
        v[key].as_f64().expect("frozen numeric field")
    }

    fn pose(v: &Value) -> CameraSolution {
        CameraSolution {
            ra_deg: number(v, "ra_deg"),
            dec_deg: number(v, "dec_deg"),
            roll_deg: number(v, "roll_deg"),
            focal_px: number(v, "focal_px"),
            k1: number(v, "k1"),
            width: u32::try_from(v["width"].as_u64().expect("width")).expect("u32 width"),
            height: u32::try_from(v["height"].as_u64().expect("height")).expect("u32 height"),
        }
    }

    #[test]
    #[ignore = "explicit development-only frozen 2x2 replay"]
    fn export() -> Result<(), Box<dyn std::error::Error>> {
        let plan: Value =
            serde_json::from_slice(&std::fs::read(std::env::var("STARGLYPH_REFINEMENT_PLAN")?)?)?;
        assert_eq!(plan["id"], "wm_r_132162731");
        assert_eq!(plan["holdout"], false);
        let saved = &plan["inputs"];
        assert_eq!(saved["control"]["catalog"], saved["huber"]["catalog"]);
        assert_eq!(saved["control"]["detections"], saved["huber"]["detections"]);
        let stars = VerifyStars {
            list: serde_json::from_value(saved["control"]["catalog"].clone())?,
            by_id: HashMap::new(),
        };
        // match_predictions only reads list; no adapter or tetra3 search in replay.
        let dets: Vec<_> = saved["control"]["detections"]
            .as_array()
            .expect("detections")
            .iter()
            .map(|d| Detection {
                x: number(d, "x"),
                y: number(d, "y"),
                flux: number(d, "flux") as f32,
                peak: number(d, "peak") as f32,
                snr: number(d, "snr") as f32,
                elongation: number(d, "elongation") as f32,
                rank: u32::try_from(d["rank"].as_u64().expect("rank")).expect("u32 rank"),
                area: u32::try_from(d["area"].as_u64().expect("area")).expect("u32 area"),
            })
            .collect();
        let mut cases = Vec::new();
        for pose_arm in ["control", "huber"] {
            let initial = pose(&saved[pose_arm]["camera"]);
            assert_eq!(
                saved[pose_arm]["original_size"],
                json!([initial.width, initial.height]),
                "lift not supported in this diagnostic"
            );
            let v = match_predictions(&initial, &stars, &dets, VERIFY_RADIUS_PX);
            // Tolerant float checks belong in the Python report; identities must match exactly.
            assert_eq!(
                verification(&v)["detection_indices"],
                saved[pose_arm]["verification"]["detection_indices"]
            );
            for match_arm in ["control", "huber"] {
                let rows: Vec<Match> = saved[match_arm]["verification"]["matches"]
                    .as_array()
                    .expect("matches")
                    .iter()
                    .map(|m| Match {
                        world: serde_json::from_value(m["world"].clone()).expect("world"),
                        px: m["xy"][0].as_f64().expect("x"),
                        py: m["xy"][1].as_f64().expect("y"),
                    })
                    .collect();
                let started = Instant::now();
                let (p, v, steps) = run(&initial, &rows, &stars, &dets);
                cases.push(json!({"pose_arm":pose_arm,"match_arm":match_arm,"camera":camera(&p),
                    "verification":verification(&v),"steps":steps,"elapsed_ms":started.elapsed().as_secs_f64()*1000.}));
            }
        }
        std::fs::write(
            std::env::var("STARGLYPH_REFINEMENT_OUTPUT")?,
            serde_json::to_vec_pretty(&json!({"cases":cases}))?,
        )?;
        Ok(())
    }
}
