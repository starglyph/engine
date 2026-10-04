//! Frozen-pair diagnostic; calls the production optimizer without rematching.
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

impl SavedCamera {
    fn camera(&self) -> CameraSolution {
        CameraSolution {
            ra_deg: self.ra_deg,
            dec_deg: self.dec_deg,
            roll_deg: self.roll_deg,
            focal_px: self.focal_px,
            k1: self.k1,
            width: self.width,
            height: self.height,
        }
    }
}

#[derive(Deserialize)]
struct SavedMatch {
    world: [f64; 3],
    xy: [f64; 2],
}
#[derive(Deserialize)]
struct Case {
    name: String,
    initial: SavedCamera,
    matches: Vec<SavedMatch>,
    prior_weight: Option<f64>,
    expected: Option<SavedCamera>,
}
#[derive(Deserialize)]
struct Input {
    split: String,
    id: String,
    cases: Vec<Case>,
    probe_worlds: Vec<[f64; 3]>,
}

fn camera_json(camera: &CameraSolution) -> Value {
    let r = camera.rotation();
    json!({"ra_deg":camera.ra_deg,"dec_deg":camera.dec_deg,"roll_deg":camera.roll_deg,
        "focal_px":camera.focal_px,"k1":camera.k1,"width":camera.width,"height":camera.height,
        "world_to_camera":(0..3).map(|i| (0..3).map(|j| r[(i,j)]).collect::<Vec<_>>()).collect::<Vec<_>>()})
}

fn matrix_rows(matrix: &DMatrix<f64>) -> Vec<Vec<f64>> {
    (0..matrix.nrows())
        .map(|i| (0..matrix.ncols()).map(|j| matrix[(i, j)]).collect())
        .collect()
}

#[test]
#[ignore = "frozen development pair fits: STARGLYPH_PAIR_INPUT and STARGLYPH_PAIR_OUTPUT"]
fn fixed_pairs() -> Result<(), Box<dyn std::error::Error>> {
    let input: Input =
        serde_json::from_slice(&std::fs::read(std::env::var("STARGLYPH_PAIR_INPUT")?)?)?;
    assert_eq!(input.split, "development");
    assert_eq!(input.id, "wm_r_143159342");
    let probes: Vec<_> = input
        .probe_worlds
        .iter()
        .map(|world| Match {
            world: *world,
            px: 0.,
            py: 0.,
        })
        .collect();
    let mut results = Vec::new();
    for case in input.cases {
        let initial = case.initial.camera();
        let matches: Vec<_> = case
            .matches
            .iter()
            .map(|m| Match {
                world: m.world,
                px: m.xy[0],
                py: m.xy[1],
            })
            .collect();
        assert!(
            matches.len() >= 8,
            "keep the same five free parameters in every diagnostic case"
        );
        let natural_weight = k1_reg_weight(initial.fov_x_deg());
        let weight = case.prior_weight.unwrap_or(natural_weight);
        // In this production function scale multiplies only the k1 prior.
        // Fix that weight across initial cameras to keep an identical objective.
        let start = Instant::now();
        let fitted = refine_pose_with_scale(&initial, &matches, weight / natural_weight);
        let elapsed_ms = start.elapsed().as_secs_f64() * 1000.;
        let reproduced = if let Some(expected) = case.expected {
            let expected = expected.camera();
            assert!((fitted.focal_px - expected.focal_px).abs() < 1e-8);
            assert!((fitted.k1 - expected.k1).abs() < 1e-10);
            assert!((fitted.rotation() - expected.rotation()).norm() < 1e-10);
            true
        } else {
            false
        };
        let params = [
            fitted.ra_deg,
            fitted.dec_deg,
            fitted.roll_deg,
            fitted.focal_px,
            fitted.k1,
        ];
        let r = residuals(&params, true, weight, fitted.width, fitted.height, &matches);
        let jac = numeric_jacobian(
            &params,
            true,
            weight,
            fitted.width,
            fitted.height,
            &matches,
            &r,
        );
        let probe_r = residuals(&params, true, 0., fitted.width, fitted.height, &probes);
        let probe_jac = numeric_jacobian(
            &params,
            true,
            0.,
            fitted.width,
            fitted.height,
            &probes,
            &probe_r,
        );
        results.push(
            json!({"name":case.name,"camera":camera_json(&fitted),"initial":camera_json(&initial),
            "prior_weight":weight,"prior_scale":weight/natural_weight,"elapsed_ms":elapsed_ms,
            "matches":matches.len(),"residuals":r.as_slice(),"jacobian":matrix_rows(&jac),
            "probe_jacobian":matrix_rows(&probe_jac),"reproduced_previous_first_fit":reproduced}),
        );
    }
    std::fs::write(
        std::env::var("STARGLYPH_PAIR_OUTPUT")?,
        serde_json::to_vec_pretty(&json!({
            "id":input.id,"split":input.split,"cases":results
        }))?,
    )?;
    Ok(())
}
