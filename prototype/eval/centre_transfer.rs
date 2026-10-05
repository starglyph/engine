//! Isolated adapter/refinement audit. The production implementation is called
//! directly; only the coordinate representation of fixed pairs is varied.
use super::*;
use serde::Deserialize;
use serde_json::{json, Value};

#[path = "centre_transfer_fixture.rs"]
mod fixture;

#[derive(Deserialize)]
struct Plan {
    sizes: Vec<[u32; 2]>,
    fovs_deg: Vec<f64>,
    sky_tan_deg: Vec<[f64; 3]>,
    centre_offsets_px: Vec<f64>,
    representation_shifts_px: Vec<f64>,
    supports: Vec<String>,
}

fn camera(p: &CameraSolution) -> Value {
    json!({"ra_deg":p.ra_deg,"dec_deg":p.dec_deg,"roll_deg":p.roll_deg,
        "focal_px":p.focal_px,"k1":p.k1,"width":p.width,"height":p.height})
}

fn residuals_for(pose: &CameraSolution, points: &[fixture::Point], shift: f64) -> Vec<[f64; 2]> {
    let rot = pose.rotation();
    points
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
            .expect("visible synthetic probe");
            [x - shift - p.xy[0], y - shift - p.xy[1]]
        })
        .collect()
}

fn metrics(errors: &[[f64; 2]], keep: impl Fn(usize) -> bool) -> Value {
    let norms: Vec<_> = errors
        .iter()
        .enumerate()
        .filter(|(i, _)| keep(*i))
        .map(|(_, e)| e[0].hypot(e[1]))
        .collect();
    if norms.is_empty() {
        return json!({"count":0});
    }
    json!({"count":norms.len(),"rms_px":(norms.iter().map(|x|x*x).sum::<f64>()/norms.len() as f64).sqrt(),
        "max_px":norms.into_iter().fold(0_f64,f64::max)})
}

fn snapshot(
    pose: &CameraSolution,
    train: &[fixture::Point],
    probes: &[fixture::Point],
    shift: f64,
) -> Value {
    let training = residuals_for(pose, train, shift);
    let errors = residuals_for(pose, probes, shift);
    let (w, h) = (f64::from(pose.width), f64::from(pose.height));
    json!({"camera":camera(pose),"training":metrics(&training, |_|true),
        "probes":metrics(&errors, |_|true),"left":metrics(&errors, |i|probes[i].xy[0]<w*0.1),
        "right":metrics(&errors, |i|probes[i].xy[0]>w*0.9),"top":metrics(&errors, |i|probes[i].xy[1]<h*0.1),
        "bottom":metrics(&errors, |i|probes[i].xy[1]>h*0.9),"probe_residuals_px":errors})
}

fn trial(sol: &Solution, train: &[fixture::Point], probes: &[fixture::Point], shift: f64) -> Value {
    let detections: Vec<_> = train
        .iter()
        .map(|p| Detection {
            x: p.xy[0] + shift,
            y: p.xy[1] + shift,
            flux: 100.,
            peak: 1.,
            snr: 100.,
            area: 1,
            elongation: 1.,
            rank: 0,
        })
        .collect();
    let stars = VerifyStars {
        list: vec![],
        by_id: train
            .iter()
            .enumerate()
            .map(|(i, p)| (i as i64, p.world))
            .collect(),
    };
    let matches: Vec<_> = train
        .iter()
        .map(|p| Match {
            world: p.world,
            px: p.xy[0] + shift,
            py: p.xy[1] + shift,
        })
        .collect();
    let start = Instant::now();
    let initial = pose_from_solution(sol, &detections, &stars).expect("ideal adapter input");
    let adapted_us = start.elapsed().as_secs_f64() * 1e6;
    let start = Instant::now();
    let refined = refine_pose(&initial, &matches);
    let fit_us = start.elapsed().as_secs_f64() * 1e6;
    json!({"representation_shift_px":shift,"fixed_pairs":matches.len(),"adapter_us":adapted_us,
        "fit_us":fit_us,"prior_weight":k1_reg_weight(initial.fov_x_deg()),
        "before":snapshot(&initial,train,probes,shift),"after":snapshot(&refined,train,probes,shift)})
}

#[test]
fn matched_centre_control_recovers_known_tan_field() {
    let sol = fixture::solution([1600, 1200], 55., [30., 20., 15.], 0.);
    let row = trial(
        &sol,
        &fixture::training(&sol, false),
        &fixture::probes(&sol),
        0.5,
    );
    assert!(row["after"]["probes"]["max_px"].as_f64().expect("metric") < 0.01);
}

#[test]
#[ignore = "explicit frozen synthetic experiment"]
fn export() -> Result<(), Box<dyn std::error::Error>> {
    let plan: Plan =
        serde_json::from_slice(&std::fs::read(std::env::var("STARGLYPH_CENTRE_PLAN")?)?)?;
    let mut rows = vec![];
    for size in plan.sizes {
        for fov in &plan.fovs_deg {
            for sky in &plan.sky_tan_deg {
                for offset in &plan.centre_offsets_px {
                    let sol = fixture::solution(size, *fov, *sky, *offset);
                    for support in &plan.supports {
                        assert!(support == "full" || support == "upper");
                        let train = fixture::training(&sol, support == "upper");
                        let probes = fixture::probes(&sol);
                        for shift in &plan.representation_shifts_px {
                            let mut row = trial(&sol, &train, &probes, *shift);
                            let id = format!(
                                "{}x{}-f{}-ra{}-centre{}-{}-shift{}",
                                size[0], size[1], fov, sky[0], offset, support, shift
                            );
                            row["id"] = json!(id);
                            row["size"] = json!(size);
                            row["truth_fov_deg"] = json!(fov);
                            row["truth_sky_tan_deg"] = json!(sky);
                            row["truth_centre_offset_from_geometric_px"] = json!(offset);
                            row["support"] = json!(support);
                            row["centres_aligned"] = json!((offset + shift - 0.5).abs() < 1e-12);
                            rows.push(row);
                        }
                    }
                }
            }
        }
    }
    std::fs::write(
        std::env::var("STARGLYPH_CENTRE_OUTPUT")?,
        serde_json::to_vec_pretty(&json!({"cases":rows}))?,
    )?;
    Ok(())
}
