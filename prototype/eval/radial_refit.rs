//! Isolated fixed-pair diagnostic, never linked into the production solver.
use super::*;
use serde::Deserialize;
use serde_json::{json, Value};
use std::cell::Cell;

#[derive(Default)]
struct Budget {
    iterations: Cell<usize>,
    evaluations: Cell<usize>,
    stop: Cell<&'static str>,
}

// Generated from the current production LM loop and finite-difference Jacobian.
include!("radial_optimizer.rs");

/// The entire radial path must stay monotone, not just its endpoint.
fn monotone(k1: f64, k2: f64, r2: f64) -> bool {
    let derivative = |t: f64| 1. + 3. * k1 * t + 5. * k2 * t * t;
    let mut minimum = derivative(0.).min(derivative(r2));
    if k2 > 0. {
        let vertex = (-3. * k1 / (10. * k2)).clamp(0., r2);
        minimum = minimum.min(derivative(vertex));
    }
    minimum.is_finite() && minimum > 0.
}

fn project_radial(camera: &CameraSolution, k2: f64, world: [f64; 3]) -> Option<(f64, f64)> {
    let rot = camera.rotation();
    project_rotated(camera, &rot, k2, world)
}

fn project_rotated(
    camera: &CameraSolution,
    rot: &nalgebra::Matrix3<f64>,
    k2: f64,
    world: [f64; 3],
) -> Option<(f64, f64)> {
    // Exact nesting, including the existing negative-k1 guard.
    if k2 == 0. {
        return geom::project(
            rot,
            camera.focal_px,
            camera.k1,
            camera.width,
            camera.height,
            world,
        );
    }
    let cam = rot * nalgebra::Vector3::from(world);
    if cam.z <= 0.05 || !camera.focal_px.is_finite() || camera.focal_px <= 0. {
        return None;
    }
    let u = cam.x / cam.z;
    let v = cam.y / cam.z;
    let r2 = u * u + v * v;
    if !monotone(camera.k1, k2, r2) {
        return None;
    }
    let factor = 1. + camera.k1 * r2 + k2 * r2 * r2;
    let (cx, cy) = camera.principal_point();
    let x = cx + camera.focal_px * u * factor;
    let y = cy - camera.focal_px * v * factor;
    (x.is_finite() && y.is_finite()).then_some((x, y))
}

fn residuals(
    p: &[f64],
    free_k1: bool,
    weight: f64,
    width: u32,
    height: u32,
    matches: &[Match],
) -> DVector<f64> {
    if p.len() != 6 {
        return super::residuals(p, free_k1, weight, width, height, matches);
    }
    let camera = CameraSolution {
        ra_deg: p[0],
        dec_deg: p[1],
        roll_deg: p[2],
        focal_px: p[3],
        k1: p[4],
        width,
        height,
    };
    let mut r = DVector::zeros(2 * matches.len() + 2);
    let rot = camera.rotation();
    for (i, m) in matches.iter().enumerate() {
        if let Some((x, y)) = project_rotated(&camera, &rot, p[5], m.world) {
            r[2 * i] = x - m.px;
            r[2 * i + 1] = y - m.py;
        } else {
            r[2 * i] = 1e3;
            r[2 * i + 1] = 1e3;
        }
    }
    r[2 * matches.len()] = weight * p[4];
    r[2 * matches.len() + 1] = weight * p[5];
    r
}

#[derive(Deserialize)]
struct SavedCamera {
    focal_px: f64,
    k1: f64,
    width: u32,
    height: u32,
    world_to_camera: [[f64; 3]; 3],
}
impl SavedCamera {
    fn camera(&self) -> CameraSolution {
        let matrix = nalgebra::Matrix3::from_fn(|i, j| self.world_to_camera[i][j]);
        let (ra_deg, dec_deg, roll_deg) = geom::rotation_to_pose(&matrix);
        let camera = CameraSolution {
            ra_deg,
            dec_deg,
            roll_deg,
            focal_px: self.focal_px,
            k1: self.k1,
            width: self.width,
            height: self.height,
        };
        assert!((camera.rotation() - matrix).norm() < 1e-12);
        camera
    }
}
#[derive(Deserialize)]
struct Source {
    world: [f64; 3],
    xy: [f64; 2],
    outside_both_inputs: bool,
}
#[derive(Deserialize)]
struct Frame {
    id: String,
    initial: SavedCamera,
    sources: Vec<Source>,
}
#[derive(Deserialize)]
struct Input {
    split: String,
    frames: Vec<Frame>,
}

fn camera_json(camera: &CameraSolution, k2: f64) -> Value {
    let r = camera.rotation();
    json!({"ra_deg":camera.ra_deg,"dec_deg":camera.dec_deg,"roll_deg":camera.roll_deg,
        "focal_px":camera.focal_px,"k1":camera.k1,"k2":k2,"width":camera.width,"height":camera.height,
        "world_to_camera":(0..3).map(|i| (0..3).map(|j| r[(i,j)]).collect::<Vec<_>>()).collect::<Vec<_>>()})
}

#[test]
#[ignore = "frozen development diagnostic: STARGLYPH_RADIAL_INPUT and STARGLYPH_RADIAL_OUTPUT"]
fn fixed_pairs() -> Result<(), Box<dyn std::error::Error>> {
    let input: Input =
        serde_json::from_slice(&std::fs::read(std::env::var("STARGLYPH_RADIAL_INPUT")?)?)?;
    assert_eq!(input.split, "development");
    assert_eq!(
        input
            .frames
            .iter()
            .map(|f| f.id.as_str())
            .collect::<Vec<_>>(),
        ["wm_r_132162731", "wm_r_143159342", "wm_r_149276071"]
    );
    let mut frames = Vec::new();
    for frame in input.frames {
        let initial = frame.initial.camera();
        let matches: Vec<_> = frame
            .sources
            .iter()
            .filter(|s| !s.outside_both_inputs)
            .map(|s| Match {
                world: s.world,
                px: s.xy[0],
                py: s.xy[1],
            })
            .collect();
        assert!(matches.len() >= 8);
        let native = refine_pose_with_scale(&initial, &matches, 1.);
        let mut cases = Vec::new();
        for extra in [false, true] {
            let budget = Budget::default();
            let start = Instant::now();
            let params = fit(&initial, &matches, extra, &budget);
            let elapsed_ms = start.elapsed().as_secs_f64() * 1000.;
            let camera = to_solution(&params, true, &initial);
            let k2 = if extra { params[5] } else { 0. };
            if !extra {
                assert_eq!(
                    camera, native,
                    "five-parameter control must exactly match production"
                );
            }
            let projected: Vec<_> = frame
                .sources
                .iter()
                .map(|s| project_radial(&camera, k2, s.world))
                .collect();
            let weight = k1_reg_weight(initial.fov_x_deg());
            let residual = residuals(&params, true, weight, camera.width, camera.height, &matches);
            cases.push(json!({"name":if extra {"k1_k2"} else {"k1"}, "camera":camera_json(&camera,k2),
                "matches":matches.len(),"prior_weight":weight,"elapsed_ms":elapsed_ms,
                "iterations":budget.iterations.get(),"residual_evaluations":budget.evaluations.get(),
                "stop":budget.stop.get(),"control_exact":!extra,"projected_xy":projected,
                "objective":residual.norm_squared(),"fit_residuals":&residual.as_slice()[..2*matches.len()]}));
        }
        frames.push(json!({"id":frame.id,"initial":camera_json(&initial,0.),"cases":cases}));
    }
    std::fs::write(
        std::env::var("STARGLYPH_RADIAL_OUTPUT")?,
        serde_json::to_vec_pretty(&json!({"split":"development","frames":frames}))?,
    )?;
    Ok(())
}

#[test]
fn rejects_fold_even_after_derivative_recovers() {
    assert!(!monotone(-1., 0.1, 10.));
    assert!(monotone(-0.1, 0.1, 10.));
    assert!(!monotone(-0.5, 0., 1.));
}

#[test]
fn nested_projection_retains_existing_guard() {
    let camera = CameraSolution {
        ra_deg: 250.,
        dec_deg: -5.,
        roll_deg: 17.,
        focal_px: 800.,
        k1: -0.2,
        width: 1200,
        height: 1600,
    };
    for ra in [250., 255., 280., 310., 70.] {
        let world = geom::radec_to_unit(ra, -8.);
        assert_eq!(
            project_radial(&camera, 0., world),
            geom::project(
                &camera.rotation(),
                camera.focal_px,
                camera.k1,
                camera.width,
                camera.height,
                world
            )
        );
    }
}

#[test]
fn synthetic_extra_term_recovery() {
    let camera = CameraSolution {
        ra_deg: 250.,
        dec_deg: -5.,
        roll_deg: 17.,
        focal_px: 800.,
        k1: -0.05,
        width: 1200,
        height: 1600,
    };
    let matches: Vec<_> = (-4..=4)
        .flat_map(|i| (-4..=4).map(move |j| (i, j)))
        .map(|(i, j)| {
            let direction =
                nalgebra::Vector3::new(f64::from(i) * 0.18, f64::from(j) * 0.18, 1.).normalize();
            let world = camera.rotation().transpose() * direction;
            let world = [world.x, world.y, world.z];
            let (px, py) = project_radial(&camera, 0.04, world).expect("synthetic monotonic ray");
            Match { world, px, py }
        })
        .collect();
    let budget = Budget::default();
    let p = fit(&camera, &matches, true, &budget);
    assert!((p[5] - 0.04).abs() < 0.001);
    assert!((p[4] + 0.05).abs() < 0.001);
    assert!(budget.iterations.get() <= 30);
    assert!(budget.evaluations.get() <= 1 + 30 * (7 + 10));
}
