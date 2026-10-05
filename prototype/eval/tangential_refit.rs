//! Isolated nonlinear validation of the frozen p1/p2 diagnostic; not production.
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
include!("tangential_optimizer.rs");

// A conservative lower bound for the smallest distortion-Jacobian eigenvalue
// along the entire segment from the optical axis to (x,y), not just its endpoint.
fn ray_margin(x: f64, y: f64, k1: f64, p1: f64, p2: f64) -> f64 {
    let a = 2. * p1 * y + 6. * p2 * x;
    let b = 2. * p1 * x + 2. * p2 * y;
    let d = 6. * p1 * y + 2. * p2 * x;
    let tangent_min = (a + d) * 0.5 - ((a - d) * 0.5).hypot(b);
    1. + tangent_min.min(0.) + (3. * k1 * (x * x + y * y)).min(0.)
}

fn project_tangential(
    camera: &CameraSolution,
    rot: &nalgebra::Matrix3<f64>,
    p1: f64,
    p2: f64,
    world: [f64; 3],
) -> Option<(f64, f64)> {
    // Always retain the current camera's guards and exact nested projection.
    let base = geom::project(
        rot,
        camera.focal_px,
        camera.k1,
        camera.width,
        camera.height,
        world,
    )?;
    if p1 == 0. && p2 == 0. {
        return Some(base);
    }
    let cam = rot * nalgebra::Vector3::from(world);
    let x = cam.x / cam.z;
    let y = -cam.y / cam.z;
    let r2 = x * x + y * y;
    let margin = ray_margin(x, y, camera.k1, p1, p2);
    if !margin.is_finite() || margin <= 0. {
        return None;
    }
    let px = base.0 + camera.focal_px * (2. * p1 * x * y + p2 * (r2 + 2. * x * x));
    let py = base.1 + camera.focal_px * (p1 * (r2 + 2. * y * y) + 2. * p2 * x * y);
    (px.is_finite() && py.is_finite()).then_some((px, py))
}

fn residuals(
    p: &[f64],
    free_k1: bool,
    weight: f64,
    width: u32,
    height: u32,
    matches: &[Match],
) -> DVector<f64> {
    if p.len() != 7 {
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
    let rot = camera.rotation();
    let mut r = DVector::zeros(2 * matches.len() + 1);
    for (i, m) in matches.iter().enumerate() {
        if let Some((x, y)) = project_tangential(&camera, &rot, p[5], p[6], m.world) {
            r[2 * i] = x - m.px;
            r[2 * i + 1] = y - m.py;
        } else {
            r[2 * i] = 1e3;
            r[2 * i + 1] = 1e3;
        }
    }
    r[2 * matches.len()] = weight * p[4];
    r
}

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
    prior_weight: f64,
    extra: bool,
    expected: Option<SavedCamera>,
}
#[derive(Deserialize)]
struct Input {
    id: String,
    split: String,
    cases: Vec<Case>,
    probe_worlds: Vec<[f64; 3]>,
}

fn camera_json(camera: &CameraSolution, p1: f64, p2: f64) -> Value {
    let r = camera.rotation();
    json!({"ra_deg":camera.ra_deg,"dec_deg":camera.dec_deg,"roll_deg":camera.roll_deg,"focal_px":camera.focal_px,
        "k1":camera.k1,"p1":p1,"p2":p2,"width":camera.width,"height":camera.height,
        "world_to_camera":(0..3).map(|i|(0..3).map(|j|r[(i,j)]).collect::<Vec<_>>()).collect::<Vec<_>>()})
}

#[test]
#[ignore = "frozen development inputs: STARGLYPH_TANGENTIAL_INPUT and STARGLYPH_TANGENTIAL_OUTPUT"]
fn fixed_pairs() -> Result<(), Box<dyn std::error::Error>> {
    let input: Input = serde_json::from_slice(&std::fs::read(std::env::var(
        "STARGLYPH_TANGENTIAL_INPUT",
    )?)?)?;
    assert_eq!(input.id, "wm_r_132162731");
    assert_eq!(input.split, "development");
    assert_eq!(input.cases.len(), 18);
    let mut cases = Vec::new();
    for c in input.cases {
        let initial = c.initial.camera();
        let matches: Vec<_> = c
            .matches
            .iter()
            .map(|m| Match {
                world: m.world,
                px: m.xy[0],
                py: m.xy[1],
            })
            .collect();
        assert_eq!(matches.len(), 22);
        let budget = Budget::default();
        let start = Instant::now();
        let params = fit(&initial, &matches, c.extra, c.prior_weight, &budget);
        let elapsed_ms = start.elapsed().as_secs_f64() * 1000.;
        let camera = to_solution(&params, true, &initial);
        let (p1, p2) = if c.extra {
            (params[5], params[6])
        } else {
            (0., 0.)
        };
        let mut control_exact = None;
        if let Some(expected) = c.expected {
            assert_eq!(
                camera,
                expected.camera(),
                "saved control must reproduce exactly"
            );
            control_exact = Some(true);
        }
        let rot = camera.rotation();
        let probes: Vec<_> = input
            .probe_worlds
            .iter()
            .map(|&w| project_tangential(&camera, &rot, p1, p2, w))
            .collect();
        let fit_xy: Vec<_> = matches
            .iter()
            .map(|m| project_tangential(&camera, &rot, p1, p2, m.world))
            .collect();
        let min_margin = matches
            .iter()
            .map(|m| m.world)
            .chain(input.probe_worlds.iter().copied())
            .map(|w| {
                let ray = rot * nalgebra::Vector3::from(w);
                ray_margin(ray.x / ray.z, -ray.y / ray.z, camera.k1, p1, p2)
            })
            .fold(f64::INFINITY, f64::min);
        let residual = residuals(
            &params,
            true,
            c.prior_weight,
            camera.width,
            camera.height,
            &matches,
        );
        cases.push(json!({"name":c.name,"camera":camera_json(&camera,p1,p2),"control_exact":control_exact,
            "extra":c.extra,"matches":matches.len(),"prior_weight":c.prior_weight,"iterations":budget.iterations.get(),
            "residual_evaluations":budget.evaluations.get(),"stop":budget.stop.get(),"elapsed_ms":elapsed_ms,
            "objective":residual.norm_squared(),"residuals":residual.as_slice(),"fit_projected_xy":fit_xy,
            "probe_projected_xy":probes,"min_ray_margin":min_margin}));
    }
    std::fs::write(
        std::env::var("STARGLYPH_TANGENTIAL_OUTPUT")?,
        serde_json::to_vec_pretty(&json!({"id":input.id,"split":input.split,"cases":cases}))?,
    )?;
    Ok(())
}

fn synthetic_camera() -> CameraSolution {
    CameraSolution {
        ra_deg: 250.,
        dec_deg: -5.,
        roll_deg: 17.,
        focal_px: 800.,
        k1: 0.,
        width: 1200,
        height: 1600,
    }
}
fn synthetic_pairs(camera: &CameraSolution, p1: f64, p2: f64) -> Vec<Match> {
    (-4..=4)
        .flat_map(|i| (-4..=4).map(move |j| (i, j)))
        .map(|(i, j)| {
            let ray =
                nalgebra::Vector3::new(f64::from(i) * 0.18, f64::from(j) * 0.18, 1.).normalize();
            let w = camera.rotation().transpose() * ray;
            let world = [w.x, w.y, w.z];
            let (px, py) = project_tangential(camera, &camera.rotation(), p1, p2, world)
                .expect("synthetic valid ray");
            Match { world, px, py }
        })
        .collect()
}

#[test]
fn zero_terms_nest_and_preserve_negative_k1_guard() {
    let mut c = synthetic_camera();
    c.k1 = -0.2;
    for ra in [250., 255., 280., 310., 70.] {
        let w = geom::radec_to_unit(ra, -8.);
        assert_eq!(
            project_tangential(&c, &c.rotation(), 0., 0., w),
            geom::project(&c.rotation(), c.focal_px, c.k1, c.width, c.height, w)
        );
    }
    assert!(ray_margin(1., 0., 0., 0., -1.) < 0.);
    assert!(ray_margin(1., 1., -0.03, 0.002, -0.003) > 0.);
}

#[test]
fn production_five_parameter_loop_is_exact() {
    let c = synthetic_camera();
    let pairs = synthetic_pairs(&c, 0.002, -0.003);
    let b = Budget::default();
    let weight = 2.5;
    let p = fit(&c, &pairs, false, weight, &b);
    let native = refine_pose_with_scale(&c, &pairs, weight / k1_reg_weight(c.fov_x_deg()));
    assert_eq!(to_solution(&p, true, &c), native);
}

#[test]
fn synthetic_seven_parameter_recovery_and_budget() {
    let c = synthetic_camera();
    let pairs = synthetic_pairs(&c, 0.002, -0.003);
    let b = Budget::default();
    let p = fit(&c, &pairs, true, 2.5, &b);
    assert!((p[5] - 0.002).abs() < 1e-7);
    assert!((p[6] + 0.003).abs() < 1e-7);
    assert!(residuals(&p, true, 2.5, c.width, c.height, &pairs).norm() < 1e-5);
    assert!(b.iterations.get() <= 30);
    assert!(b.evaluations.get() <= 1 + 30 * (8 + 10));
}

#[test]
fn p1_p2_derivative_has_image_y_down_sign() {
    let c = synthetic_camera();
    let ray = nalgebra::Vector3::new(0.2, -0.3, 1.).normalize();
    let w = c.rotation().transpose() * ray;
    let world = [w.x, w.y, w.z];
    let zero = project_tangential(&c, &c.rotation(), 0., 0., world).expect("ray");
    for (p1, p2, expected) in [(1e-4, 0., [0.12, 0.31]), (0., 1e-4, [0.21, 0.12])] {
        let value = project_tangential(&c, &c.rotation(), p1, p2, world).expect("ray");
        assert!((value.0 - zero.0 - c.focal_px * 1e-4 * expected[0]).abs() < 1e-10);
        assert!((value.1 - zero.1 - c.focal_px * 1e-4 * expected[1]).abs() < 1e-10);
    }
}
