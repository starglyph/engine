//! Ideal TAN fixtures generated through tetra3's coordinate API, independently
//! of Starglyph projection/refinement. No image, catalogue search or acceptance.
use tetra3::{CameraModel, Quaternion, Solution};

pub struct Point {
    pub world: [f64; 3],
    pub xy: [f64; 2],
}

pub fn solution(size: [u32; 2], fov: f64, sky: [f64; 3], offset: f64) -> Solution {
    let mut camera = CameraModel::from_fov(fov.to_radians(), size[0], size[1]);
    camera.crpix = [offset; 2];
    Solution {
        // Adapter and TAN methods do not read quaternion/CD or acceptance fields.
        // This is an ideal input record, not a result of solve_from_centroids.
        qicrs2cam: Quaternion::new(1., 0., 0., 0.),
        fov_rad: fov.to_radians() as f32,
        num_matches: 20,
        rmse_rad: 0.,
        p90e_rad: 0.,
        max_err_rad: 0.,
        prob: 0.,
        solve_time_ms: 0.,
        parity_flip: false,
        matched_catalog_ids: (0..20).collect(),
        matched_centroid_indices: (0..20).collect(),
        image_width: size[0],
        image_height: size[1],
        cd_matrix: [[0.; 2]; 2],
        crval_rad: [sky[0].to_radians(), sky[1].to_radians()],
        camera_model: camera,
        theta_rad: sky[2].to_radians(),
    }
}

pub fn point(solution: &Solution, fraction: [f64; 2]) -> Point {
    let size = [solution.image_width, solution.image_height].map(f64::from);
    let xy = [fraction[0] * size[0], fraction[1] * size[1]];
    let midpoint = size.map(|d| (d - 1.) / 2.);
    let (ra, dec) = solution.pixel_to_world(xy[0] - midpoint[0], xy[1] - midpoint[1]);
    let (x, y) = solution.world_to_pixel(ra, dec).expect("visible TAN point");
    assert!((x + midpoint[0] - xy[0]).abs() < 1e-8);
    assert!((y + midpoint[1] - xy[1]).abs() < 1e-8);
    let (ra, dec) = (ra.to_radians(), dec.to_radians());
    Point {
        world: [dec.cos() * ra.cos(), dec.cos() * ra.sin(), dec.sin()],
        xy,
    }
}

pub fn training(solution: &Solution, upper_only: bool) -> Vec<Point> {
    let ys = if upper_only {
        [0.08, 0.16, 0.24, 0.32]
    } else {
        [0.1, 0.3, 0.7, 0.9]
    };
    ys.into_iter()
        .flat_map(|y| [0.1, 0.3, 0.5, 0.7, 0.9].map(|x| point(solution, [x, y])))
        .collect()
}

pub fn probes(solution: &Solution) -> Vec<Point> {
    (0..7)
        .flat_map(|y| {
            (0..9).map(move |x| {
                point(
                    solution,
                    [
                        0.03 + 0.94 * f64::from(x) / 8.,
                        0.04 + 0.92 * f64::from(y) / 6.,
                    ],
                )
            })
        })
        .collect()
}

#[test]
fn tan_control_and_training_probe_separation() {
    let sol = solution([1989, 1498], 85., [210., -35., -27.], 0.);
    let center = point(
        &sol,
        [(1989. - 1.) / (2. * 1989.), (1498. - 1.) / (2. * 1498.)],
    );
    let expected = [
        (-35_f64).to_radians().cos() * 210_f64.to_radians().cos(),
        (-35_f64).to_radians().cos() * 210_f64.to_radians().sin(),
        (-35_f64).to_radians().sin(),
    ];
    for (actual, expected) in center.world.iter().zip(expected) {
        assert!((actual - expected).abs() < 1e-12);
    }
    for upper in [false, true] {
        let train = training(&sol, upper);
        let probe = probes(&sol);
        assert_eq!(train.len(), 20);
        assert_eq!(probe.len(), 63);
        assert!(train.iter().all(|t| probe.iter().all(|p| t.xy != p.xy)));
    }
}
