//! Celestial geometry: unit sphere, camera pose, pinhole projection.

use std::f64::consts::PI;

use nalgebra::{Matrix3, RowVector3, Vector3};

const DEG_TO_RAD: f64 = PI / 180.0;
const RAD_TO_DEG: f64 = 180.0 / PI;
const BEHIND_CAMERA_Z: f64 = 0.05;

/// Solved camera pose and intrinsics for overlay generation.
#[derive(Debug, Clone, PartialEq)]
pub struct CameraSolution {
    pub ra_deg: f64,
    pub dec_deg: f64,
    pub roll_deg: f64,
    pub focal_px: f64,
    pub k1: f64,
    pub width: u32,
    pub height: u32,
}

impl CameraSolution {
    /// Horizontal field of view in degrees.
    #[must_use]
    pub fn fov_x_deg(&self) -> f64 {
        2.0 * (self.width as f64 / (2.0 * self.focal_px)).atan() * RAD_TO_DEG
    }

    /// Vertical field of view in degrees.
    #[must_use]
    pub fn fov_y_deg(&self) -> f64 {
        2.0 * (self.height as f64 / (2.0 * self.focal_px)).atan() * RAD_TO_DEG
    }

    /// World-to-camera rotation (rows: right, up, forward).
    #[must_use]
    pub fn rotation(&self) -> Matrix3<f64> {
        pose_to_rotation(self.ra_deg, self.dec_deg, self.roll_deg)
    }

    #[must_use]
    pub fn principal_point(&self) -> (f64, f64) {
        principal_point(self.width, self.height)
    }
}

/// Equatorial J2000 unit vector from RA/Dec in degrees.
#[must_use]
pub fn radec_to_unit(ra_deg: f64, dec_deg: f64) -> [f64; 3] {
    let ra = ra_deg * DEG_TO_RAD;
    let dec = dec_deg * DEG_TO_RAD;
    let cos_dec = dec.cos();
    [cos_dec * ra.cos(), cos_dec * ra.sin(), dec.sin()]
}

/// RA/Dec in degrees from a unit vector on the celestial sphere.
#[must_use]
pub fn unit_to_radec(unit: [f64; 3]) -> (f64, f64) {
    let dec = unit[2].clamp(-1.0, 1.0).asin() * RAD_TO_DEG;
    let ra = unit[1].atan2(unit[0]) * RAD_TO_DEG;
    let ra_deg = if ra < 0.0 { ra + 360.0 } else { ra };
    (ra_deg, dec)
}

/// Principal point matching `simulator-core` (`width/2`, `height/2`).
#[must_use]
pub fn principal_point(width: u32, height: u32) -> (f64, f64) {
    (width as f64 * 0.5, height as f64 * 0.5)
}

/// World-to-camera rotation; rows are [right; up; forward] in world coordinates.
#[must_use]
pub fn pose_to_rotation(ra_deg: f64, dec_deg: f64, roll_deg: f64) -> Matrix3<f64> {
    let forward = radec_to_unit(ra_deg, dec_deg);
    let world_up = if forward[2].abs() > 0.99 {
        [0.0, 1.0, 0.0]
    } else {
        [0.0, 0.0, 1.0]
    };
    let mut right = normalize(cross(forward, world_up));
    let mut up = normalize(cross(right, forward));

    let roll = roll_deg * DEG_TO_RAD;
    let cos_roll = roll.cos();
    let sin_roll = roll.sin();
    let rolled_right = [
        right[0] * cos_roll + up[0] * sin_roll,
        right[1] * cos_roll + up[1] * sin_roll,
        right[2] * cos_roll + up[2] * sin_roll,
    ];
    let rolled_up = [
        -right[0] * sin_roll + up[0] * cos_roll,
        -right[1] * sin_roll + up[1] * cos_roll,
        -right[2] * sin_roll + up[2] * cos_roll,
    ];
    right = normalize(rolled_right);
    up = normalize(rolled_up);

    Matrix3::from_rows(&[
        RowVector3::new(right[0], right[1], right[2]),
        RowVector3::new(up[0], up[1], up[2]),
        RowVector3::new(forward[0], forward[1], forward[2]),
    ])
}

/// Recover boresight pose from a world-to-camera rotation matrix.
#[must_use]
pub fn rotation_to_pose(rot: &Matrix3<f64>) -> (f64, f64, f64) {
    let forward = normalize_arr([rot[(2, 0)], rot[(2, 1)], rot[(2, 2)]]);
    let (ra_deg, dec_deg) = unit_to_radec(forward);
    let unrolled = pose_to_rotation(ra_deg, dec_deg, 0.0);
    let right0 = unrolled.row(0).transpose();
    let up0 = unrolled.row(1).transpose();
    let right = Vector3::new(rot[(0, 0)], rot[(0, 1)], rot[(0, 2)]);
    let roll_deg = right.dot(&up0).atan2(right.dot(&right0)) * RAD_TO_DEG;
    (ra_deg, dec_deg, roll_deg)
}

/// Project a world-unit direction to pixel coordinates; `None` when behind the camera.
#[must_use]
pub fn project(
    rot: &Matrix3<f64>,
    f: f64,
    k1: f64,
    width: u32,
    height: u32,
    world_unit: [f64; 3],
) -> Option<(f64, f64)> {
    let w = Vector3::new(world_unit[0], world_unit[1], world_unit[2]);
    let cam = rot * w;
    let xc = cam.x;
    let yc = cam.y;
    let zc = cam.z;
    if zc <= BEHIND_CAMERA_Z {
        return None;
    }
    let u = xc / zc;
    let v = yc / zc;
    let r2 = u * u + v * v;
    // For k1 < 0 the distorted radius r·(1 + k1·r²) peaks at r² = -1/(3·k1) and then
    // folds back, mirroring far off-axis points through the frame centre. Treat points
    // beyond the monotonic range as invisible.
    if k1 < 0.0 && r2 >= -1.0 / (3.0 * k1) {
        return None;
    }
    let factor = 1.0 + k1 * r2;
    let (cx, cy) = principal_point(width, height);
    let x_px = cx + f * u * factor;
    let y_px = cy - f * v * factor;
    Some((x_px, y_px))
}

/// Unproject a pixel to a world-unit direction (inverse of [`project`]).
#[must_use]
pub fn unproject(
    rot: &Matrix3<f64>,
    f: f64,
    k1: f64,
    width: u32,
    height: u32,
    x_px: f64,
    y_px: f64,
) -> [f64; 3] {
    let (cx, cy) = principal_point(width, height);
    let mut u = (x_px - cx) / f;
    let mut v = (cy - y_px) / f;
    if k1.abs() > f64::EPSILON {
        for _ in 0..3 {
            let r2 = u * u + v * v;
            let scale = 1.0 + k1 * r2;
            u = (x_px - cx) / (f * scale);
            v = (cy - y_px) / (f * scale);
        }
    }
    let cam = Vector3::new(u, v, 1.0).normalize();
    let world = rot.transpose() * cam;
    normalize_arr([world.x, world.y, world.z])
}

/// Angular separation between unit directions, in degrees.
#[must_use]
pub fn angular_sep(u1: [f64; 3], u2: [f64; 3]) -> f64 {
    let dot = u1[0] * u2[0] + u1[1] * u2[1] + u1[2] * u2[2];
    dot.clamp(-1.0, 1.0).acos() * RAD_TO_DEG
}

/// Spherical linear interpolation between unit vectors.
#[must_use]
pub fn slerp(u1: [f64; 3], u2: [f64; 3], t: f64) -> [f64; 3] {
    let mut dot = u1[0] * u2[0] + u1[1] * u2[1] + u1[2] * u2[2];
    let mut v2 = u2;
    if dot < 0.0 {
        dot = -dot;
        v2 = [-u2[0], -u2[1], -u2[2]];
    }
    if dot > 0.9995 {
        return normalize_arr([
            u1[0] + t * (v2[0] - u1[0]),
            u1[1] + t * (v2[1] - u1[1]),
            u1[2] + t * (v2[2] - u1[2]),
        ]);
    }
    let omega = dot.clamp(-1.0, 1.0).acos();
    let sin_omega = omega.sin();
    if sin_omega.abs() < f64::EPSILON {
        return u1;
    }
    let a = ((1.0 - t) * omega).sin() / sin_omega;
    let b = (t * omega).sin() / sin_omega;
    normalize_arr([
        a * u1[0] + b * v2[0],
        a * u1[1] + b * v2[1],
        a * u1[2] + b * v2[2],
    ])
}

#[must_use]
fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

#[must_use]
fn normalize(v: [f64; 3]) -> [f64; 3] {
    normalize_arr(v)
}

#[must_use]
fn normalize_arr(v: [f64; 3]) -> [f64; 3] {
    let norm = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if norm <= f64::EPSILON {
        [0.0, 0.0, 0.0]
    } else {
        [v[0] / norm, v[1] / norm, v[2] / norm]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn radec_unit_roundtrip() {
        let ra = 123.45;
        let dec = -33.2;
        let unit = radec_to_unit(ra, dec);
        let (ra2, dec2) = unit_to_radec(unit);
        assert!((ra - ra2).abs() < 1e-10);
        assert!((dec - dec2).abs() < 1e-10);
    }

    #[test]
    fn pose_rotation_roundtrip() {
        let ra = 83.6;
        let dec = 5.0;
        let roll = 12.3;
        let rot = pose_to_rotation(ra, dec, roll);
        let (ra2, dec2, roll2) = rotation_to_pose(&rot);
        assert!((ra - ra2).abs() < 1e-8);
        assert!((dec - dec2).abs() < 1e-8);
        assert!((roll - roll2).abs() < 1e-6);
    }

    #[test]
    fn project_unproject_roundtrip_k0() {
        let rot = pose_to_rotation(45.0, 20.0, 5.0);
        let f = 800.0;
        let w = 740;
        let h = 576;
        let world = radec_to_unit(50.0, 15.0);
        let (x, y) = project(&rot, f, 0.0, w, h, world).expect("visible");
        let back = unproject(&rot, f, 0.0, w, h, x, y);
        assert!(angular_sep(world, back) < 1e-6);
    }

    #[test]
    fn slerp_ten_degree_segment_has_at_least_ten_pieces() {
        let u1 = radec_to_unit(0.0, 0.0);
        let u2 = radec_to_unit(10.0, 0.0);
        let sep = angular_sep(u1, u2);
        assert!(sep > 9.9);
        let pieces = (sep / 1.0).ceil() as usize;
        assert!(pieces >= 10);
    }

    #[test]
    fn simulator_projection_cross_test() {
        use simulator_core::camera::{CameraExtrinsics, CameraIntrinsics};
        use simulator_core::catalog::Star as SimStar;
        use simulator_core::projection::project_star;

        let width = 740u32;
        let height = 576u32;
        let focal = 812.5f64;
        let ra = 83.633;
        let dec = 22.0145;
        let roll = 5.2;

        let intrinsics = CameraIntrinsics {
            width_px: width,
            height_px: height,
            fx: focal as f32,
            fy: focal as f32,
            cx: width as f32 * 0.5,
            cy: height as f32 * 0.5,
        };
        let extrinsics = CameraExtrinsics {
            ra_deg: ra as f32,
            dec_deg: dec as f32,
            roll_deg: roll as f32,
            position: [0.0, 0.0, 0.0],
        };
        let rot = pose_to_rotation(ra, dec, roll);

        let test_stars = [
            (ra, dec),
            (ra + 5.0, dec + 3.0),
            (ra - 8.0, dec - 2.0),
            (ra + 15.0, dec - 10.0),
            (ra - 20.0, dec + 5.0),
        ];

        for (i, (s_ra, s_dec)) in test_stars.into_iter().enumerate() {
            let sim_star = SimStar {
                id: format!("star{i}"),
                ra_deg: s_ra as f32,
                dec_deg: s_dec as f32,
                mag_v: 1.0,
            };
            let projected = project_star(&sim_star, &intrinsics, &extrinsics);
            let world = radec_to_unit(s_ra, s_dec);
            let ours = project(&rot, focal, 0.0, width, height, world).expect("visible");
            assert!(
                (projected.x_px as f64 - ours.0).abs() < 1e-3,
                "x mismatch star {i}: sim={} ours={}",
                projected.x_px,
                ours.0
            );
            assert!(
                (projected.y_px as f64 - ours.1).abs() < 1e-3,
                "y mismatch star {i}: sim={} ours={}",
                projected.y_px,
                ours.1
            );
        }
    }
}
