use serde::{Deserialize, Serialize};

use crate::config::CameraConfig;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CameraIntrinsics {
    pub width_px: u32,
    pub height_px: u32,
    pub fx: f32,
    pub fy: f32,
    pub cx: f32,
    pub cy: f32,
}

impl CameraIntrinsics {
    #[must_use]
    pub fn from_camera_config(config: &CameraConfig) -> Self {
        let fov_rad = config.fov_deg.to_radians();
        let fx = (config.width_px as f32 * 0.5) / (fov_rad * 0.5).tan();
        let fy = fx;
        let cx = config.width_px as f32 * 0.5;
        let cy = config.height_px as f32 * 0.5;
        Self {
            width_px: config.width_px,
            height_px: config.height_px,
            fx,
            fy,
            cx,
            cy,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CameraExtrinsics {
    pub ra_deg: f32,
    pub dec_deg: f32,
    pub roll_deg: f32,
    pub position: [f32; 3],
}

#[derive(Debug, Clone, Copy)]
pub struct CameraBasis {
    pub right: [f32; 3],
    pub up: [f32; 3],
    pub forward: [f32; 3],
}

impl CameraExtrinsics {
    #[must_use]
    pub fn basis(&self) -> CameraBasis {
        let forward = spherical_to_cartesian(self.ra_deg, self.dec_deg);
        let world_up = if forward[2].abs() > 0.99 {
            [0.0, 1.0, 0.0]
        } else {
            [0.0, 0.0, 1.0]
        };
        let mut right = normalize(cross(forward, world_up));
        let mut up = normalize(cross(right, forward));

        let roll = self.roll_deg.to_radians();
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

        CameraBasis { right, up, forward }
    }
}

#[must_use]
pub fn spherical_to_cartesian(ra_deg: f32, dec_deg: f32) -> [f32; 3] {
    let ra = ra_deg.to_radians();
    let dec = dec_deg.to_radians();
    let cos_dec = dec.cos();
    [cos_dec * ra.cos(), cos_dec * ra.sin(), dec.sin()]
}

#[must_use]
fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

#[must_use]
fn normalize(v: [f32; 3]) -> [f32; 3] {
    let norm = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if norm <= f32::EPSILON {
        [0.0, 0.0, 0.0]
    } else {
        [v[0] / norm, v[1] / norm, v[2] / norm]
    }
}
