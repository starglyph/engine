use rand::Rng;

use crate::camera::CameraExtrinsics;
use crate::config::CameraSamplingConfig;

pub fn sample_camera_pose(config: &CameraSamplingConfig, rng: &mut impl Rng) -> CameraExtrinsics {
    let ra_deg = rng.random_range(config.ra_range_deg.0..config.ra_range_deg.1);
    let dec_deg = rng.random_range(config.dec_range_deg.0..config.dec_range_deg.1);
    let roll_deg = rng.random_range(config.roll_range_deg.0..config.roll_range_deg.1);

    CameraExtrinsics {
        ra_deg,
        dec_deg,
        roll_deg,
        position: [0.0, 0.0, 0.0],
    }
}
