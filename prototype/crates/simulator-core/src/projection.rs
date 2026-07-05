use crate::camera::{CameraExtrinsics, CameraIntrinsics};
use crate::catalog::Star;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Visibility {
    Visible,
    OutOfFrustum,
    BehindCamera,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectedStar {
    pub id: String,
    pub ra_deg: f32,
    pub dec_deg: f32,
    pub mag_v: f32,
    pub x_px: f32,
    pub y_px: f32,
    pub visibility: Visibility,
}

impl ProjectedStar {
    #[must_use]
    pub fn is_visible(&self) -> bool {
        self.visibility == Visibility::Visible
    }
}

pub fn project_star(
    star: &Star,
    intrinsics: &CameraIntrinsics,
    extrinsics: &CameraExtrinsics,
) -> ProjectedStar {
    let basis = extrinsics.basis();
    let star_vector = crate::camera::spherical_to_cartesian(star.ra_deg, star.dec_deg);
    let x_cam = star_vector[0] * basis.right[0]
        + star_vector[1] * basis.right[1]
        + star_vector[2] * basis.right[2];
    let y_cam =
        star_vector[0] * basis.up[0] + star_vector[1] * basis.up[1] + star_vector[2] * basis.up[2];
    let z_cam = star_vector[0] * basis.forward[0]
        + star_vector[1] * basis.forward[1]
        + star_vector[2] * basis.forward[2];

    let (x_px, y_px, visibility) = if z_cam <= 0.0 {
        (f32::NAN, f32::NAN, Visibility::BehindCamera)
    } else {
        let x_px = intrinsics.fx * (x_cam / z_cam) + intrinsics.cx;
        let y_px = intrinsics.cy - intrinsics.fy * (y_cam / z_cam);
        let in_bounds = x_px >= 0.0
            && x_px < intrinsics.width_px as f32
            && y_px >= 0.0
            && y_px < intrinsics.height_px as f32;
        let visibility = if in_bounds {
            Visibility::Visible
        } else {
            Visibility::OutOfFrustum
        };
        (x_px, y_px, visibility)
    };

    ProjectedStar {
        id: star.id.clone(),
        ra_deg: star.ra_deg,
        dec_deg: star.dec_deg,
        mag_v: star.mag_v,
        x_px,
        y_px,
        visibility,
    }
}

#[must_use]
pub fn project_catalog(
    stars: &[Star],
    intrinsics: &CameraIntrinsics,
    extrinsics: &CameraExtrinsics,
) -> Vec<ProjectedStar> {
    stars
        .iter()
        .map(|star| project_star(star, intrinsics, extrinsics))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::camera::CameraExtrinsics;
    use crate::camera::CameraIntrinsics;
    use crate::catalog::Star;
    use crate::config::CameraConfig;

    #[test]
    fn center_star_projects_to_principal_point() {
        let intrinsics = CameraIntrinsics::from_camera_config(&CameraConfig {
            width_px: 1000,
            height_px: 1000,
            fov_deg: 60.0,
        });
        let extrinsics = CameraExtrinsics {
            ra_deg: 0.0,
            dec_deg: 0.0,
            roll_deg: 0.0,
            position: [0.0, 0.0, 0.0],
        };
        let star = Star {
            id: "center".to_string(),
            ra_deg: 0.0,
            dec_deg: 0.0,
            mag_v: 1.0,
        };

        let projected = project_star(&star, &intrinsics, &extrinsics);
        assert_eq!(projected.visibility, Visibility::Visible);
        assert!((projected.x_px - intrinsics.cx).abs() < 1e-4);
        assert!((projected.y_px - intrinsics.cy).abs() < 1e-4);
    }

    #[test]
    fn behind_camera_is_classified() {
        let intrinsics = CameraIntrinsics::from_camera_config(&CameraConfig {
            width_px: 1000,
            height_px: 1000,
            fov_deg: 60.0,
        });
        let extrinsics = CameraExtrinsics {
            ra_deg: 0.0,
            dec_deg: 0.0,
            roll_deg: 0.0,
            position: [0.0, 0.0, 0.0],
        };
        let star = Star {
            id: "behind".to_string(),
            ra_deg: 180.0,
            dec_deg: 0.0,
            mag_v: 1.0,
        };

        let projected = project_star(&star, &intrinsics, &extrinsics);
        assert_eq!(projected.visibility, Visibility::BehindCamera);
    }

    #[test]
    fn deterministic_projection_for_fixed_inputs() {
        let intrinsics = CameraIntrinsics::from_camera_config(&CameraConfig {
            width_px: 1000,
            height_px: 1000,
            fov_deg: 60.0,
        });
        let extrinsics = CameraExtrinsics {
            ra_deg: 45.0,
            dec_deg: -10.0,
            roll_deg: 7.5,
            position: [0.0, 0.0, 0.0],
        };
        let star = Star {
            id: "deterministic".to_string(),
            ra_deg: 41.2,
            dec_deg: -8.9,
            mag_v: 2.0,
        };

        let first = project_star(&star, &intrinsics, &extrinsics);
        let second = project_star(&star, &intrinsics, &extrinsics);
        assert_eq!(first.visibility, second.visibility);
        assert!((first.x_px - second.x_px).abs() < f32::EPSILON);
        assert!((first.y_px - second.y_px).abs() < f32::EPSILON);
    }
}
