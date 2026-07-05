use image::{GrayImage, Luma};

use crate::config::RenderConfig;
use crate::projection::ProjectedStar;

#[derive(Debug, Clone)]
pub struct RenderFrame {
    pub width_px: u32,
    pub height_px: u32,
    pub dynamic_range_max: u16,
    pub pixels: Vec<f32>,
    pub rendered_stars: usize,
}

impl RenderFrame {
    #[must_use]
    pub fn to_gray_image(&self) -> GrayImage {
        let scale = if self.dynamic_range_max == 0 {
            0.0
        } else {
            255.0 / f32::from(self.dynamic_range_max)
        };
        let mut image = GrayImage::new(self.width_px, self.height_px);
        for (idx, pixel) in self.pixels.iter().copied().enumerate() {
            let x = (idx as u32) % self.width_px;
            let y = (idx as u32) / self.width_px;
            let u8_pixel = (pixel * scale).clamp(0.0, 255.0) as u8;
            image.put_pixel(x, y, Luma([u8_pixel]));
        }
        image
    }
}

/// Pogson-style mapping from apparent magnitude to linear intensity: `I = I_ref * 10^{-0.4 (m - m_ref)}`.
/// Documented for dataset consumers in `docs/data-contracts.md` (dataset v1).
#[must_use]
pub fn magnitude_to_intensity(magnitude: f32, config: &RenderConfig) -> f32 {
    let relative_flux = 10.0_f32.powf(-0.4 * (magnitude - config.reference_magnitude));
    config.reference_intensity * relative_flux
}

#[must_use]
pub fn render_stars(
    width_px: u32,
    height_px: u32,
    stars: &[ProjectedStar],
    config: &RenderConfig,
) -> RenderFrame {
    let pixel_count = (width_px as usize) * (height_px as usize);
    let mut pixels =
        vec![f32::from(config.dynamic_range_max) * config.background_level; pixel_count];
    let rendered_stars = stars.iter().filter(|star| star.is_visible()).count();
    let radius = (config.psf_sigma_px * 3.0).ceil() as i32;
    let sigma_squared = (config.psf_sigma_px * config.psf_sigma_px).max(1e-6);

    for star in stars.iter().filter(|star| star.is_visible()) {
        let peak_intensity =
            magnitude_to_intensity(star.mag_v, config).min(f32::from(config.dynamic_range_max));
        let center_x = star.x_px.round() as i32;
        let center_y = star.y_px.round() as i32;
        for dy in -radius..=radius {
            let y = center_y + dy;
            if y < 0 || y >= height_px as i32 {
                continue;
            }
            for dx in -radius..=radius {
                let x = center_x + dx;
                if x < 0 || x >= width_px as i32 {
                    continue;
                }
                let distance2 = (dx * dx + dy * dy) as f32;
                let weight = (-distance2 / (2.0 * sigma_squared)).exp();
                let idx = (y as usize) * width_px as usize + x as usize;
                pixels[idx] += peak_intensity * weight;
            }
        }
    }
    pixels
        .iter_mut()
        .for_each(|pixel| *pixel = pixel.clamp(0.0, f32::from(config.dynamic_range_max)));

    RenderFrame {
        width_px,
        height_px,
        dynamic_range_max: config.dynamic_range_max,
        pixels,
        rendered_stars,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::projection::{ProjectedStar, Visibility};

    fn visible_star(id: &str, x_px: f32, y_px: f32, mag_v: f32) -> ProjectedStar {
        ProjectedStar {
            id: id.to_string(),
            ra_deg: 0.0,
            dec_deg: 0.0,
            mag_v,
            x_px,
            y_px,
            visibility: Visibility::Visible,
        }
    }

    #[test]
    fn magnitude_mapping_preserves_brighter_higher_intensity() {
        let config = RenderConfig::default();
        let bright = magnitude_to_intensity(1.0, &config);
        let faint = magnitude_to_intensity(6.0, &config);
        assert!(
            bright > faint,
            "brighter star (lower magnitude) should map to higher intensity"
        );
    }

    #[test]
    fn brighter_star_has_higher_peak_than_fainter_star_same_frame() {
        let config = RenderConfig::default();
        let stars = [
            visible_star("a", 20.0, 22.0, 1.0),
            visible_star("b", 52.0, 54.0, 8.0),
        ];
        let frame = render_stars(64, 64, &stars, &config);
        let idx_bright = 22_usize * 64 + 20;
        let idx_faint = 54_usize * 64 + 52;
        assert!(
            frame.pixels[idx_bright] > frame.pixels[idx_faint],
            "peak at brighter star center should exceed peak at fainter star center"
        );
    }

    #[test]
    fn saturated_rendering_clips_to_dynamic_range_max() {
        let config = RenderConfig {
            psf_sigma_px: 0.9,
            dynamic_range_max: 100,
            reference_magnitude: 0.0,
            reference_intensity: 50_000.0,
            background_level: 0.02,
        };
        let stars = [visible_star("hot", 30.0, 30.0, -3.0)];
        let frame = render_stars(64, 64, &stars, &config);
        let cap = f32::from(config.dynamic_range_max);
        assert!(
            frame.pixels.iter().all(|&p| p <= cap && p >= 0.0),
            "all samples must stay within [0, dynamic_range_max] after peak clip and PSF accumulation"
        );
    }
}
