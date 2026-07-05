use anyhow::{Context, Result};
use image::codecs::jpeg::JpegEncoder;
use image::{DynamicImage, GrayImage, ImageFormat};
use rand::Rng;
use serde::{Deserialize, Serialize};

use crate::config::DegradationConfig;
use crate::rendering::RenderFrame;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppliedDegradations {
    pub shot_noise: bool,
    pub read_noise: bool,
    pub blur_sigma_px: f32,
    pub jpeg_quality: u8,
}

pub fn apply_baseline_degradations(
    mut frame: RenderFrame,
    config: &DegradationConfig,
    rng: &mut impl Rng,
) -> Result<(RenderFrame, AppliedDegradations)> {
    if config.shot_noise {
        apply_shot_noise(&mut frame, config.shot_noise_scale, rng);
    }
    if config.read_noise {
        apply_read_noise(&mut frame, config.read_noise_sigma, rng);
    }
    if config.blur_sigma_px > 0.0 {
        apply_gaussian_blur(&mut frame, config.blur_sigma_px);
    }
    if config.jpeg_quality < 100 {
        apply_jpeg_artifacts(&mut frame, config.jpeg_quality)?;
    }

    frame.pixels.iter_mut().for_each(|pixel| {
        *pixel = pixel.clamp(0.0, f32::from(frame.dynamic_range_max));
    });

    Ok((
        frame,
        AppliedDegradations {
            shot_noise: config.shot_noise,
            read_noise: config.read_noise,
            blur_sigma_px: config.blur_sigma_px,
            jpeg_quality: config.jpeg_quality,
        },
    ))
}

fn apply_shot_noise(frame: &mut RenderFrame, shot_noise_scale: f32, rng: &mut impl Rng) {
    if shot_noise_scale <= 0.0 {
        return;
    }
    for pixel in &mut frame.pixels {
        let std_dev = pixel.sqrt() * shot_noise_scale;
        *pixel += gaussian_sample(rng, 0.0, std_dev);
    }
}

fn apply_read_noise(frame: &mut RenderFrame, sigma: f32, rng: &mut impl Rng) {
    if sigma <= 0.0 {
        return;
    }
    for pixel in &mut frame.pixels {
        *pixel += gaussian_sample(rng, 0.0, sigma);
    }
}

fn apply_gaussian_blur(frame: &mut RenderFrame, sigma: f32) {
    let radius = (sigma * 3.0).ceil() as i32;
    if radius <= 0 {
        return;
    }
    let kernel = gaussian_kernel(radius, sigma.max(1e-3));
    let width = frame.width_px as usize;
    let height = frame.height_px as usize;

    let mut horizontal = vec![0.0_f32; frame.pixels.len()];
    for y in 0..height {
        for x in 0..width {
            let mut acc = 0.0;
            for (idx, weight) in kernel.iter().enumerate() {
                let offset = idx as i32 - radius;
                let sample_x = (x as i32 + offset).clamp(0, width as i32 - 1) as usize;
                acc += frame.pixels[y * width + sample_x] * weight;
            }
            horizontal[y * width + x] = acc;
        }
    }

    let mut blurred = vec![0.0_f32; frame.pixels.len()];
    for y in 0..height {
        for x in 0..width {
            let mut acc = 0.0;
            for (idx, weight) in kernel.iter().enumerate() {
                let offset = idx as i32 - radius;
                let sample_y = (y as i32 + offset).clamp(0, height as i32 - 1) as usize;
                acc += horizontal[sample_y * width + x] * weight;
            }
            blurred[y * width + x] = acc;
        }
    }
    frame.pixels = blurred;
}

fn apply_jpeg_artifacts(frame: &mut RenderFrame, jpeg_quality: u8) -> Result<()> {
    let gray = frame.to_gray_image();
    let mut encoded = Vec::new();
    JpegEncoder::new_with_quality(&mut encoded, jpeg_quality)
        .encode_image(&DynamicImage::ImageLuma8(gray))
        .context("failed to JPEG encode degraded frame")?;

    let decoded = image::load_from_memory_with_format(&encoded, ImageFormat::Jpeg)
        .context("failed to decode JPEG degraded frame")?
        .to_luma8();
    frame.pixels = u8_image_to_dynamic(decoded, frame.dynamic_range_max);
    Ok(())
}

fn u8_image_to_dynamic(image: GrayImage, dynamic_range_max: u16) -> Vec<f32> {
    let scale = f32::from(dynamic_range_max) / 255.0;
    image
        .pixels()
        .map(|pixel| f32::from(pixel.0[0]) * scale)
        .collect()
}

fn gaussian_kernel(radius: i32, sigma: f32) -> Vec<f32> {
    let mut kernel = Vec::with_capacity((radius * 2 + 1) as usize);
    let mut sum = 0.0;
    for i in -radius..=radius {
        let distance2 = (i * i) as f32;
        let weight = (-distance2 / (2.0 * sigma * sigma)).exp();
        kernel.push(weight);
        sum += weight;
    }
    kernel.iter_mut().for_each(|weight| *weight /= sum);
    kernel
}

fn gaussian_sample(rng: &mut impl Rng, mean: f32, std_dev: f32) -> f32 {
    if std_dev <= 0.0 {
        return mean;
    }
    let u1 = rng.random::<f32>().max(f32::MIN_POSITIVE);
    let u2 = rng.random::<f32>();
    let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
    mean + z * std_dev
}
