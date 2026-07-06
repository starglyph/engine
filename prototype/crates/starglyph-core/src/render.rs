//! Rasterize solve results for humans: percentile-stretched grayscale frame
//! with overlay geometry (constellations, grid, stars, planets) and detection
//! markers drawn on top. Shared by the CLI (`--overlay-png`, `--debug-png`)
//! and the HTTP service (overlay PNG responses).

use image::{ImageBuffer, Rgb, RgbImage};

use crate::contracts::{SolveDetection, SolveOverlay, SolveReport};
use crate::image_input::FrameImage;

const CONSTELLATION_COLOR: Rgb<u8> = Rgb([0, 255, 255]);
const GRID_COLOR: Rgb<u8> = Rgb([80, 80, 80]);
const STAR_COLOR: Rgb<u8> = Rgb([255, 215, 0]);
const PLANET_COLOR: Rgb<u8> = Rgb([255, 194, 77]);
const INLIER_COLOR: Rgb<u8> = Rgb([0, 255, 0]);
const OUTLIER_COLOR: Rgb<u8> = Rgb([140, 140, 140]);

/// Grayscale base image: frame luma stretched to the p10..p99.95 range so
/// faint stars survive the 8-bit quantization.
#[must_use]
pub fn stretched_base(frame: &FrameImage) -> RgbImage {
    let p10 = percentile(&frame.gray, 0.10);
    let p9995 = percentile(&frame.gray, 0.9995);
    let span = (p9995 - p10).max(1e-6);

    let mut img: RgbImage = ImageBuffer::new(frame.width, frame.height);
    for y in 0..frame.height {
        for x in 0..frame.width {
            let v = frame.gray[(y * frame.width + x) as usize];
            let stretched = ((v - p10) / span).clamp(0.0, 1.0);
            let byte = (stretched * 255.0).round() as u8;
            img.put_pixel(x, y, Rgb([byte, byte, byte]));
        }
    }
    img
}

/// Draw overlay geometry: constellation lines, RA/Dec grid, star circles,
/// planet diamonds. Grid goes under the star/planet markers.
pub fn draw_overlay(img: &mut RgbImage, overlay: &SolveOverlay) {
    for constellation in &overlay.constellations {
        for line in &constellation.lines {
            for window in line.windows(2) {
                draw_line_bresenham(
                    img,
                    window[0][0],
                    window[0][1],
                    window[1][0],
                    window[1][1],
                    CONSTELLATION_COLOR,
                );
            }
        }
    }
    for grid_line in &overlay.grid {
        for window in grid_line.points.windows(2) {
            draw_line_bresenham(
                img,
                window[0][0],
                window[0][1],
                window[1][0],
                window[1][1],
                GRID_COLOR,
            );
        }
    }
    for star in &overlay.stars {
        draw_circle_outline(img, star.x, star.y, 4.0, STAR_COLOR);
    }
    for planet in &overlay.planets {
        draw_diamond(img, planet.x, planet.y, 7.0, PLANET_COLOR);
    }
}

/// Draw detection markers: green = matched inlier, gray = unmatched.
pub fn draw_detections(img: &mut RgbImage, detections: &[SolveDetection]) {
    for det in detections {
        let color = if det.inlier {
            INLIER_COLOR
        } else {
            OUTLIER_COLOR
        };
        draw_circle_outline(img, det.x, det.y, 6.0, color);
    }
}

/// Render a solved frame: overlay geometry (when present) plus detection
/// markers over the stretched base.
#[must_use]
pub fn render_report(frame: &FrameImage, report: &SolveReport) -> RgbImage {
    let mut img = stretched_base(frame);
    if let Some(overlay) = &report.overlay {
        draw_overlay(&mut img, overlay);
    }
    draw_detections(&mut img, &report.detections);
    img
}

/// Encode an image as PNG bytes (for HTTP responses; files go via `save`).
pub fn encode_png(img: &RgbImage) -> Result<Vec<u8>, image::ImageError> {
    let mut bytes = Vec::new();
    image::DynamicImage::ImageRgb8(img.clone()).write_to(
        &mut std::io::Cursor::new(&mut bytes),
        image::ImageFormat::Png,
    )?;
    Ok(bytes)
}

/// Value at quantile `p` (0..=1) of `values`, by sorting a copy.
#[must_use]
pub fn percentile(values: &[f32], p: f64) -> f32 {
    let mut sorted: Vec<f32> = values.to_vec();
    sorted.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let idx = ((sorted.len() as f64 - 1.0) * p).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

/// Bresenham line clipped to the image bounds.
pub fn draw_line_bresenham(img: &mut RgbImage, x0: f64, y0: f64, x1: f64, y1: f64, color: Rgb<u8>) {
    let width = img.width() as i32;
    let height = img.height() as i32;
    let mut x0 = x0.round() as i32;
    let mut y0 = y0.round() as i32;
    let x1 = x1.round() as i32;
    let y1 = y1.round() as i32;
    let dx = (x1 - x0).abs();
    let dy = -(y1 - y0).abs();
    let sx = if x0 < x1 { 1 } else { -1 };
    let sy = if y0 < y1 { 1 } else { -1 };
    let mut err = dx + dy;

    loop {
        if x0 >= 0 && y0 >= 0 && x0 < width && y0 < height {
            img.put_pixel(x0 as u32, y0 as u32, color);
        }
        if x0 == x1 && y0 == y1 {
            break;
        }
        let e2 = 2 * err;
        if e2 >= dy {
            err += dy;
            x0 += sx;
        }
        if e2 <= dx {
            err += dx;
            y0 += sy;
        }
    }
}

/// Circle outline sampled densely enough to be gap-free at any radius.
pub fn draw_circle_outline(img: &mut RgbImage, cx: f64, cy: f64, radius: f64, color: Rgb<u8>) {
    let width = img.width() as i32;
    let height = img.height() as i32;
    let steps = ((2.0 * std::f64::consts::PI * radius).ceil() as i32).max(32);
    for i in 0..steps {
        let theta = 2.0 * std::f64::consts::PI * f64::from(i) / f64::from(steps);
        let x = (cx + radius * theta.cos()).round() as i32;
        let y = (cy + radius * theta.sin()).round() as i32;
        if x >= 0 && y >= 0 && x < width && y < height {
            img.put_pixel(x as u32, y as u32, color);
        }
    }
}

/// Diamond outline with a center dot (planet marker).
pub fn draw_diamond(img: &mut RgbImage, cx: f64, cy: f64, radius: f64, color: Rgb<u8>) {
    let r = radius;
    let edges = [
        (cx, cy - r, cx + r, cy),
        (cx + r, cy, cx, cy + r),
        (cx, cy + r, cx - r, cy),
        (cx - r, cy, cx, cy - r),
    ];
    for (x0, y0, x1, y1) in edges {
        draw_line_bresenham(img, x0, y0, x1, y1, color);
    }
    let cx_i = cx.round() as i32;
    let cy_i = cy.round() as i32;
    if cx_i >= 0 && cy_i >= 0 && cx_i < img.width() as i32 && cy_i < img.height() as i32 {
        img.put_pixel(cx_i as u32, cy_i as u32, color);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::contracts::{SolveReport, SolveStatus};

    fn test_frame() -> FrameImage {
        let width = 32;
        let height = 24;
        let mut gray = vec![0.02_f32; (width * height) as usize];
        gray[(10 * width + 12) as usize] = 0.9;
        FrameImage {
            width,
            height,
            gray,
            source_name: "test".to_string(),
            exif: None,
        }
    }

    #[test]
    fn render_report_matches_frame_dimensions_and_marks_detections() {
        let frame = test_frame();
        let report = SolveReport {
            status: SolveStatus::Failed,
            failure: None,
            pose: None,
            fov: None,
            quality: None,
            timing_ms: None,
            detections: vec![SolveDetection {
                x: 12.0,
                y: 10.0,
                flux: 1.0,
                snr: 10.0,
                inlier: true,
            }],
            overlay: None,
        };
        let img = render_report(&frame, &report);
        assert_eq!((img.width(), img.height()), (frame.width, frame.height));
        // Detection circle of radius 6 around (12, 10) must hit (18, 10).
        assert_eq!(*img.get_pixel(18, 10), INLIER_COLOR);
    }

    #[test]
    fn encode_png_produces_png_magic() {
        let img = stretched_base(&test_frame());
        let bytes = encode_png(&img).expect("png encoding");
        assert_eq!(&bytes[..8], b"\x89PNG\r\n\x1a\n");
    }
}
