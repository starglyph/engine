use image::GrayImage;

use crate::config::DetectionConfig;
use crate::contracts::{DetectionCandidate, DetectionMetrics, DetectionStageResult};
use crate::io::TruthStar;

pub fn detect_stars(image: &GrayImage, config: &DetectionConfig) -> DetectionStageResult {
    let width = image.width() as i32;
    let height = image.height() as i32;
    let mut raw = Vec::new();

    for y in 1..(height - 1) {
        for x in 1..(width - 1) {
            let center = image.get_pixel(x as u32, y as u32).0[0];
            if center < config.min_peak_value {
                continue;
            }
            if !is_local_maximum(image, x, y, center) {
                continue;
            }
            raw.push((x as f32, y as f32, f32::from(center)));
        }
    }

    raw.sort_by(|a, b| {
        b.2.total_cmp(&a.2)
            .then_with(|| a.1.total_cmp(&b.1))
            .then_with(|| a.0.total_cmp(&b.0))
    });

    let mut selected: Vec<(f32, f32, f32)> = Vec::with_capacity(config.max_candidates);
    let nms_radius = config.non_max_radius_px as f32;
    for candidate in raw {
        if selected.len() >= config.max_candidates {
            break;
        }
        if selected.iter().any(|(sx, sy, _)| {
            let dx = sx - candidate.0;
            let dy = sy - candidate.1;
            (dx * dx + dy * dy).sqrt() <= nms_radius
        }) {
            continue;
        }
        selected.push(candidate);
    }

    let candidates = selected
        .into_iter()
        .enumerate()
        .map(|(idx, (x_px, y_px, intensity))| DetectionCandidate {
            x_px,
            y_px,
            intensity,
            rank: idx + 1,
        })
        .collect::<Vec<_>>();

    DetectionStageResult {
        candidates,
        metrics: None,
    }
}

pub fn evaluate_detection_metrics(
    detections: &[DetectionCandidate],
    truth_stars: &[TruthStar],
    tolerance_px: f32,
) -> DetectionMetrics {
    let mut truth_taken = vec![false; truth_stars.len()];
    let mut tp = 0_usize;
    let tolerance2 = tolerance_px * tolerance_px;

    for detection in detections {
        let mut best_idx: Option<usize> = None;
        let mut best_dist2 = f32::MAX;
        for (idx, truth) in truth_stars.iter().enumerate() {
            if truth_taken[idx] {
                continue;
            }
            let dx = detection.x_px - truth.x_px;
            let dy = detection.y_px - truth.y_px;
            let dist2 = dx * dx + dy * dy;
            if dist2 <= tolerance2
                && (dist2 < best_dist2
                    || (dist2 == best_dist2
                        && truth.star_id < truth_stars[best_idx.unwrap_or(idx)].star_id))
            {
                best_dist2 = dist2;
                best_idx = Some(idx);
            }
        }
        if let Some(idx) = best_idx {
            truth_taken[idx] = true;
            tp += 1;
        }
    }

    let fp = detections.len().saturating_sub(tp);
    let fn_ = truth_stars.len().saturating_sub(tp);
    let precision = if tp + fp == 0 {
        0.0
    } else {
        tp as f32 / (tp + fp) as f32
    };
    let recall = if tp + fn_ == 0 {
        0.0
    } else {
        tp as f32 / (tp + fn_) as f32
    };

    DetectionMetrics {
        tp,
        fp,
        fn_,
        precision,
        recall,
        tolerance_px,
    }
}

fn is_local_maximum(image: &GrayImage, x: i32, y: i32, center: u8) -> bool {
    for yy in (y - 1)..=(y + 1) {
        for xx in (x - 1)..=(x + 1) {
            if xx == x && yy == y {
                continue;
            }
            if image.get_pixel(xx as u32, yy as u32).0[0] > center {
                return false;
            }
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{GrayImage, Luma};

    #[test]
    fn deterministic_detection_order_for_fixed_image() {
        let mut image = GrayImage::new(16, 16);
        image.put_pixel(4, 4, Luma([180]));
        image.put_pixel(12, 8, Luma([200]));
        image.put_pixel(7, 13, Luma([190]));

        let config = DetectionConfig {
            min_peak_value: 20,
            non_max_radius_px: 1,
            max_candidates: 10,
            truth_match_tolerance_px: 2.5,
        };

        let first = detect_stars(&image, &config);
        let second = detect_stars(&image, &config);
        assert_eq!(first.candidates.len(), 3);
        assert_eq!(first.candidates[0].x_px, 12.0);
        assert_eq!(first.candidates, second.candidates);
    }

    #[test]
    fn computes_precision_and_recall_with_tolerance() {
        let detections = vec![
            DetectionCandidate {
                x_px: 10.0,
                y_px: 10.0,
                intensity: 100.0,
                rank: 1,
            },
            DetectionCandidate {
                x_px: 20.0,
                y_px: 20.0,
                intensity: 90.0,
                rank: 2,
            },
        ];
        let truth = vec![
            TruthStar {
                star_id: "a".to_string(),
                x_px: 10.4,
                y_px: 9.9,
            },
            TruthStar {
                star_id: "b".to_string(),
                x_px: 28.0,
                y_px: 20.0,
            },
        ];
        let metrics = evaluate_detection_metrics(&detections, &truth, 1.0);
        assert_eq!(metrics.tp, 1);
        assert_eq!(metrics.fp, 1);
        assert_eq!(metrics.fn_, 1);
        assert!((metrics.precision - 0.5).abs() < f32::EPSILON);
        assert!((metrics.recall - 0.5).abs() < f32::EPSILON);
    }
}
