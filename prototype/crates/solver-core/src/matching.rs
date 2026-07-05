use std::collections::HashSet;

use simulator_core::camera::CameraIntrinsics;

use crate::config::MatchingConfig;
use crate::contracts::{
    DetectionCandidate, MatchHypothesis, MatchingStageResult, StarCorrespondence,
};
use crate::io::CatalogEntry;

pub fn match_catalog_hypotheses(
    detections: &[DetectionCandidate],
    catalog: &[CatalogEntry],
    intrinsics: &CameraIntrinsics,
    config: &MatchingConfig,
) -> MatchingStageResult {
    if detections.len() < 3 || catalog.len() < 3 {
        return MatchingStageResult {
            ranked_hypotheses: Vec::new(),
            accepted_hypothesis_index: None,
            accepted_correspondences: Vec::new(),
            ambiguity: false,
            no_accept_reason: Some("insufficient points for matching".to_string()),
        };
    }

    let selected = detections
        .iter()
        .take(config.top_k_detections)
        .cloned()
        .collect::<Vec<_>>();
    let obs_vectors = selected
        .iter()
        .map(|det| pixel_to_camera_vector(det.x_px, det.y_px, intrinsics))
        .collect::<Vec<_>>();
    let obs_descriptors = build_neighbor_descriptors(&obs_vectors, config.descriptor_neighbors);
    let catalog_descriptors = build_catalog_descriptors(catalog, config.descriptor_neighbors);

    let mut ranked_pairs = Vec::new();
    for (det_idx, descriptor) in obs_descriptors.iter().enumerate() {
        let mut scores = catalog_descriptors
            .iter()
            .enumerate()
            .map(|(cat_idx, cat_desc)| {
                let distance = descriptor_distance(descriptor, cat_desc);
                let sigma = config.descriptor_tolerance_deg.to_radians().max(1e-6);
                let similarity = (-distance / sigma).exp();
                (cat_idx, similarity)
            })
            .collect::<Vec<_>>();
        scores.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        ranked_pairs.push((det_idx, scores));
    }

    let seed_variants = ranked_pairs
        .first()
        .map(|(_, ranked)| {
            ranked
                .iter()
                .take(3)
                .map(|entry| entry.0)
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();

    let mut hypotheses = Vec::new();
    for (variant_idx, first_catalog_idx) in seed_variants.into_iter().enumerate() {
        let mut used_catalog = HashSet::new();
        let mut correspondences = Vec::new();
        for (det_idx, ranked) in &ranked_pairs {
            let mut picked = None;
            for (candidate_idx, similarity) in ranked {
                if *det_idx == 0 && *candidate_idx != first_catalog_idx {
                    continue;
                }
                if used_catalog.insert(*candidate_idx) {
                    picked = Some((*candidate_idx, *similarity));
                    break;
                }
            }
            if let Some((cat_idx, similarity)) = picked {
                correspondences.push(StarCorrespondence {
                    detection_index: *det_idx,
                    star_id: catalog[cat_idx].star_id.clone(),
                    image_point_px: [selected[*det_idx].x_px, selected[*det_idx].y_px],
                    catalog_direction: catalog[cat_idx].world_vector,
                    similarity,
                });
            }
        }
        if correspondences.len() < 3 {
            continue;
        }
        let avg_similarity = correspondences.iter().map(|c| c.similarity).sum::<f32>()
            / correspondences.len() as f32;
        let coverage = correspondences.len() as f32 / selected.len() as f32;
        let confidence = (avg_similarity * coverage).clamp(0.0, 1.0);
        hypotheses.push(MatchHypothesis {
            id: format!("hypothesis-{}", variant_idx + 1),
            confidence,
            score: avg_similarity,
            correspondences,
        });
    }

    hypotheses.sort_by(|a, b| {
        b.confidence
            .total_cmp(&a.confidence)
            .then_with(|| a.id.cmp(&b.id))
    });
    hypotheses.truncate(config.max_ranked_hypotheses);

    let (accepted_hypothesis_index, no_accept_reason, ambiguity) =
        choose_acceptance(&hypotheses, config);
    let accepted_correspondences = accepted_hypothesis_index
        .and_then(|idx| hypotheses.get(idx))
        .map(|hyp| hyp.correspondences.clone())
        .unwrap_or_default();

    MatchingStageResult {
        ranked_hypotheses: hypotheses,
        accepted_hypothesis_index,
        accepted_correspondences,
        ambiguity,
        no_accept_reason,
    }
}

fn choose_acceptance(
    hypotheses: &[MatchHypothesis],
    config: &MatchingConfig,
) -> (Option<usize>, Option<String>, bool) {
    let Some(top1) = hypotheses.first() else {
        return (None, Some("no hypotheses produced".to_string()), false);
    };
    if top1.confidence < config.absolute_accept_threshold {
        return (
            None,
            Some(format!(
                "top confidence {:.3} below threshold {:.3}",
                top1.confidence, config.absolute_accept_threshold
            )),
            false,
        );
    }
    let ambiguous = hypotheses
        .get(1)
        .map(|top2| (top1.confidence - top2.confidence) < config.ambiguity_margin)
        .unwrap_or(false);
    if ambiguous {
        return (
            None,
            Some(format!(
                "top hypotheses too close (margin {:.3} < {:.3})",
                top1.confidence - hypotheses[1].confidence,
                config.ambiguity_margin
            )),
            true,
        );
    }
    (Some(0), None, false)
}

fn build_neighbor_descriptors(vectors: &[[f32; 3]], count: usize) -> Vec<Vec<f32>> {
    vectors
        .iter()
        .enumerate()
        .map(|(idx, vector)| {
            let mut angles = vectors
                .iter()
                .enumerate()
                .filter(|(other_idx, _)| *other_idx != idx)
                .map(|(_, other)| angular_distance(*vector, *other))
                .collect::<Vec<_>>();
            angles.sort_by(f32::total_cmp);
            angles.truncate(count);
            angles
        })
        .collect()
}

fn build_catalog_descriptors(catalog: &[CatalogEntry], count: usize) -> Vec<Vec<f32>> {
    let vectors = catalog
        .iter()
        .map(|entry| entry.world_vector)
        .collect::<Vec<_>>();
    build_neighbor_descriptors(&vectors, count)
}

fn descriptor_distance(lhs: &[f32], rhs: &[f32]) -> f32 {
    let len = lhs.len().min(rhs.len());
    if len == 0 {
        return f32::MAX;
    }
    lhs.iter()
        .zip(rhs.iter())
        .take(len)
        .map(|(a, b)| (a - b).abs())
        .sum::<f32>()
        / len as f32
}

fn angular_distance(a: [f32; 3], b: [f32; 3]) -> f32 {
    let dot = (a[0] * b[0] + a[1] * b[1] + a[2] * b[2]).clamp(-1.0, 1.0);
    dot.acos()
}

pub fn pixel_to_camera_vector(x_px: f32, y_px: f32, intrinsics: &CameraIntrinsics) -> [f32; 3] {
    let x = (x_px - intrinsics.cx) / intrinsics.fx;
    let y = (intrinsics.cy - y_px) / intrinsics.fy;
    normalize([x, y, 1.0])
}

fn normalize(v: [f32; 3]) -> [f32; 3] {
    let n = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt().max(1e-6);
    [v[0] / n, v[1] / n, v[2] / n]
}

#[cfg(test)]
mod tests {
    use super::*;
    use simulator_core::camera::CameraIntrinsics;
    use simulator_core::catalog::baseline_catalog;
    use simulator_core::config::CameraConfig;

    fn intrinsics() -> CameraIntrinsics {
        CameraIntrinsics::from_camera_config(&CameraConfig {
            width_px: 1024,
            height_px: 768,
            fov_deg: 62.0,
        })
    }

    #[test]
    fn returns_ranked_hypotheses() {
        let detections = vec![
            DetectionCandidate {
                x_px: 200.0,
                y_px: 200.0,
                intensity: 220.0,
                rank: 1,
            },
            DetectionCandidate {
                x_px: 430.0,
                y_px: 180.0,
                intensity: 180.0,
                rank: 2,
            },
            DetectionCandidate {
                x_px: 300.0,
                y_px: 420.0,
                intensity: 170.0,
                rank: 3,
            },
            DetectionCandidate {
                x_px: 520.0,
                y_px: 380.0,
                intensity: 160.0,
                rank: 4,
            },
        ];
        let catalog = crate::io::catalog_with_vectors(&baseline_catalog());
        let result = match_catalog_hypotheses(
            &detections,
            &catalog,
            &intrinsics(),
            &MatchingConfig::default(),
        );
        assert!(!result.ranked_hypotheses.is_empty());
        assert!(
            result.ranked_hypotheses[0].confidence
                >= result.ranked_hypotheses.last().unwrap().confidence
        );
    }

    #[test]
    fn flags_low_confidence_as_no_accept() {
        let config = MatchingConfig {
            absolute_accept_threshold: 0.95,
            ..MatchingConfig::default()
        };
        let detections = vec![
            DetectionCandidate {
                x_px: 200.0,
                y_px: 200.0,
                intensity: 220.0,
                rank: 1,
            },
            DetectionCandidate {
                x_px: 430.0,
                y_px: 180.0,
                intensity: 180.0,
                rank: 2,
            },
            DetectionCandidate {
                x_px: 300.0,
                y_px: 420.0,
                intensity: 170.0,
                rank: 3,
            },
        ];
        let catalog = crate::io::catalog_with_vectors(&baseline_catalog());
        let result = match_catalog_hypotheses(&detections, &catalog, &intrinsics(), &config);
        assert!(result.accepted_hypothesis_index.is_none());
        assert!(result.no_accept_reason.is_some());
    }
}
