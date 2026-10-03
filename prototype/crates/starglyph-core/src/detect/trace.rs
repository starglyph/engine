//! Opt-in source tracing. Probe association is geometric, never a star identity.
use std::collections::BTreeMap;

use super::{Blob, DetectConfig, DetectResult, Detection};
use crate::{image_input::FrameImage, sky_mask::SkyMask};

/// Measurements reached by a component; absent values were not evaluated.
#[derive(Debug, Default, Clone, serde::Serialize)]
pub struct ComponentTrace {
    pub component: usize,
    pub bounds: [i32; 4],
    pub area: usize,
    pub core_area: Option<u32>,
    pub peak: Option<f32>,
    pub measured_area: Option<usize>,
    pub elongation: Option<f32>,
    pub centroid: Option<[f64; 2]>,
    pub positive_blob_flux: Option<f32>,
    pub photometry_flux: Option<f32>,
    pub concentration: Option<f32>,
    pub concentration_screened: bool,
    pub outcome: &'static str,
    pub rank_before_top_k: Option<usize>,
}

/// Nearest threshold-component pixel within the requested radius. Several
/// probes can share one component, which is useful evidence of merging.
#[derive(Debug, serde::Serialize)]
pub struct ProbeTrace {
    pub point: [f64; 2],
    pub distance_to_component: Option<f64>,
    pub component: Option<ComponentTrace>,
}

pub(super) struct TraceCollector {
    probes: Vec<[f64; 2]>,
    radius: f64,
    associations: Vec<Option<(f64, usize)>>,
    pub components: BTreeMap<usize, ComponentTrace>,
}

impl TraceCollector {
    pub fn associate(&mut self, blobs: &[Blob]) {
        for &[x, y] in &self.probes {
            let mut best = None;
            for (id, blob) in blobs.iter().enumerate() {
                if x < blob.min_x as f64 - self.radius
                    || x > blob.max_x as f64 + self.radius
                    || y < blob.min_y as f64 - self.radius
                    || y > blob.max_y as f64 + self.radius
                {
                    continue;
                }
                for &(px, py) in &blob.pixels {
                    let distance = (x - px as f64).hypot(y - py as f64);
                    if distance <= self.radius && best.is_none_or(|(d, _)| distance < d) {
                        best = Some((distance, id));
                    }
                }
            }
            self.associations.push(best);
            if let Some((_, id)) = best {
                let blob = &blobs[id];
                self.components.entry(id).or_insert_with(|| ComponentTrace {
                    component: id,
                    bounds: [blob.min_x, blob.min_y, blob.max_x, blob.max_y],
                    area: blob.pixels.len(),
                    outcome: "pending",
                    ..ComponentTrace::default()
                });
            }
        }
    }

    pub fn finish(&mut self, detections: &[Detection], limit: usize) {
        for trace in self
            .components
            .values_mut()
            .filter(|t| t.outcome == "accepted")
        {
            trace.rank_before_top_k = detections
                .iter()
                .position(|d| Some([d.x, d.y]) == trace.centroid);
            trace.outcome = match trace.rank_before_top_k {
                None => "deduplicated",
                Some(rank) if rank >= limit => "top_k",
                _ => "selected",
            };
        }
    }
}

/// Trace selected source positions through the exact detector. Coordinates and
/// radius are in working pixels; callers must validate finite in-frame probes.
/// This is a separate diagnostic pass and must not be included in solve timing.
pub fn detect_stars_with_trace(
    frame: &FrameImage,
    config: &DetectConfig,
    mask: Option<&SkyMask>,
    probes: &[[f64; 2]],
    radius: f64,
) -> (DetectResult, Vec<ProbeTrace>) {
    let mut collector = TraceCollector {
        probes: probes.to_vec(),
        radius,
        associations: Vec::new(),
        components: BTreeMap::new(),
    };
    let result = super::detect_stars_impl(frame, config, mask, Some(&mut collector));
    let traces = probes
        .iter()
        .zip(&collector.associations)
        .map(|(&point, association)| ProbeTrace {
            point,
            distance_to_component: association.map(|(d, _)| d),
            component: association.and_then(|(_, id)| collector.components.get(&id).cloned()),
        })
        .collect();
    (result, traces)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::detect::tests::{flat_background, gaussian_bump, make_frame};

    #[test]
    fn tracing_preserves_detection_and_identifies_top_k_and_rejection() {
        let mut gray = flat_background(128, 128, 0.1);
        gaussian_bump(128, 128, 30.0, 30.0, 0.7, 1.4, &mut gray);
        gaussian_bump(128, 128, 80.0, 80.0, 0.4, 1.4, &mut gray);
        let frame = make_frame(128, 128, gray);
        let probes = [[30.0, 30.0], [80.0, 80.0], [60.0, 30.0]];
        let config = DetectConfig {
            max_detections: 1,
            ..DetectConfig::default()
        };
        let (result, traces) = detect_stars_with_trace(&frame, &config, None, &probes, 2.0);
        assert_eq!(result, super::super::detect_stars(&frame, &config));
        assert_eq!(traces[0].component.as_ref().unwrap().outcome, "selected");
        assert_eq!(traces[1].component.as_ref().unwrap().outcome, "top_k");
        assert!(traces[2].component.is_none());
        let config = DetectConfig {
            min_core_area: 1000,
            ..config
        };
        let (result, traces) = detect_stars_with_trace(&frame, &config, None, &probes, 2.0);
        assert_eq!(result, super::super::detect_stars(&frame, &config));
        let rejected = traces[0].component.as_ref().unwrap();
        assert_eq!(rejected.outcome, "core_area");
        assert!(rejected.centroid.is_none()); // Don't invent uncomputed measurements.
    }

    #[test]
    fn associates_actual_pixels_not_large_bounding_boxes() {
        let blobs = [super::super::blob_from_pixels(vec![(1, 1), (1, 2), (8, 8)])];
        let mut trace = TraceCollector {
            probes: vec![[4.0, 4.0], [1.0, 1.0], [1.0, 2.0]],
            radius: 1.0,
            associations: Vec::new(),
            components: BTreeMap::new(),
        };
        trace.associate(&blobs);
        assert!(trace.associations[0].is_none());
        assert_eq!(
            trace.associations[1].unwrap().1,
            trace.associations[2].unwrap().1
        );
        assert_eq!(trace.components.len(), 1);
    }
}
