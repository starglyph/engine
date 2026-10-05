//! Experimental radius-aware Huber location for clustered pair roll angles.
//! No detection IDs, verification hits, known noise or corruption labels enter.
use super::robust_roll;

const HUBER_C: f64 = 1.345;
const MAD_TO_SIGMA: f64 = 1.4826;
const SCALE_FLOOR_PX: f64 = 1e-6;
const MAX_STEPS: usize = 8;

pub(super) fn estimate(pairs: &[(f64, f64)]) -> Option<f64> {
    if pairs
        .iter()
        .any(|(a, r)| !a.is_finite() || !r.is_finite() || *r < 0.)
    {
        return None;
    }
    let informative: Vec<_> = pairs.iter().copied().filter(|(_, r)| *r > 0.).collect();
    let angles: Vec<_> = informative.iter().map(|(a, _)| *a).collect();
    let mut location = robust_roll::estimate(&angles)?;
    let mut residuals: Vec<_> = informative
        .iter()
        .map(|(a, r)| r * wrap(a - location))
        .collect();
    let center = median(&mut residuals);
    let mut deviations: Vec<_> = residuals.iter().map(|r| (r - center).abs()).collect();
    let scale = (MAD_TO_SIGMA * median(&mut deviations)).max(SCALE_FLOOR_PX);
    let radius_scale = informative.iter().map(|(_, r)| *r).fold(0_f64, f64::max);
    if !scale.is_finite() || !radius_scale.is_finite() {
        return None;
    }
    for _ in 0..MAX_STEPS {
        let mut numerator = 0.;
        let mut denominator = 0.;
        for &(angle, radius) in &informative {
            let delta = wrap(angle - location);
            let error_px = (radius * delta).abs();
            let robust_weight = if error_px <= HUBER_C * scale {
                1.
            } else {
                HUBER_C * scale / error_px
            };
            let weight = (radius / radius_scale).powi(2) * robust_weight;
            numerator += weight * delta;
            denominator += weight;
        }
        if !numerator.is_finite() || !denominator.is_finite() || denominator <= 0. {
            return None;
        }
        let step = numerator / denominator;
        location = wrap(location + step);
        if step.abs() <= 1e-12 {
            break;
        }
    }
    Some(location)
}

fn wrap(angle: f64) -> f64 {
    angle.sin().atan2(angle.cos())
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    let middle = values.len() / 2;
    if values.len().is_multiple_of(2) {
        (values[middle - 1] + values[middle]) / 2.
    } else {
        values[middle]
    }
}

#[test]
fn quadratic_region_weights_by_squared_radius() {
    let pairs = [(-0.001, 100.), (0.001, 200.)];
    let result = estimate(&pairs).expect("valid pairs");
    assert!((result - 0.0006).abs() < 1e-12);
    let scaled = pairs.map(|(a, r)| (a, r * 10.));
    assert!((estimate(&scaled).expect("scaled") - result).abs() < 1e-12);
}

#[test]
fn boundary_permutation_and_corrupted_minority() {
    let mut pairs = vec![(179.9_f64.to_radians(), 300.); 12];
    pairs.push((-175_f64.to_radians(), 500.));
    let original = estimate(&pairs).expect("finite pairs");
    assert!(wrap(original - 179.9_f64.to_radians()).abs() < 1e-8);
    for rotation in [0., 0.7, -2.8] {
        let mut rotated: Vec<_> = pairs.iter().map(|(a, r)| (a + rotation, *r)).collect();
        rotated.reverse();
        assert!(wrap(estimate(&rotated).expect("rotated") - original - rotation).abs() < 1e-12);
    }
}

#[test]
fn center_has_no_roll_information_and_invalid_input_rejects() {
    assert!(estimate(&[]).is_none());
    assert!(estimate(&[(0., 0.)]).is_none());
    assert!(estimate(&[(f64::NAN, 1.)]).is_none());
    assert!(estimate(&[(0., -1.)]).is_none());
    assert!(estimate(&[(0., f64::INFINITY)]).is_none());
    assert!(
        (estimate(&[(0.3, 10.), (-2., 0.)]).expect("one informative pair") - 0.3).abs() < 1e-12
    );
}
