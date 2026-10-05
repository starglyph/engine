//! Experimental angular median; no verification scores or source identities.
//! Angles are unwrapped about their circular mean. This is a local estimator
//! for clustered candidate rolls, not a global circular L1 optimizer.
pub(super) fn estimate(angles: &[f64]) -> Option<f64> {
    if angles.is_empty() || angles.iter().any(|a| !a.is_finite()) {
        return None;
    }
    let anchor = angles
        .iter()
        .map(|a| a.sin())
        .sum::<f64>()
        .atan2(angles.iter().map(|a| a.cos()).sum::<f64>());
    let mut offsets: Vec<_> = angles.iter().map(|a| wrap(a - anchor)).collect();
    offsets.sort_by(f64::total_cmp);
    let middle = offsets.len() / 2;
    let median = if offsets.len() % 2 == 0 {
        (offsets[middle - 1] + offsets[middle]) / 2.
    } else {
        offsets[middle]
    };
    Some(wrap(anchor + median))
}

fn wrap(angle: f64) -> f64 {
    angle.sin().atan2(angle.cos())
}

#[test]
fn wrap_boundary_permutation_and_rotation() {
    let input = [179.8_f64, -179.9, 179.9, -179.8, -170.].map(f64::to_radians);
    let original = estimate(&input).expect("finite angles");
    assert!((original.to_degrees() + 179.9).abs() < 1e-10);
    for rotation in [0., 0.7, -2.8, 3.2] {
        let mut rotated = input.map(|v| v + rotation);
        rotated.reverse();
        assert!(wrap(estimate(&rotated).expect("rotated") - original - rotation).abs() < 1e-12);
    }
}

#[test]
fn majority_even_count_and_invalid_input() {
    assert!(estimate(&[]).is_none());
    assert!(estimate(&[f64::NAN]).is_none());
    assert!(estimate(&[f64::INFINITY]).is_none());
    assert!((estimate(&[0.1, 0.1, 0.1, 0.8, 0.9]).expect("majority") - 0.1).abs() < 1e-12);
    assert!((estimate(&[0.1, 0.2, 0.3, 0.9]).expect("even") - 0.25).abs() < 1e-12);
}
