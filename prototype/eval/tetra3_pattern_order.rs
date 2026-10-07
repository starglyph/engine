//! Experimental priority only: preserve every cluster-buster survivor.
pub fn farthest_first(items: &[usize], positions: &[[f32; 2]]) -> Vec<usize> {
    if items.is_empty()
        || items.iter().any(|&i| {
            positions
                .get(i)
                .is_none_or(|p| p.iter().any(|v| !v.is_finite()))
        })
    {
        return items.to_vec();
    }
    let mut output = Vec::with_capacity(items.len());
    let mut used = vec![false; items.len()];
    let mut distance = vec![f64::INFINITY; items.len()];
    let mut next = 0; // Brightest survivor remains first; ties keep brightness order.
    for _ in 0..items.len() {
        let selected = positions[items[next]];
        used[next] = true;
        output.push(items[next]);
        for (i, &index) in items.iter().enumerate() {
            let p = positions[index];
            let dx = f64::from(p[0]) - f64::from(selected[0]);
            let dy = f64::from(p[1]) - f64::from(selected[1]);
            distance[i] = distance[i].min(dx * dx + dy * dy);
        }
        let mut best = None;
        for i in 0..items.len() {
            if !used[i] && best.is_none_or(|j| distance[i] > distance[j]) {
                best = Some(i);
            }
        }
        if let Some(index) = best {
            next = index;
        }
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prioritizes_distance_without_dropping_survivors() {
        let xy = [[0., 0.], [1., 0.], [10., 0.], [4., 0.], [5., 0.]];
        assert_eq!(farthest_first(&[0, 1, 2, 3], &xy), [0, 2, 3, 1]);
        // Point 4 did not survive thinning and must not be reintroduced.
    }

    #[test]
    fn ties_keep_original_brightness_order() {
        let xy = [[0., 0.], [-1., 0.], [1., 0.], [0., 0.]];
        assert_eq!(farthest_first(&[0, 2, 1, 3], &xy), [0, 2, 1, 3]);
    }

    #[test]
    fn translation_and_uniform_scale_preserve_priority() {
        let xy = [[2., 4.], [3., 4.], [12., 4.], [6., 4.]];
        let transformed = xy.map(|p| [p[0] * 2. + 10., p[1] * 2. - 10.]);
        assert_eq!(
            farthest_first(&[0, 1, 2, 3], &xy),
            farthest_first(&[0, 1, 2, 3], &transformed)
        );
    }

    #[test]
    fn empty_single_and_invalid_inputs_do_not_panic() {
        assert!(farthest_first(&[], &[]).is_empty());
        assert_eq!(farthest_first(&[0], &[[1., 2.]]), [0]);
        assert_eq!(farthest_first(&[0, 1], &[[f32::NAN, 2.], [0., 0.]]), [0, 1]);
        assert_eq!(farthest_first(&[3], &[]), [3]);
    }
}
