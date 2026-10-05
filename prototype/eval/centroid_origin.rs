//! Injected only into the isolated candidate workspace.
use super::*;

#[test]
fn projected_boresight_is_tetra3_origin_for_even_and_odd_sizes() {
    for (width, height) in [(740, 576), (1989, 1498), (3000, 4000)] {
        let (x, y) = geom::principal_point(width, height);
        let detection = Detection {
            x,
            y,
            flux: 10.,
            peak: 1.,
            snr: 20.,
            area: 1,
            elongation: 1.,
            rank: 0,
        };
        let centre = detection_to_centroid(&detection, width, height);
        assert_eq!((centre.x, centre.y), (0., 0.));
        let corner = detection_to_centroid(
            &Detection {
                x: 0.,
                y: 0.,
                ..detection
            },
            width,
            height,
        );
        assert_eq!(
            (corner.x, corner.y),
            (-(width as f32) / 2., -(height as f32) / 2.)
        );
    }
}
