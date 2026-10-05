//! Synthetic input corruption is separate from the experimental estimator.
use super::fixture::{self, Point};
use tetra3::Solution;

pub struct Input {
    pub worlds: Vec<[f64; 3]>,
    pub detections: Vec<[f64; 2]>,
    pub probes: Vec<Point>,
    pub corrupted_indices: Vec<usize>,
}

struct Random(u64);

impl Random {
    fn uniform(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^= z >> 31;
        ((z >> 11) as f64 + 1.) / ((1_u64 << 53) as f64 + 2.)
    }

    fn normal_pair(&mut self) -> [f64; 2] {
        let r = (-2. * self.uniform().ln()).sqrt();
        let theta = std::f64::consts::TAU * self.uniform();
        [r * theta.cos(), r * theta.sin()]
    }
}

pub fn input(
    sol: &Solution,
    upper: bool,
    count: usize,
    sigma: f64,
    seed: u64,
    bad_count: usize,
    bad_angle_deg: f64,
) -> Input {
    let grid = fixture::training(sol, upper);
    let train: Vec<_> = (0..count).map(|i| &grid[i * grid.len() / count]).collect();
    let mut random = Random(20261005 + seed);
    let detections = train
        .iter()
        .map(|p| {
            let noise = random.normal_pair();
            // Align known geometric centre without changing the tested adapter.
            [
                p.xy[0] + 0.5 + sigma * noise[0],
                p.xy[1] + 0.5 + sigma * noise[1],
            ]
        })
        .collect();
    let mut order: Vec<_> = (0..count).collect();
    for i in (1..count).rev() {
        let j = (random.uniform() * (i + 1) as f64).floor() as usize;
        order.swap(i, j);
    }
    let corrupted_indices = order[..bad_count].to_vec();
    let angle = bad_angle_deg.to_radians() * if seed.is_multiple_of(2) { 1. } else { -1. };
    let (s, c) = angle.sin_cos();
    let [w, h] = [sol.image_width, sol.image_height].map(f64::from);
    let (cx, cy) = ((w - 1.) / 2., (h - 1.) / 2.);
    let worlds = train
        .iter()
        .enumerate()
        .map(|(i, p)| {
            if corrupted_indices.contains(&i) {
                let (x, y) = (p.xy[0] - cx, p.xy[1] - cy);
                fixture::point(sol, [(cx + c * x - s * y) / w, (cy + s * x + c * y) / h]).world
            } else {
                p.world
            }
        })
        .collect();
    Input {
        worlds,
        detections,
        probes: fixture::probes(sol),
        corrupted_indices,
    }
}

#[test]
fn corruption_changes_only_selected_catalogue_pairs() {
    let sol = fixture::solution([1600, 1200], 55., [30., 20., 15.], 0.);
    let clean = input(&sol, false, 13, 0.5, 4, 0, 0.);
    let bad = input(&sol, false, 13, 0.5, 4, 3, 5.);
    assert_eq!(clean.detections, bad.detections);
    assert_eq!(bad.corrupted_indices.len(), 3);
    for i in 0..13 {
        assert_eq!(
            clean.worlds[i] != bad.worlds[i],
            bad.corrupted_indices.contains(&i)
        );
    }
    for (a, b) in clean.probes.iter().zip(bad.probes) {
        assert_eq!(a.world, b.world);
        assert_eq!(a.xy, b.xy);
    }
}
