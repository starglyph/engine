//! Experimental sky selection and explicitly enabled detector interventions.

/// A simple sky polygon in EXIF-oriented image-edge coordinates normalized to
/// [0, 1]. Clouds belong to sky. The caller validates image identity/dimensions.
#[derive(Debug, Clone)]
pub struct SkyMask {
    polygon: Vec<[f64; 2]>,
    sky_statistics: bool,
    sky_fill: bool,
}

impl SkyMask {
    /// Construct a non-degenerate polygon. Vertices must describe a simple ring.
    pub fn new(polygon: Vec<[f64; 2]>) -> Result<Self, &'static str> {
        if polygon.len() < 3
            || polygon
                .iter()
                .flatten()
                .any(|v| !v.is_finite() || !(0.0..=1.0).contains(v))
        {
            return Err("sky polygon needs at least three finite vertices in [0, 1]");
        }
        let area: f64 = polygon
            .iter()
            .zip(polygon.iter().cycle().skip(1))
            .map(|(a, b)| a[0] * b[1] - b[0] * a[1])
            .sum();
        if area.abs() < 1e-12 {
            return Err("degenerate sky polygon");
        }
        for i in 0..polygon.len() {
            let a = polygon[i];
            let b = polygon[(i + 1) % polygon.len()];
            if a == b {
                return Err("duplicate adjacent sky polygon vertices");
            }
            for j in (i + 2)..polygon.len() {
                if i == 0 && j + 1 == polygon.len() {
                    continue;
                }
                let c = polygon[j];
                let d = polygon[(j + 1) % polygon.len()];
                if segments_intersect(a, b, c, d) {
                    return Err("sky polygon must be a simple ring");
                }
            }
        }
        Ok(Self {
            polygon,
            sky_statistics: false,
            sky_fill: false,
        })
    }

    /// Opt into estimating column/row, mesh and noise statistics from sky pixels.
    /// Does not change adaptive threshold fill accounting or acceptance thresholds.
    pub fn with_sky_statistics(mut self) -> Self {
        self.sky_statistics = true;
        self
    }

    /// Enable sky-only background statistics and adaptive-fill accounting.
    /// Component labeling and solve acceptance thresholds remain unchanged.
    pub fn with_sky_fill(mut self) -> Self {
        self.sky_statistics = true;
        self.sky_fill = true;
        self
    }

    pub(crate) fn uses_sky_fill(&self) -> bool {
        self.sky_fill
    }

    pub(crate) fn uses_sky_statistics(&self) -> bool {
        self.sky_statistics
    }

    /// Scanline rasterization of pixel centers; no resizing or masked pixel fill.
    pub(crate) fn rasterize(&self, width: u32, height: u32) -> Vec<bool> {
        let mut raster = vec![false; width as usize * height as usize];
        for y in 0..height {
            let py = (f64::from(y) + 0.5) / f64::from(height);
            let mut crossings = Vec::new();
            for (a, b) in self.polygon.iter().zip(self.polygon.iter().cycle().skip(1)) {
                if (a[1] > py) != (b[1] > py) {
                    crossings.push(a[0] + (py - a[1]) * (b[0] - a[0]) / (b[1] - a[1]));
                }
            }
            crossings.sort_by(f64::total_cmp);
            for pair in crossings.as_chunks::<2>().0 {
                let start = ((pair[0] * f64::from(width) - 0.5).floor() + 1.)
                    .clamp(0., f64::from(width)) as u32;
                let end = (pair[1] * f64::from(width) - 0.5)
                    .ceil()
                    .clamp(0., f64::from(width)) as u32;
                for x in start..end {
                    raster[y as usize * width as usize + x as usize] = true;
                }
                // Match the conservative boundary convention, including horizontal edges.
                for x in [start, end.saturating_sub(1)] {
                    if x < width && !self.contains(f64::from(x), f64::from(y), width, height) {
                        raster[y as usize * width as usize + x as usize] = false;
                    }
                }
            }
            for (a, b) in self.polygon.iter().zip(self.polygon.iter().cycle().skip(1)) {
                if a[1] == b[1] && (py - a[1]).abs() <= 1e-12 {
                    for x in 0..width {
                        let px = (f64::from(x) + 0.5) / f64::from(width);
                        if px >= a[0].min(b[0]) && px <= a[0].max(b[0]) {
                            raster[y as usize * width as usize + x as usize] = false;
                        }
                    }
                }
            }
        }
        raster
    }

    /// Test a zero-based pixel centroid at any resolution of the same image.
    /// Half-pixel mapping preserves image-edge geometry across solver scales.
    /// Points exactly on the polygon boundary are conservatively excluded.
    pub fn contains(&self, x: f64, y: f64, width: u32, height: u32) -> bool {
        if width == 0 || height == 0 || !x.is_finite() || !y.is_finite() {
            return false;
        }
        let x = (x + 0.5) / f64::from(width);
        let y = (y + 0.5) / f64::from(height);
        let mut inside = false;
        for (a, b) in self.polygon.iter().zip(self.polygon.iter().cycle().skip(1)) {
            let cross = (x - a[0]) * (b[1] - a[1]) - (y - a[1]) * (b[0] - a[0]);
            if cross.abs() <= 1e-12
                && x >= a[0].min(b[0])
                && x <= a[0].max(b[0])
                && y >= a[1].min(b[1])
                && y <= a[1].max(b[1])
            {
                return false;
            }
            if (a[1] > y) != (b[1] > y) && x < (b[0] - a[0]) * (y - a[1]) / (b[1] - a[1]) + a[0] {
                inside = !inside;
            }
        }
        inside
    }
}

fn segments_intersect(a: [f64; 2], b: [f64; 2], c: [f64; 2], d: [f64; 2]) -> bool {
    let cross = |a: [f64; 2], b: [f64; 2], c: [f64; 2]| {
        (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    };
    (0..2).all(|i| a[i].min(b[i]) <= c[i].max(d[i]) && c[i].min(d[i]) <= a[i].max(b[i]))
        && cross(a, b, c) * cross(a, b, d) <= 0.
        && cross(c, d, a) * cross(c, d, b) <= 0.
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn raster_matches_centroid_membership_for_concave_and_edge_cases() {
        for polygon in [
            vec![[0., 0.], [1., 0.], [1., 1.], [0.5, 0.5], [0., 1.]],
            vec![[0., 0.], [1., 0.], [1., 0.5], [0., 0.5]],
        ] {
            let mask = SkyMask::new(polygon).unwrap();
            for (w, h) in [(9, 9), (31, 17), (40, 24)] {
                let raster = mask.rasterize(w, h);
                for y in 0..h {
                    for x in 0..w {
                        assert_eq!(
                            raster[(y * w + x) as usize],
                            mask.contains(f64::from(x), f64::from(y), w, h),
                            "{x},{y} at {w}x{h}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn selection_preserves_pixel_centers_at_different_scales() {
        let mask = SkyMask::new(vec![[0., 0.], [1., 0.], [1., 0.5], [0., 0.5]]).unwrap();
        for (x, y) in [(20., 15.), (20., 80.), (50., 49.5)] {
            assert_eq!(
                mask.contains(x, y, 100, 100),
                mask.contains((x + 0.5) * 4. - 0.5, (y + 0.5) * 4. - 0.5, 400, 400)
            );
        }
        assert!(mask.contains(20., 15., 100, 100));
        assert!(!mask.contains(20., 80., 100, 100));
        assert!(!mask.contains(50., 49.5, 100, 100));
        assert!(SkyMask::new(vec![[0., 0.]; 3]).is_err());
        assert!(SkyMask::new(vec![[0., 0.], [1., 1.], [0., 1.], [0.7, 0.]]).is_err());
        assert!(SkyMask::new(vec![[f64::NAN, 0.]; 3]).is_err());
    }
}
