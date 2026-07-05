//! Overlay geometry generation from a solved camera pose.

use std::collections::HashSet;

use crate::catalog::{Catalog, Star};
use crate::constellations::ConstellationSet;
use crate::contracts::{
    OverlayConstellation, OverlayGridLine, OverlayPlanet, OverlayStar, SolveOverlay,
};
use crate::ephem::{moon_position, planet_positions, PlanetPosition};
use crate::geom::{angular_sep, project, radec_to_unit, slerp, CameraSolution};

const TESSELLATION_DEG: f64 = 1.0;
const FRAME_MARGIN_PX: f64 = 60.0;

/// Options controlling overlay content.
#[derive(Debug, Clone, PartialEq)]
pub struct OverlayOptions {
    pub epoch_years: Option<f64>,
    pub jd_utc: Option<f64>,
    pub include_grid: bool,
    pub star_mag_limit: f32,
    pub labeled_star_mag_limit: f32,
    pub max_stars: usize,
}

impl Default for OverlayOptions {
    fn default() -> Self {
        Self {
            epoch_years: None,
            jd_utc: None,
            include_grid: false,
            star_mag_limit: 5.0,
            labeled_star_mag_limit: 6.0,
            max_stars: 40,
        }
    }
}

/// Build full overlay geometry for a solved pose.
#[must_use]
pub fn build_overlay(
    pose: &CameraSolution,
    catalog: &Catalog,
    cons: &ConstellationSet,
    opts: &OverlayOptions,
) -> SolveOverlay {
    let rot = pose.rotation();
    let constellations = build_constellations(pose, cons, &rot);
    let stars = build_stars(pose, catalog, opts, &rot);
    let grid = if opts.include_grid {
        build_grid(pose, &rot)
    } else {
        Vec::new()
    };
    let planets = build_planets(pose, opts.jd_utc, &rot);

    SolveOverlay {
        constellations,
        stars,
        planets,
        grid,
    }
}

fn build_constellations(
    pose: &CameraSolution,
    cons: &ConstellationSet,
    rot: &nalgebra::Matrix3<f64>,
) -> Vec<OverlayConstellation> {
    cons.constellations()
        .iter()
        .filter_map(|c| {
            let lines = c
                .polylines
                .iter()
                .flat_map(|polyline| project_polyline(pose, rot, polyline))
                .collect::<Vec<_>>();
            if lines.is_empty() {
                return None;
            }
            let label_xy = constellation_label_xy(&lines, pose.width, pose.height);
            Some(OverlayConstellation {
                abbr: c.abbr.clone(),
                name: c.name.clone(),
                lines,
                label_xy,
            })
        })
        .collect()
}

fn constellation_label_xy(lines: &[Vec<[f64; 2]>], width: u32, height: u32) -> Option<[f64; 2]> {
    let w = width as f64;
    let h = height as f64;
    let inside: Vec<[f64; 2]> = lines
        .iter()
        .flatten()
        .copied()
        .filter(|&[x, y]| x >= 0.0 && x < w && y >= 0.0 && y < h)
        .collect();
    if inside.len() < 2 {
        return None;
    }
    let n = inside.len() as f64;
    let sx: f64 = inside.iter().map(|p| p[0]).sum();
    let sy: f64 = inside.iter().map(|p| p[1]).sum();
    Some([sx / n, sy / n])
}

fn build_stars(
    pose: &CameraSolution,
    catalog: &Catalog,
    opts: &OverlayOptions,
    rot: &nalgebra::Matrix3<f64>,
) -> Vec<OverlayStar> {
    let mut seen = HashSet::new();
    let mut candidates: Vec<&Star> = Vec::new();

    for star in catalog.brighter_than(opts.star_mag_limit) {
        if seen.insert(star.id) {
            candidates.push(star);
        }
    }
    for star in catalog.brighter_than(opts.labeled_star_mag_limit) {
        if star.proper.is_some() && seen.insert(star.id) {
            candidates.push(star);
        }
    }

    let w = pose.width as f64;
    let h = pose.height as f64;
    let mut projected: Vec<OverlayStar> = candidates
        .into_iter()
        .filter_map(|star| {
            let unit = star_unit(star, opts.epoch_years);
            let (x, y) = project(rot, pose.focal_px, pose.k1, pose.width, pose.height, unit)?;
            if x < 0.0 || x >= w || y < 0.0 || y >= h {
                return None;
            }
            Some(OverlayStar {
                x,
                y,
                mag: f64::from(star.mag),
                label: star.proper.clone(),
                hip: star.hip.unwrap_or(0),
            })
        })
        .collect();

    projected.sort_by(|a, b| {
        a.mag
            .partial_cmp(&b.mag)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    projected.truncate(opts.max_stars);
    projected
}

fn star_unit(star: &Star, epoch_years: Option<f64>) -> [f64; 3] {
    match epoch_years {
        Some(epoch) => star.unit_at_epoch(epoch),
        None => radec_to_unit(star.ra_deg, star.dec_deg),
    }
}

fn build_planets(
    pose: &CameraSolution,
    jd_utc: Option<f64>,
    rot: &nalgebra::Matrix3<f64>,
) -> Vec<OverlayPlanet> {
    let Some(jd) = jd_utc else {
        return Vec::new();
    };

    let w = pose.width as f64;
    let h = pose.height as f64;
    let mut bodies: Vec<PlanetPosition> = planet_positions(jd);
    bodies.push(moon_position(jd));

    let mut out = Vec::new();
    for body in bodies {
        if body.name == "Sun" {
            // Night-sky plate solve: if the Sun is in frame, skip it silently.
            if let Some((x, y)) = project_body(pose, rot, &body) {
                if x >= 0.0 && x < w && y >= 0.0 && y < h {
                    continue;
                }
            }
            continue;
        }

        let Some((x, y)) = project_body(pose, rot, &body) else {
            continue;
        };
        if x < 0.0 || x >= w || y < 0.0 || y >= h {
            continue;
        }

        out.push(OverlayPlanet {
            x,
            y,
            name: body.name.to_string(),
            mag: body.mag,
            approx: body.name == "Moon",
        });
    }
    out
}

fn project_body(
    pose: &CameraSolution,
    rot: &nalgebra::Matrix3<f64>,
    body: &PlanetPosition,
) -> Option<(f64, f64)> {
    let unit = radec_to_unit(body.ra_deg, body.dec_deg);
    project(rot, pose.focal_px, pose.k1, pose.width, pose.height, unit)
}

fn build_grid(pose: &CameraSolution, rot: &nalgebra::Matrix3<f64>) -> Vec<OverlayGridLine> {
    let mut grid = Vec::new();

    let mut ra = 0.0;
    while ra < 360.0 {
        let mut polyline = Vec::new();
        let mut dec = -80.0;
        while dec <= 80.0 {
            polyline.push([ra, dec]);
            dec += 2.0;
        }
        grid.extend(
            project_polyline(pose, rot, &polyline)
                .into_iter()
                .map(|points| OverlayGridLine {
                    kind: "ra".to_string(),
                    value_deg: ra,
                    points,
                }),
        );
        ra += 15.0;
    }

    let mut dec = -80.0;
    while dec <= 80.0 {
        let mut polyline = Vec::new();
        let mut ra = 0.0;
        while ra < 360.0 {
            polyline.push([ra, dec]);
            ra += 2.0;
        }
        grid.extend(
            project_polyline(pose, rot, &polyline)
                .into_iter()
                .map(|points| OverlayGridLine {
                    kind: "dec".to_string(),
                    value_deg: dec,
                    points,
                }),
        );
        dec += 10.0;
    }

    grid
}

fn project_polyline(
    pose: &CameraSolution,
    rot: &nalgebra::Matrix3<f64>,
    vertices: &[[f64; 2]],
) -> Vec<Vec<[f64; 2]>> {
    let tessellated = tessellate_polyline(vertices);
    let screen_points: Vec<Option<[f64; 2]>> = tessellated
        .iter()
        .map(|&[ra, dec]| {
            let unit = radec_to_unit(ra, dec);
            project(rot, pose.focal_px, pose.k1, pose.width, pose.height, unit).map(|(x, y)| [x, y])
        })
        .collect();
    split_screen_polyline(&screen_points, pose.width, pose.height)
}

fn tessellate_polyline(vertices: &[[f64; 2]]) -> Vec<[f64; 2]> {
    if vertices.is_empty() {
        return Vec::new();
    }
    let mut out = vec![vertices[0]];
    for window in vertices.windows(2) {
        let [ra1, dec1] = window[0];
        let [ra2, dec2] = window[1];
        let u1 = radec_to_unit(ra1, dec1);
        let u2 = radec_to_unit(ra2, dec2);
        let sep = angular_sep(u1, u2);
        let pieces = (sep / TESSELLATION_DEG).ceil() as usize;
        if pieces <= 1 {
            out.push([ra2, dec2]);
        } else {
            for i in 1..=pieces {
                let t = f64::from(i as u32) / f64::from(pieces as u32);
                let u = slerp(u1, u2, t);
                let ra = u[1].atan2(u[0]).to_degrees().rem_euclid(360.0);
                let dec = u[2].clamp(-1.0, 1.0).asin().to_degrees();
                out.push([ra, dec]);
            }
        }
    }
    out
}

fn split_screen_polyline(
    points: &[Option<[f64; 2]>],
    width: u32,
    height: u32,
) -> Vec<Vec<[f64; 2]>> {
    let margin = FRAME_MARGIN_PX;
    let min_x = -margin;
    let max_x = width as f64 + margin;
    let min_y = -margin;
    let max_y = height as f64 + margin;

    let mut segments = Vec::new();
    let mut current = Vec::new();

    for pt in points {
        match pt {
            Some([x, y]) if *x >= min_x && *x <= max_x && *y >= min_y && *y <= max_y => {
                current.push([*x, *y]);
            }
            _ => {
                if current.len() >= 2 {
                    segments.push(std::mem::take(&mut current));
                } else {
                    current.clear();
                }
            }
        }
    }
    if current.len() >= 2 {
        segments.push(current);
    }
    segments
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::*;
    use crate::constellations::ConstellationSet;

    fn data_root() -> std::path::PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../data")
    }

    fn orion_pose() -> CameraSolution {
        CameraSolution {
            ra_deg: 83.6,
            dec_deg: 5.0,
            roll_deg: 0.0,
            focal_px: 800.0,
            k1: 0.0,
            width: 740,
            height: 576,
        }
    }

    #[test]
    fn orion_overlay_has_constellation_and_stars() {
        let catalog_path = data_root().join("catalogs/hyg_v3.csv");
        let lines = data_root().join("celestial/constellations.lines.json");
        let names = data_root().join("celestial/constellations.json");
        if !catalog_path.exists() || !lines.exists() || !names.exists() {
            return;
        }
        let catalog = Catalog::load(&catalog_path).expect("catalog");
        let cons = ConstellationSet::load(&lines, &names).expect("constellations");
        let overlay = build_overlay(&orion_pose(), &catalog, &cons, &OverlayOptions::default());
        let has_ori = overlay.constellations.iter().any(|c| c.abbr == "Ori");
        assert!(has_ori, "expected Orion among constellations");
        assert!(overlay.stars.len() >= 3, "expected at least 3 stars");
        let betelgeuse = overlay.stars.iter().any(|s| {
            s.label.as_deref() == Some("Betelgeuse")
                && s.x >= 0.0
                && s.x < 740.0
                && s.y >= 0.0
                && s.y < 576.0
        });
        assert!(betelgeuse, "expected labeled Betelgeuse in frame");
    }

    #[test]
    fn tessellation_produces_at_least_ten_points_for_ten_degree_segment() {
        let vertices = [[0.0, 0.0], [10.0, 0.0]];
        let tess = tessellate_polyline(&vertices);
        assert!(
            tess.len() >= 11,
            "expected >=11 tessellated points, got {}",
            tess.len()
        );
    }

    #[test]
    fn emitted_points_within_frame_margin() {
        let catalog_path = data_root().join("catalogs/hyg_v3.csv");
        let lines = data_root().join("celestial/constellations.lines.json");
        let names = data_root().join("celestial/constellations.json");
        if !catalog_path.exists() || !lines.exists() || !names.exists() {
            return;
        }
        let catalog = Catalog::load(&catalog_path).expect("catalog");
        let cons = ConstellationSet::load(&lines, &names).expect("constellations");
        let overlay = build_overlay(
            &orion_pose(),
            &catalog,
            &cons,
            &OverlayOptions {
                include_grid: true,
                ..OverlayOptions::default()
            },
        );
        let margin = FRAME_MARGIN_PX;
        let w = orion_pose().width as f64 + margin;
        let h = orion_pose().height as f64 + margin;
        for c in &overlay.constellations {
            for line in &c.lines {
                for &[x, y] in line {
                    assert!(x >= -margin && x <= w && y >= -margin && y <= h);
                }
            }
        }
        for g in &overlay.grid {
            for &[x, y] in &g.points {
                assert!(x >= -margin && x <= w && y >= -margin && y <= h);
            }
        }
    }

    #[test]
    fn grid_off_by_default_on_when_enabled() {
        let catalog_path = data_root().join("catalogs/hyg_v3.csv");
        let lines = data_root().join("celestial/constellations.lines.json");
        let names = data_root().join("celestial/constellations.json");
        if !catalog_path.exists() || !lines.exists() || !names.exists() {
            return;
        }
        let catalog = Catalog::load(&catalog_path).expect("catalog");
        let cons = ConstellationSet::load(&lines, &names).expect("constellations");
        let off = build_overlay(&orion_pose(), &catalog, &cons, &OverlayOptions::default());
        assert!(off.grid.is_empty());
        let on = build_overlay(
            &orion_pose(),
            &catalog,
            &cons,
            &OverlayOptions {
                include_grid: true,
                ..OverlayOptions::default()
            },
        );
        assert!(!on.grid.is_empty());
    }

    #[test]
    fn jupiter_at_boresight_when_jd_set() {
        use crate::ephem::julian_day_utc;

        let jd = julian_day_utc(2011, 9, 21, 0, 0, 0.0);
        let jupiter = planet_positions(jd)
            .into_iter()
            .find(|p| p.name == "Jupiter")
            .expect("Jupiter");

        let pose = CameraSolution {
            ra_deg: jupiter.ra_deg,
            dec_deg: jupiter.dec_deg,
            roll_deg: 0.0,
            focal_px: 800.0,
            k1: 0.0,
            width: 740,
            height: 576,
        };

        let catalog_path = data_root().join("catalogs/hyg_v3.csv");
        let lines = data_root().join("celestial/constellations.lines.json");
        let names = data_root().join("celestial/constellations.json");
        if !catalog_path.exists() || !lines.exists() || !names.exists() {
            return;
        }
        let catalog = Catalog::load(&catalog_path).expect("catalog");
        let cons = ConstellationSet::load(&lines, &names).expect("constellations");

        let with_jd = build_overlay(
            &pose,
            &catalog,
            &cons,
            &OverlayOptions {
                jd_utc: Some(jd),
                ..OverlayOptions::default()
            },
        );
        let jup = with_jd
            .planets
            .iter()
            .find(|p| p.name == "Jupiter")
            .expect("Jupiter marker");
        let (cx, cy) = pose.principal_point();
        let dx = jup.x - cx;
        let dy = jup.y - cy;
        assert!(
            dx * dx + dy * dy < 4.0,
            "Jupiter at ({}, {}), expected near ({}, {})",
            jup.x,
            jup.y,
            cx,
            cy
        );

        let without_jd = build_overlay(&pose, &catalog, &cons, &OverlayOptions::default());
        assert!(without_jd.planets.is_empty());
    }

    #[test]
    fn moon_marker_is_approximate() {
        use crate::ephem::julian_day_utc;

        let jd = julian_day_utc(2011, 9, 21, 0, 0, 0.0);
        let moon = moon_position(jd);
        let pose = CameraSolution {
            ra_deg: moon.ra_deg,
            dec_deg: moon.dec_deg,
            roll_deg: 0.0,
            focal_px: 800.0,
            k1: 0.0,
            width: 740,
            height: 576,
        };

        let catalog_path = data_root().join("catalogs/hyg_v3.csv");
        let lines = data_root().join("celestial/constellations.lines.json");
        let names = data_root().join("celestial/constellations.json");
        if !catalog_path.exists() || !lines.exists() || !names.exists() {
            return;
        }
        let catalog = Catalog::load(&catalog_path).expect("catalog");
        let cons = ConstellationSet::load(&lines, &names).expect("constellations");

        let overlay = build_overlay(
            &pose,
            &catalog,
            &cons,
            &OverlayOptions {
                jd_utc: Some(jd),
                ..OverlayOptions::default()
            },
        );
        let moon_marker = overlay
            .planets
            .iter()
            .find(|p| p.name == "Moon")
            .expect("Moon marker");
        assert!(moon_marker.approx);
    }
}
