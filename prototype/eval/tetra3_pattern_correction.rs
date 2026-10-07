//! Diagnostic adapter: existing Brown–Conrady correction and tetra3 pattern math.
use crate::distortion::RadialDistortion;
use crate::solver::pattern::{compute_edge_ratios, compute_sorted_edge_angles};
use crate::solver::pixel_scale_from_fov;
use crate::SolverDatabase;
use serde_json::{json, Value};

type Error = Box<dyn std::error::Error>;

fn number(v: &Value, key: &str) -> Result<f64, Error> {
    v[key]
        .as_f64()
        .filter(|x| x.is_finite())
        .ok_or_else(|| format!("finite {key} required").into())
}

fn distortion(camera: &Value) -> Result<RadialDistortion, Error> {
    let f = number(camera, "focal_px")?;
    if f <= 0.0 {
        return Err("positive focal length required".into());
    }
    Ok(RadialDistortion::with_tangential(
        number(camera, "k1")? / (f * f),
        0.0,
        0.0,
        number(camera, "p1")? / f,
        number(camera, "p2")? / f,
    ))
}

fn ratios(xy: &[[f32; 2]; 4], fov_rad: f32) -> [f32; 5] {
    let ps = pixel_scale_from_fov(1600, f64::from(fov_rad)) as f32;
    // Same f32 conversion as solve_at_fov; checked against every saved49 window.
    let vectors = xy.map(|[cx, cy]| {
        let x = cx * ps;
        let y = cy * ps;
        let z = 1.0_f32;
        let norm = (x * x + y * y + z * z).sqrt();
        [x / norm, y / norm, z / norm]
    });
    compute_edge_ratios(&compute_sorted_edge_angles(&vectors))
}

pub fn measure(input: &Value) -> Result<Value, Error> {
    let db = SolverDatabase::load_from_file(input["database"].as_str().ok_or("database")?)?;
    let fovs: Vec<f32> = serde_json::from_value(input["fovs_rad"].clone())?;
    let mut output = Vec::new();
    for case in input["cases"].as_array().ok_or("cases")? {
        let ids: [i64; 4] = serde_json::from_value(case["catalog_ids"].clone())?;
        let indices: Vec<_> = ids
            .iter()
            .map(|id| db.star_catalog_ids.iter().position(|v| v == id))
            .collect();
        let [Some(a), Some(b), Some(c), Some(d)] = indices.as_slice() else {
            output.push(json!({"name":case["name"],"error":"catalog_star_absent"}));
            continue;
        };
        let vectors = [*a, *b, *c, *d].map(|i| db.star_vectors[i]);
        let catalog = compute_edge_ratios(&compute_sorted_edge_angles(&vectors));
        let xy: [[f64; 2]; 4] = serde_json::from_value(case["xy"].clone())?;
        let raw = xy.map(|[x, y]| [x as f32 - 799.5, y as f32 - 520.5]);
        let mut arms = Vec::new();
        for arm in input["arms"].as_array().ok_or("arms")? {
            let offset = number(arm, "origin_offset")?;
            let model = distortion(&arm["camera"])?;
            let ideal = raw.map(|[x, y]| {
                let (u, v) = model.undistort(f64::from(x) - offset, f64::from(y) - offset);
                [u as f32, v as f32]
            });
            let roundtrip = raw
                .iter()
                .zip(ideal)
                .map(|([x, y], [u, v])| {
                    let (a, b) = model.distort(f64::from(u), f64::from(v));
                    (a - (f64::from(*x) - offset)).hypot(b - (f64::from(*y) - offset))
                })
                .fold(0.0_f64, f64::max);
            let curves: Vec<_> = fovs
                .iter()
                .map(|&fov| {
                    let image = ratios(&ideal, fov);
                    let pass = (0..5).all(|i| {
                        catalog[i] > image[i] - 0.006_f32 && catalog[i] < image[i] + 0.006_f32
                    });
                    let delta = (0..5)
                        .map(|i| (f64::from(image[i]) - f64::from(catalog[i])).abs())
                        .fold(0.0_f64, f64::max);
                    json!({"ratios":image,"max_abs_delta":delta,"ratios_pass":pass})
                })
                .collect();
            arms.push(json!({"name":arm["name"],"ideal_xy":ideal,"roundtrip_after_f32_px":roundtrip,"curves":curves}));
        }
        output.push(json!({"name":case["name"],"catalog_ratios":catalog,"catalog_vectors":vectors,"arms":arms}));
    }
    let mut checks = Vec::new();
    for arm in input["arms"].as_array().ok_or("arms")? {
        let model = distortion(&arm["camera"])?;
        let mut forward = 0.0_f64;
        let mut inverse = 0.0_f64;
        for row in arm["projection_checks"]
            .as_array()
            .ok_or("projection checks")?
        {
            let ideal: [f64; 2] = serde_json::from_value(row["ideal"].clone())?;
            let expected: [f64; 2] = serde_json::from_value(row["observed"].clone())?;
            let (x, y) = model.distort(ideal[0], ideal[1]);
            forward = forward.max((x - expected[0]).hypot(y - expected[1]));
            let (u, v) = model.undistort(expected[0], expected[1]);
            inverse = inverse.max((u - ideal[0]).hypot(v - ideal[1]));
        }
        checks
            .push(json!({"arm":arm["name"],"forward_error_px":forward,"inverse_error_px":inverse}));
    }
    Ok(json!({"cases":output,"projection_checks":checks}))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn normalized_to_pixel_units_preserve_tangential_sign() {
        let model = distortion(&json!({"focal_px":1000.,"k1":0.1,"p1":0.01,"p2":0.02})).unwrap();
        let (x, y) = model.distort(200., -300.);
        assert!((x - 205.6).abs() < 1e-10);
        assert!((y + 303.2).abs() < 1e-10);
        let (u, v) = model.undistort(x, y);
        assert!((u - 200.).hypot(v + 300.) < 1e-8);
        assert!(distortion(&json!({"focal_px":0.,"k1":0.,"p1":0.,"p2":0.})).is_err());
    }
}
