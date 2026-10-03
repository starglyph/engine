//! Standalone offline caller of tetra3; consumes already frozen development lists.
use serde_json::{json, Value};
use std::{fs, time::Instant};
use tetra3::{CameraModel, Centroid, SolveConfig, SolverDatabase};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    let input: Value = serde_json::from_slice(&fs::read(args.get(1).ok_or("input path")?)?)?;
    let dbs: Vec<_> = input["databases"].as_array().ok_or("databases")?.iter()
        .map(|p| SolverDatabase::load_from_file(p.as_str().expect("database path")))
        .collect::<Result<_, _>>()?;
    let mut results = Vec::new();
    for case in input["cases"].as_array().ok_or("cases")? {
        let name = case["name"].as_str().ok_or("name")?;
        // Only original native development tiers and the positive control.
        if !matches!(name, "4080-default-control" | "4080-deep-control" | "positive-control") { continue; }
        let w = case["width"].as_u64().ok_or("width")? as u32;
        let h = case["height"].as_u64().ok_or("height")? as u32;
        let detections = case["search"].as_array().ok_or("search")?;
        let centroids: Vec<_> = detections.iter().map(|d| Centroid {
            x: d["x"].as_f64().expect("x") as f32 - (w-1) as f32/2.,
            y: d["y"].as_f64().expect("y") as f32 - (h-1) as f32/2.,
            mass: Some(d["flux"].as_f64().expect("flux") as f32), cov: None,
        }).collect();
        let ids = case["search_ids"].as_array().map(|v| v.iter().map(|i| i.as_i64()).collect())
            .unwrap_or_else(|| vec![None; centroids.len()]);
        tetra3::research_trace::set_ids(ids);
        let mut prefixes: Vec<_> = [8,10,12,16,20,30].iter().map(|&k| k.min(centroids.len())).filter(|&k| k>=4).collect();
        prefixes.dedup();
        let mut logs = Vec::new();
        for (db, fov, error) in [(0,22.,5.),(0,15.,5.25),(0,25.,8.75),(0,40.,14.),(0,60.,21.),(1,23.,8.),(2,42.,13.),(3,68.,21.)] {
            for &k in &prefixes {
                tetra3::research_trace::emit("attempt", json!({"case": name, "db":db,"k":k,"fov":fov}));
                let config = SolveConfig {
                    fov_max_error_rad: Some((error as f32).to_radians()), match_radius:0.01,
                    match_threshold:1e-3, solve_timeout_ms:Some(2500), match_max_error:None,
                    ..SolveConfig::with_camera_model(CameraModel::from_fov((fov as f64).to_radians(),w,h))
                };
                let started = Instant::now();
                let result = dbs[db].solve_from_centroids(&centroids[..k], &config);
                let summary = match result {
                    Ok(s) => json!({"matches":s.num_matches,"prob":s.prob,"catalog_ids":s.matched_catalog_ids,
                        "centroid_indices":s.matched_centroid_indices,"fov_rad":s.fov_rad,"crval_rad":s.crval_rad}),
                    Err(e) => json!({"error":format!("{:?}",e.status)}),
                };
                logs.push(json!({"db":db,"k":k,"fov":fov,"elapsed_ms":started.elapsed().as_secs_f64()*1000.,"result":summary}));
            }
        }
        results.push(json!({"name":name,"attempts":logs}));
    }
    fs::write(args.get(2).ok_or("output path")?, serde_json::to_vec_pretty(&json!({"cases":results}))?)?;
    Ok(())
}
