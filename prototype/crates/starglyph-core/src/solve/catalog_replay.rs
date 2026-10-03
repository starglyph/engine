//! Offline catalogue diagnosis. No production switches or database generation.
use super::*;
use serde::Deserialize;
use std::collections::HashSet;

#[derive(Deserialize)]
struct Input {
    image: PathBuf,
    catalog: PathBuf,
    databases: Vec<PathBuf>,
    probe_ids: Vec<i64>,
    cases: Vec<Case>,
}

#[derive(Deserialize)]
struct Case {
    name: String,
    width: u32,
    height: u32,
    search: Vec<Point>,
    verification: Vec<Point>,
}

#[derive(Deserialize)]
struct Point {
    x: f64,
    y: f64,
    flux: f32,
}

fn detections(points: &[Point]) -> Vec<Detection> {
    points
        .iter()
        .enumerate()
        .map(|(rank, p)| Detection {
            x: p.x,
            y: p.y,
            flux: p.flux,
            peak: 0.0,
            snr: 0.0,
            area: 1,
            elongation: 1.0,
            rank: rank as u32,
        })
        .collect()
}

/// Matched indices belong to the SEARCH list. Only after deriving its pose
/// may verification use the unchanged full detector list.
fn verify_separately(
    sol: &Solution,
    search: &[Detection],
    verification: &[Detection],
    stars: &VerifyStars,
    label: &'static str,
) -> Option<Candidate> {
    let mut candidate = make_candidate(sol, search, stars, label)?;
    candidate.verify = match_predictions(&candidate.pose, stars, verification, VERIFY_RADIUS_PX);
    Some(candidate)
}

#[test]
#[ignore = "offline research: STARGLYPH_CATALOG_INPUT and STARGLYPH_CATALOG_OUTPUT"]
fn catalog_patterns_and_replay() -> Result<(), Box<dyn std::error::Error>> {
    let input: Input =
        serde_json::from_slice(&std::fs::read(std::env::var("STARGLYPH_CATALOG_INPUT")?)?)?;
    let frame = FrameImage::load(&input.image)?;
    let opts = SolveOptions::default().resolved_for(&frame);
    // This protocol is deliberately blind; no external WCS enters SolveConfig.
    assert!(opts.fov_hint_deg.is_none() && opts.attitude_hint.is_none());
    assert_eq!(input.databases.len(), 4);
    let catalog = Catalog::load(&input.catalog)?;
    let epoch = frame
        .acquisition_timestamp()
        .map(|t| crate::ephem::epoch_years(t.to_jd_utc(opts.utc_offset_hours)));
    let stars = VerifyStars::build(&catalog, epoch);
    let ids: HashSet<_> = input.probe_ids.iter().copied().collect();
    let mut databases = Vec::new();
    let mut summaries = Vec::new();
    for path in &input.databases {
        let db = tetra3::SolverDatabase::load_from_file(path.to_str().ok_or("non-UTF8 path")?)?;
        let mut patterns = Vec::new();
        let mut participation = std::collections::BTreeMap::new();
        for i in 0..db.pattern_catalog.len() {
            let entry = db.pattern_catalog.get(i);
            if entry.is_empty() {
                continue;
            }
            let pattern = entry.star_indices.map(|s| db.star_catalog_ids[s as usize]);
            for id in pattern {
                if ids.contains(&id) {
                    *participation.entry(id).or_insert(0usize) += 1;
                }
            }
            if pattern.iter().all(|id| ids.contains(id)) {
                patterns.push(pattern);
            }
        }
        summaries.push(serde_json::json!({
            "file": path.file_name().and_then(|s| s.to_str()), "stars": db.star_catalog_ids.len(),
            "properties": db.props,
            "patterns_total": db.props.num_patterns,
            "probe_ids_present": input.probe_ids.iter().filter(|id| db.star_catalog_ids.contains(id)).collect::<Vec<_>>(),
            "probe_pattern_participation": participation, "probe_patterns": patterns,
        }));
        databases.push(db);
    }
    let mut results = Vec::new();
    for case in input.cases {
        let search = detections(&case.search);
        let verification = detections(&case.verification);
        let centroids: Vec<_> = search
            .iter()
            .map(|d| detection_to_centroid(d, case.width, case.height))
            .collect();
        let prefixes = ladder_prefixes(search.len());
        let mut attempts: Vec<_> = build_attempts(&opts).into_iter().map(|a| (0, a)).collect();
        for (index, center) in BLIND_DENSE_CENTERS.iter().enumerate() {
            let (lo, hi) = dense_band_bounds(DbKind::dense_for_center(*center));
            attempts.push((
                index + 1,
                Attempt {
                    label: "dense",
                    fov_deg: (lo + hi) * 0.5,
                    fov_err_deg: (hi - lo) * 0.5 + 1.0,
                    attitude_hint: None,
                },
            ));
        }
        let started = Instant::now();
        let mut logs = Vec::new();
        let mut softs = Vec::new();
        let mut accepted = None;
        'attempts: for (db, attempt) in attempts {
            for &k in &prefixes {
                let t = Instant::now();
                let result = databases[db].solve_from_centroids(
                    &centroids[..k],
                    &attempt.solve_config(case.width, case.height),
                );
                let row = match result {
                    Err(e) => serde_json::json!({"error": format!("{:?}", e.status)}),
                    Ok(sol) => {
                        let candidate =
                            verify_separately(&sol, &search, &verification, &stars, attempt.label);
                        let row = serde_json::json!({
                            "matches": sol.num_matches, "prob": sol.prob,
                            "matched_catalog_ids": sol.matched_catalog_ids,
                            "matched_search_indices": sol.matched_centroid_indices,
                            "fov_deg": sol.fov_rad.to_degrees(),
                            "verification": candidate.as_ref().map(|c| serde_json::json!({
                                "hits": c.verify.hits, "predicted": c.verify.predicted,
                                "log_odds": c.verify.log_odds, "rms_px": c.verify.rms_px,
                            })),
                        });
                        if let Some(c) = candidate {
                            if sol.num_matches >= HARD_MIN_MATCHES
                                && sol.prob < HARD_MAX_PROB
                                && c.verify.hits >= VERIFY_MIN_HITS
                            {
                                accepted = Some(c);
                            } else {
                                softs.push(c);
                            }
                        }
                        row
                    }
                };
                logs.push(serde_json::json!({"database": db, "label": attempt.label,
                    "fov_deg": attempt.fov_deg, "k": k, "elapsed_ms": t.elapsed().as_secs_f64()*1000.0, "result": row}));
                if accepted.is_some() {
                    break 'attempts;
                }
            }
        }
        let chosen = MatchOutcome {
            accepted,
            softs,
            best_fov: None,
        }
        .into_chosen();
        eprintln!("CATALOG REPLAY {} accepted={}", case.name, chosen.is_some());
        results.push(serde_json::json!({
            "name": case.name, "search_count": search.len(), "verification_count": verification.len(),
            "elapsed_ms": started.elapsed().as_secs_f64()*1000.0,
            "attempts": logs, "chosen_before_refinement": chosen.map(|c| serde_json::json!({
                "hits": c.verify.hits, "log_odds": c.verify.log_odds, "rms_px": c.verify.rms_px,
                "pose": {"ra_deg": c.pose.ra_deg, "dec_deg": c.pose.dec_deg,
                    "roll_deg": c.pose.roll_deg, "focal_px": c.pose.focal_px,
                    "width": c.pose.width, "height": c.pose.height},
                "attitude_quat": c.attitude_quat,
            })),
        }));
    }
    std::fs::write(
        std::env::var("STARGLYPH_CATALOG_OUTPUT")?,
        serde_json::to_vec_pretty(&serde_json::json!({
            "databases": summaries, "cases": results, "epoch_years": epoch,
        }))?,
    )?;
    Ok(())
}

#[test]
#[ignore = "offline regression: uses the positive development replay fixture"]
fn separate_verification_preserves_pose_and_uses_full_list(
) -> Result<(), Box<dyn std::error::Error>> {
    let input: Input =
        serde_json::from_slice(&std::fs::read(std::env::var("STARGLYPH_CATALOG_INPUT")?)?)?;
    let case = input.cases.first().ok_or("missing positive case")?;
    let search = detections(&case.search);
    let stars = VerifyStars::build(&Catalog::load(&input.catalog)?, None);
    let db =
        tetra3::SolverDatabase::load_from_file(input.databases[0].to_str().ok_or("invalid path")?)?;
    let centroids: Vec<_> = search
        .iter()
        .map(|d| detection_to_centroid(d, case.width, case.height))
        .collect();
    for attempt in build_attempts(&SolveOptions::default()) {
        for k in ladder_prefixes(search.len()) {
            let Ok(sol) = db.solve_from_centroids(
                &centroids[..k],
                &attempt.solve_config(case.width, case.height),
            ) else {
                continue;
            };
            let Some(original) = make_candidate(&sol, &search, &stars, attempt.label) else {
                continue;
            };
            if original.verify.hits < VERIFY_MIN_HITS {
                continue;
            }
            let same = verify_separately(&sol, &search, &search, &stars, attempt.label)
                .ok_or("no candidate")?;
            assert_eq!(same.pose, original.pose);
            assert_eq!(same.verify.hits, original.verify.hits);
            assert_eq!(same.verify.log_odds, original.verify.log_odds);
            assert_eq!(same.verify.rms_px, original.verify.rms_px);
            // Search indices must not accidentally address the verification list.
            let mut reversed = search.clone();
            reversed.reverse();
            let reversed = verify_separately(&sol, &search, &reversed, &stars, attempt.label)
                .ok_or("no candidate")?;
            assert_eq!(reversed.pose, original.pose);
            assert_eq!(reversed.verify.hits, original.verify.hits);
            let empty = verify_separately(&sol, &search, &[], &stars, attempt.label)
                .ok_or("no candidate")?;
            assert_eq!(empty.pose, original.pose);
            assert_eq!(empty.verify.hits, 0);
            assert!(!empty.verify.is_verified());
            return Ok(());
        }
    }
    Err("positive fixture did not produce a verified candidate".into())
}

#[test]
#[ignore = "offline regression: compares research replay with the production matcher"]
fn control_matches_production() -> Result<(), Box<dyn std::error::Error>> {
    let input: Input =
        serde_json::from_slice(&std::fs::read(std::env::var("STARGLYPH_CATALOG_INPUT")?)?)?;
    let result: serde_json::Value =
        serde_json::from_slice(&std::fs::read(std::env::var("STARGLYPH_CATALOG_OUTPUT")?)?)?;
    let frame = FrameImage::load(&input.image)?;
    let opts = SolveOptions {
        cache_dir: input.databases[0]
            .parent()
            .ok_or("no cache parent")?
            .to_path_buf(),
        ..SolveOptions::default()
    }
    .resolved_for(&frame);
    let catalog = Catalog::load(&input.catalog)?;
    let epoch = frame
        .acquisition_timestamp()
        .map(|t| crate::ephem::epoch_years(t.to_jd_utc(opts.utc_offset_hours)));
    let verify = VerifyStars::build(&catalog, epoch);
    let mut engine = Engine::ensure(&catalog, DbKind::Bootstrap, &opts.cache_dir, &mut |_| {})?;
    let mut checked = 0;
    for case in &input.cases {
        if !case.name.ends_with("-control") {
            continue;
        }
        let found = run_matching(
            &mut engine,
            &catalog,
            &verify,
            &detections(&case.search),
            &opts,
            case.width,
            case.height,
            false,
        )
        .into_chosen();
        let expected = result["cases"]
            .as_array()
            .ok_or("no results")?
            .iter()
            .find(|c| c["name"] == case.name)
            .ok_or("no case")?;
        let expected = &expected["chosen_before_refinement"];
        assert_eq!(found.is_none(), expected.is_null(), "{}", case.name);
        if let Some(c) = found {
            assert_eq!(serde_json::json!(c.verify.hits), expected["hits"]);
            assert_eq!(serde_json::json!(c.verify.log_odds), expected["log_odds"]);
            assert_eq!(serde_json::json!(c.verify.rms_px), expected["rms_px"]);
        }
        checked += 1;
    }
    assert!(checked > 0);
    Ok(())
}
