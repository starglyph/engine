//! Diagnostic only: injected into an isolated copy of solve.rs as a test module.
use super::*;
use serde_json::{json, Value};

fn camera(p: &CameraSolution) -> Value {
    let r = p.rotation();
    json!({"width":p.width, "height":p.height, "focal_px":p.focal_px, "k1":p.k1,
        "ra_deg":p.ra_deg, "dec_deg":p.dec_deg, "roll_deg":p.roll_deg,
        "world_to_camera":(0..3).map(|i| (0..3).map(|j| r[(i,j)]).collect::<Vec<_>>()).collect::<Vec<_>>()})
}

fn snapshot(name: &str, pose: &CameraSolution, pairs: &[Match], stars: &VerifyStars) -> Value {
    let rotation = pose.rotation();
    let rows: Vec<_> = pairs
        .iter()
        .map(|m| {
            let mut ids: Vec<_> = stars
                .by_id
                .iter()
                .filter_map(|(id, unit)| (*unit == m.world).then_some(*id))
                .collect();
            ids.sort_unstable();
            let projected = geom::project(
                &rotation,
                pose.focal_px,
                pose.k1,
                pose.width,
                pose.height,
                m.world,
            );
            json!({"hyg_ids":ids,"world":m.world,"xy":[m.px,m.py],"projected_xy":projected})
        })
        .collect();
    json!({"stage":name,"camera":camera(pose),"pairs":rows})
}

// Deliberately mirrors the unchanged production sequence; the final pose and
// detections are checked against the saved production report outside this test.
fn stages(
    c: &Candidate,
    detections: &[Detection],
    stars: &VerifyStars,
    original: &FrameImage,
) -> Vec<Value> {
    let mut result = vec![snapshot("candidate", &c.pose, &c.verify.matches, stars)];
    let mut refined = refine_pose(&c.pose, &c.verify.matches);
    result.push(snapshot(
        "initial_refine",
        &refined,
        &c.verify.matches,
        stars,
    ));
    if refined.fov_x_deg() > K1_TAPER_START_FOV_DEG {
        for radius in WIDE_REMATCH_RADII_PX {
            let rematch = match_predictions(&refined, stars, detections, radius);
            result.push(snapshot(
                &format!("rematch_{radius}_before"),
                &refined,
                &rematch.matches,
                stars,
            ));
            if rematch.matches.len() >= REMATCH_MIN_MATCHES {
                refined = refine_pose(&refined, &rematch.matches);
            }
            result.push(snapshot(
                &format!("rematch_{radius}_after"),
                &refined,
                &rematch.matches,
                stars,
            ));
        }
    }
    let final_match = match_predictions(&refined, stars, detections, FINAL_RADIUS_PX);
    result.push(snapshot(
        "working_final",
        &refined,
        &final_match.matches,
        stars,
    ));
    let sx = f64::from(original.width) / f64::from(refined.width);
    let sy = f64::from(original.height) / f64::from(refined.height);
    let mut lifted_detections = detections.to_vec();
    for d in &mut lifted_detections {
        d.x = (d.x + 0.5) * sx - 0.5;
        d.y = (d.y + 0.5) * sy - 0.5;
    }
    let lifted: Vec<_> = final_match
        .matches
        .iter()
        .map(|m| Match {
            world: m.world,
            px: (m.px + 0.5) * sx - 0.5,
            py: (m.py + 0.5) * sy - 0.5,
        })
        .collect();
    refined.width = original.width;
    refined.height = original.height;
    refined.focal_px *= sx;
    result.push(snapshot("lift_before_refine", &refined, &lifted, stars));
    refined = refine_pose_with_scale(&refined, &lifted, sx.max(sy));
    result.push(snapshot("lift_after_refine", &refined, &lifted, stars));
    let final_match = match_predictions(
        &refined,
        stars,
        &lifted_detections,
        FINAL_RADIUS_PX * sx.max(sy),
    );
    result.push(snapshot(
        "original_final",
        &refined,
        &final_match.matches,
        stars,
    ));
    result
}

#[test]
#[ignore = "development-only research: requires STARGLYPH_GEOMETRY_INPUT and OUTPUT"]
fn candidate_geometry() -> Result<(), Box<dyn std::error::Error>> {
    let input: Value =
        serde_json::from_slice(&std::fs::read(std::env::var("STARGLYPH_GEOMETRY_INPUT")?)?)?;
    assert_eq!(input["split"], "development");
    assert_eq!(input["id"], "wm_r_143159342");
    let original = FrameImage::load(std::path::Path::new(
        input["image"].as_str().ok_or("image path missing")?,
    ))?;
    let working = original
        .resized(WORKING_MAX_EDGE)
        .ok_or("working image required")?;
    let opts = SolveOptions {
        cache_dir: input["cache"].as_str().ok_or("cache path missing")?.into(),
        ..SolveOptions::default()
    }
    .resolved_for(&working);
    let epoch = working
        .acquisition_timestamp()
        .map(|t| crate::ephem::epoch_years(t.to_jd_utc(opts.utc_offset_hours)));
    assert!(
        epoch.is_none() && opts.fov_hint_deg.is_none(),
        "this replay requires the frozen blind inputs"
    );
    let catalog = Catalog::load(std::path::Path::new(
        input["catalog"].as_str().ok_or("catalog path missing")?,
    ))?;
    let stars = VerifyStars::build(&catalog, epoch);
    let mut engine = Engine::ensure(&catalog, DbKind::Bootstrap, &opts.cache_dir, &mut |_| {})?;
    let mut tiers = Vec::new();
    for (tier, config) in [
        ("default", DetectConfig::default()),
        ("deep", deep_detect_config()),
    ] {
        let detections = detect_stars_with_mask(&working, &config, None).detections;
        eprintln!("GEOMETRY TIER {tier}");
        let start = Instant::now();
        let outcome = run_matching(
            &mut engine,
            &catalog,
            &stars,
            &detections,
            &opts,
            working.width,
            working.height,
            true,
        );
        let matching_ms = start.elapsed().as_secs_f64() * 1000.0;
        let chosen_index = if outcome.accepted.is_some() {
            Some(outcome.softs.len())
        } else {
            outcome
                .softs
                .iter()
                .enumerate()
                .filter(|(_, c)| c.verify.is_verified())
                .max_by(|(_, a), (_, b)| a.verify.log_odds.total_cmp(&b.verify.log_odds))
                .map(|(i, _)| i)
        };
        let mut candidates = Vec::new();
        for (i, c) in outcome
            .softs
            .iter()
            .chain(outcome.accepted.iter())
            .enumerate()
        {
            let start = Instant::now();
            let snapshots = stages(c, &detections, &stars, &original);
            candidates.push(
                json!({"index":i,"label":c.label,"chosen":Some(i)==chosen_index,
                "hard":outcome.accepted.is_some() && i==outcome.softs.len(),
                "verified":c.verify.is_verified(),"hits":c.verify.hits,"log_odds":c.verify.log_odds,
                "rms_px":c.verify.rms_px,"attitude_quat":c.attitude_quat,"stages":snapshots,
                "refinement_ms":start.elapsed().as_secs_f64()*1000.0}),
            );
        }
        tiers.push(json!({"tier":tier,"detections":detections,"width":working.width,"height":working.height,
            "matching_ms":matching_ms,"chosen_index":chosen_index,"candidates":candidates}));
    }
    std::fs::write(
        std::env::var("STARGLYPH_GEOMETRY_OUTPUT")?,
        serde_json::to_vec_pretty(&json!({
            "id":input["id"],"split":"development","tiers":tiers,
            "interpretation":"both tiers replayed with unchanged search budget; unchosen refinements diagnostic only"
        }))?,
    )?;
    Ok(())
}
