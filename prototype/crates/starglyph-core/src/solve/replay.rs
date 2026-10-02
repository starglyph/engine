//! Explicit, offline research replay of saved detections through the real matcher.
use super::*;
use serde::Deserialize;

#[derive(Deserialize)]
struct Input {
    image: PathBuf,
    catalog: PathBuf,
    cache: PathBuf,
    width: u32,
    height: u32,
    #[serde(default)]
    extra_prefixes: Vec<usize>,
    #[serde(default)]
    extra_offsets: Vec<usize>,
    #[serde(default)]
    extra_dense: bool,
    cases: Vec<Case>,
}

#[derive(Deserialize)]
struct Case {
    name: String,
    detections: Vec<SavedDetection>,
}

#[derive(Deserialize)]
struct SavedDetection {
    x: f64,
    y: f64,
    flux: f32,
    peak: f32,
    snr: f32,
    area: u32,
    elongation: f32,
    rank: u32,
}

#[test]
#[ignore = "research replay: requires STARGLYPH_REPLAY_INPUT and STARGLYPH_REPLAY_OUTPUT"]
fn saved_detections() -> Result<(), Box<dyn std::error::Error>> {
    let input: Input =
        serde_json::from_slice(&std::fs::read(std::env::var("STARGLYPH_REPLAY_INPUT")?)?)?;
    let frame = FrameImage::load(&input.image)?;
    let opts = SolveOptions {
        cache_dir: input.cache,
        ..SolveOptions::default()
    }
    .resolved_for(&frame);
    let epoch = frame
        .acquisition_timestamp()
        .map(|t| crate::ephem::epoch_years(t.to_jd_utc(opts.utc_offset_hours)));
    let catalog = Catalog::load(&input.catalog)?;
    let verify = VerifyStars::build(&catalog, epoch);
    let mut engine = Engine::ensure(&catalog, DbKind::Bootstrap, &opts.cache_dir, &mut |_| {})?;
    let mut results = Vec::new();
    for case in input.cases {
        let detections: Vec<_> = case
            .detections
            .into_iter()
            .map(|d| Detection {
                x: d.x,
                y: d.y,
                flux: d.flux,
                peak: d.peak,
                snr: d.snr,
                area: d.area,
                elongation: d.elongation,
                rank: d.rank,
            })
            .collect();
        eprintln!("REPLAY {}", case.name);
        let matching_started = Instant::now();
        let chosen = run_matching(
            &mut engine,
            &catalog,
            &verify,
            &detections,
            &opts,
            input.width,
            input.height,
            true,
        )
        .into_chosen();
        let matching_ms = matching_started.elapsed().as_secs_f64() * 1000.0;
        let mut extra = Vec::new();
        let centroids: Vec<_> = detections
            .iter()
            .map(|d| detection_to_centroid(d, input.width, input.height))
            .collect();
        let mut attempts: Vec<_> = build_attempts(&opts)
            .into_iter()
            .map(|a| (DbKind::Bootstrap, a))
            .collect();
        if input.extra_dense && opts.allow_dense_band {
            let centers = opts
                .fov_hint_deg
                .map_or_else(|| BLIND_DENSE_CENTERS.to_vec(), |f| vec![f]);
            for center in centers {
                if center < MIN_DENSE_CENTER_DEG {
                    continue;
                }
                let kind = DbKind::dense_for_center(center);
                let (lo, hi) = dense_band_bounds(kind);
                attempts.push((
                    kind,
                    Attempt {
                        label: "dense",
                        fov_deg: (lo + hi) * 0.5,
                        fov_err_deg: (hi - lo) * 0.5 + 1.0,
                        attitude_hint: opts.attitude_hint,
                    },
                ));
            }
        }
        for (kind, attempt) in attempts {
            if input.extra_prefixes.is_empty() {
                break;
            }
            engine.ensure_kind(&catalog, kind, &opts.cache_dir, &mut |_| {})?;
            for offset in std::iter::once(0).chain(input.extra_offsets.iter().copied()) {
                if offset >= centroids.len() {
                    continue;
                }
                for &requested in &input.extra_prefixes {
                    let k = requested.min(centroids.len() - offset);
                    if k < MIN_DETECTIONS {
                        continue;
                    }
                    let started = Instant::now();
                    let result = engine
                        .get(kind)
                        .expect("database ensured")
                        .solve_from_centroids(
                            &centroids[offset..offset + k],
                            &attempt.solve_config(input.width, input.height),
                        );
                    let summary = match result {
                        Ok(sol) => {
                            let candidate =
                                make_candidate(&sol, &detections, &verify, attempt.label);
                            serde_json::json!({
                                "matches": sol.num_matches, "prob": sol.prob,
                                "candidate": candidate.map(|c| serde_json::json!({
                                    "accepted": (sol.num_matches >= HARD_MIN_MATCHES && sol.prob < HARD_MAX_PROB
                                        && c.verify.hits >= VERIFY_MIN_HITS) || c.verify.is_verified(),
                                    "hits": c.verify.hits, "log_odds": c.verify.log_odds,
                                    "rms_px": c.verify.rms_px, "attitude_quat": c.attitude_quat,
                                })),
                            })
                        }
                        Err(e) => serde_json::json!({"error": format!("{:?}", e.status)}),
                    };
                    extra.push(serde_json::json!({"label": attempt.label, "k": k,
                    "offset": offset, "fov_deg": attempt.fov_deg,
                    "elapsed_ms": started.elapsed().as_secs_f64() * 1000.0, "result": summary}));
                }
            }
        }
        results.push(serde_json::json!({
            "name": case.name,
            "matching_ms": matching_ms,
            "extra_prefixes": extra,
            "chosen_before_refinement": chosen.map(|c| serde_json::json!({
                "label": c.label, "hits": c.verify.hits,
                "log_odds": c.verify.log_odds, "rms_px": c.verify.rms_px,
                "attitude_quat": c.attitude_quat,
            })),
        }));
    }
    std::fs::write(
        std::env::var("STARGLYPH_REPLAY_OUTPUT")?,
        serde_json::to_vec_pretty(&serde_json::json!({
            "fov_hint_deg": opts.fov_hint_deg, "epoch_years": epoch, "cases": results,
        }))?,
    )?;
    Ok(())
}
