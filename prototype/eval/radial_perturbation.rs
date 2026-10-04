//! Coherent coordinate perturbations with unchanged starts, priors and LM.
use super::*;

#[derive(Deserialize)]
struct CellInput {
    name: String,
    training_indices: Vec<usize>,
}
#[derive(Deserialize)]
struct Expected {
    name: String,
    params: Vec<f64>,
    projected_xy: Vec<[f64; 2]>,
}
#[derive(Deserialize)]
struct PerturbFrame {
    #[serde(flatten)]
    frame: Frame,
    cells: Vec<CellInput>,
    expected: Vec<Expected>,
}
#[derive(Deserialize)]
struct PerturbInput {
    split: String,
    deltas: Vec<[f64; 2]>,
    frames: Vec<PerturbFrame>,
}

fn shifted(matches: &[Match], indices: &[usize], delta: [f64; 2]) -> Vec<Match> {
    assert!(indices.iter().all(|&i| i < matches.len()));
    matches
        .iter()
        .enumerate()
        .map(|(i, m)| {
            let take = indices.contains(&i);
            Match {
                world: m.world,
                px: m.px + if take { delta[0] } else { 0. },
                py: m.py + if take { delta[1] } else { 0. },
            }
        })
        .collect()
}

fn evaluate(frame: &Frame, initial: &CameraSolution, matches: &[Match], extra: bool) -> Value {
    let budget = Budget::default();
    let start = Instant::now();
    let params = fit(initial, matches, extra, &budget);
    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.;
    let camera = to_solution(&params, true, initial);
    let k2 = if extra { params[5] } else { 0. };
    let projected: Vec<_> = frame
        .sources
        .iter()
        .map(|s| project_radial(&camera, k2, s.world))
        .collect();
    let r = residuals(
        &params,
        true,
        k1_reg_weight(initial.fov_x_deg()),
        camera.width,
        camera.height,
        matches,
    );
    let fit_rms = (r.rows(0, 2 * matches.len()).norm_squared() / matches.len() as f64).sqrt();
    json!({"params":params,"camera":camera_json(&camera,k2),"projected_xy":projected,
        "elapsed_ms":elapsed_ms,"iterations":budget.iterations.get(),"residual_evaluations":budget.evaluations.get(),
        "stop":budget.stop.get(),"objective":r.norm_squared(),"fit_rms_px":fit_rms,
        "bound_hit":params[4..].iter().any(|v| v.abs()>=0.5-1e-12)})
}

#[test]
#[ignore = "frozen cell perturbations: STARGLYPH_PERTURB_INPUT and STARGLYPH_PERTURB_OUTPUT"]
fn experiment() -> Result<(), Box<dyn std::error::Error>> {
    let input: PerturbInput =
        serde_json::from_slice(&std::fs::read(std::env::var("STARGLYPH_PERTURB_INPUT")?)?)?;
    assert_eq!(input.split, "development");
    assert_eq!(
        input
            .frames
            .iter()
            .map(|f| f.frame.id.as_str())
            .collect::<Vec<_>>(),
        ["wm_r_132162731", "wm_r_143159342", "wm_r_149276071"]
    );
    assert_eq!(
        input.deltas,
        vec![
            [0.1, 0.],
            [-0.1, 0.],
            [0., 0.1],
            [0., -0.1],
            [1., 0.],
            [-1., 0.],
            [0., 1.],
            [0., -1.]
        ]
    );
    let mut frames = Vec::new();
    for entry in &input.frames {
        let frame = &entry.frame;
        let initial = frame.initial.camera();
        let matches: Vec<_> = frame
            .sources
            .iter()
            .filter(|s| !s.outside_both_inputs)
            .map(|s| Match {
                world: s.world,
                px: s.xy[0],
                py: s.xy[1],
            })
            .collect();
        let mut covered = vec![0; matches.len()];
        assert_eq!(entry.cells.len(), 9);
        for cell in &entry.cells {
            for &i in &cell.training_indices {
                covered[i] += 1;
            }
        }
        assert!(covered.iter().all(|n| *n == 1));
        let mut cases = Vec::new();
        for expected in &entry.expected {
            let extra = expected.name == "k1_k2";
            let baseline = evaluate(frame, &initial, &matches, extra);
            assert_eq!(baseline["params"], json!(expected.params));
            assert_eq!(baseline["projected_xy"], json!(expected.projected_xy));
            let mut trials = Vec::new();
            for cell in &entry.cells {
                if cell.training_indices.is_empty() {
                    trials.push(json!({"cell":cell.name,"status":"skipped_empty_cell"}));
                    continue;
                }
                for delta in &input.deltas {
                    let changed = shifted(&matches, &cell.training_indices, *delta);
                    trials.push(
                        json!({"cell":cell.name,"delta_xy":delta,"status":"completed",
                        "fit":evaluate(frame,&initial,&changed,extra)}),
                    );
                }
            }
            cases.push(json!({"name":expected.name,"baseline":baseline,"trials":trials}));
        }
        frames.push(json!({"id":frame.id,"training_count":matches.len(),"cases":cases}));
    }
    std::fs::write(
        std::env::var("STARGLYPH_PERTURB_OUTPUT")?,
        serde_json::to_vec_pretty(&json!({"split":"development","frames":frames}))?,
    )?;
    Ok(())
}

#[test]
fn shifts_only_selected_coordinates_and_keeps_catalog_vectors() {
    let base: Vec<_> = (0..4)
        .map(|i| Match {
            world: [1., f64::from(i), 0.],
            px: f64::from(i),
            py: 10.,
        })
        .collect();
    let result = shifted(&base, &[1, 3], [0.1, -1.]);
    for i in 0..4 {
        assert_eq!(base[i].world, result[i].world);
        assert_eq!(base[i].px, f64::from(i as u32));
        assert_eq!(result[i].px, base[i].px + if i % 2 == 1 { 0.1 } else { 0. });
        assert_eq!(result[i].py, if i % 2 == 1 { 9. } else { 10. });
    }
}

#[test]
fn empty_cell_does_not_change_observations() {
    let base = [Match {
        world: [1., 0., 0.],
        px: 20.,
        py: 30.,
    }];
    let result = shifted(&base, &[], [1., 0.]);
    assert_eq!((result[0].px, result[0].py), (20., 30.));
}
