//! Export derivatives at frozen cameras; never call fit or rematch.
use super::*;

#[derive(Deserialize)]
struct FixedCase {
    name: String,
    params: Vec<f64>,
    prior_weight: f64,
    projected_xy: Vec<[f64; 2]>,
    fit_residuals: Vec<f64>,
}
#[derive(Deserialize)]
struct FixedFrame {
    id: String,
    width: u32,
    height: u32,
    sources: Vec<Source>,
    cases: Vec<FixedCase>,
}
#[derive(Deserialize)]
struct FixedInput {
    split: String,
    frames: Vec<FixedFrame>,
}

fn rows(matrix: &DMatrix<f64>) -> Vec<Vec<f64>> {
    (0..matrix.nrows())
        .map(|i| matrix.row(i).iter().copied().collect())
        .collect()
}

fn derivatives(
    case: &FixedCase,
    frame: &FixedFrame,
    matches: &[Match],
    multiplier: f64,
) -> DMatrix<f64> {
    let p = &case.params;
    let r0 = residuals(
        p,
        true,
        case.prior_weight,
        frame.width,
        frame.height,
        matches,
    );
    if multiplier == 1. {
        return numeric_jacobian(
            p,
            true,
            case.prior_weight,
            frame.width,
            frame.height,
            matches,
            &r0,
        );
    }
    let mut jac = DMatrix::zeros(r0.len(), p.len());
    for j in 0..p.len() {
        let step = multiplier
            * if j == 3 {
                (p[3].abs() * 1e-4).max(1e-3)
            } else {
                1e-4
            };
        let mut pp = p.clone();
        pp[j] += step;
        let r1 = residuals(
            &pp,
            true,
            case.prior_weight,
            frame.width,
            frame.height,
            matches,
        );
        for i in 0..r0.len() {
            jac[(i, j)] = (r1[i] - r0[i]) / step;
        }
    }
    jac
}

#[test]
#[ignore = "frozen derivative export: STARGLYPH_SENSITIVITY_INPUT and STARGLYPH_SENSITIVITY_OUTPUT"]
fn export() -> Result<(), Box<dyn std::error::Error>> {
    let input: FixedInput = serde_json::from_slice(&std::fs::read(std::env::var(
        "STARGLYPH_SENSITIVITY_INPUT",
    )?)?)?;
    assert_eq!(input.split, "development");
    assert_eq!(
        input
            .frames
            .iter()
            .map(|f| f.id.as_str())
            .collect::<Vec<_>>(),
        ["wm_r_132162731", "wm_r_143159342", "wm_r_149276071"]
    );
    let started = Instant::now();
    let mut frames = Vec::new();
    for frame in &input.frames {
        let probes: Vec<_> = frame
            .sources
            .iter()
            .map(|s| Match {
                world: s.world,
                px: s.xy[0],
                py: s.xy[1],
            })
            .collect();
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
        let mut cases = Vec::new();
        for case in &frame.cases {
            assert_eq!(case.params.len(), if case.name == "k1" { 5 } else { 6 });
            let residual = residuals(
                &case.params,
                true,
                case.prior_weight,
                frame.width,
                frame.height,
                &matches,
            );
            assert_eq!(
                &residual.as_slice()[..2 * matches.len()],
                case.fit_residuals
            );
            let probe_r = residuals(
                &case.params,
                true,
                case.prior_weight,
                frame.width,
                frame.height,
                &probes,
            );
            assert_eq!(case.projected_xy.len(), probes.len());
            let mut projection_error: f64 = 0.;
            for (i, m) in probes.iter().enumerate() {
                projection_error = projection_error
                    .max((probe_r[2 * i] + m.px - case.projected_xy[i][0]).abs())
                    .max((probe_r[2 * i + 1] + m.py - case.projected_xy[i][1]).abs());
            }
            assert!(projection_error < 1e-10);
            let steps: Vec<_> = [0.5, 1., 2.]
                .into_iter()
                .map(|multiplier| {
                    let jac = derivatives(case, frame, &matches, multiplier);
                    let probe = derivatives(case, frame, &probes, multiplier);
                    json!({"multiplier":multiplier,"jacobian":rows(&jac),
                    "probe_jacobian":rows(&probe.rows(0,2*probes.len()).into_owned())})
                })
                .collect();
            cases.push(
                json!({"name":case.name,"matches":matches.len(),"params":case.params,
                "prior_rows":case.params.len()-4,"projection_max_abs_px":projection_error,
                "fit_residuals_exact":true,"steps":steps}),
            );
        }
        frames.push(json!({"id":frame.id,"cases":cases}));
    }
    std::fs::write(
        std::env::var("STARGLYPH_SENSITIVITY_OUTPUT")?,
        serde_json::to_vec_pretty(
            &json!({"split":"development","frames":frames,"elapsed_s":started.elapsed().as_secs_f64()}),
        )?,
    )?;
    Ok(())
}
