#!/usr/bin/env python3
"""Frozen-objective stationarity and initialization check on development Orion."""
import argparse
import copy
import os
from pathlib import Path
import shutil

import numpy as np

from collection_run import ROOT, digest, write_json
from compare_local_wcs import project, stats
from robustness_candidate_trace import command
from robustness_orion_fit import METHODS, WIDTH, HEIGHT, RID, read, previous, validate as validate42
from robustness_reference_audit import selected
from robustness_refinement_report import lift
from robustness_tangent_fit import correction

OLD = ROOT/'prototype/artifacts/robustness-iteration-42'
ARMS = ('replay', 'continue', 'alternate')


def build_input(original, fitted, alternate):
    if (original['id'], original['split'], fitted['id'], fitted['split']) != (RID, 'development', RID, 'development'):
        raise ValueError('frozen development required')
    old = {c['name']: c for c in fitted['cases']}
    cases = []
    for method in METHODS:
        name = 'distributed16_'+method
        source = next(c for c in original['cases'] if c['name'] == name)
        saved = old[name]
        if len(source['matches']) != 16 or len({m['source_id'] for m in source['matches']}) != 16:
            raise ValueError('sixteen distinct frozen pairs required')
        for arm in ARMS:
            case = copy.deepcopy(source)
            case.update(name=arm+'_'+method, prior_weight=saved['prior_weight'], expected=None)
            if arm == 'replay':
                case['expected'] = saved['camera']
            else:
                case['initial'] = copy.deepcopy(saved['camera'] if arm == 'continue' else alternate)
            if any(case['initial'][k] != source['initial'][k] for k in ('width', 'height')):
                raise ValueError('start in different pixel system')
            cases.append(case)
    return dict(id=RID, split='development', cases=cases, probe_worlds=original['probe_worlds'])


def stationarity(case):
    """Local Gauss-Newton diagnostic with the saved nonzero prior residual."""
    j, r, jp = (np.asarray(case[k], float) for k in ('jacobian', 'residuals', 'probe_jacobian'))
    if j.shape != (33, 5) or r.shape != (33,) or jp.shape != (95, 5) or np.any(jp[-1] != 0):
        raise ValueError('frozen16 pairs and47 probe derivatives required')
    delta, info = correction(j[:-1], r[:-1], j[-1], r[-1])
    remaining = r+j@delta
    change = (jp[:-1]@delta).reshape(-1, 2)*[WIDTH/case['camera']['width'], HEIGHT/case['camera']['height']]
    reduction = float((r@r-remaining@remaining)/(r@r))
    maximum = float(np.linalg.norm(change, axis=1).max())
    norms = np.linalg.norm(j, axis=0)
    return dict(delta_parameters=delta.tolist(), information=info, objective=float(r@r),
        linear_objective=float(remaining@remaining), relative_linear_reduction=reduction,
        normalized_gradient=(j.T@r/(norms*np.linalg.norm(r))).tolist(),
        max_linear_probe_shift_original_px=maximum,
        stationary_diagnostic=maximum <= .01 and abs(reduction) <= 1e-8)


def prepare(out):
    prior = validate42(OLD)
    checks = read(previous('42-checks'))
    for name in ('fits', 'input', 'results'):
        p = OLD/(name+'.json')
        if digest(p) != checks['artifact_hashes'][str(p.relative_to(ROOT))]:
            raise ValueError('saved fit changed: '+name)
    trace = read(previous('41-trace'))
    if trace['steps'][-1]['stage'] != 'final':
        raise ValueError('saved final working camera required')
    inp = build_input(read(OLD/'input.json'), read(OLD/'fits.json'), trace['steps'][-1]['camera'])
    if out.exists():
        raise ValueError('fresh output required')
    out.mkdir(parents=True)
    shutil.copy2(OLD/'pair-test', out/'pair-test')
    write_json(out/'input.json', inp)
    paths = [Path(__file__).resolve(), out/'input.json', out/'pair-test', OLD/'protocol.json',
        *(OLD/(n+'.json') for n in ('fits', 'input', 'results')), previous('42-checks'),
        previous('40-results'), previous('41-trace'),
        *(Path(__file__).with_name(n) for n in ('robustness_orion_fit.py', 'robustness_tangent_fit.py',
          'robustness_observability.py', 'robustness_refinement_report.py', 'compare_local_wcs.py',
          'robustness_candidate_trace.py', 'collection_run.py'))]
    write_json(out/'protocol.json', dict(iteration=43, id=RID, split='development', holdout=False,
        manual_diagnostic=True, selection=prior['selection'], check_groups=prior['check_groups'],
        case_names=[c['name'] for c in inp['cases']],
        alternate_start='iteration41 final working camera, before native fit; no WCS-derived start or parameter sweep',
        objective='same16 ordered pairs per centroid method; same explicit saved prior weight3.5993484637547715 and k1 target0; same5 free parameters',
        budget='9 unchanged production LM calls:3 exact replay+3 continuation+3 alternate; no rematch;60s wall limit',
        criteria=['three replay cameras reproduce iteration42 with existing harness tolerances',
            'local stationarity diagnostic: max Gauss-Newton shift on47<=0.01 original px AND abs(relative objective reduction)<=1e-8, including prior',
            'converged for tested starts: every pair of output projections differs<=0.01 original px on all47 AND relative objective gap<=1e-8',
            'report all9 fits, all47 sources and fixed fit16/check31 groups; no camera selection by check residuals',
            'retain numerical ambiguity; local derivative and two starts do not prove global optimality or model impossibility'],
        hashes={str(p.relative_to(ROOT)): digest(p) for p in paths}))


def validate(out):
    p = read(out/'protocol.json')
    if (p['id'], p['split'], p['holdout']) != (RID, 'development', False):
        raise ValueError('development only')
    selected()
    validate42(OLD)
    for name, sha in p['hashes'].items():
        if digest(ROOT/name) != sha:
            raise ValueError('frozen input changed: '+name)
    return p


def run(out):
    validate(out)
    if (out/'run.json').exists():
        raise ValueError('fresh run required')
    command(out, 'run', [str(out/'pair-test'), '--exact', 'solve::pair_refit::fixed_pairs', '--ignored'],
        {**os.environ, 'STARGLYPH_PAIR_INPUT': str(out/'input.json'), 'STARGLYPH_PAIR_OUTPUT': str(out/'fits.json')}, 60)
    validate(out)


def summarize(out):
    p = validate(out)
    fitted, inp = read(out/'fits.json'), read(out/'input.json')
    if (fitted['id'], fitted['split']) != (RID, 'development') or [c['name'] for c in fitted['cases']] != p['case_names']:
        raise ValueError('fit scope changed')
    sources = read(previous('40-results'))['sources']
    groups = sorted({g for s in sources for g in s['groups']})
    check = np.array([s['id'] in p['selection']['check_ids'] for s in sources])
    def projection(cam):
        rotated = np.array(inp['probe_worlds'])@np.array(cam['world_to_camera']).T
        r2 = np.sum((rotated[:, :2]/rotated[:, 2, None])**2, axis=1)
        if np.any(rotated[:, 2] <= 0) or (cam['k1'] < 0 and np.any(r2 >= -1/(3*cam['k1']))):
            raise ValueError('source outside monotone camera domain')
        return lift(project(cam, [s['radec'] for s in sources]), cam, WIDTH, HEIGHT)
    rows = []
    for case, original in zip(fitted['cases'], inp['cases']):
        if case['matches'] != 16 or case['prior_weight'] != original['prior_weight']:
            raise ValueError('objective changed')
        if case['name'].startswith('replay_') and not case['reproduced_previous_first_fit']:
            raise ValueError('replay failed')
        method = next(m for m in METHODS if case['name'].endswith(m))
        cam = case['camera']
        units = np.array([m['world'] for m in original['matches']])
        angles = np.rad2deg(np.column_stack((np.arctan2(units[:, 1], units[:, 0]), np.arcsin(units[:, 2]))))
        residual = project(cam, angles)-[m['xy'] for m in original['matches']]
        if not np.allclose(residual.ravel(), case['residuals'][:-1], rtol=0, atol=1e-8):
            raise ValueError('Rust/Python residual mismatch')
        if not np.isclose(case['residuals'][-1], cam['k1']*case['prior_weight'], rtol=0, atol=1e-12):
            raise ValueError('nonzero prior residual lost')
        pred = projection(cam)
        errors = np.linalg.norm(pred-[s['coordinates'][method] for s in sources], axis=1)
        def grouped(subset):
            return {g: stats(errors[m]) if m.any() else dict(count=0)
                for g in groups for m in [subset & np.array([g in s['groups'] for s in sources]) ]}
        rows.append(dict(name=case['name'], camera=cam, prior_weight=case['prior_weight'], prior_scale=case['prior_scale'],
            elapsed_ms=case['elapsed_ms'], matches=16, replay_reproduced=case['reproduced_previous_first_fit'],
            stationarity=stationarity(case), predicted_xy=pred.tolist(), errors_px=errors.tolist(),
            fit16=stats(errors[~check]), check_groups=grouped(check), all_groups=grouped(np.ones(47, bool)),
            initial_predicted_xy=projection(case['initial']).tolist()))
    comparisons = []
    for i, method in enumerate(METHODS):
        three = rows[i*3:i*3+3]
        for a, b in ((0, 1), (0, 2), (1, 2)):
            ca, cb = three[a], three[b]
            d = stats(np.linalg.norm(np.array(ca['predicted_xy'])-cb['predicted_xy'], axis=1))
            oa, ob = (c['stationarity']['objective'] for c in (ca, cb))
            gap = abs(oa-ob)/max(oa, ob)
            comparisons.append(dict(method=method, arms=[ARMS[a], ARMS[b]], projection_difference=d,
                relative_objective_gap=gap, converged=d['max_px'] <= .01 and gap <= 1e-8))
    write_json(out/'results.json', dict(iteration=43, id=RID, split='development', holdout=False,
        protocol_sha256=digest(out/'protocol.json'), selection=p['selection'], cases=rows, comparisons=comparisons,
        all_tested_starts_converged=all(c['converged'] for c in comparisons),
        all_stationary_diagnostic=all(c['stationarity']['stationary_diagnostic'] for c in rows),
        sources=[dict(id=s['id'], hyg_id=s['hyg_id'], name=s['name'], groups=s['groups']) for s in sources],
        run=read(out/'run.json'), new_fits=9, new_solver_calls=0, new_wcs_calls=0, process_failures=0, timeouts=0,
        production_changed=False, independent_ground_truth=False, automatic_improvement_confirmed=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'validate', 'run', 'summarize'])
    parser.add_argument('--out-dir', type=Path, required=True)
    a = parser.parse_args()
    globals()[a.stage](a.out_dir.resolve())
