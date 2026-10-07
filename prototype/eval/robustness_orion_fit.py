#!/usr/bin/env python3
"""Development-only distributed fixed-pair fit with the existing production LM."""
import argparse
import json
import os
from pathlib import Path
import shutil

import numpy as np

from collection_run import ROOT, digest, write_json
from compare_local_wcs import project, stats
from robustness_candidate_trace import command
from robustness_centroid_origin import prepare as prepare_workspace
from robustness_reference_audit import RID, MANIFEST, read, selected
from robustness_refinement_report import lift

PUBLIC = ROOT/'docs/experiments'
METHODS = ('external_centroid', 'native_r8', 'native_r12')
WIDTH, HEIGHT = 4485, 2920


def previous(name):
    return PUBLIC/f'robustness-iteration-{name}.json'


def selection(sources, candidates):
    """One brightest reviewed star per measured-position cell; no residual input."""
    mags = {p['id']: p['mag'] for p in candidates}
    cells = {}
    for s in sorted(sources, key=lambda s: (mags[s['id']], s['hyg_id'])):
        x, y = s['coordinates']['external_centroid']
        if not (0 <= x < WIDTH and 0 <= y < HEIGHT):
            raise ValueError('source outside photograph')
        cell = (int(4*y/HEIGHT), int(4*x/WIDTH))
        cells.setdefault(cell, s['id'])
    if set(cells) != {(y, x) for y in range(4) for x in range(4)}:
        raise ValueError('all sixteen cells required; no substitution')
    fit = [cells[c] for c in sorted(cells)]
    check = [s['id'] for s in sources if s['id'] not in fit]
    if len(sources) != 47 or len(set(fit+check)) != 47:
        raise ValueError('47 distinct reviewed sources required')
    return dict(fit_ids=fit, check_ids=check,
        cells=[dict(row=y, column=x, source_id=cells[y, x]) for y, x in sorted(cells)])


def working_xy(xy, camera):
    return ((np.asarray(xy)+.5)*[camera['width']/WIDTH, camera['height']/HEIGHT]-.5).tolist()


def build_input(sources, chosen, trace):
    initial = trace['input']['camera']
    replay = trace['steps'][0]
    if replay['stage'] != 'first_lm' or len(replay['fit_matches']) != 16:
        raise ValueError('saved first16 fit required')
    ra, dec = np.deg2rad([s['radec'] for s in sources]).T
    worlds = np.column_stack((np.cos(dec)*np.cos(ra), np.cos(dec)*np.sin(ra), np.sin(dec))).tolist()
    cases = [dict(name='reproduce_local16', initial=initial, matches=replay['fit_matches'],
        expected=replay['camera'], prior_weight=None)]
    for method in METHODS:
        matches = [dict(source_id=s['id'], hyg_id=s['hyg_id'], world=w,
            xy=working_xy(s['coordinates'][method], initial))
            for s, w in zip(sources, worlds) if s['id'] in chosen['fit_ids']]
        if len(matches) != 16:
            raise ValueError('sixteen fixed pairs required')
        cases.append(dict(name='distributed16_'+method, initial=initial, matches=matches,
            expected=None, prior_weight=None))
    return dict(id=RID, split='development', cases=cases, probe_worlds=worlds)


def adapted_harness(text):
    guard = 'assert_eq!(input.id, "wm_r_143159342");'
    if text.count(guard) != 1:
        raise ValueError('existing fixed-pair harness guard changed')
    return text.replace(guard, 'assert_eq!(input.id, "'+RID+'");')


def prepare(out):
    rec = selected()
    sources = read(previous('40-results'))['sources']
    candidates = read(previous('40-candidates'))['frames'][0]['points']
    trace = read(previous('41-trace'))
    for name, checks in [('40-results', '40-checks'), ('40-candidates', '40-checks'), ('41-trace', '41-checks')]:
        path = previous(name)
        if digest(path) != read(previous(checks))['artifact_hashes'][str(path.relative_to(ROOT))]:
            raise ValueError('published evidence changed: '+name)
    chosen = selection(sources, candidates)
    inp = build_input(sources, chosen, trace)
    prepare_workspace(out)
    pipeline = out/'pipeline'
    target = pipeline/'crates/starglyph-core/src/solve.rs'
    original = ROOT/'prototype/crates/starglyph-core/src/solve.rs'
    target.write_text(original.read_text()+'\n#[cfg(test)]\nmod pair_refit;\n')
    (target.parent/'solve/centroid_origin.rs').unlink()
    (target.parent/'solve/pair_refit.rs').write_text(adapted_harness(Path(__file__).with_name('pair_refit.rs').read_text()))
    write_json(out/'input.json', inp)
    write_json(out/'selection.json', chosen)
    command(out, 'build', ['cargo', 'test', '--offline', '--locked', '--release', '--manifest-path',
        str(pipeline/'Cargo.toml'), '-p', 'starglyph-core', '--features', 'serde_json/float_roundtrip',
        '--lib', '--no-run', '--message-format', 'json'],
        {**os.environ, 'CARGO_TARGET_DIR': str(ROOT/'prototype/target')}, 600)
    artifacts = [json.loads(s) for s in (out/'build.log').read_text().splitlines() if s.startswith('{')]
    binaries = [a['executable'] for a in artifacts if a.get('reason') == 'compiler-artifact'
        and a.get('executable') and a['target']['name'] == 'starglyph_core']
    if len(binaries) != 1:
        raise ValueError('unique core test binary required')
    shutil.copy2(binaries[0], out/'pair-test')
    paths = [Path(__file__).resolve(), out/'input.json', out/'selection.json', out/'pair-test',
        *(previous(n) for n in ['40-results', '40-candidates', '40-checks', '41-trace', '41-checks']),
        *(MANIFEST.parent/n for n in ['manifest.json', 'robustness-results.json', 'collection-wcs-review.json']),
        MANIFEST.parent/rec['file'],
        *(Path(__file__).with_name(n) for n in ['pair_refit.rs', 'collection_run.py', 'compare_local_wcs.py',
            'robustness_candidate_trace.py', 'robustness_centroid_origin.py', 'robustness_reference_audit.py',
            'robustness_refinement_report.py'])]
    for base in (ROOT/'prototype', pipeline):
        paths.extend([base/'Cargo.toml', base/'Cargo.lock'])
        for name in ('starglyph-core', 'starglyph-cli', 'simulator-core'):
            paths.append(base/'crates'/name/'Cargo.toml')
            paths.extend((base/'crates'/name/'src').rglob('*.rs'))
    groups = sorted({g for s in sources for g in s['groups']})
    write_json(out/'protocol.json', dict(iteration=42, id=RID, split='development', holdout=False,
        manual_diagnostic=True, selection=chosen, case_names=[c['name'] for c in inp['cases']],
        selection_rule='brightest reviewed source per4x4 measured-position cell, tie HYG ID; no residual ranking',
        check_groups={g: [s['id'] for s in sources if s['id'] in chosen['check_ids'] and g in s['groups']] for g in groups},
        budget='4 production5parameter LM calls: local16 replay +3 distributed16 centroid methods; same initial and natural prior; no rematching/restarts;60s process limit',
        criteria=['local16 replay reproduces saved first LM within existing harness tolerances',
            'distributed16 favourable if check31 RMS falls>=10percent vs BOTH local16 and saved native final, with no nonempty check group RMS regression>0.5px vs either',
            'criterion must pass separately for all3 centroid methods',
            'report all47 residuals and old edge groups, plus disjoint check intersections; no exclusion after fitting',
            'verify saved residuals against Python projection and reject non-monotone or behind-camera projections',
            'manual identities and changed membership confound spatial coverage with correctness; no automatic improvement or absolute ground truth claim'],
        hashes={str(p.relative_to(ROOT)): digest(p) for p in paths}))


def validate(out):
    p = read(out/'protocol.json')
    if (p['id'], p['split'], p['holdout']) != (RID, 'development', False):
        raise ValueError('development only')
    selected()
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


def favourable(after, before):
    return (after['all_reviewed']['rms_px'] <= .9*before['all_reviewed']['rms_px'] and
        all(v['count'] == 0 or v['rms_px'] <= before[k]['rms_px']+.5 for k, v in after.items()))


def summarize(out):
    p = validate(out)
    inp, fitted = read(out/'input.json'), read(out/'fits.json')
    if fitted['id'] != RID or fitted['split'] != 'development' or [c['name'] for c in fitted['cases']] != p['case_names']:
        raise ValueError('fit scope changed')
    if not fitted['cases'][0]['reproduced_previous_first_fit']:
        raise ValueError('local replay failed')
    sources = read(previous('40-results'))['sources']
    groups = sorted({g for s in sources for g in s['groups']})
    check = np.array([s['id'] in p['selection']['check_ids'] for s in sources])
    fits = ~check
    rows = []
    for case, original in zip(fitted['cases'], inp['cases']):
        cam = case['camera']
        if case['matches'] != 16 or case['prior_scale'] != 1. or case['initial'] != fitted['cases'][0]['initial']:
            raise ValueError('count, initial or prior changed')
        rotated = np.array(inp['probe_worlds'])@np.array(cam['world_to_camera']).T
        r2 = np.sum((rotated[:, :2]/rotated[:, 2, None])**2, axis=1)
        if np.any(rotated[:, 2] <= 0) or (cam['k1'] < 0 and np.any(r2 >= -1/(3*cam['k1']))):
            raise ValueError('reviewed source outside monotone camera domain')
        units = np.array([m['world'] for m in original['matches']])
        angles = np.rad2deg(np.column_stack((np.arctan2(units[:, 1], units[:, 0]), np.arcsin(units[:, 2]))))
        observed = project(cam, angles)-[m['xy'] for m in original['matches']]
        if not np.allclose(observed.ravel(), np.array(case['residuals'])[:-1], rtol=0, atol=1e-8):
            raise ValueError('Rust/Python fit residual mismatch')
        pred = lift(project(cam, [s['radec'] for s in sources]), cam, WIDTH, HEIGHT)
        measures = {}
        for method in METHODS:
            positions = np.array([s['coordinates'][method] for s in sources])
            errors = np.linalg.norm(pred-positions, axis=1)
            native = np.linalg.norm(np.array([s['predicted_xy']['control'] for s in sources])-positions, axis=1)
            masks = {g: np.array([g in s['groups'] for s in sources]) for g in groups}
            def grouped(values, subset):
                return {g: stats(values[m & subset]) if (m & subset).any() else dict(count=0) for g, m in masks.items()}
            measures[method] = dict(errors_px=errors.tolist(), all_groups=grouped(errors, np.ones(len(sources), dtype=bool)),
                check_groups=grouped(errors, check), native_final_check_groups=grouped(native, check),
                distributed_fit16=stats(errors[fits]))
        rows.append(dict(name=case['name'], camera=cam, matches=16, elapsed_ms=case['elapsed_ms'],
            prior_weight=case['prior_weight'], prior_scale=case['prior_scale'],
            local16_reproduced=case['reproduced_previous_first_fit'], predicted_xy=pred.tolist(), measurements=measures,
            fit_residuals_working_px=observed.tolist()))
    comparisons = []
    for row, method in zip(rows[1:], METHODS):
        a = row['measurements'][method]
        comparisons.append(dict(method=method,
            favourable_vs_local16=favourable(a['check_groups'], rows[0]['measurements'][method]['check_groups']),
            favourable_vs_native_final=favourable(a['check_groups'], a['native_final_check_groups'])))
    write_json(out/'results.json', dict(iteration=42, id=RID, split='development', holdout=False,
        protocol_sha256=digest(out/'protocol.json'), selection=p['selection'], cases=rows, comparisons=comparisons,
        distributed_fit_favourable=all(c['favourable_vs_local16'] and c['favourable_vs_native_final'] for c in comparisons),
        sources=[dict(id=s['id'], hyg_id=s['hyg_id'], name=s['name'], groups=s['groups']) for s in sources],
        run=read(out/'run.json'), new_fits=4, new_solver_calls=0, new_wcs_calls=0, process_failures=0, timeouts=0,
        independent_ground_truth=False, production_changed=False, automatic_improvement_confirmed=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'validate', 'run', 'summarize'])
    parser.add_argument('--out-dir', type=Path, required=True)
    args = parser.parse_args()
    globals()[args.stage](args.out_dir.resolve())
