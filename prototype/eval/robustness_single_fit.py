#!/usr/bin/env python3
"""Frozen wide-pair subset fit; reuse the existing optimizer and diagnostic module."""
import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np

from collection_run import ROOT, digest, write_json
from compare_local_wcs import project, stats
from robustness_geometry import selected
from robustness_pair_report import sensitivity
from robustness_refinement_report import lift
from robustness_wide_pairs import RID, reviewed
from robustness_wide_replay import validate as validate_prior
from tetra3_footprint import prepare as prepare_pipeline

PRIOR = ROOT/'prototype/artifacts/robustness-iteration-10/fixed'
REVIEW = ROOT/'prototype/artifacts/robustness-iteration-10/review'


def build_cases(trace, points):
    if trace['id'] != RID or trace['split'] != 'development':
        raise ValueError('development trace required')
    cases = []
    for name, label in [('drop_detection_all_stages', 'wide_34'), ('drop_second_leaf', 'without_leaf_33')]:
        old = next(c for c in trace['cases'] if c['name'] == name)
        before = next(s for s in old['stages'] if s['stage'] == 'rematch_10_before')
        after = next(s for s in old['stages'] if s['stage'] == 'rematch_10_after')
        cases.append(dict(name=label, initial=before['camera'],
            matches=[dict(hyg_ids=p['hyg_ids'], world=p['world'], xy=p['xy']) for p in before['pairs']],
            prior_weight=None, expected=after['camera']))
    if cases[0]['initial'] != cases[1]['initial']:
        raise ValueError('common initial camera required')
    if len(points) != 34 or [len(c['matches']) for c in cases] != [34,33]:
        raise ValueError('fixed wide selection required')
    for pair, point in zip(cases[0]['matches'], points):
        if (pair['hyg_ids'] != [point['hyg_id']] or pair['xy'] != point['detector_xy_working']
                or pair['world'] != point['world']):
            raise ValueError('review/pair identity mismatch')
    single = copy.deepcopy(cases[0])
    single.update(name='single_sources_25', expected=None,
                  matches=[p for p, q in zip(single['matches'], points) if q['label'] == 'visible_source'])
    if len(single['matches']) != 25:
        raise ValueError('frozen 25-source subset required')
    return cases+[single]


def prepare(out, registry):
    selected([RID])
    points = reviewed(REVIEW)
    validate_prior(PRIOR)
    geometry_path = ROOT/'docs/experiments/robustness-iteration-10-geometry.json'
    old = json.loads(geometry_path.read_text())
    if digest(PRIOR/'trace.json') != old['trace_sha256']:
        raise ValueError('prior trace changed')
    cases = build_cases(json.loads((PRIOR/'trace.json').read_text()), points)
    sources = old['sources']
    ra, dec = np.deg2rad([[s['ra_deg'], s['dec_deg']] for s in sources]).T
    worlds = np.column_stack((np.cos(dec)*np.cos(ra), np.cos(dec)*np.sin(ra), np.sin(dec))).tolist()
    prepare_pipeline(out, registry)
    project = out/'pipeline'
    for name in ('solve.rs', 'track.rs'):
        shutil.copyfile(registry/'src/solver'/name, project/'tetra3/src/solver'/name)
    manifest = project/'Cargo.toml'
    content = manifest.read_text()
    if content.count('serde_json = "1"') != 1:
        raise ValueError('serde declaration changed')
    manifest.write_text(content.replace('serde_json = "1"',
        'serde_json = { version = "1", features = ["float_roundtrip"] }'))
    solve = project/'crates/starglyph-core/src/solve.rs'
    solve.write_text(solve.read_text()+'\n#[cfg(test)]\nmod pair_refit;\n')
    rust = ROOT/'prototype/eval/pair_refit.rs'
    shutil.copyfile(rust, solve.parent/'solve/pair_refit.rs')
    with (out/'lock-resolution.log').open('w') as log:
        subprocess.run(['cargo', 'update', '--offline', '--workspace', '--manifest-path', str(manifest)],
                       cwd=ROOT/'prototype', stdout=log, stderr=subprocess.STDOUT, check=True)
    write_json(out/'input.json', dict(id=RID, split='development', cases=cases, probe_worlds=worlds))
    files = [Path(__file__).resolve(), rust, geometry_path, PRIOR/'trace.json', REVIEW/'review.json',
        REVIEW/'candidates.json', REVIEW/'protocol.json', out/'input.json', ROOT/'prototype/Cargo.lock',
        ROOT/'data/samples/sky-samples/manifest.json']
    files += [ROOT/'prototype/eval'/name for name in ('robustness_pair_report.py', 'robustness_refinement_report.py',
        'compare_local_wcs.py', 'robustness_wide_pairs.py', 'robustness_wide_replay.py', 'tetra3_footprint.py')]
    files += list((project/'crates').glob('*/src/**/*.rs'))+list((project/'tetra3/src').rglob('*.rs'))
    files += list((project/'crates').glob('*/Cargo.toml'))+[manifest, project/'Cargo.lock', project/'tetra3/Cargo.toml']
    write_json(out/'protocol.json', dict(iteration=11, id=RID, split='development',
        cases=[c['name'] for c in cases], hashes={str(p.resolve()):digest(p) for p in files},
        review_sha256=digest(REVIEW/'review.json'), selected_pair_ids=[p['id'] for p in points if p['label']=='visible_source'],
        criteria=['reproduce both previous wide fits', 'report all 40 fixed probes and 14 outside previous inputs',
                  'report every edge group and per-source regression; no selection by fitted residual'],
        unchanged=['common initial camera', 'five free parameters and natural k1 prior',
                   'retained pair order, detector coordinates and catalogue vectors', '40 scoring probes'],
        interpretation='one fixed-pair subset diagnostic, no rematch or acceptance/solve-rate test', holdout=False))


def validate(out):
    p = json.loads((out/'protocol.json').read_text())
    if p['id'] != RID or p['split'] != 'development' or p['holdout']:
        raise ValueError('development required')
    selected([RID])
    for path, sha in p['hashes'].items():
        if digest(Path(path)) != sha:
            raise ValueError('frozen input changed: '+path)
    return p


def run(out):
    validate(out)
    project = out/'pipeline'
    command = ['cargo', 'test', '--offline', '--locked', '--release', '--manifest-path', str(project/'Cargo.toml'),
               '-p', 'starglyph-core', 'solve::pair_refit::fixed_pairs', '--', '--ignored', '--nocapture']
    env = {**os.environ, 'CARGO_TARGET_DIR':str(project/'target'),
           'STARGLYPH_PAIR_INPUT':str(out/'input.json'), 'STARGLYPH_PAIR_OUTPUT':str(out/'fits.json')}
    started = time.monotonic()
    with (out/'run.log').open('w') as log:
        try:
            proc = subprocess.run(command, cwd=ROOT/'prototype', env=env, stdout=log,
                                  stderr=subprocess.STDOUT, timeout=240, check=False)
            status = dict(status='completed' if proc.returncode == 0 else 'failed', exit_code=proc.returncode)
        except subprocess.TimeoutExpired:
            status = dict(status='timeout', exit_code=None)
    status.update(command=command, elapsed_s=time.monotonic()-started)
    write_json(out/'run.json', status)
    if status['status'] != 'completed':
        raise RuntimeError('fit failed; see run.log')


def report(out):
    protocol = validate(out)
    sources = json.loads((ROOT/'docs/experiments/robustness-iteration-10-geometry.json').read_text())['sources']
    xy = np.array([s['xy'] for s in sources])
    radec = [[s['ra_deg'], s['dec_deg']] for s in sources]
    masks = dict(all_reviewed=np.ones(len(sources), dtype=bool),
        outside_both_inputs=np.array([s['outside_both_inputs'] for s in sources]),
        left_10pct=xy[:,0]<300, right_10pct=xy[:,0]>2700, top_10pct=xy[:,1]<400, bottom_10pct=xy[:,1]>3600)
    fits = json.loads((out/'fits.json').read_text())
    if fits['id'] != RID or fits['split'] != 'development' or [c['name'] for c in fits['cases']] != protocol['cases']:
        raise ValueError('fit selection changed')
    if not all(c['reproduced_previous_first_fit'] for c in fits['cases'][:2]):
        raise ValueError('control reproduction required')
    inputs = json.loads((out/'input.json').read_text())['cases']
    results = []
    for case, source in zip(fits['cases'], inputs):
        if case['matches'] != len(source['matches']):
            raise ValueError('fit size changed')
        camera = case['camera']
        errors = np.linalg.norm(lift(project(camera, radec), camera, 3000, 4000)-xy, axis=1)
        residuals = np.array(case['residuals'])[:-1].reshape(-1,2)*2.5
        results.append(dict(name=case['name'], matches=case['matches'], camera=camera,
            elapsed_ms=case['elapsed_ms'], prior_weight=case['prior_weight'], prior_scale=case['prior_scale'],
            reproduced_control=case['reproduced_previous_first_fit'],
            fit_rms_original_px=float(np.sqrt(np.mean(np.sum(residuals**2, axis=1)))),
            fit_pairs=[dict(hyg_ids=p['hyg_ids'], xy_original=((np.array(p['xy'])+.5)*2.5-.5).tolist(),
                            residual_original_xy=r.tolist()) for p, r in zip(source['matches'], residuals)],
            groups={k:stats(errors[m]) if m.any() else dict(count=0) for k,m in masks.items()},
            source_errors_px=errors.tolist(), sensitivity=sensitivity(case['jacobian'], case['probe_jacobian'])))
    write_json(out/'geometry.json', dict(iteration=11, id=RID, split='development',
        protocol_sha256=digest(out/'protocol.json'), fits_sha256=digest(out/'fits.json'), sources=sources, cases=results,
        limitations=['wide fixed fit only, not full rematch or final camera',
                    'source identities remain conditional; subset selection is manual, not an automatic filter',
                    'unchanged 40 probes including 14 outside previous inputs; no absolute ground truth']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'run', 'report'])
    parser.add_argument('out', type=Path)
    parser.add_argument('--registry', type=Path)
    args = parser.parse_args()
    if args.command == 'prepare':
        if args.registry is None:
            parser.error('--registry required')
        prepare(args.out.resolve(), args.registry.resolve())
    elif args.command == 'run':
        run(args.out.resolve())
    else:
        report(args.out.resolve())
