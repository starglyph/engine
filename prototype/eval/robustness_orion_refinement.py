#!/usr/bin/env python3
"""Trace one frozen development control through the existing observer."""
import argparse
import csv
import gzip
import os
from pathlib import Path

import numpy as np

from collection_run import ROOT, digest, write_json
from compare_local_wcs import project, stats
from robustness_candidate_trace import command, traces
from robustness_refinement_trace import events
from robustness_refinement_crossover_report import close
from robustness_refinement_report import lift
from robustness_reference_audit import RID, MANIFEST, CATALOG, read, selected, reports
from robustness_reference_coverage import extent

OLD = ROOT/'prototype/artifacts/robustness-iteration-25/fixed'
BINARY = OLD/'binaries/control'
AUDIT = ROOT/'docs/experiments/robustness-iteration-40-results.json'


def prepare(out):
    rec = selected()
    if out.exists():
        raise ValueError('fresh output required')
    source = MANIFEST.parent/rec['file']
    if digest(source) != rec['clean_sha256']:
        raise ValueError('photo changed')
    old_path = ROOT/'docs/experiments/robustness-iteration-25-protocol.json'
    old = read(old_path)
    for p in [BINARY, OLD/'solve-control.rs']:
        if digest(p) != old['hashes'][str(p.relative_to(ROOT))]:
            raise ValueError('prior observer changed')
    checks = read(ROOT/'docs/experiments/robustness-iteration-40-checks.json')
    if digest(AUDIT) != checks['artifact_hashes'][str(AUDIT.relative_to(ROOT))]:
        raise ValueError('reviewed sources changed')
    out.mkdir(parents=True)
    write_json(out/'input.json', dict(id=RID, split='development', holdout=False,
        image=str(source), source_sha256=digest(source), baseline_report=str(reports()['control']),
        source_ids=[s['id'] for s in read(AUDIT)['sources']]))
    paths = [Path(__file__).resolve(), MANIFEST, CATALOG, source, old_path, BINARY,
        OLD/'solve-control.rs', OLD/'pipeline/crates/starglyph-core/src/solve/refinement_trace.rs',
        AUDIT, reports()['control'], out/'input.json',
        ROOT/'docs/experiments/robustness-iteration-40-checks.json',
        *(MANIFEST.parent/n for n in ['robustness-results.json', 'collection-wcs-review.json']),
        *(Path(__file__).with_name(n) for n in ['robustness_candidate_trace.py', 'robustness_refinement_trace.py',
          'robustness_refinement_crossover_report.py', 'robustness_refinement_report.py',
          'robustness_reference_audit.py', 'robustness_reference_coverage.py', 'compare_local_wcs.py', 'collection_run.py']),
        *(ROOT/'prototype/artifacts/cache').glob('*.bin')]
    write_json(out/'protocol.json', dict(iteration=41, id=RID, split='development', holdout=False,
        experiment='observation only: one original control solve, chosen candidate and existing refinement stages',
        budget=dict(processes=1, wall_s=120, per_match_attempt_ms=2500),
        criteria=['final camera, quality and ordered detector DTO exactly reproduce iteration40 control',
            'retain every candidate trace and timeout; no alternative choice or parameter tuning',
            'all 47 frozen reviewed sources at every stage, same three centroid methods and groups',
            'fit identities bound to HYG units; proximity alone never certifies catalog identity',
            'scale intermediate projections with existing pixel-center lift; native report measured directly',
            'no WCS run, no holdout, no new algorithm or accepted ground truth'],
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
    p = validate(out)
    if (out/'run.json').exists():
        raise ValueError('fresh run required')
    command(out, 'run', [str(BINARY), 'eval', '--manifest', str(MANIFEST), '--ids', RID,
        '--catalog', str(CATALOG), '--out-dir', str(out/'control')],
        {**os.environ, 'STARGLYPH_SOLVE_DEBUG': '1'}, p['budget']['wall_s'])
    validate(out)


def summarize(out):
    validate(out)
    actual = read(out/'control/solve-reports'/f'{RID}.json')
    previous = read(reports()['control'])
    checks = {k: close(actual[k], previous[k], tolerance=0) for k in ['camera', 'source_sha256']}
    for k in ['quality', 'detections', 'status']:
        checks[k] = close(actual['report'][k], previous['report'][k], tolerance=0)
    inp, paths = events(out/'run.log', 'REFINEMENT_INPUT '), events(out/'run.log', 'REFINEMENT_PATH ')
    if len(inp) != 1 or len(paths) != 1:
        raise ValueError('expected one chosen refinement path')
    inp, steps = inp[0], paths[0]
    candidates = traces(out/'run.log')
    chosen = [c for c in candidates if all(c['camera'][k] == inp['camera'][k] for k in c['camera'])]
    if len(chosen) != 1:
        raise ValueError('chosen candidate not unique')
    write_json(out/'trace.json', dict(input=inp, steps=steps, candidates=candidates, chosen=chosen[0]))
    sources = read(AUDIT)['sources']
    width, height = actual['width'], actual['height']
    dets = lift([[d['x'], d['y']] for d in inp['detections']], inp['camera'], width, height)
    dto = np.array([[d['x'], d['y']] for d in actual['report']['detections']])
    if not np.allclose(dets, dto, rtol=0, atol=.00501):
        raise ValueError('working detections do not lift to final DTO')
    world = np.array([s['radec'] for s in sources])
    xy = np.array([s['coordinates']['external_centroid'] for s in sources])
    nearest = np.linalg.norm(xy[:, None, :]-dets[None, :, :], axis=2)
    with gzip.open(CATALOG, 'rt') as f:
        catalog = [s for s in csv.DictReader(f) if s['mag'] and float(s['mag']) <= 6.8]
    angles = np.array([[float(s['rarad']), float(s['decrad'])] if s.get('rarad') and s.get('decrad')
        else np.deg2rad([float(s['ra'])*15, float(s['dec'])]) for s in catalog])
    ra, dec = angles.T
    units = np.column_stack((np.cos(dec)*np.cos(ra), np.cos(dec)*np.sin(ra), np.sin(dec)))

    def pairs(rows, camera):
        result = []
        for m in rows:
            ids = np.flatnonzero(np.linalg.norm(units-m['world'], axis=1) < 1e-12)
            if len(ids) != 1:
                raise ValueError('catalog unit identity ambiguous')
            star = catalog[int(ids[0])]
            pos = lift(m['xy'], camera, width, height)
            d = np.linalg.norm(xy-pos, axis=1)
            n = int(np.argmin(d))
            result.append(dict(hyg_id=int(star['id']), name=star['proper'] or star['bf'], xy=pos.tolist(),
                nearest_reviewed_id=sources[n]['id'], reviewed_distance_px=float(d[n]),
                reviewed_identity_agrees=int(star['id']) == sources[n]['hyg_id'] if d[n] < 12 else None))
        return result

    stages = [('candidate', inp['camera'], inp['verification']['matches'], None)]
    for s in steps:
        v = s.get('input', s.get('verification'))
        stages.append((s['stage'], s['camera'], s.get('fit_matches') if v is None else v['matches'], s.get('radius_px')))
    stages.append(('native_final', actual['camera'], None, None))
    groups = sorted({g for s in sources for g in s['groups']})
    measured = []
    for name, cam, matches, radius in stages:
        pred = lift(project(cam, world), cam, width, height) if name != 'native_final' else project(cam, world)
        bound = None if matches is None else pairs(matches, cam)
        geometry = {}
        for method in ['external_centroid', 'native_r8', 'native_r12']:
            errors = np.linalg.norm(pred-np.array([s['coordinates'][method] for s in sources]), axis=1)
            geometry[method] = dict(errors_px=errors.tolist(), groups={g: stats(errors[[g in s['groups'] for s in sources]]) for g in groups})
        measured.append(dict(stage=name, radius_working_px=radius, camera=cam,
            pair_role='final membership' if name in ['final', 'native_final'] else 'input pairs for this fit' if name != 'candidate' else 'initial verification',
            pairs=bound, pair_extent=None if bound is None else extent([dict(x=m['xy'][0], y=m['xy'][1]) for m in bound], width, height),
            geometry=geometry, predicted_xy=pred.tolist(),
            projection_to_nearest_physical_detection_px=[float(np.linalg.norm(pred[i]-dets[int(nearest[i].argmin())])) for i in range(len(sources))]))
    write_json(out/'results.json', dict(iteration=41, id=RID, split='development', holdout=False,
        protocol_sha256=digest(out/'protocol.json'), checks=checks,
        run=read(out/'run.json'), candidate_count=len(candidates), chosen_label=inp['label'],
        original_size=inp['original_size'], working_size=[inp['camera']['width'], inp['camera']['height']],
        stages=measured, sources=[dict(id=s['id'], hyg_id=s['hyg_id'], name=s['name'], groups=s['groups'],
            nearest_detection_index=int(nearest[i].argmin()), nearest_detection_distance_px=float(nearest[i].min())) for i, s in enumerate(sources)],
        independent_ground_truth=False, production_changed=False, automatic_improvement_confirmed=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'validate', 'run', 'summarize'])
    parser.add_argument('--out-dir', type=Path, required=True)
    a = parser.parse_args()
    globals()[a.stage](a.out_dir.resolve())
