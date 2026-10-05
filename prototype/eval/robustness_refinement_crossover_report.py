#!/usr/bin/env python3
"""Check observer/replay equivalence and report the fixed crossover geometry."""
import argparse
import json
from pathlib import Path

import numpy as np

from collection_run import ROOT, digest, write_json
from compare_local_wcs import project, stats
from robustness_centroid_origin_report import geometry_groups
from robustness_refinement_trace import ARMS, FRAME, validate


def close(a, b, path='', tolerance=1e-8):
    """Exact structure/discrete identities; fixed absolute float tolerance."""
    if isinstance(a, dict):
        if not isinstance(b, dict) or a.keys() != b.keys():
            raise ValueError('keys differ: '+path)
        return max([0.]+[close(a[k],b[k],path+'/'+k,tolerance) for k in a])
    if isinstance(a, list):
        if not isinstance(b,list) or len(a) != len(b):
            raise ValueError('length differs: '+path)
        return max([0.]+[close(x,y,f'{path}/{i}',tolerance) for i,(x,y) in enumerate(zip(a,b))])
    if isinstance(a, float):
        error = abs(a-b)
        if not np.isfinite(error) or error > tolerance:
            raise ValueError(f'numeric mismatch {path}: {a} != {b}')
        return error
    if a != b:
        raise ValueError('identity differs: '+path)
    return 0.


def summarize(out):
    p = validate(out)
    replay_protocol = json.loads((out/'replay-protocol.json').read_text())
    for name, sha in replay_protocol['hashes'].items():
        if digest(ROOT/name) != sha:
            raise ValueError('replay input changed: '+name)
    load = lambda name: json.loads((out/name).read_text())
    inputs = load('replay-plan.json')['inputs']
    observed = load('observed-paths.json')
    candidates = load('chosen-candidates.json')
    cases = load('replay-results.json')['cases']
    if [dict(pose_arm=c['pose_arm'],match_arm=c['match_arm']) for c in cases] != replay_protocol['cases']:
        raise ValueError('crossover cases changed')
    prior = json.loads((ROOT/'docs/experiments/robustness-iteration-24-case-132162731.json').read_text())
    checks = {}
    for arm in ARMS:
        report = load(f'{arm}/solve-reports/{FRAME}.json')
        checks[arm] = dict(iteration24_camera_max_delta=close(report['camera'],prior['arms'][arm]['camera']),
            iteration24_quality_max_delta=close(report['report']['quality'],prior['arms'][arm]['quality']),
            inlier_identity_max_delta=close([i for i,d in enumerate(report['report']['detections']) if d['inlier']],prior['arms'][arm]['final_inlier_indices']))
        diagonal = next(c for c in cases if c['pose_arm']==c['match_arm']==arm)
        checks[arm]['diagonal_steps_max_delta'] = close(diagonal['steps'],observed[arm])
        checks[arm]['report_vs_trace_camera_max_delta'] = close(report['camera'],{k:diagonal['camera'][k] for k in report['camera']})
    close(inputs['control']['catalog'],inputs['huber']['catalog'],tolerance=0)
    close(inputs['control']['detections'],inputs['huber']['detections'],tolerance=0)
    # Search durations may vary; all other serialized Solution fields are compared.
    solution_keys = candidates['control']['solution'].keys()
    timing_keys = [k for k in solution_keys if 'time' in k]
    solutions = {a:{k:v for k,v in candidates[a]['solution'].items() if k not in timing_keys} for a in ARMS}
    solution_equal = solutions['control'] == solutions['huber']
    pair_equal = [{k:r[k] for k in ['catalog_id','detection_index','world','xy']} for r in candidates['control']['pairs']] == [{k:r[k] for k in ['catalog_id','detection_index','world','xy']} for r in candidates['huber']['pairs']]
    frozen = json.loads((ROOT/'docs/experiments/robustness-iteration-13-geometry.json').read_text())
    sources = next(f['sources'] for f in frozen['frames'] if f['id']==FRAME)
    alternate = json.loads((ROOT/'docs/experiments/robustness-iteration-5-geometry.json').read_text())
    alternate = {s['source_id']:s for f in alternate['frames'] if f['id']==FRAME for s in f['sources']}
    xy = np.array([s['xy'] for s in sources])
    radec = np.array([s['radec'] for s in sources])
    dets = np.array([[d['x'],d['y']] for d in inputs['control']['detections']])
    cam = inputs['control']['camera']
    masks = geometry_groups(xy,[dets],cam['width'],cam['height'],12)
    positions = dict(external_centroid=xy, **{f'native_r{r}':np.array([alternate[s['source_id']][f'native_r{r}_xy'] for s in sources]) for r in [8,12]})

    def geometry(camera):
        predicted = project(camera,radec)
        return {centroid:{name:stats(np.linalg.norm(predicted[mask]-pos[mask],axis=1)) for name,mask in masks.items() if mask.any()} for centroid,pos in positions.items()}

    for case in cases:
        case['conditional_geometry'] = geometry(case['camera'])
        for step in case['steps']:
            step['conditional_geometry'] = geometry(step['camera'])
    write_json(out/'results.json',dict(iteration=25,id=FRAME,split='development',holdout=False,
        protocol_sha256=digest(out/'protocol.json'),replay_protocol_sha256=digest(out/'replay-protocol.json'),
        automatic_improvement_confirmed=False,diagnostic_only=True,checks=checks,
        same_tetra_solution_except_timing=solution_equal,excluded_solution_timing_keys=timing_keys,
        same_ordered_tetra_pairs=pair_equal,pair_count=len(candidates['control']['pairs']),
        detections_equal=True,detection_count=len(dets),catalog_equal=True,catalog_count=len(inputs['control']['catalog']),
        input={a:dict(camera=inputs[a]['camera'],verification=inputs[a]['verification'],
            conditional_geometry=geometry(inputs[a]['camera'])) for a in ARMS},
        runs={a:load(f'run-{a}.json') for a in ARMS},cases=cases,
        source_ids=[s['source_id'] for s in sources],
        limitations=['conditional reviewed star identities, no accepted independent ground truth',
            'one previously inspected development regression; no generalization claim',
            'crossovers bypass acceptance diagnostically, never count as new solves',
            'all hypotheses/parameters inherited unchanged from iteration24']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir',type=Path,required=True)
    summarize(parser.parse_args().out_dir.resolve())
