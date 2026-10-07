#!/usr/bin/env python3
"""Assess frozen pattern-order pilot; retain failed outcomes and all field groups."""
import argparse
import copy
import json
from pathlib import Path
import re

import numpy as np

from collection_run import ROOT, digest, write_json
from compare_local_wcs import project, stats
from robustness_candidate_trace import traces
from robustness_matcher_funnel import COUNTERS, validate_counts
from robustness_orion_fit import METHODS, RID, previous, read
from robustness_orion_pattern_order import ARMS, validate
from robustness_refinement_crossover_report import close
from robustness_refinement_trace import events


def attempted(text):
    attempts = []; orders = []; funnels = []
    for line in text.splitlines():
        if line.startswith('SGORDER '):
            row = json.loads(line[8:])
            if len(set(row['retained'])) != len(row['retained']) or sorted(row['retained']) != sorted(row['priority']):
                raise ValueError('thinning membership changed')
            if row['retained'] and row['retained'][0] != row['priority'][0]:
                raise ValueError('brightest survivor changed')
            orders.append(row)
        elif line.startswith('SGTRACE '):
            row = json.loads(line[8:])
            if row['stage'] == 'funnel':
                validate_counts(row['data']); funnels.append(row['data'])
        elif re.match(r'^  \[.*\] k=\d+', line):
            if len(orders) != len(funnels):
                raise ValueError('unpaired order and search counters')
            for order, funnel in zip(orders, funnels):
                if order['fov_deg'] != funnel['fov_deg'] or len(order['retained']) != funnel['retained']:
                    raise ValueError('FOV/counter attribution mismatch')
            attempts.append(dict(outcome=line.strip(), sweeps=[dict(order=o, funnel=f) for o, f in zip(orders, funnels)]))
            orders = []; funnels = []
    if orders or funnels or not attempts:
        raise ValueError('unattributed search observations')
    return attempts


def candidate_identity(c):
    result = copy.deepcopy(c)
    result['solution'].pop('solve_time_ms', None)
    return result


def compare_control(actual, expected):
    # close() returns maximum numeric error (zero on exact replay), not bool.
    for key in ('camera', 'source_sha256'):
        close(actual[key], expected[key], tolerance=0)
    for key in ('quality', 'detections', 'status'):
        close(actual['report'][key], expected['report'][key], tolerance=0)


def assess(before, after):
    if before is None or after is None:
        return dict(passed=False, reason='missing solved geometry')
    result = {}
    for method in METHODS:
        b, a = before[method]['groups'], after[method]['groups']
        if b.keys() != a.keys() or any(b[k]['count'] != a[k]['count'] for k in b):
            raise ValueError('assessment membership changed')
        key = 'outside_both_starglyph_inputs'
        if b[key]['count'] != 18:
            raise ValueError('frozen18 outside detector inputs required')
        ratio = a[key]['rms_px']/b[key]['rms_px']
        regressions = {k: a[k]['rms_px']-v['rms_px'] for k, v in b.items()
                       if v['count'] and a[k]['rms_px']-v['rms_px'] > .5}
        result[method] = dict(outside18_rms_ratio=ratio, regressions=regressions, passed=ratio <= .9 and not regressions)
    return dict(passed=all(v['passed'] for v in result.values()), methods=result)


def freeze(out):
    validate(out)
    if (out/'assessment-protocol.json').exists() or any((out/(a+'.json')).exists() for a in ARMS):
        raise ValueError('freeze assessment before pilot')
    paths = [Path(__file__).resolve(), Path(__file__).with_name('robustness_refinement_crossover_report.py'),
             Path(__file__).with_name('robustness_refinement_trace.py'), Path(__file__).with_name('compare_local_wcs.py'),
             Path(__file__).with_name('robustness_orion_fit.py')]
    write_json(out/'assessment-protocol.json', dict(iteration=48, id=RID, split='development', holdout=False,
        protocol_sha256=digest(out/'protocol.json'),
        hashes={str(p.relative_to(ROOT)): digest(p) for p in paths}))


def run(out):
    validate(out); p = read(out/'assessment-protocol.json')
    if (p['id'], p['split'], p['holdout']) != (RID, 'development', False) or p['protocol_sha256'] != digest(out/'protocol.json'):
        raise ValueError('assessment scope changed')
    for name, sha in p['hashes'].items():
        if digest(ROOT/name) != sha:
            raise ValueError('assessment code changed')
    if (out/'results.json').exists():
        raise ValueError('fresh assessment required')
    sources = read(previous('40-results'))['sources']; old = read(previous('41-native-report'))
    original_candidates = read(previous('41-trace'))['candidates']
    groups = sorted({g for s in sources for g in s['groups']}); rows = {}
    for arm in ARMS:
        native = read(out/arm/'solve-reports'/f'{RID}.json'); log = out/(arm+'.log')
        candidates = traces(log); attempts = attempted(log.read_text())
        shared_input = all(c['detections'] == original_candidates[0]['detections'] for c in candidates)
        if arm == 'control':
            compare_control(native, old)
            if [candidate_identity(c) for c in candidates] != [candidate_identity(c) for c in original_candidates]:
                raise ValueError('control candidate identities or geometry did not reproduce41')
            if any(s['order']['retained'] != s['order']['priority'] for a in attempts for s in a['sweeps']):
                raise ValueError('control priority changed')
        geometry = None; pred = None
        if native['report']['status'] == 'solved':
            pred = project(native['camera'], [s['radec'] for s in sources])
            geometry = {}
            for method in METHODS:
                residual = pred-[s['coordinates'][method] for s in sources]; errors = np.linalg.norm(residual, axis=1)
                geometry[method] = dict(residual_xy_px=residual.tolist(), groups={g: stats(errors[[g in s['groups'] for s in sources]]) for g in groups})
        refinement_inputs = events(log, 'REFINEMENT_INPUT ')
        for inp in refinement_inputs:
            inp.pop('catalog', None)
        trace = dict(candidates=candidates, refinement_inputs=refinement_inputs,
                     refinement_paths=events(log, 'REFINEMENT_PATH '), attempts=attempts)
        write_json(out/(arm+'-trace.json'), trace)
        write_json(out/(arm+'-native-report.json'), native)
        dets = native['report']['detections']
        rows[arm] = dict(status=native['report']['status'], quality=native['report']['quality'],
            detections=len(dets), inlier_indices=[i for i, d in enumerate(dets) if d['inlier']],
            camera=native.get('camera'), candidates=len(candidates), candidate_input_matches41=shared_input,
            attempts=len(attempts), timeouts=sum('err=Timeout' in a['outcome'] for a in attempts),
            counters={k: sum(s['funnel'][k] for a in attempts for s in a['sweeps']) for k in COUNTERS},
            changed_priority_sweeps=sum(s['order']['retained'] != s['order']['priority'] for a in attempts for s in a['sweeps']),
            predicted_xy_original=None if pred is None else pred.tolist(), geometry=geometry,
            process=read(out/(arm+'.json')))
    comparison = assess(rows['control']['geometry'], rows['farthest']['geometry'])
    write_json(out/'results.json', dict(iteration=48, id=RID, split='development', holdout=False,
        protocol_sha256=digest(out/'protocol.json'), assessment_protocol_sha256=digest(out/'assessment-protocol.json'),
        arms=rows, comparison=comparison, control_exact_replay=True,
        sources=[dict(id=s['id'], hyg_id=s['hyg_id'], name=s['name'], groups=s['groups']) for s in sources],
        pilot_geometry_passed=comparison['passed'] and rows['farthest']['candidate_input_matches41'],
        new_solver_calls=2, new_wcs_calls=0, production_changed=False,
        independent_ground_truth=False, automatic_improvement_confirmed=False))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=('freeze', 'run')); p.add_argument('--out-dir', type=Path, required=True)
    a = p.parse_args(); globals()[a.stage](a.out_dir.resolve())
