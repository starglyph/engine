#!/usr/bin/env python3
"""Separate database absence, lookup-window exclusion and observed rejection."""
import argparse
from collections import Counter
import copy
import json
from pathlib import Path
import re

from collection_run import ROOT, digest, write_json
from robustness_orion_fit import RID, previous, read
from robustness_orion_pattern_trace import validate


def trace_events(text, targets):
    by_indices = {tuple(sorted(t['detection_indices'])): t['name'] for t in targets}
    by_ids = {tuple(sorted(t['catalog_ids'])): t['name'] for t in targets}
    attempts = []; window = None; lookup = None
    for line in text.splitlines():
        if not line.startswith('SGTRACE '):
            continue
        row = json.loads(line[8:]); stage, data = row['stage'], row['data']
        if stage == 'attempt':
            attempts.append(dict(context=data, thinning=[], windows=[])); window = lookup = None
        elif stage in ('thinning', 'image_pattern', 'target_window', 'lookup', 'ratios', 'pairing', 'rotation', 'verification_pool', 'verification'):
            if not attempts:
                raise ValueError('unattributed trace event')
            attempt = attempts[-1]
            if stage == 'thinning':
                attempt['thinning'].append(data); window = lookup = None
            elif stage == 'image_pattern':
                key = tuple(sorted(data['input_indices']))
                if key not in by_indices:
                    raise ValueError('unselected image pattern')
                window = dict(target=by_indices[key], image=data, lookups=[])
                attempt['windows'].append(window); lookup = None
            elif stage == 'target_window':
                if window is None or data['input_indices'] != window['image']['input_indices'] or data['fov'] != window['image']['fov']:
                    raise ValueError('lookup window attribution mismatch')
                window['window'] = data
            elif stage == 'lookup':
                if window is None or data['fov'] != window['image']['fov']:
                    raise ValueError('lookup without image pattern')
                lookup = dict(catalog_target=by_ids[tuple(sorted(data['ids']))], lookup=data)
                window['lookups'].append(lookup)
            else:
                if lookup is None:
                    raise ValueError('stage without catalogue lookup')
                lookup[stage] = data
    if not attempts or any('window' not in w for a in attempts for w in a['windows']):
        raise ValueError('missing target windows')
    return attempts


def classify(window, inventory):
    """Window calculations describe necessary conditions, not forced solver calls."""
    metrics = inventory['metrics']; bounds = window['window']
    lookups = [l for l in window['lookups'] if l['catalog_target'] == window['target']]
    indexed = bool(inventory['entries'])
    if metrics is None:
        return dict(indexed=indexed, blocker='catalog_star_absent', lookups=len(lookups))
    key_inside = all(lo <= k <= hi for k,lo,hi in zip(metrics['key'],bounds['key_min'],bounds['key_max']))
    ratio_inside = all(lo < r < hi for r,lo,hi in zip(metrics['ratios'],bounds['ratio_min'],bounds['ratio_max']))
    delta = [a-b for a,b in zip(window['image']['ratios'],metrics['ratios'])]
    if not indexed:
        blocker = 'not_indexed'
    elif not lookups:
        blocker = 'lookup_window' if not key_inside else 'lookup_not_observed'
    elif any(l.get('verification',{}).get('pass') for l in lookups):
        blocker = 'verification_passed'
    elif any('verification' in l for l in lookups):
        blocker = 'verification_rejected'
    elif any(l.get('ratios',{}).get('pass') for l in lookups):
        blocker = 'after_ratios_unresolved'
    elif any('ratios' in l for l in lookups):
        blocker = 'edge_ratios'
    else:
        blocker = 'fov'
    return dict(indexed=indexed, key_inside=key_inside, ratios_inside=ratio_inside,
        ratio_delta=delta, max_abs_ratio_delta=max(map(abs,delta)), tolerance=bounds['tolerance'],
        blocker=blocker, lookups=len(lookups))


def freeze(out):
    validate(out)
    if (out/'run.json').exists() or (out/'assessment-protocol.json').exists():
        raise ValueError('freeze assessment before observation')
    source=Path(__file__).resolve()
    write_json(out/'assessment-protocol.json',dict(id=RID,split='development',holdout=False,
        protocol_sha256=digest(out/'protocol.json'),hashes={str(source.relative_to(ROOT)):digest(source)}))


def summarize(out):
    validate(out)
    p=read(out/'assessment-protocol.json')
    if (p['id'],p['split'],p['holdout']) != (RID,'development',False) or p['protocol_sha256'] != digest(out/'protocol.json'):
        raise ValueError('assessment scope changed')
    for name,sha in p['hashes'].items():
        if digest(ROOT/name) != sha:raise ValueError('assessment code changed')
    if (out/'results.json').exists():
        raise ValueError('fresh summary required')
    inp=read(out/'input.json');raw=read(out/'raw.json')
    if len(raw['cases']) != 1 or raw['cases'][0]['name'] != 'orion-control':
        raise ValueError('one Orion case required')
    parsed=trace_events((out/'run.log').read_text(),inp['targets'])
    actual=raw['cases'][0]['attempts'];old=read(previous('48-control-trace'))['attempts']
    if not len(actual)==len(old)==len(parsed)==48:
        raise ValueError('48 attempts required')
    candidates=[]
    for a,b,t in zip(actual,old,parsed):
        expected_error=re.search(r'err=(\w+)',b['outcome'])
        if a['result'].get('error') != (expected_error[1] if expected_error else None):
            raise ValueError('attempt outcome not reproduced')
        if t['context'] != dict(case='orion-control',db=a['db'],k=a['k'],fov=a['fov']):
            raise ValueError('attempt attribution mismatch')
        if [dict(fov=s['order']['fov_deg'],kept_input_indices=s['order']['retained']) for s in b['sweeps']] != t['thinning']:
            raise ValueError('thinning or FOV did not reproduce48')
        if 'solution' in a['result']:
            c=copy.deepcopy(a['result']['solution']);c.pop('solve_time_ms',None);candidates.append(c)
    expected=[]
    for c in read(previous('41-trace'))['candidates']:
        s=copy.deepcopy(c['solution']);s.pop('solve_time_ms',None);expected.append(s)
    if candidates != expected:
        raise ValueError('full candidate replay failed')
    rows=[]
    for t in parsed:
        db=t['context']['db']
        for w in t['windows']:
            index=next(i for i,target in enumerate(inp['targets']) if target['name']==w['target'])
            inv=raw['inventory'][db]['targets'][index]
            if sorted(inv['catalog_ids']) != sorted(inp['targets'][index]['catalog_ids']):
                raise ValueError('inventory identity mismatch')
            if inv['metrics'] is not None and any(e['key_hash16'] != (inv['metrics']['key_hash'] & 65535) for e in inv['entries']):
                raise ValueError('inventory key hash mismatch')
            rows.append(dict(attempt=t['context'],**w,assessment=classify(w,inv)))
    summaries={}
    for target in inp['targets']:
        subset=[r for r in rows if r['target']==target['name']]
        summaries[target['name']]=dict(windows=len(subset),blockers=dict(Counter(r['assessment']['blocker'] for r in subset)),
            per_database={str(db):dict(windows=sum(r['attempt']['db']==db for r in subset),
                blockers=dict(Counter(r['assessment']['blocker'] for r in subset if r['attempt']['db']==db))) for db in range(4)})
    write_json(out/'results.json',dict(iteration=49,id=RID,split='development',holdout=False,
        protocol_sha256=digest(out/'protocol.json'),targets=inp['targets'],inventory=raw['inventory'],
        summaries=summaries,windows=rows,attempts=actual,thinning=[dict(context=t['context'],sweeps=t['thinning']) for t in parsed],
        replay_48_outcomes=True,replay_full_solutions=True,replay_thinning=True,
        run=read(out/'run.json'),new_tetra3_attempts=48,new_pipeline_calls=0,new_wcs_calls=0,
        production_changed=False,independent_ground_truth=False,automatic_improvement_confirmed=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=('freeze','summarize'));p.add_argument('--out-dir',type=Path,required=True)
    a=p.parse_args();globals()[a.stage](a.out_dir.resolve())
