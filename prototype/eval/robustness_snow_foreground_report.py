#!/usr/bin/env python3
"""Preserve iteration55 observations after a failed strict reproduction gate.

Reporting was added after the run; the original protocol and failure criterion
remain unchanged. This does not retry measurements or certify the comparison.
"""
import argparse
from collections import Counter
from pathlib import Path
import re

from collection_run import ROOT, digest, write_json
from robustness_snow_foreground import (RID, POSITIVE, BASELOG, WCS, validate,
    read, previous, parse_trace, aggregate, failure_stage)


def control_changes(cases, old):
    actual = [(c['name'], a) for c in cases for a in c['attempts']]
    if len(actual) != len(old):
        raise ValueError('control attempt count changed')
    return [dict(case=name, db=a['db'], k=a['k'], fov=a['fov'], before=b,
                 after=a['result'].get('error', 'candidate'))
            for (name, a), b in zip(actual, old) if a['result'].get('error', 'candidate') != b]


def report(out):
    p = validate(out)
    if (out/'results.json').exists() or (out/'report-protocol.json').exists():
        raise ValueError('fresh report required')
    paths = [out/f'{n}.json' for n in ['protocol', 'input', 'raw', 'run', 'visual-review']]
    paths += [out/'run.log', Path(__file__).resolve(), Path(__file__).with_name('test_robustness_snow_foreground_report.py')]
    write_json(out/'report-protocol.json', dict(iteration=55, stage='post-run reporting repair',
        reason='original strict summarize refused publication after two control timeouts; preserve all observed outcomes and mark comparison invalid, never waive the criterion',
        hashes={str(f.relative_to(ROOT)): digest(f) for f in paths}))
    raw = read(out/'raw.json')['cases']; trace = parse_trace((out/'run.log').read_text())
    if [c['name'] for c in raw] != p['case_names'] or len(trace) != 432 or any(len(c['attempts']) != 48 for c in raw):
        raise ValueError('frozen run scope changed')
    cursor = 0; cases = []
    for c in raw:
        attempts = []; fovs = []
        for a in c['attempts']:
            t = trace[cursor]; cursor += 1
            if t['context'] != dict(case=c['name'], db=a['db'], k=a['k'], fov=a['fov']):
                raise ValueError('trace attribution mismatch')
            counts = aggregate(t['fovs']); fovs.extend(t['fovs'])
            if counts['finalized'] != int('error' not in a['result']):
                raise ValueError('finalization mismatch')
            attempts.append(a | dict(funnel=counts, stage=failure_stage(a['result'], counts)))
        cases.append(dict(name=c['name'], statuses=dict(Counter(a['result'].get('error', 'candidate') for a in attempts)),
            stages=dict(Counter(a['stage'] for a in attempts)), funnel=aggregate(fovs),
            elapsed_ms=sum(a['elapsed_ms'] for a in attempts), attempts=attempts))
    old = re.findall(r'err=(\w+)', BASELOG.read_text())
    if len(old) != 192:
        raise ValueError('expected192 original control failures')
    changes = control_changes(cases[:4], old)
    positive = next(c for c in read(previous('38-results'))['cases'] if c['name'] == 'positive-control')
    differences = [dict(db=a['db'], k=a['k'], fov=a['fov'], before=b['result'], after=a['result'])
                   for a, b in zip(cases[-1]['attempts'], positive['attempts']) if a['result'] != b['result']]
    positive_ok = not any('error' not in d['before'] for d in differences)
    write_json(out/'results.json', dict(iteration=55, ids=[RID, POSITIVE], split='development', holdout=False,
        protocol_sha256=digest(out/'protocol.json'), report_protocol_sha256=digest(out/'report-protocol.json'),
        comparison_valid=not changes and positive_ok, strict_control_reproduced=not changes,
        control_status_matches=192-len(changes), control_differences=changes,
        positive_candidates_reproduced=positive_ok, positive_differences=differences,
        target_candidates=sum(c['statuses'].get('candidate', 0) for c in cases[4:8]),
        cases=cases, selections=read(out/'input.json')['selections'], external_wcs_status=read(WCS)['status'],
        run=read(out/'run.json'), new_full_solver_calls=0, new_wcs_calls=0,
        production_changed=False, independent_ground_truth=False, automatic_improvement_confirmed=False))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--out-dir', type=Path, required=True)
    report(p.parse_args().out_dir.resolve())
