#!/usr/bin/env python3
"""Observe frozen development prefixes through the existing tetra3 trace harness."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re
import shutil
import statistics

from collection_run import ROOT, digest, select_records, write_json
from robustness_candidate_trace import command
from tetra3_internal import prepare as prepare_harness

RID = 'wm_r_134291495'
POSITIVE = 'wm_r_112929230'
MANIFEST = ROOT / 'data/samples/sky-samples/manifest.json'
DIAG = ROOT / f'prototype/artifacts/robustness-iteration-1/diagnosis/{RID}/solve-reports/{RID}.json'
BASELOG = ROOT / f'prototype/artifacts/robustness-iteration-24/real/development/control/{RID}.log'
POS = ROOT / 'prototype/artifacts/robustness-iteration-3/positive-input.json'
WCS = ROOT / f'prototype/artifacts/robustness-baseline/wcs-run/{RID}/reference.json'


def read(path):
    return json.loads(path.read_text())


def selection():
    records = select_records(read(MANIFEST), 'development', [RID, POSITIVE])
    if {r['id'] for r in records} != {RID, POSITIVE}:
        raise ValueError('both development records required')
    return records


def prepare(out, registry):
    records = selection()
    if out.exists():
        raise ValueError('fresh output directory required')
    cases = []
    for d in read(DIAG)['detection_diagnostics']:
        cases.append(dict(name=f"{d['width']}-{d['tier']}", id=RID,
                          width=d['width'], height=d['height'], total_detections=len(d['result']['detections']),
                          search=d['result']['detections'][:d['max_detections']]))
    positive = read(POS)
    cases.append(positive['cases'][0] | dict(id=POSITIVE))
    if len(cases) != 5 or len({c['name'] for c in cases}) != 5:
        raise ValueError('four tiers and one positive required')
    prepare_harness(out, registry)
    # Only widen the standalone caller's case selection, not the solver.
    caller = out / 'harness/src/main.rs'
    code = caller.read_text()
    anchor = '        if !matches!(name, "4080-default-control" | "4080-deep-control" | "positive-control") { continue; }'
    if code.count(anchor) != 1:
        raise ValueError('caller selection changed')
    caller.write_text(code.replace(anchor, '        // All cases in this separately frozen development input.'))
    write_json(out / 'input.json', dict(split='development', ids=[RID, POSITIVE], databases=positive['databases'], cases=cases))
    paths = [DIAG, BASELOG, POS, WCS, MANIFEST, out / 'input.json',
             *(ROOT / 'data/samples/sky-samples' / n for n in ['robustness-results.json', 'collection-wcs-review.json']),
             *(Path(__file__).with_name(n) for n in ['robustness_thinning_trace.py', 'tetra3_internal.py',
               'tetra3_internal.rs', 'tetra3_internal_trace.rs', 'collection_run.py', 'robustness_candidate_trace.py']),
             *(Path(p) for p in positive['databases']), caller, out / 'harness/tetra3/src/solver/solve.rs',
             out / 'harness/tetra3/src/research_trace.rs', out / 'harness/Cargo.toml',
             ROOT / 'prototype/Cargo.lock', ROOT / 'prototype/crates/starglyph-core/src/solve.rs']
    for r in records:
        image = MANIFEST.parent / r['file']
        if digest(image) != r['clean_sha256']:
            raise ValueError('image changed')
        paths.append(image)
    write_json(out / 'protocol.json', dict(iteration=36, ids=[RID, POSITIVE], split='development', holdout=False,
        scope='observation only: existing tetra3 thinning hook; no detector or acceptance changes',
        case_names=[c['name'] for c in cases], budget=dict(attempts=240, per_attempt_ms=2500, wall_s=180),
        criteria=['192 target outcomes reproduce saved log in order', 'positive control returns at least one tetra3 candidate',
                  'all attempts have attributed thinning events; count every swept FOV, including TooFew',
                  'no solve-rate or geometry improvement claim; unresolved WCS remains unresolved'],
        hashes={str(p.relative_to(ROOT)): digest(p) for p in sorted(set(paths))}))


def validate(out):
    p = read(out / 'protocol.json')
    if (p['split'], p['ids'], p['holdout']) != ('development', [RID, POSITIVE], False):
        raise ValueError('frozen development required')
    selection()
    for path, sha in p['hashes'].items():
        if digest(ROOT / path) != sha:
            raise ValueError('frozen input changed: ' + path)
    return p


def build(out):
    validate(out)
    command(out, 'build', ['cargo', 'build', '--release', '--offline', '--manifest-path', str(out / 'harness/Cargo.toml')], timeout=600)
    shutil.copy2(out / 'harness/target/release/starglyph-tetra3-internal-trace', out / 'trace')
    write_json(out / 'binary.json', dict(sha256=digest(out / 'trace'), protocol_sha256=digest(out / 'protocol.json')))


def run(out):
    validate(out)
    b = read(out / 'binary.json')
    if b != dict(sha256=digest(out / 'trace'), protocol_sha256=digest(out / 'protocol.json')):
        raise ValueError('binary freeze changed')
    if (out / 'run.json').exists():
        raise ValueError('fresh run required')
    command(out, 'run', [str(out / 'trace'), str(out / 'input.json'), str(out / 'raw.json')], timeout=180)


def read_thinning(text):
    attempts = []
    for line in text.splitlines():
        if not line.startswith('SGTRACE '):
            continue
        row = json.loads(line[8:])
        if row['stage'] == 'attempt':
            attempts.append(dict(context=row['data'], sweeps=[]))
        elif row['stage'] == 'thinning':
            if not attempts:
                raise ValueError('unattributed thinning')
            data = row['data']; k = attempts[-1]['context']['k']
            indices = data['kept_input_indices']
            if len(indices) != len(set(indices)) or any(i < 0 or i >= k for i in indices):
                raise ValueError('invalid retained index')
            attempts[-1]['sweeps'].append(data)
    if not attempts or any(not a['sweeps'] for a in attempts):
        raise ValueError('missing thinning events')
    return attempts


def summarize(out):
    protocol = validate(out)
    inp = read(out / 'input.json'); raw = read(out / 'raw.json')['cases']
    trace = read_thinning((out / 'run.log').read_text())
    if [c['name'] for c in raw] != protocol['case_names'] or len(trace) != 240:
        raise ValueError('case/attempt scope changed')
    rows = []; cursor = 0
    for case, result in zip(inp['cases'], raw):
        attempts = []
        for a in result['attempts']:
            t = trace[cursor]; cursor += 1
            if t['context'] != dict(case=case['name'], db=a['db'], k=a['k'], fov=a['fov']):
                raise ValueError('trace/result attribution mismatch')
            groups = {}
            for s in t['sweeps']:
                groups.setdefault(tuple(s['kept_input_indices']), []).append(s['fov'])
            attempts.append(a | dict(sweep_count=len(t['sweeps']),
                retained=[dict(indices=list(k), fovs_deg=v) for k, v in groups.items()],
                pattern_capable_sweeps=sum(len(s['kept_input_indices']) >= 4 for s in t['sweeps'])))
        points = case['search']; first = points[:30]; w = case['width']; h = case['height']
        rows.append(dict(name=case['name'], id=case['id'], width=w, height=h,
            detections=len(points), total_detections=case.get('total_detections'),
            first30=dict(outer_horizontal_fifths=sum(d['x'] < .2*w or d['x'] >= .8*w for d in first),
                occupied_4x4_cells=len({(min(3,int(d['x']/w*4)),min(3,int(d['y']/h*4))) for d in first}),
                elongation_median=statistics.median(d['elongation'] for d in first) if 'elongation' in first[0] else None),
            statuses=dict(Counter(a['result'].get('error','candidate') for a in attempts)), attempts=attempts))
    old = re.findall(r'err=(\w+)', BASELOG.read_text())
    new = [a['result'].get('error','candidate') for r in rows[:4] for a in r['attempts']]
    if len(old) != 192 or old != new:
        raise ValueError('saved 192 outcomes not reproduced')
    if not rows[-1]['statuses'].get('candidate',0):
        raise ValueError('positive control did not return candidate')
    write_json(out / 'results.json', dict(iteration=36, split='development', holdout=False,
        ids=[RID, POSITIVE], protocol_sha256=digest(out / 'protocol.json'), cases=rows,
        baseline_outcomes_reproduced=True, positive_candidate_control_passed=True,
        elapsed_s=read(out / 'run.json')['elapsed_s'], external_wcs=read(WCS)['status'],
        production_changed=False, independent_ground_truth=False, automatic_improvement_confirmed=False))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['prepare','build','run','validate','summarize'])
    p.add_argument('--out-dir', type=Path, required=True)
    p.add_argument('--registry', type=Path)
    args = p.parse_args()
    if args.stage == 'prepare':
        if args.registry is None: p.error('--registry required for prepare')
        prepare(args.out_dir.resolve(), args.registry)
    else:
        globals()[args.stage](args.out_dir.resolve())
