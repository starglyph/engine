#!/usr/bin/env python3
"""Fixed manual foreground exclusion before top-K; matcher diagnostic only."""
import argparse
from collections import Counter
from pathlib import Path
import re

from collection_run import ROOT, digest, select_records, write_json
from robustness_candidate_trace import command
from robustness_matcher_funnel import parse_trace, aggregate
from robustness_orion_fit import previous, read

RID = 'wm_r_16053617'
POSITIVE = 'wm_r_112929230'
MANIFEST = ROOT/'data/samples/sky-samples/manifest.json'
DIAG = ROOT/f'prototype/artifacts/robustness-iteration-1/diagnosis/{RID}/solve-reports/{RID}.json'
BASELOG = ROOT/f'prototype/artifacts/robustness-iteration-24/real/development/control/{RID}.log'
WCS = ROOT/f'prototype/artifacts/robustness-baseline/wcs-run/{RID}/reference.json'
BINARY = ROOT/'prototype/artifacts/robustness-iteration-38/trace'
FRACTION = .45


def selection():
    rows = select_records(read(MANIFEST), 'development', [RID, POSITIVE])
    if {r['id'] for r in rows} != {RID, POSITIVE} or any(r['track'] != 'solver' for r in rows):
        raise ValueError('two fixed development solver records required')
    return rows


def sky_indices(points, height, original_height):
    """Pixel-centre lift; preserve full-frame coordinates, photometry and order."""
    if height <= 0 or original_height <= 0:
        raise ValueError('positive image dimensions required')
    return [i for i, p in enumerate(points)
            if (p['y']+.5)*original_height/height-.5 < FRACTION*original_height]


def make_cases(diag, original_height):
    cases = []; selections = []
    for arm in ['control', 'manual_sky']:
        for d in diag:
            points = d['result']['detections']; limit = d['max_detections']
            indices = list(range(len(points))) if arm == 'control' else sky_indices(points, d['height'], original_height)
            selected = indices[:limit]
            if len(selected) < 30:
                raise ValueError('fixed six prefixes require at least30 sources')
            name = f"{d['width']}-{d['tier']}-{arm}"
            cases.append(dict(name=name, id=RID, width=d['width'], height=d['height'], search=[points[i] for i in selected]))
            selections.append(dict(name=name, arm=arm, total_detections=len(points), eligible=len(indices),
                max_detections=limit, original_indices=selected,
                removed_from_original_top30=[i for i in range(30) if i not in indices],
                newly_promoted_from_below_top_k=[i for i in selected if i >= limit]))
    return cases, selections


def prepare(out):
    records = selection()
    if out.exists():
        raise ValueError('fresh output directory required')
    checks = read(previous('38-checks'))
    paths = [BINARY, previous('38-input'), previous('38-results')]
    for p in paths:
        if digest(p) != checks['artifact_hashes'][str(p.relative_to(ROOT))]:
            raise ValueError('saved matcher evidence changed')
    rec = next(r for r in records if r['id'] == RID)
    cases, selections = make_cases(read(DIAG)['detection_diagnostics'], rec['height'])
    if len(cases) != 8:
        raise ValueError('four original branches required')
    old = read(previous('38-input'))
    cases.append(next(c for c in old['cases'] if c['name'] == 'positive-control'))
    protected = read(previous('54-checks'))['protected_hashes'].copy()
    for r in records:
        protected[str((MANIFEST.parent/r['file']).relative_to(ROOT))] = r['clean_sha256']
    for name, sha in protected.items():
        if digest(ROOT/name) != sha:
            raise ValueError('protected data changed')
        paths.append(ROOT/name)
    out.mkdir(parents=True)
    write_json(out/'input.json', dict(split='development', ids=[RID, POSITIVE], databases=old['databases'], cases=cases, selections=selections))
    paths += [DIAG, BASELOG, WCS, MANIFEST, out/'input.json', previous('38-checks'), previous('54-checks'),
        *(Path(p) for p in old['databases']), Path(__file__).resolve(),
        *(Path(__file__).with_name(n) for n in ['test_robustness_snow_foreground.py', 'robustness_matcher_funnel.py',
            'collection_run.py', 'robustness_candidate_trace.py', 'robustness_diagnose.py', 'robustness_orion_fit.py'])]
    write_json(out/'protocol.json', dict(iteration=55, ids=[RID, POSITIVE], split='development', holdout=False,
        target=RID, case_names=[c['name'] for c in cases], original_size=[rec['width'], rec['height']],
        manual_region=dict(x_fraction=[0, 1], y_fraction=[0, FRACTION], rule='original pixel-centre y < 0.45*original height'),
        hypothesis='foreground competition before top-K explains failure; remove only below fixed sky boundary from full saved detector list',
        change='manual diagnostic only; unchanged dimensions, pixels, detections, photometry, order, thresholds and budgets; refill same top40/50 before existing prefixes',
        budget=dict(attempts=432, target_per_arm=192, positive=48, per_attempt_ms=2500, wall_s=180),
        criteria=['192 control statuses reproduce saved24 log in order; retain timing-sensitive differences as invalid reproduction if any',
            'same48 positive attempts; saved successful candidate identities reproduce, all other outcomes including timeouts reported',
            'single predefined boundary, no radius/threshold/region tuning; compare all four branches at same per-attempt budget',
            'a new tetra3 candidate supports search-level rescue only; identities and unused field checks required before geometry or solve-rate claims',
            'if no candidate, foreground exclusion alone is insufficient; retain stage counters and best rejected hypothesis without accepting it',
            'no new WCS, full solver, ground truth, negatives or holdout; production unchanged'],
        protected_hashes=protected, hashes={str(p.relative_to(ROOT)): digest(p) for p in sorted(set(paths))}))


def validate(out):
    p = read(out/'protocol.json')
    if (p['ids'], p['split'], p['holdout']) != ([RID, POSITIVE], 'development', False):
        raise ValueError('development only')
    selection()
    for name, sha in p['hashes'].items():
        if digest(ROOT/name) != sha:
            raise ValueError('frozen input changed: '+name)
    return p


def render(out):
    p = validate(out)
    from PIL import Image
    from robustness_diagnose import preview
    rec = next(r for r in selection() if r['id'] == RID)
    with Image.open(MANIFEST.parent/rec['file']) as im:
        for c in read(out/'input.json')['cases'][:8]:
            points = [d | dict(x=(d['x']+.5)*im.width/c['width']-.5,
                               y=(d['y']+.5)*im.height/c['height']-.5) for d in c['search']]
            preview(im, c['name'], points, out)


def run(out):
    validate(out)
    if (out/'run.json').exists():
        raise ValueError('fresh run required')
    command(out, 'run', [str(BINARY), str(out/'input.json'), str(out/'raw.json')], timeout=180)
    validate(out)


def failure_stage(result, counts):
    if 'error' not in result:
        return 'candidate'
    if result['error'] == 'Timeout':
        return 'timeout'
    for name, stage in [('patterns', 'no_patterns'), ('hash_candidates', 'no_hash_candidates'),
                        ('fov_pass', 'fov'), ('ratios_pass', 'edge_ratios'), ('verifications', 'svd'),
                        ('probability_pass', 'verification_probability')]:
        if counts[name] == 0:
            return stage
    return 'finalization'


def summarize(out):
    p = validate(out); inp = read(out/'input.json'); raw = read(out/'raw.json')['cases']
    trace = parse_trace((out/'run.log').read_text())
    if len(trace) != 432 or [c['name'] for c in raw] != p['case_names'] or any(len(c['attempts']) != 48 for c in raw):
        raise ValueError('scope changed')
    cursor = 0; rows = []
    for c in raw:
        attempts = []; all_fovs = []
        for a in c['attempts']:
            t = trace[cursor]; cursor += 1
            if t['context'] != dict(case=c['name'], db=a['db'], k=a['k'], fov=a['fov']):
                raise ValueError('attempt attribution mismatch')
            counts = aggregate(t['fovs']); all_fovs.extend(t['fovs'])
            if counts['finalized'] != int('error' not in a['result']):
                raise ValueError('finalization mismatch')
            attempts.append(a | dict(funnel=counts, stage=failure_stage(a['result'], counts)))
        rows.append(dict(name=c['name'], statuses=dict(Counter(a['result'].get('error', 'candidate') for a in attempts)),
            stages=dict(Counter(a['stage'] for a in attempts)), elapsed_ms=sum(a['elapsed_ms'] for a in attempts),
            funnel=aggregate(all_fovs), attempts=attempts))
    old = re.findall(r'err=(\w+)', BASELOG.read_text())
    new = [a['result'].get('error', 'candidate') for c in rows[:4] for a in c['attempts']]
    if len(old) != 192 or old != new:
        raise ValueError('192 control outcomes not reproduced')
    positive = next(c for c in read(previous('38-results'))['cases'] if c['name'] == 'positive-control')
    differences = [dict(db=a['db'], k=a['k'], fov=a['fov'], before=b['result'], after=a['result'])
        for a, b in zip(rows[-1]['attempts'], positive['attempts']) if a['result'] != b['result']]
    if any('error' not in d['before'] for d in differences):
        raise ValueError('successful positive candidate changed')
    write_json(out/'results.json', dict(iteration=55, ids=[RID, POSITIVE], split='development', holdout=False,
        protocol_sha256=digest(out/'protocol.json'), cases=rows, selections=inp['selections'],
        control_outcomes_reproduced=192, positive_differences=differences, external_wcs_status=read(WCS)['status'],
        target_candidates=sum(c['statuses'].get('candidate', 0) for c in rows[4:8]), run=read(out/'run.json'),
        new_full_solver_calls=0, new_wcs_calls=0, production_changed=False,
        independent_ground_truth=False, automatic_improvement_confirmed=False))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['prepare', 'render', 'run', 'summarize', 'validate'])
    p.add_argument('--out-dir', type=Path, required=True)
    a = p.parse_args(); globals()[a.stage](a.out_dir.resolve())
