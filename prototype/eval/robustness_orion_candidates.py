#!/usr/bin/env python3
"""Audit both saved Orion candidates without matching, fitting or new WCS."""
import argparse
from pathlib import Path
import re
import time

import numpy as np

from collection_run import ROOT, digest, write_json
from compare_local_wcs import stats
from robustness_candidate_trace_report import project, pair_metrics
from robustness_orion_fit import METHODS, WIDTH, HEIGHT, RID, previous, read
from robustness_reference_audit import selected
from robustness_reference_coverage import extent
from robustness_refinement_report import lift
from robustness_tangential_basis import geometry

SAVED_SOLVE = ROOT/'prototype/artifacts/robustness-iteration-25/fixed/solve-control.rs'
SAVED_LOG = ROOT/'prototype/artifacts/robustness-iteration-41/run.log'
CURRENT_SOLVE = ROOT/'prototype/crates/starglyph-core/src/solve.rs'


def pair_key(pair):
    """An index alone does not establish equality of correspondences."""
    return pair['detection_index'], tuple(pair['world']), tuple(pair['xy'])


def bind_pairs(rows, trace, saved):
    original = trace['input']['verification']['matches']
    bound = saved['stages'][0]['pairs']
    if len(original) != 16 or len(bound) != 16:
        raise ValueError('saved initial16 binding required')
    # Refinement input omits detector indices; recover them by exact coordinates,
    # preserving the input order used by the published catalog binding.
    detections = trace['chosen']['detections']
    indices = {tuple(d[:2]): i for i, d in enumerate(detections)}
    if len(indices) != len(detections):
        raise ValueError('ambiguous detector coordinates')
    lookup = {pair_key(dict(p, detection_index=indices[tuple(p['xy'])])): b
              for p, b in zip(original, bound)}
    if len(lookup) != 16 or len({pair_key(p) for p in rows}) != len(rows):
        raise ValueError('duplicate correspondence')
    result = []
    for p in rows:
        if pair_key(p) not in lookup:
            raise ValueError('pair lacks an exact saved catalog binding')
        b = lookup[pair_key(p)]
        pos = lift(p['xy'], trace['input']['camera'], WIDTH, HEIGHT)
        if not np.allclose(pos, b['xy'], rtol=0, atol=1e-8):
            raise ValueError('bound coordinate mismatch')
        result.append(dict(**b, detection_index=p['detection_index']))
    return result


def coverage(xy):
    return extent([dict(x=p[0], y=p[1]) for p in xy], WIDTH, HEIGHT)


def thresholds(source):
    names = ('HARD_MIN_MATCHES', 'HARD_MAX_PROB', 'VERIFY_MIN_HITS', 'VERIFY_LOG_ODDS_MIN', 'VERIFY_RADIUS_PX')
    return {n: float(re.search(r'const '+n+r': \w+ = ([\d.e-]+);', source)[1]) for n in names}


def prepare(out):
    selected()
    if out.exists():
        raise ValueError('fresh output required')
    paths = []
    for name in ('40-results', '41-trace', '41-results', '46-assessment'):
        path = previous(name); checks = previous(name[:2]+'-checks')
        if digest(path) != read(checks)['artifact_hashes'][str(path.relative_to(ROOT))]:
            raise ValueError('published evidence changed: '+name)
        paths.extend([path, checks])
    original = read(previous('41-protocol'))
    for path in (SAVED_SOLVE, SAVED_LOG):
        expected = (original['hashes'] if path == SAVED_SOLVE else read(previous('41-checks'))['artifact_hashes'])[str(path.relative_to(ROOT))]
        if digest(path) != expected:
            raise ValueError('saved observer evidence changed')
    protected = read(previous('46-checks'))['protected_hashes']
    for name, sha in protected.items():
        if digest(ROOT/name) != sha:
            raise ValueError('protected input changed')
    old, current = thresholds(SAVED_SOLVE.read_text()), thresholds(CURRENT_SOLVE.read_text())
    if old != current:
        raise ValueError('acceptance thresholds changed')
    paths += [previous('41-protocol'), SAVED_SOLVE, SAVED_LOG, CURRENT_SOLVE, Path(__file__).resolve(),
        *(ROOT/n for n in protected), *(Path(__file__).with_name(n) for n in (
            'collection_run.py', 'compare_local_wcs.py', 'robustness_candidate_trace_report.py',
            'robustness_orion_fit.py', 'robustness_reference_audit.py', 'robustness_reference_coverage.py',
            'robustness_refinement_report.py', 'robustness_tangential_basis.py'))]
    out.mkdir(parents=True)
    write_json(out/'protocol.json', dict(iteration=47, id=RID, split='development', holdout=False,
        scope='descriptive audit of both saved candidates, no parameter or camera selection',
        thresholds=old, criteria=['retain both candidates in original order and all47 reviewed sources',
            'compare full correspondence identity including detection index, world and xy, ignoring only order',
            'bind pairs only through exact saved41 catalog binding; unmatched or ambiguous identities stay unresolved',
            'replay Rust adapter projections within1e-8 working px and chosen47 projections within1e-8 original px',
            'same three centroid methods and frozen46 spatial groups; no optimization or independent GT claim'],
        budget=dict(new_solver_calls=0, new_wcs_calls=0, new_fits=0),
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
    started = time.monotonic(); protocol = validate(out)
    if (out/'results.json').exists():
        raise ValueError('fresh run required')
    trace = read(previous('41-trace')); saved = read(previous('41-results'))
    sources = read(previous('40-results'))['sources']
    groups = read(previous('46-assessment'))['groups']; ids = [s['id'] for s in sources]
    if len(trace['candidates']) != 2 or saved['id'] != RID or saved['split'] != 'development':
        raise ValueError('two saved development candidates required')
    ra, dec = np.deg2rad([s['radec'] for s in sources]).T
    worlds = np.column_stack((np.cos(dec)*np.cos(ra), np.cos(dec)*np.sin(ra), np.sin(dec)))
    rows = []; sets = []; t = protocol['thresholds']
    for i, c in enumerate(trace['candidates']):
        if c['camera']['k1'] != 0 or c['detections'] != trace['chosen']['detections']:
            raise ValueError('same detector input and undistorted candidate required')
        v = c['verification']; sol = c['solution']; cam = c['camera']
        if v['hits'] != len(v['matches']) or sol['num_matches'] != len(c['pairs']):
            raise ValueError('pair count mismatch')
        sets.append({pair_key(p) for p in v['matches']})
        tetra = bind_pairs(c['pairs'], trace, saved); verified = bind_pairs(v['matches'], trace, saved)
        pred = lift(project(cam, worlds), cam, WIDTH, HEIGHT)
        chosen = c == trace['chosen']
        if chosen and not np.allclose(pred, saved['stages'][0]['predicted_xy'], rtol=0, atol=1e-8):
            raise ValueError('chosen geometry replay failed')
        residual = project(cam, [p['world'] for p in v['matches']])-[p['xy'] for p in v['matches']]
        internal = stats(np.linalg.norm(residual, axis=1))
        if abs(internal['rms_px']-v['rms_px']) > 1e-8:
            raise ValueError('verification RMS replay failed')
        rows.append(dict(index=i, chosen=chosen, camera=cam, solution=sol,
            verification={k: value for k, value in v.items() if k != 'matches'},
            hard_eligible=sol['num_matches'] >= t['HARD_MIN_MATCHES'] and sol['prob'] < t['HARD_MAX_PROB'] and v['hits'] >= t['VERIFY_MIN_HITS'],
            soft_verified=v['hits'] >= t['VERIFY_MIN_HITS'] and v['log_odds'] >= t['VERIFY_LOG_ODDS_MIN'],
            tetra_pairs=tetra, verification_pairs=verified,
            tetra_coverage=coverage([p['xy'] for p in tetra]), verification_coverage=coverage([p['xy'] for p in verified]),
            adapter_replay=pair_metrics(c), verification_rms_replayed_px=internal['rms_px'],
            predicted_xy_original=pred.tolist(), geometry={m: dict(residual_xy_px=(pred-[s['coordinates'][m] for s in sources]).tolist(),
                groups=geometry(pred-[s['coordinates'][m] for s in sources], ids, groups)) for m in METHODS}))
    if sum(c['chosen'] for c in rows) != 1:
        raise ValueError('unique observed choice required')
    dets = lift(np.array(trace['chosen']['detections'])[:, :2], trace['chosen']['camera'], WIDTH, HEIGHT)
    attempts = [s.strip() for s in SAVED_LOG.read_text().splitlines() if s.startswith('  [')]
    result = dict(iteration=47, id=RID, split='development', holdout=False, protocol_sha256=digest(out/'protocol.json'),
        candidates=rows, verification_sets_equal=sets[0] == sets[1],
        common_verification_pairs=len(sets[0] & sets[1]), union_verification_pairs=len(sets[0] | sets[1]),
        detector_xy_original=dets.tolist(), detector_coverage=coverage(dets),
        detector_prefix_coverage={str(k): coverage(dets[:k]) for k in (20, 30, 40)},
        sources=[dict(id=s['id'], hyg_id=s['hyg_id'], name=s['name'], coordinates=s['coordinates']) for s in sources],
        group_ids=groups, saved_attempts=attempts, elapsed_s=time.monotonic()-started,
        new_solver_calls=0, new_wcs_calls=0, new_fits=0, production_changed=False,
        independent_ground_truth=False, automatic_improvement_confirmed=False)
    validate(out); write_json(out/'results.json', result)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=('prepare', 'run', 'validate')); p.add_argument('--out-dir', type=Path, required=True)
    a = p.parse_args(); globals()[a.stage](a.out_dir.resolve())
