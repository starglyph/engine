#!/usr/bin/env python3
"""One visually justified exclusion using the already verified iteration-8 replay."""
import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import time

from collection_run import ROOT, digest, write_json
from robustness_rematch import validate as validate_replay
from robustness_rematch_report import assert_same_stages, reused_pairs
from robustness_refinement_report import summarize_stage, compact_report
from robustness_wide_pairs import RID, TRACE, GEOMETRY, reviewed

REPLAY = ROOT/'prototype/artifacts/robustness-iteration-8/verified'


def cases_for(points, saved):
    rejected = [p for p in points if p['label'] == 'not_visible']
    if len(rejected) != 1 or rejected[0]['hyg_id'] != 81442:
        raise ValueError('exactly the visually confirmed second leaf required')
    original = next(c for c in saved['cases'] if c['name'] == 'drop_detection_all_stages')
    candidate = copy.deepcopy(original)
    xy = rejected[0]['detector_xy_working']
    found = [i for i, d in enumerate(candidate['detections']) if [d['x'], d['y']] == xy]
    if len(found) != 1 or any(m['xy'] == xy for m in candidate['matches']):
        raise ValueError('second leaf must be unique and absent from initial fit')
    candidate['name'] = 'drop_second_leaf'
    del candidate['detections'][found[0]]
    return [copy.deepcopy(original), candidate], rejected[0]


def prepare(out, review):
    points = reviewed(review)
    old_protocol = validate_replay(REPLAY)
    saved = json.loads((REPLAY/'input.json').read_text())
    cases, leaf = cases_for(points, saved)
    if out.exists():
        raise ValueError('fresh output directory required')
    out.mkdir(parents=True)
    saved['cases'] = cases
    write_json(out/'input.json', saved)
    files = [Path(__file__).resolve(), Path(__file__).with_name('robustness_wide_pairs.py'),
        review/'review.json', review/'candidates.json', review/'protocol.json', out/'input.json',
        TRACE, GEOMETRY, REPLAY/'protocol.json', ROOT/'prototype/eval/robustness_rematch_report.py',
        ROOT/'prototype/eval/robustness_refinement_report.py']
    hashes = dict(old_protocol['hashes'])
    hashes.update({str(p.resolve()):digest(p) for p in files})
    write_json(out/'protocol.json', dict(iteration=10, id=RID, split='development',
        cases=[c['name'] for c in cases], hashes=hashes, leaf_xy=leaf['detector_xy_original'],
        criteria=['reproduce all 12 stages of the 39-detection control',
                  'retain all 40 prior probes and edge/outside-input groups',
                  'track physical leaf/Albireo reuse, without selecting by residual'],
        unchanged=['initial camera and ten initial pairs', 'remaining detection order and coordinates',
                   'radius schedule', 'optimizer', 'acceptance code'],
        interpretation='manual diagnostic exclusion only; no automatic solve-rate evaluation', holdout=False))


def validate(out):
    p = json.loads((out/'protocol.json').read_text())
    if p['id'] != RID or p['split'] != 'development' or p['holdout']:
        raise ValueError('development required')
    validate_replay(REPLAY)
    for path, sha in p['hashes'].items():
        if digest(Path(path)) != sha:
            raise ValueError('frozen input changed: '+path)
    return p


def run(out):
    validate(out)
    # Same unchanged workspace and Cargo features, not a different build variant.
    command = json.loads((REPLAY/'run.json').read_text())['command']
    env = {**os.environ, 'CARGO_TARGET_DIR':str(REPLAY/'pipeline/target'),
        'STARGLYPH_REMATCH_INPUT':str(out/'input.json'), 'STARGLYPH_REMATCH_OUTPUT':str(out/'trace.json')}
    started = time.monotonic()
    with (out/'run.log').open('w') as log:
        try:
            process = subprocess.run(command, cwd=ROOT/'prototype', env=env, stdout=log,
                                     stderr=subprocess.STDOUT, timeout=240, check=False)
            status = dict(status='completed' if process.returncode == 0 else 'failed', exit_code=process.returncode)
        except subprocess.TimeoutExpired:
            status = dict(status='timeout', exit_code=None)
    status.update(elapsed_s=time.monotonic()-started, command=command)
    write_json(out/'run.json', status)
    if status['status'] != 'completed':
        raise RuntimeError('replay failed; see run.log')


def report(out):
    protocol = validate(out)
    trace = json.loads((out/'trace.json').read_text())
    if trace['id'] != RID or trace['split'] != 'development' or [c['name'] for c in trace['cases']] != protocol['cases']:
        raise ValueError('trace selection changed')
    previous = next(c for c in json.loads(TRACE.read_text())['cases'] if c['name'] == protocol['cases'][0])
    assert_same_stages(trace['cases'][0]['stages'], previous['stages'])
    prior = json.loads(GEOMETRY.read_text())
    cases = []
    for case in trace['cases']:
        stages = []
        for raw in case['stages']:
            stage = summarize_stage(raw, prior['sources'], 3000, 4000)
            stage['flow'] = dict(albireo=reused_pairs(raw, prior['albireo_detection_xy']),
                first_leaf=reused_pairs(raw, prior['leaf_xy']), second_leaf=reused_pairs(raw, protocol['leaf_xy']))
            stages.append(stage)
        cases.append(dict(name=case['name'], n_detections=case['n_detections'], elapsed_ms=case['elapsed_ms'], stages=stages))
    compact = compact_report(dict(candidates=cases))
    write_json(out/'geometry.json', dict(iteration=10, id=RID, split='development',
        protocol_sha256=digest(out/'protocol.json'), trace_sha256=digest(out/'trace.json'),
        baseline_all_stages_exact=trace['cases'][0]['stages'] == previous['stages'],
        sources=prior['sources'], cases=compact['candidates'], fit_pair_columns=compact['fit_pair_columns'],
        interpretation='conditional geometry on unchanged probes; manual exclusion, no absolute ground truth or new acceptance'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'run', 'report'])
    parser.add_argument('out', type=Path)
    parser.add_argument('--review', type=Path)
    args = parser.parse_args()
    if args.command == 'prepare':
        if args.review is None:
            parser.error('--review required')
        prepare(args.out.resolve(), args.review.resolve())
    elif args.command == 'run':
        run(args.out.resolve())
    else:
        report(args.out.resolve())
