#!/usr/bin/env python3
"""Trace one existing smartphone regression; never select collection or holdout."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from collection_run import ROOT, digest, write_json
from robustness_centroid_origin import BEFORE, AFTER, prepare as prepare_workspace

FRAME = '20260831_214709'
PREFIX = 'CANDIDATE_TRACE '
HOOK_BEFORE = '''    Some(Candidate {
        label,
        pose,
        verify: result,
        attitude_quat: Some(quat),
    })'''
HOOK_AFTER = '''    let candidate = Candidate {
        label,
        pose,
        verify: result,
        attitude_quat: Some(quat),
    };
    candidate_trace::emit(sol, detections, verify, &candidate);
    Some(candidate)'''


def instrument(source, arm):
    if arm not in ('control', 'centered') or source.count(HOOK_BEFORE) != 1 or source.count(BEFORE) != 1:
        raise ValueError('unexpected source or arm')
    if arm == 'centered':
        source = source.replace(BEFORE, AFTER)
    return source.replace(HOOK_BEFORE, HOOK_AFTER) + '\nmod candidate_trace;\n'


def command(out, name, args, env=None, timeout=300):
    started = time.monotonic()
    with (out / (name + '.log')).open('w') as log:
        try:
            process = subprocess.run(args, cwd=ROOT/'prototype', env=env, stdout=log,
                                     stderr=subprocess.STDOUT, timeout=timeout)
            code = process.returncode
        except subprocess.TimeoutExpired:
            code = None
    row = dict(command=args, exit_code=code, elapsed_s=time.monotonic()-started)
    write_json(out/(name+'.json'), row)
    if code != 0:
        raise RuntimeError(f'{name} failed: {code}; see saved log')
    return row


def prepare(out):
    prepare_workspace(out)
    pipeline = out/'pipeline'
    source = ROOT/'prototype/crates/starglyph-core/src/solve.rs'
    target = pipeline/'crates/starglyph-core/src/solve.rs'
    module = Path(__file__).with_name('candidate_trace.rs')
    shutil.copyfile(module, target.parent/'solve/candidate_trace.rs')
    (out/'binaries').mkdir()
    env = {**os.environ, 'CARGO_TARGET_DIR':str(pipeline/'target')}
    for arm in ('control','centered'):
        target.write_text(instrument(source.read_text(), arm))
        shutil.copyfile(target, out/f'solve-{arm}.rs')
        command(out, 'build-'+arm, ['cargo','build','--offline','--locked','--release',
            '--manifest-path',str(pipeline/'Cargo.toml'),'-p','starglyph-cli'], env)
        shutil.copy2(pipeline/'target/release/starglyph',out/'binaries'/arm)
    files = [Path(__file__).resolve(),module,source,out/'solve-control.rs',out/'solve-centered.rs',
        ROOT/'data/input/smartphone/manifest.json',ROOT/f'data/input/smartphone/{FRAME}.jpg',
        ROOT/'data/catalogs/hyg_v42.csv.gz',out/'binaries/control',out/'binaries/centered',
        ROOT/'docs/experiments/robustness-iteration-18-results.json',
        ROOT/'docs/experiments/robustness-iteration-18-checks.json']
    for base in (ROOT/'prototype',pipeline):
        files.extend([base/'Cargo.toml',base/'Cargo.lock'])
        for crate in ('starglyph-core','starglyph-cli','simulator-core'):
            files.extend((base/'crates'/crate/'src').rglob('*.rs'))
            files.append(base/'crates'/crate/'Cargo.toml')
    files.extend((ROOT/'prototype/artifacts/cache').glob('*.bin'))
    write_json(out/'protocol.json',dict(iteration=19,id=FRAME,track='legacy smartphone regression',
        holdout=False,algorithm_change='same isolated two-line candidate as iteration18',
        scope='observe all tetra3 candidates before verification acceptance/refine; no tuning',
        wall_limit_s=120,timing='instrumented diagnostic run; no performance inference',
        hashes={str(p.relative_to(ROOT)):digest(p) for p in sorted(set(files))}))


def validate(out):
    p=json.loads((out/'protocol.json').read_text())
    if p['id']!=FRAME or p['holdout']:
        raise ValueError('only frozen legacy frame is allowed')
    for name,sha in p['hashes'].items():
        if digest(ROOT/name)!=sha:
            raise ValueError('changed frozen input: '+name)
    return p


def run(out):
    p=validate(out)
    for arm in ('control','centered'):
        if (out/f'run-{arm}.json').exists():
            raise ValueError('run already exists; use fresh output')
    rows={}
    for arm in ('control','centered'):
        rows[arm]=command(out,'run-'+arm,[str(out/'binaries'/arm),'eval','--manifest',
            '../data/input/smartphone/manifest.json','--ids',FRAME,'--catalog','../data/catalogs/hyg_v42.csv.gz',
            '--out-dir',str(out/arm)],{**os.environ,'STARGLYPH_SOLVE_DEBUG':'1'},p['wall_limit_s'])
    write_json(out/'runs.json',rows)


def traces(path):
    return [json.loads(line[len(PREFIX):]) for line in path.read_text().splitlines() if line.startswith(PREFIX)]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['prepare','run'])
    parser.add_argument('--out-dir',type=Path,required=True)
    args=parser.parse_args()
    globals()[args.stage](args.out_dir.resolve())


if __name__=='__main__':
    main()
