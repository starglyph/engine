#!/usr/bin/env python3
"""Frozen single-frame roll influence using unchanged production Rust functions."""
import argparse
import json
import os
from pathlib import Path
import shutil

from collection_run import ROOT, digest, write_json
from robustness_candidate_trace import command
from robustness_centroid_origin import prepare as prepare_workspace

FRAME = '20260831_214709'
MODULE = '\n#[cfg(test)]\nmod roll_influence;\n'


def prepare(out):
    prepare_workspace(out)
    pipeline = out/'pipeline'
    source = ROOT/'prototype/crates/starglyph-core/src/solve.rs'
    target = pipeline/'crates/starglyph-core/src/solve.rs'
    target.write_text(source.read_text()+MODULE)
    (target.parent/'solve/centroid_origin.rs').unlink()
    module = Path(__file__).with_name('roll_influence.rs')
    shutil.copyfile(module, target.parent/'solve/roll_influence.rs')
    manifest = pipeline/'Cargo.toml'
    text = manifest.read_text()
    if text.count('serde_json = "1"') != 1:
        raise ValueError('unexpected serde manifest')
    manifest.write_text(text.replace('serde_json = "1"',
        'serde_json = { version = "1", features = ["float_roundtrip"] }'))
    trace = ROOT/'docs/experiments/robustness-iteration-19-results.json'
    image = ROOT/f'data/input/smartphone/{FRAME}.jpg'
    catalog = ROOT/'data/catalogs/hyg_v42.csv.gz'
    write_json(out/'input.json', dict(id=FRAME,holdout=False,image=str(image),catalog=str(catalog),
        first_candidates=json.loads(trace.read_text())['first_candidates']))
    files = [Path(__file__).resolve(), module, Path(__file__).with_name('collection_run.py'),
        Path(__file__).with_name('robustness_candidate_trace.py'),
        Path(__file__).with_name('robustness_centroid_origin.py'),trace,image,catalog,out/'input.json',
        ROOT/'docs/experiments/robustness-iteration-21-results.json']
    files.extend(ROOT/'data/samples/sky-samples'/name for name in
        ['manifest.json','robustness-results.json','collection-wcs-review.json'])
    for base in [ROOT/'prototype',pipeline]:
        files.extend([base/'Cargo.toml',base/'Cargo.lock'])
        for crate in ['starglyph-core','starglyph-cli','simulator-core']:
            files.append(base/'crates'/crate/'Cargo.toml')
            files.extend((base/'crates'/crate/'src').rglob('*.rs'))
    write_json(out/'protocol.json',dict(iteration=22,id=FRAME,holdout=False,
        cases=28,scope='two saved first candidates: baseline and each of 13 single-pair omissions from roll only',
        production_change=False,new_searches=0,refinements=0,
        criteria=['reproduce saved baseline camera/metrics to 1e-10 and exact verification matches',
                  'rebuild full epoch catalogue; saved pair worlds agree to 1e-14',
                  'retain all 50 detections and full magnitude-6.8 verification catalogue in every case',
                  'same verification radius and hard/soft thresholds; no parameter selection',
                  'report all omissions including regressions; original search evidence is not recomputed',
                  'counts are counterfactual sensitivity, not new solves or independent geometry validation'],
        detection_metadata='x,y,flux exact saved values; unused peak/snr/area/elongation placeholders; rank enumeration',
        wall_limit_s=120,build_wall_limit_s=600,
        hashes={str(p.relative_to(ROOT)):digest(p) for p in sorted(set(files))}))


def validate(out):
    plan=json.loads((out/'protocol.json').read_text())
    if plan['id']!=FRAME or plan['holdout'] or plan['cases']!=28:
        raise ValueError('only fixed legacy frame and 28 cases allowed')
    for name,sha in plan['hashes'].items():
        if digest(ROOT/name)!=sha:
            raise ValueError('changed frozen input: '+name)
    return plan


def run(out):
    plan=validate(out)
    if (out/'run.json').exists():
        raise ValueError('run already attempted; use fresh output')
    pipeline=out/'pipeline'
    env={**os.environ,'CARGO_TARGET_DIR':str(pipeline/'target')}
    args=['cargo','test','--offline','--locked','--release','--manifest-path',str(pipeline/'Cargo.toml'),
          '-p','starglyph-core','--lib']
    command(out,'build',args+['--no-run','--message-format=json'],env,plan['build_wall_limit_s'])
    artifacts=[]
    for line in (out/'build.log').read_text().splitlines():
        if line.startswith('{'):
            row=json.loads(line)
            if row.get('reason')=='compiler-artifact' and row.get('executable') and row['target']['name']=='starglyph_core':
                artifacts.append(row['executable'])
    if len(artifacts)!=1:
        raise ValueError('expected one core test executable')
    binary=out/'roll-test'
    shutil.copy2(artifacts[0],binary)
    write_json(out/'build-artifact.json',dict(binary=str(binary.relative_to(ROOT)),sha256=digest(binary)))
    env.update(STARGLYPH_ROLL_PLAN=str(out/'input.json'),STARGLYPH_ROLL_OUTPUT=str(out/'results.json'))
    command(out,'run',[str(binary),'solve::roll_influence::export','--exact','--ignored','--nocapture'],
            env,plan['wall_limit_s'])
    validate(out)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['prepare','run','validate'])
    parser.add_argument('--out-dir',type=Path,required=True)
    args=parser.parse_args()
    globals()[args.stage](args.out_dir.resolve())


if __name__=='__main__':
    main()
