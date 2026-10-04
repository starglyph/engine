#!/usr/bin/env python3
"""Freeze and run fixed correspondence fits; no image matching or new WCS."""
import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np

from collection_run import ROOT, digest, write_json
from robustness_pairs import RID, PRIOR, initial_candidate, load_traces, reviewed
from tetra3_footprint import prepare as prepare_pipeline


def build_cases(traces, points):
    candidates={a:initial_candidate(traces,a) for a in ('control','footprint')}
    starts={a:c['stages'][0]['camera'] for a,c in candidates.items()}
    pairs={a:[dict(world=p['world'],xy=p['xy']) for p in c['stages'][0]['pairs']] for a,c in candidates.items()}
    foot=starts['footprint']
    fov=np.rad2deg(2*np.arctan(foot['width']/(2*foot['focal_px'])))
    weight=float(10+(2.5-10)*np.clip((fov-30)/40,0,1))
    cases=[]
    for arm in starts:
        cases.append(dict(name='reproduce_'+arm,initial=starts[arm],matches=pairs[arm],prior_weight=None,
                          expected=candidates[arm]['stages'][1]['camera']))
    for initial in starts:
        for pair_set in pairs:
            cases.append(dict(name=f'matrix_{initial}_{pair_set}',initial=starts[initial],matches=pairs[pair_set],prior_weight=weight))
    for dropped in range(len(points)):
        cases.append(dict(name=f'leave_one_out_{dropped}',initial=foot,
            matches=[p for i,p in enumerate(pairs['footprint']) if i!=dropped],prior_weight=weight))
    visible=[p['id'] for p in points if p['label']=='visible_source']
    stellar=[p['id'] for p in points if p['label'] in ('visible_source','blend')]
    for label,indices in [('stellar_including_blend',stellar),('reviewed_single_sources',visible)]:
        for initial in starts:
            cases.append(dict(name=f'{label}_{initial}',initial=starts[initial],
                matches=[pairs['footprint'][i] for i in indices],prior_weight=weight))
    for radius in (8,12):
        measured=copy.deepcopy(pairs['footprint'])
        for i,p in enumerate(points):
            measured[i]['xy']=((np.array(p[f'native_r{radius}_xy'])+.5)/2.5-.5).tolist()
        cases.append(dict(name=f'pixels_r{radius}_stellar',initial=foot,
            matches=[measured[i] for i in stellar],prior_weight=weight))
    if any(len(c['matches'])<8 for c in cases):
        raise ValueError('review leaves too few pairs for fixed five-parameter diagnostic')
    return cases


def prepare(out, review, registry):
    points=reviewed(review)
    traces=load_traces()
    if out.exists():
        raise ValueError('fresh output directory required')
    cases=build_cases(traces,points)
    geometry_path=ROOT/'docs/experiments/robustness-iteration-5-geometry.json'
    frame=next(f for f in json.loads(geometry_path.read_text())['frames'] if f['id']==RID)
    original=json.loads((ROOT/'docs/experiments/robustness-iteration-5-review.json').read_text())
    source=next(f for f in original['frames'] if f['id']==RID)
    radec=np.deg2rad([[next(p['ra_deg'] for p in source['points'] if p['id']==s['source_id']),
                      next(p['dec_deg'] for p in source['points'] if p['id']==s['source_id'])] for s in frame['sources']])
    ra,dec=radec.T
    worlds=np.column_stack((np.cos(dec)*np.cos(ra),np.cos(dec)*np.sin(ra),np.sin(dec))).tolist()
    prepare_pipeline(out,registry)
    project=out/'pipeline'
    for name in ('solve.rs','track.rs'):
        shutil.copyfile(registry/'src/solver'/name,project/'tetra3/src/solver'/name)
    solve=project/'crates/starglyph-core/src/solve.rs'
    solve.write_text(solve.read_text()+'\n#[cfg(test)]\nmod pair_refit;\n')
    rust=Path(__file__).with_name('pair_refit.rs')
    shutil.copyfile(rust,solve.parent/'solve/pair_refit.rs')
    write_json(out/'input.json',dict(id=RID,split='development',cases=cases,probe_worlds=worlds))
    files=[Path(__file__),rust,Path(__file__).with_name('robustness_pairs.py'),review/'review.json',
           review/'candidates.json',review/'review-protocol.json',out/'input.json',geometry_path,
           ROOT/'docs/experiments/robustness-iteration-5-review.json',ROOT/'prototype/crates/starglyph-core/src/solve.rs']
    files += list((project/'crates').glob('*/src/**/*.rs'))+list((project/'tetra3/src').rglob('*.rs'))
    files += [project/'Cargo.toml',project/'Cargo.lock',project/'tetra3/Cargo.toml']
    files += [PRIOR/a/'trace.json' for a in ('control','footprint')]
    write_json(out/'protocol.json',dict(iteration=7,id=RID,split='development',cases=[c['name'] for c in cases],
        hashes={str(p.resolve()):digest(p) for p in files},review_sha256=digest(review/'review.json'),
        parameters='same five-parameter optimizer; shared k1 prior for comparisons; no rematch',
        design='two first-fit reproduction checks; 2x2 start/pair matrix; all 11 leave-one-out; visually selected subsets; image-only centroid sensitivity',
        criteria=['reproduce both previous first fits','report fixed 40 review sources and 14 outside-input probes',
                  'retain every leave-one-out outcome without choosing a new production heuristic'],
        ground_truth=False,holdout=False))


def run(out):
    protocol=json.loads((out/'protocol.json').read_text())
    if protocol['split']!='development' or protocol['id']!=RID:
        raise ValueError('development required')
    for path,expected in protocol['hashes'].items():
        if digest(Path(path))!=expected:
            raise ValueError('frozen input changed: '+path)
    project=out/'pipeline'
    env={**os.environ,'CARGO_TARGET_DIR':str(project/'target'),'STARGLYPH_PAIR_INPUT':str(out/'input.json'),
         'STARGLYPH_PAIR_OUTPUT':str(out/'fits.json')}
    command=['cargo','test','--offline','--release','--manifest-path',str(project/'Cargo.toml'),'-p','starglyph-core',
             'solve::pair_refit::fixed_pairs','--','--ignored','--nocapture']
    start=time.monotonic()
    with (out/'run.log').open('w') as log:
        try:
            result=subprocess.run(command,cwd=ROOT/'prototype',env=env,stdout=log,stderr=subprocess.STDOUT,timeout=240,check=False)
            status=dict(exit_code=result.returncode,status='completed' if result.returncode==0 else 'failed')
        except subprocess.TimeoutExpired:
            status=dict(exit_code=None,status='timeout')
    status.update(elapsed_s=time.monotonic()-start,command=command)
    write_json(out/'run.json',status)
    if status['status']!='completed':
        raise RuntimeError('fixed-pair replay failed; see run.log')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['prepare','run']);p.add_argument('out',type=Path)
    p.add_argument('--review',type=Path);p.add_argument('--registry',type=Path)
    a=p.parse_args()
    if a.command=='prepare':
        if a.review is None or a.registry is None:p.error('--review and --registry required')
        prepare(a.out.resolve(),a.review.resolve(),a.registry.resolve())
    else:run(a.out.resolve())
