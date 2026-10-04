#!/usr/bin/env python3
"""Freeze three fixed-candidate rematch counterfactuals on one development frame."""
import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from collection_run import ROOT, digest, write_json
from robustness_pairs import RID, PRIOR, load_traces, initial_candidate
from robustness_geometry import selected
from tetra3_footprint import prepare as prepare_pipeline


def build_cases(candidate,detections,leaf):
    if leaf['label']!='not_visible' or leaf['hyg_id']!=80761:
        raise ValueError('visually rejected foreground pair required')
    pairs=candidate['stages'][0]['pairs']
    bad=[i for i,p in enumerate(pairs) if p['hyg_ids']==[leaf['hyg_id']]]
    found=[i for i,d in enumerate(detections) if [d['x'],d['y']]==leaf['detector_xy_working']]
    if len(bad)!=1 or len(found)!=1 or pairs[bad[0]]['xy']!=leaf['detector_xy_working']:
        raise ValueError('foreground pair/detection identity mismatch')
    all_pairs=[dict(hyg_ids=p['hyg_ids'],world=p['world'],xy=p['xy']) for p in pairs]
    clean_pairs=[p for i,p in enumerate(all_pairs) if i!=bad[0]]
    cases=[dict(name='baseline',matches=all_pairs,detections=detections),
           dict(name='drop_initial_pair',matches=clean_pairs,detections=detections),
           dict(name='drop_detection_all_stages',matches=clean_pairs,
                detections=[d for i,d in enumerate(detections) if i!=found[0]])]
    return copy.deepcopy(cases),found[0]


def prepare(out,registry):
    traces=load_traces()
    records=selected([RID])
    review_path=ROOT/'docs/experiments/robustness-iteration-7-review.json'
    checks=json.loads((ROOT/'docs/experiments/robustness-iteration-7-checks.json').read_text())
    if digest(review_path)!=checks['review']['review_sha256']:
        raise ValueError('iteration-7 review changed')
    review=json.loads(review_path.read_text())
    if review['split']!='development' or review['frames'][0]['id']!=RID:
        raise ValueError('development review required')
    leaf=next(p for p in review['frames'][0]['points'] if p['hyg_id']==80761)
    candidate=initial_candidate(traces,'footprint')
    tier=next(t for t in traces['footprint']['tiers'] if t['tier']=='default')
    cases,index=build_cases(candidate,tier['detections'],leaf)
    image=ROOT/'data/samples/sky-samples'/records[0]['file']
    if digest(image)!=records[0]['clean_sha256']:
        raise ValueError('image changed')
    prepare_pipeline(out,registry)
    project=out/'pipeline'
    # The fixture must round-trip every f64 exactly, including detection positions.
    # This parser feature applies only to the isolated diagnostic workspace.
    manifest=project/'Cargo.toml'
    manifest_text=manifest.read_text()
    if manifest_text.count('serde_json = "1"')!=1:raise ValueError('serde_json declaration changed')
    manifest.write_text(manifest_text.replace('serde_json = "1"',
        'serde_json = { version = "1", features = ["float_roundtrip"] }'))
    for name in ('solve.rs','track.rs'):
        shutil.copyfile(registry/'src/solver'/name,project/'tetra3/src/solver'/name)
    solve=project/'crates/starglyph-core/src/solve.rs'
    solve.write_text(solve.read_text()+'\n#[cfg(test)]\nmod geometry_replay;\n#[cfg(test)]\nmod rematch_replay;\n')
    observer=ROOT/'prototype/eval/geometry_replay.rs'
    source=observer.read_text()
    if source.count('fn stages(')!=1:raise ValueError('stage observer signature changed')
    (solve.parent/'solve/geometry_replay.rs').write_text(source.replace('fn stages(','pub(super) fn stages('))
    rust=Path(__file__).with_name('rematch_replay.rs')
    shutil.copyfile(rust,solve.parent/'solve/rematch_replay.rs')
    catalog=ROOT/'data/catalogs/hyg_v42.csv.gz'
    write_json(out/'input.json',dict(id=RID,split='development',image=str(image),catalog=str(catalog),
        initial=candidate['stages'][0]['camera'],cases=cases))
    # Resolve the copied workspace before freezing its lock, so reruns can use --locked.
    with (out/'lock-resolution.log').open('w') as log:
        subprocess.run(['cargo','update','--offline','--workspace','--manifest-path',str(project/'Cargo.toml')],
                       cwd=ROOT/'prototype',stdout=log,stderr=subprocess.STDOUT,check=True)
    files=[Path(__file__),Path(__file__).with_name('robustness_rematch_report.py'),rust,observer,review_path,ROOT/'docs/experiments/robustness-iteration-7-checks.json',image,catalog,
           ROOT/'data/samples/sky-samples/manifest.json',ROOT/'prototype/Cargo.lock',out/'input.json',
           ROOT/'docs/experiments/robustness-iteration-6-geometry.json',
           ROOT/'docs/experiments/robustness-iteration-5-geometry.json',ROOT/'docs/experiments/robustness-iteration-5-review.json',
           PRIOR/'footprint/trace.json']
    files+=list((project/'crates').glob('*/src/**/*.rs'))+list((project/'tetra3/src').rglob('*.rs'))
    files+=list((project/'crates').glob('*/Cargo.toml'))
    files += [project/'Cargo.toml',project/'Cargo.lock',project/'tetra3/Cargo.toml']
    write_json(out/'protocol.json',dict(iteration=8,id=RID,split='development',cases=[c['name'] for c in cases],
        hashes={str(p.resolve()):digest(p) for p in files},leaf_detection_index=index,leaf_xy_original=leaf['detector_xy_original'],
        criteria=['reproduce every baseline stage','track both catalogue identity and physical detection reuse',
                  'retain all fixed iteration-5 probes including edge regressions'],
        unchanged=['fixed candidate','radii','optimizer','catalogue','all remaining detections and their order'],
        interpretation='manual source exclusion diagnostic; no automated acceptance or solve-rate test',holdout=False))


def validate(out):
    protocol=json.loads((out/'protocol.json').read_text())
    if protocol['id']!=RID or protocol['split']!='development':raise ValueError('development required')
    selected([RID])
    for path,sha in protocol['hashes'].items():
        if digest(Path(path))!=sha:raise ValueError('frozen input changed: '+path)
    return protocol


def run(out):
    validate(out)
    project=out/'pipeline'
    command=['cargo','test','--offline','--locked','--release','--manifest-path',str(project/'Cargo.toml'),'-p','starglyph-core',
             'solve::rematch_replay::fixed_candidate','--','--ignored','--nocapture']
    env={**os.environ,'CARGO_TARGET_DIR':str(project/'target'),'STARGLYPH_REMATCH_INPUT':str(out/'input.json'),
         'STARGLYPH_REMATCH_OUTPUT':str(out/'trace.json')}
    started=time.monotonic()
    with (out/'run.log').open('w') as log:
        try:
            result=subprocess.run(command,cwd=ROOT/'prototype',env=env,stdout=log,stderr=subprocess.STDOUT,timeout=240,check=False)
            status=dict(status='completed' if result.returncode==0 else 'failed',exit_code=result.returncode)
        except subprocess.TimeoutExpired:status=dict(status='timeout',exit_code=None)
    status.update(command=command,elapsed_s=time.monotonic()-started)
    write_json(out/'run.json',status)
    if status['status']!='completed':raise RuntimeError('rematch replay failed; see run.log')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['prepare','run']);p.add_argument('out',type=Path)
    p.add_argument('--registry',type=Path);a=p.parse_args()
    if a.command=='prepare':
        if a.registry is None:p.error('--registry required')
        prepare(a.out.resolve(),a.registry.resolve())
    else:run(a.out.resolve())
