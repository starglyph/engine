#!/usr/bin/env python3
"""Bounded local-field identity leads from frozen detections; no ground truth."""
import argparse
from collections import Counter
from pathlib import Path
import json

from collection_run import ROOT, digest, select_records, write_json
from robustness_candidate_trace import command
from robustness_matcher_funnel import parse_trace, aggregate

RID='wm_r_134291495'
MANIFEST=ROOT/'data/samples/sky-samples/manifest.json'
DIAG=ROOT/f'prototype/artifacts/robustness-iteration-1/diagnosis/{RID}/solve-reports/{RID}.json'
BINARY=ROOT/'prototype/artifacts/robustness-iteration-38/trace'
REGIONS=[('left',[0,0,2000,1800]),('center',[1800,0,4200,1900]),('right',[4016,0,6016,1800])]


def read(p):return json.loads(p.read_text())


def selected():
    r=select_records(read(MANIFEST),'development',[RID])
    if len(r)!=1 or r[0]['id']!=RID:raise ValueError('one development record required')
    return r[0]


def subfield(detections,box,limit=40):
    x0,y0,x1,y1=box
    if x1<=x0 or y1<=y0:raise ValueError('positive rectangle required')
    eligible=[d for d in detections if x0<=d['x']<x1 and y0<=d['y']<y1]
    return [d|dict(x=d['x']-x0,y=d['y']-y0,rank=i)for i,d in enumerate(eligible[:limit])], [d['rank']for d in eligible[:limit]]


def prepare(out):
    record=selected()
    if out.exists():raise ValueError('fresh output directory required')
    checks=read(ROOT/'docs/experiments/robustness-iteration-38-checks.json')
    if digest(BINARY)!=checks['artifact_hashes'][str(BINARY.relative_to(ROOT))]:raise ValueError('binary changed')
    previous=ROOT/'docs/experiments/robustness-iteration-38-input.json'
    if digest(previous)!=checks['artifact_hashes'][str(previous.relative_to(ROOT))]:raise ValueError('prior input changed')
    diag=read(DIAG);old=next(d for d in diag['detection_diagnostics']if d['width']==6016 and d['tier']=='default')
    inp=read(previous)
    control=next(c for c in inp['cases']if c['name']=='6016-default-control')
    cases=[control];regions=[]
    for name,box in REGIONS:
        points,ranks=subfield(old['result']['detections'],box)
        if len(points)!=40:raise ValueError('40 local detections required')
        cases.append(dict(name=name,id=RID,width=box[2]-box[0],height=box[3]-box[1],search=points))
        regions.append(dict(name=name,bounds=box,original_ranks=ranks))
    out.mkdir(parents=True)
    write_json(out/'input.json',dict(id=RID,split='development',databases=inp['databases'],cases=cases,regions=regions))
    source=MANIFEST.parent/record['file']
    if digest(source)!=record['clean_sha256']:raise ValueError('photo changed')
    paths=[Path(__file__).resolve(),Path(__file__).with_name('robustness_matcher_funnel.py'),Path(__file__).with_name('collection_run.py'),
        Path(__file__).with_name('robustness_candidate_trace.py'),BINARY,MANIFEST,source,DIAG,previous,out/'input.json',
        ROOT/'data/catalogs/hyg_v42.csv.gz',ROOT/'prototype/artifacts/robustness-collection/noirlab-new.html',
        *(Path(p)for p in inp['databases']),*(ROOT/'data/samples/sky-samples'/n for n in ['robustness-results.json','collection-wcs-review.json'])]
    write_json(out/'protocol.json',dict(iteration=39,id=RID,split='development',holdout=False,
        scope='three predefined rectangular local fields; fixed native-default detections translated only, same top40 and matching thresholds',
        regions=REGIONS,budget=dict(attempts=192,per_attempt_ms=2500,wall_s=180),
        criteria=['48 full-frame control outcomes reproduce iteration38',
            'retain every local candidate and failure; do not tune regions or rerun successful settings',
            'local solver hypotheses are leads only; require visual identities and unused neighbouring stars before a conditional identification',
            'no accepted ground truth or whole-field projection claim; no full WCS run'],
        hashes={str(p.relative_to(ROOT)):digest(p)for p in paths}))


def validate(out):
    p=read(out/'protocol.json')
    if (p['id'],p['split'],p['holdout'])!=(RID,'development',False):raise ValueError('development only')
    selected()
    for n,sha in p['hashes'].items():
        if digest(ROOT/n)!=sha:raise ValueError('frozen input changed: '+n)
    return p


def run(out):
    validate(out)
    if (out/'run.json').exists():raise ValueError('fresh run required')
    command(out,'run',[str(BINARY),str(out/'input.json'),str(out/'raw.json')],timeout=180)


def summarize(out):
    validate(out);raw=read(out/'raw.json')['cases'];inp=read(out/'input.json')
    if [c['name']for c in raw]!=[c['name']for c in inp['cases']]or any(len(c['attempts'])!=48 for c in raw):raise ValueError('scope changed')
    trace=parse_trace((out/'run.log').read_text());i=0;cases=[]
    if len(trace)!=192:raise ValueError('attempt count changed')
    for c in raw:
        attempts=[]
        for a in c['attempts']:
            t=trace[i];i+=1
            if t['context']!=dict(case=c['name'],db=a['db'],k=a['k'],fov=a['fov']):raise ValueError('trace attribution changed')
            attempts.append(a|dict(funnel=aggregate(t['fovs'])))
        cases.append(dict(name=c['name'],statuses=dict(Counter(a['result'].get('error','candidate')for a in attempts)),attempts=attempts))
    old=read(ROOT/'docs/experiments/robustness-iteration-38-results.json')
    prior=next(c for c in old['cases']if c['name']=='6016-default-control')
    if [a['result']for a in cases[0]['attempts']]!=[a['result']for a in prior['attempts']]:raise ValueError('full-frame control changed')
    write_json(out/'results.json',dict(iteration=39,id=RID,split='development',holdout=False,protocol_sha256=digest(out/'protocol.json'),
        cases=cases,elapsed_s=read(out/'run.json')['elapsed_s'],independent_ground_truth=False,automatic_improvement_confirmed=False,production_changed=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['prepare','run','validate','summarize']);p.add_argument('--out-dir',type=Path,required=True)
    a=p.parse_args();globals()[a.stage](a.out_dir.resolve())
