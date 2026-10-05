#!/usr/bin/env python3
"""One manual source-promotion counterfactual; not an automatic selector."""
import argparse
from collections import Counter
from pathlib import Path

from collection_run import ROOT, digest, write_json
from robustness_candidate_trace import command
from robustness_central_sources import RID, DIAG, MANIFEST, read, validate as validate_trace
from robustness_thinning_trace import read_thinning

PROBES = ['P01','P03','P04','P05','P09']
PRIOR = ROOT/'prototype/artifacts/robustness-iteration-36'


def promote(detections, ranks, limit):
    if len(ranks)!=5 or len(set(ranks))!=5 or any(r<limit or r>=len(detections) for r in ranks):
        raise ValueError('five distinct sources below top-K required')
    chosen=[detections[r] for r in ranks]+detections[:limit-5]
    return [d | dict(rank=i) for i,d in enumerate(chosen)]


def prepare(out):
    validate_trace(out)
    if (out/'replay-protocol.json').exists(): raise ValueError('fresh replay required')
    raw=read(out/'trace.json')['tiers'];old=read(DIAG)['detection_diagnostics']
    previous=read(PRIOR/'input.json');binary=read(PRIOR/'binary.json')
    checks=read(ROOT/'docs/experiments/robustness-iteration-36-checks.json')
    for path in [PRIOR/'trace', PRIOR/'binary.json']:
        if digest(path)!=checks['artifact_hashes'][str(path.relative_to(ROOT))]: raise ValueError('previous binary changed')
    if digest(PRIOR/'trace')!=binary['sha256']: raise ValueError('binary changed')
    cases=[];restored=[]
    for d in old:
        name=f"{d['width']}-{d['tier']}"
        trace=next(t for t in raw if not t['quantile'] and not t['blob_concentration'] and (t['width'],t['tier'])==(d['width'],d['tier']))
        ranks=[]
        for pid in PROBES:
            c=trace['probes'][int(pid[1:])]['component']
            if not c or c['outcome']!='top_k': raise ValueError('all promoted sources must pass filters and fail only top-K')
            rank=c['rank_before_top_k'];det=d['result']['detections'][rank]
            if c['centroid'] != [det['x'],det['y']]: raise ValueError('source/rank mismatch')
            ranks.append(rank)
        control=d['result']['detections'][:d['max_detections']]
        variant=promote(d['result']['detections'],ranks,d['max_detections'])
        for arm,ds in [('control',control),('manual_promotion',variant)]:
            cases.append(dict(name=name+'-'+arm,id=RID,width=d['width'],height=d['height'],search=ds))
        restored.append(dict(name=name,probe_ids=PROBES,old_ranks=ranks))
    write_json(out/'replay-input.json',dict(id=RID,split='development',databases=previous['databases'],cases=cases,restored=restored))
    paths=[Path(__file__).resolve(),out/'protocol.json',out/'input.json',out/'results.json',out/'trace.json',out/'replay-input.json',
        PRIOR/'trace',PRIOR/'binary.json',PRIOR/'protocol.json',PRIOR/'results.json',MANIFEST,DIAG,
        Path(__file__).with_name('robustness_thinning_trace.py'),
        *(Path(p)for p in previous['databases'])]
    write_json(out/'replay-protocol.json',dict(iteration=37,id=RID,split='development',holdout=False,
        intervention='manual promotion of five visually reviewed, filter-passing sources; replace last five slots; measured coordinates and flux unchanged',
        probes=PROBES,case_names=[c['name']for c in cases],budget=dict(attempts=384,per_attempt_ms=2500,wall_s=180),
        criteria=['192 control outcomes match iteration36 in order',
            'retain all 192 paired variant statuses, timing, tetra3 matches and post-thinning source indices',
            'record whether removing short-prefix TooFew suffices to obtain a tetra3 candidate',
            'manual intervention cannot establish automatic improvement; raw candidate is not an accepted solve'],
        hashes={str(p.relative_to(ROOT)):digest(p)for p in paths}))


def validate(out):
    validate_trace(out);p=read(out/'replay-protocol.json')
    if (p['id'],p['split'],p['holdout'],p['probes'])!=(RID,'development',False,PROBES): raise ValueError('frozen development required')
    for name,sha in p['hashes'].items():
        if digest(ROOT/name)!=sha: raise ValueError('frozen replay input changed: '+name)
    return p


def run(out):
    validate(out)
    if (out/'replay-run.json').exists(): raise ValueError('fresh replay run required')
    command(out,'replay-run',[str(PRIOR/'trace'),str(out/'replay-input.json'),str(out/'replay-raw.json')],timeout=180)


def summarize(out):
    p=validate(out);raw=read(out/'replay-raw.json')['cases'];traces=read_thinning((out/'replay-run.log').read_text())
    if [c['name']for c in raw]!=p['case_names'] or len(traces)!=384 or any(len(c['attempts'])!=48 for c in raw):
        raise ValueError('replay scope changed')
    prior=read(PRIOR/'results.json')['cases'];rows=[];i=0
    for c in raw:
        attempts=[]
        for a in c['attempts']:
            t=traces[i];i+=1
            if t['context']!=dict(case=c['name'],db=a['db'],k=a['k'],fov=a['fov']): raise ValueError('trace attribution changed')
            groups={}
            for s in t['sweeps']: groups.setdefault(tuple(s['kept_input_indices']),[]).append(s['fov'])
            attempts.append(a | dict(sweep_count=len(t['sweeps']),pattern_capable_sweeps=sum(len(s['kept_input_indices'])>=4 for s in t['sweeps']),
                retained=[dict(indices=list(k),fovs_deg=v)for k,v in groups.items()]))
        if c['name'].endswith('-control'):
            previous=next(r for r in prior if r['name']==c['name'].removesuffix('-control'))
            if [a['result']for a in attempts]!=[a['result']for a in previous['attempts']]: raise ValueError('control result changed')
        rows.append(dict(name=c['name'],statuses=dict(Counter(a['result'].get('error','candidate')for a in attempts)),
            elapsed_ms=sum(a['elapsed_ms']for a in attempts),attempts=attempts))
    write_json(out/'replay-results.json',dict(iteration=37,id=RID,split='development',holdout=False,
        protocol_sha256=digest(out/'replay-protocol.json'),control_outcomes_reproduced=192,cases=rows,
        elapsed_s=read(out/'replay-run.json')['elapsed_s'],manual_diagnostic=True,automatic_improvement_confirmed=False,
        independent_ground_truth=False,production_changed=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['prepare','validate','run','summarize']);p.add_argument('--out-dir',type=Path,required=True)
    a=p.parse_args();globals()[a.stage](a.out_dir.resolve())
