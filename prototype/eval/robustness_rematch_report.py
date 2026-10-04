#!/usr/bin/env python3
"""Track physical detection reuse through a frozen rematch sequence."""
import argparse
import json
from pathlib import Path

import numpy as np

from collection_run import ROOT, digest, write_json
from robustness_pairs import PRIOR, initial_candidate, load_traces
from robustness_rematch import RID, validate
from robustness_refinement_report import lift, summarize_stage, compact_report


def assert_same_stages(actual,expected):
    if [s['stage'] for s in actual]!=[s['stage'] for s in expected]:raise ValueError('baseline stages differ')
    for a,b in zip(actual,expected):
        for key in ('width','height','focal_px','k1','world_to_camera'):
            if not np.allclose(a['camera'][key],b['camera'][key],atol=1e-8,rtol=0):
                raise ValueError('baseline camera differs at '+a['stage'])
        if len(a['pairs'])!=len(b['pairs']):raise ValueError('baseline pair count differs')
        for p,q in zip(a['pairs'],b['pairs']):
            if (p['hyg_ids']!=q['hyg_ids'] or not np.allclose(p['xy'],q['xy'],atol=1e-8,rtol=0)
                    or not np.allclose(p['world'],q['world'],atol=1e-12,rtol=0)):
                raise ValueError('baseline pairs differ')


def reused_pairs(stage,xy):
    return [p['hyg_ids'] for p in stage['pairs'] if np.linalg.norm(lift(p['xy'],stage['camera'],3000,4000)-xy)<1e-6]


def summarize(out,output):
    protocol=validate(out)
    traces=load_traces()
    previous=initial_candidate(traces,'footprint')
    trace=json.loads((out/'trace.json').read_text())
    if trace['id']!=RID or trace['split']!='development' or [c['name'] for c in trace['cases']]!=protocol['cases']:
        raise ValueError('trace selection differs')
    assert_same_stages(trace['cases'][0]['stages'],previous['stages'])
    prior=json.loads((ROOT/'docs/experiments/robustness-iteration-6-geometry.json').read_text())
    sources=prior['sources']
    albireo=next(s for s in sources if s['hyg_id']==95648)
    tier=next(t for t in traces['footprint']['tiers'] if t['tier']=='default')
    detxy=lift([[d['x'],d['y']] for d in tier['detections']],tier,3000,4000)
    albireo_det=detxy[int(np.argmin(np.linalg.norm(detxy-albireo['xy'],axis=1)))]
    leaf=np.array(protocol['leaf_xy_original'])
    cases=[]
    for case in trace['cases']:
        stages=[]
        for raw in case['stages']:
            measured=summarize_stage(raw,sources,3000,4000)
            measured['flow']=dict(albireo_detection_catalog_ids=reused_pairs(raw,albireo_det),
                leaf_detection_catalog_ids=reused_pairs(raw,leaf),
                hyg_80761_pairs=[dict(xy=lift(p['xy'],raw['camera'],3000,4000).tolist(),hyg_ids=p['hyg_ids'])
                                for p in raw['pairs'] if 80761 in p['hyg_ids']],
                pairs_near_inspected_leaf=[dict(xy=p['xy'],hyg_ids=p['hyg_ids']) for p in measured['fit_pairs']
                    if np.linalg.norm(np.array(p['xy'])-leaf)<75])
            stages.append(measured)
        cases.append(dict(name=case['name'],n_detections=case['n_detections'],elapsed_ms=case['elapsed_ms'],stages=stages))
    compact=compact_report(dict(candidates=cases))
    write_json(output,dict(iteration=8,id=RID,split='development',protocol_sha256=digest(out/'protocol.json'),
        trace_sha256=digest(out/'trace.json'),sources=sources,leaf_xy=leaf.tolist(),albireo_detection_xy=albireo_det.tolist(),
        baseline_all_stages_reproduced=True,fit_pair_columns=compact['fit_pair_columns'],cases=compact['candidates'],
        limitations=['fixed candidate with manual source exclusion; no automatic acceptance or solve-rate test',
                    'unchanged conditional iteration-5 identities, no ground-truth promotion',
                    'physical detection reuse tracked separately from catalogue ID; nearby foreground requires visual review']))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('out',type=Path);p.add_argument('--output',required=True,type=Path)
    a=p.parse_args();summarize(a.out.resolve(),a.output)
