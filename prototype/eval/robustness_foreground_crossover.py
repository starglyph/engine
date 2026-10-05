#!/usr/bin/env python3
"""One manual diagnostic exclusion; replay the existing iteration25 executable."""
import argparse
import copy
import json
import os
from pathlib import Path

import numpy as np

from collection_run import ROOT, digest, write_json
from compare_local_wcs import project, stats
from robustness_candidate_trace import command
from robustness_changed_sources import RID, PRIOR, GEOMETRY, measure
from robustness_centroid_origin_report import geometry_groups

REMOVED = 5


def drop_foreground(plan):
    if plan['id'] != RID or plan['holdout']:
        raise ValueError('frozen development frame required')
    result = copy.deepcopy(plan)
    if plan['inputs']['control']['detections'] != plan['inputs']['huber']['detections']:
        raise ValueError('different input detections')
    for arm in ['control','huber']:
        inp = result['inputs'][arm]
        if REMOVED in inp['verification']['detection_indices']:
            raise ValueError('intervention must not change first LM matches')
        if len(inp['detections']) != 40:
            raise ValueError('expected frozen full detection set')
        del inp['detections'][REMOVED]
        inp['verification']['detection_indices'] = [i-(i>REMOVED) for i in inp['verification']['detection_indices']]
    return result


def run(review_dir, out):
    if out.exists():
        raise ValueError('fresh intervention output required')
    measure(review_dir)  # Validate frozen sources and completed review first.
    review=json.loads((review_dir/'review.json').read_text())
    point=next(p for p in review['frames'][0]['points'] if p['id']==REMOVED)
    if point['label']!='not_visible' or 'foreground' not in point['note']:
        raise ValueError('confirmed foreground review required')
    plan=json.loads((PRIOR/'replay-plan.json').read_text())
    new_plan=drop_foreground(plan)
    binary=PRIOR/'replay-test'
    out.mkdir(parents=True)
    write_json(out/'input.json',new_plan)
    files=[Path(__file__).resolve(),Path(__file__).with_name('robustness_changed_sources.py'),
        Path(__file__).with_name('robustness_centroid_origin_report.py'),Path(__file__).with_name('compare_local_wcs.py'),
        Path(__file__).with_name('robustness_candidate_trace.py'),binary,PRIOR/'replay-plan.json',PRIOR/'replay-results.json',
        review_dir/'protocol.json',review_dir/'review.json',review_dir/'results.json',out/'input.json',
        GEOMETRY,ROOT/'docs/experiments/robustness-iteration-5-geometry.json']
    protocol=dict(iteration=26,id=RID,split='development',holdout=False,manual_diagnostic=True,
        intervention='delete only original detection index5 from both complete rematch inputs; first LM inputs and cameras unchanged',
        removed_detection=point['detection'],original_index_map=[i for i in range(40) if i!=REMOVED],
        unchanged=['catalogue','initial cameras','first LM matches','refinement implementation','all radii and minimum match counts'],
        criteria=['report all four crossovers, including regressions','same conditional geometry and native centroids as iteration25',
                  'no new solve or automatic improvement claim; one hand-reviewed exclusion'],
        hashes={str(p.relative_to(ROOT)):digest(p) for p in files})
    write_json(out/'protocol.json',protocol)
    timing=command(out,'replay',[str(binary),'--exact','solve::refinement_trace::replay::export','--ignored'],
        {**os.environ,'STARGLYPH_REFINEMENT_PLAN':str(out/'input.json'),
         'STARGLYPH_REFINEMENT_OUTPUT':str(out/'raw-results.json')},60)
    old=json.loads((PRIOR/'replay-results.json').read_text())['cases']
    new=json.loads((out/'raw-results.json').read_text())['cases']
    probes=next(f['sources'] for f in json.loads(GEOMETRY.read_text())['frames'] if f['id']==RID)
    alternate=next(f['sources'] for f in json.loads((ROOT/'docs/experiments/robustness-iteration-5-geometry.json').read_text())['frames'] if f['id']==RID)
    alternate={s['source_id']:s for s in alternate}
    xy=np.array([s['xy'] for s in probes]);radec=np.array([s['radec'] for s in probes])
    positions=dict(external_centroid=xy,**{f'native_r{r}':np.array([alternate[s['source_id']][f'native_r{r}_xy'] for s in probes]) for r in [8,12]})
    original_dets=plan['inputs']['control']['detections'];camera=plan['inputs']['control']['camera']
    masks=geometry_groups(xy,[np.array([[d['x'],d['y']] for d in original_dets])],camera['width'],camera['height'],12)
    rows=[]
    for before,after in zip(old,new):
        if (before['pose_arm'],before['match_arm'])!=(after['pose_arm'],after['match_arm']):
            raise ValueError('case order changed')
        if before['steps'][0]!=after['steps'][0]:
            raise ValueError('first LM changed despite identical inputs')
        predictions={a:project(c['camera'],radec) for a,c in [('before',before),('after',after)]}
        metrics={centroid:{group:dict(count=int(mask.sum()),
            **{a:stats(np.linalg.norm(pred[mask]-pos[mask],axis=1)) for a,pred in predictions.items()})
            for group,mask in masks.items() if mask.any()} for centroid,pos in positions.items()}
        for groups in metrics.values():
            for g in groups.values():g['rms_delta_px']=g['after']['rms_px']-g['before']['rms_px']
        for s in after['steps']:
            v=s.get('input',s.get('verification'))
            if v:v['original_detection_indices']=[protocol['original_index_map'][i] for i in v['detection_indices']]
        after['verification']['original_detection_indices']=[protocol['original_index_map'][i] for i in after['verification']['detection_indices']]
        rows.append(dict(pose_arm=after['pose_arm'],match_arm=after['match_arm'],first_lm_exact=True,
            before=dict(camera=before['camera'],verification=before['verification']),after=after,conditional_geometry=metrics))
    for name,sha in protocol['hashes'].items():
        if digest(ROOT/name)!=sha:raise ValueError('changed intervention input')
    write_json(out/'results.json',dict(iteration=26,id=RID,holdout=False,manual_diagnostic=True,
        automatic_improvement_confirmed=False,new_searches=0,protocol_sha256=digest(out/'protocol.json'),
        timing=timing,cases=rows,independent_ground_truth=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--review-dir',type=Path,required=True);parser.add_argument('--out-dir',type=Path,required=True)
    args=parser.parse_args();run(args.review_dir.resolve(),args.out_dir.resolve())
