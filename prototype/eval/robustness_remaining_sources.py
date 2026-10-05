#!/usr/bin/env python3
"""Review all 31 surviving wide-fit pairs on the frozen development frame."""
import argparse
import csv
import gzip
import json
from pathlib import Path

import numpy as np
from astropy.io import fits
from PIL import Image

from collection_run import ROOT, digest, write_json
from compare_local_wcs import project, stats
from robustness_geometry import selected, native_centroid, validate_review, preview
from robustness_changed_sources import RID, CATALOG, GEOMETRY, PRIOR

DROP = ROOT/'prototype/artifacts/robustness-iteration-26/drop-5'


def pair_key(index, match):
    return (index, *match['world'], *match['xy'])


def common_stage(cases):
    if len(cases) != 4:
        raise ValueError('four frozen crossovers required')
    sets=[]
    for case in cases:
        step=case['after']['steps'][1]
        if step['stage']!='rematch' or step['radius_px']!=10.:
            raise ValueError('frozen 10px stage required')
        v=step['input']
        keys={pair_key(i,m) for i,m in zip(v['original_detection_indices'],v['matches'])}
        if len(keys)!=31 or len(v['matches'])!=31 or len(v['original_detection_indices'])!=31 or any(k[0]==5 for k in keys):
            raise ValueError('expected 31 distinct pairs without foreground5')
        sets.append(keys)
    if any(s!=sets[0] for s in sets[1:]):
        raise ValueError('different wide-fit identities')
    if (cases[0]['pose_arm'],cases[0]['match_arm'])!=('control','control'):
        raise ValueError('canonical control/control path required')
    return cases[0]['after']


def load_stage():
    rec=selected([RID])[0]
    for folder in [DROP,PRIOR]:
        p=json.loads((folder/'protocol.json').read_text())
        if p['id']!=RID or p['holdout']:
            raise ValueError('development frame required')
        for name,sha in p['hashes'].items():
            if digest(ROOT/name)!=sha:raise ValueError('changed prior input: '+name)
    # Pin the published iteration26 result as well as its raw replay inputs.
    local=json.loads((DROP/'results.json').read_text())
    public=json.loads((ROOT/'docs/experiments/robustness-iteration-26-drop-results.json').read_text())
    if local!=public:raise ValueError('iteration26 local/public results differ')
    return rec,common_stage(local['cases'])


def prepare(out):
    rec,path=load_stage()
    if out.exists():raise ValueError('fresh output required')
    source=ROOT/'data/samples/sky-samples'/rec['file']
    if digest(source)!=rec['clean_sha256']:raise ValueError('source changed')
    detections=json.loads((PRIOR/'replay-plan.json').read_text())['inputs']['control']['detections']
    with gzip.open(CATALOG,'rt') as f:
        stars=[s for s in csv.DictReader(f) if s['mag'] and float(s['mag'])<=6.8]
    radec=np.array([np.rad2deg([float(s['rarad']),float(s['decrad'])])
        if s.get('rarad') and s.get('decrad') else [float(s['ra'])*15,float(s['dec'])] for s in stars])
    ra,dec=np.deg2rad(radec).T
    units=np.column_stack((np.cos(dec)*np.cos(ra),np.cos(dec)*np.sin(ra),np.sin(dec)))
    extraction=ROOT/f'prototype/artifacts/robustness-baseline/wcs-run/{RID}/downsample-2/field.axy'
    data=fits.getdata(extraction);header=fits.getheader(extraction)
    if (header['IMAGEW'],header['IMAGEH'])!=(rec['width'],rec['height']):raise ValueError('extraction dimensions differ')
    xy=np.column_stack((data['X'],data['Y'])).astype(float)-1
    probes=next(f['sources'] for f in json.loads(GEOMETRY.read_text())['frames'] if f['id']==RID)
    previous=json.loads((ROOT/'docs/experiments/robustness-iteration-26-review.json').read_text())['frames'][0]['points']
    stage=path['steps'][1]['input'];points=[]
    for n,(index,m) in enumerate(zip(stage['original_detection_indices'],stage['matches'])):
        det=detections[index];center=np.array(m['xy'])
        if [det['x'],det['y']]!=m['xy']:raise ValueError('original index mapping mismatch')
        ids=np.flatnonzero(np.linalg.norm(units-m['world'],axis=1)<1e-12)
        if len(ids)!=1:raise ValueError('ambiguous catalogue lookup')
        row=int(ids[0]);star=stars[row]
        nearest=np.argsort(np.linalg.norm(xy-center,axis=1))[:3]
        angular=np.rad2deg(np.arccos(np.clip(units@m['world'],-1,1)))
        neighbours=[dict(hyg_id=int(stars[i]['id']),mag=float(stars[i]['mag']),separation_deg=float(angular[i])) for i in np.flatnonzero((angular<.2)&(np.arange(len(stars))!=row))]
        points.append(dict(id=n,detection_index=index,hyg_id=int(star['id']),name=star['proper'] or star['bf'],
            mag=float(star['mag']),world=m['world'],radec=radec[row].tolist(),review_center=m['xy'],detection=det,
            choices=[dict(row=int(i),x=float(xy[i,0]),y=float(xy[i,1])) for i in nearest],
            prior_probes=[s for s in probes if np.linalg.norm(np.array(s['xy'])-center)<12],
            previous_review=[p for p in previous if p['id']==index],catalogue_neighbours=neighbours,
            label='pending',selected_row=None))
    frame=dict(id=RID,image=str(source.relative_to(ROOT)),source_sha256=rec['clean_sha256'],width=rec['width'],height=rec['height'],points=points)
    out.mkdir(parents=True)
    for name in ['candidates.json','review.json']:write_json(out/name,dict(split='development',frames=[frame]))
    preview(frame,out)
    files=[Path(__file__).resolve(),source,CATALOG,GEOMETRY,extraction,DROP/'protocol.json',DROP/'results.json',
        ROOT/'docs/experiments/robustness-iteration-26-drop-results.json',ROOT/'docs/experiments/robustness-iteration-26-review.json',
        PRIOR/'replay-plan.json',ROOT/'prototype/crates/starglyph-core/src/catalog.rs',
        *(ROOT/'prototype/eval'/n for n in ['robustness_geometry.py','robustness_changed_sources.py','compare_local_wcs.py','collection_run.py'])]
    files.extend(ROOT/'data/samples/sky-samples'/n for n in ['manifest.json','robustness-results.json','collection-wcs-review.json'])
    write_json(out/'protocol.json',dict(iteration=27,id=RID,split='development',holdout=False,
        selection='all31 pairs of shared 10px rematch after manual ground exclusion; canonical control/control path; no residual selection',
        detection_indices=[p['detection_index'] for p in points],hashes={str(p.relative_to(ROOT)):digest(p) for p in files},
        candidates_sha256=digest(out/'candidates.json'),
        review='physical source and ambiguity first; conditional catalogue identities, no independent WCS',
        measurements='existing native_centroid r8/r12 annulus16-22; frozen external extraction; saved camera projections only',
        criteria=['retain every ambiguous/blended point in report','no fit, no parameter or threshold search',
                  'report all31 and visually single subset, edges, centroid shifts and fixed-fit residuals']))


def reviewed(out):
    selected([RID])
    p=json.loads((out/'protocol.json').read_text())
    if (p['id'],p['split'],p['holdout'])!=(RID,'development',False):raise ValueError('development required')
    for name,sha in p['hashes'].items():
        if digest(ROOT/name)!=sha:raise ValueError('frozen input changed: '+name)
    if digest(out/'candidates.json')!=p['candidates_sha256']:raise ValueError('candidates changed')
    review=json.loads((out/'review.json').read_text())
    validate_review(json.loads((out/'candidates.json').read_text()),review)
    used=set()
    for row in review['frames'][0]['points']:
        if row['label'] not in ['visible_source','blend','ambiguous','not_visible'] or not row.get('note'):
            raise ValueError('complete visual review required')
        chosen=row['selected_row']
        if row['label']=='visible_source' and chosen is None:raise ValueError('single source needs a reviewed centroid')
        if chosen is not None:
            if chosen not in [c['row'] for c in row['choices']] or chosen in used:raise ValueError('invalid or reused extraction')
            used.add(chosen)
    return p,review['frames'][0]


def measure(out):
    protocol,frame=reviewed(out);_,path=load_stage()
    with Image.open(ROOT/frame['image']) as im:image=im.convert('RGB')
    gray=np.asarray(image,dtype=np.float32)@np.array([.2126,.7152,.0722],dtype=np.float32)
    cameras=dict(before_wide=path['steps'][0]['camera'],after_wide=path['steps'][1]['camera'],final=path['camera'])
    rows=[]
    for p in frame['points']:
        center=np.array(p['review_center']);native={f'native_r{r}':native_centroid(gray,center,r) for r in [8,12]}
        ext=next(([c['x'],c['y']] for c in p['choices'] if c['row']==p['selected_row']),None)
        positions=dict(detector=center.tolist(),**native)
        if ext is not None:positions['external']=ext
        predictions={name:project(cam,[p['radec']])[0] for name,cam in cameras.items()}
        rows.append(dict(**p,positions=positions,centroid_offsets_px={k:float(np.linalg.norm(np.array(v)-center)) for k,v in positions.items()},
            projections={k:v.tolist() for k,v in predictions.items()},
            residual_vectors={stage:{kind:(pred-pos).tolist() for kind,pos in positions.items()} for stage,pred in predictions.items()},
            errors_px={stage:{kind:float(np.linalg.norm(pred-pos)) for kind,pos in positions.items()} for stage,pred in predictions.items()}))
    def grouped(subset):
        return {stage:{kind:stats(np.array([r['errors_px'][stage][kind] for r in subset if kind in r['positions']]))
            for kind in ['detector','native_r8','native_r12','external'] if any(kind in r['positions'] for r in subset)} for stage in cameras} if subset else {}
    single=[r for r in rows if r['label']=='visible_source']
    groups=dict(all31=rows,visually_single=single,
        left_10pct=[r for r in single if r['review_center'][0]<frame['width']*.1],
        right_10pct=[r for r in single if r['review_center'][0]>frame['width']*.9],
        top_10pct=[r for r in single if r['review_center'][1]<frame['height']*.1],
        bottom_10pct=[r for r in single if r['review_center'][1]>frame['height']*.9])
    write_json(out/'results.json',dict(iteration=27,id=RID,holdout=False,independent_ground_truth=False,
        protocol_sha256=digest(out/'protocol.json'),review_sha256=digest(out/'review.json'),
        counts={k:sum(p['label']==k for p in rows) for k in ['visible_source','blend','ambiguous','not_visible']},
        camera_stages=cameras,groups={k:dict(count=len(v),metrics=grouped(v)) for k,v in groups.items()},
        points=rows,new_searches=0,new_refinements=0,algorithm_change=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['prepare','measure']);parser.add_argument('--out-dir',type=Path,required=True)
    args=parser.parse_args();globals()[args.stage](args.out_dir.resolve())
