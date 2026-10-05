#!/usr/bin/env python3
"""Frozen 31-versus-22 pair diagnostic, reusing the existing Rust LM harness."""
import argparse
import copy
import json
import os
from pathlib import Path
import shutil

import numpy as np

from collection_run import ROOT, digest, write_json
from compare_local_wcs import project, stats
from robustness_candidate_trace import command
from robustness_centroid_origin import prepare as prepare_workspace
from robustness_centroid_origin_report import geometry_groups
from robustness_pair_report import sensitivity
from robustness_remaining_sources import RID, DROP, PRIOR, GEOMETRY, load_stage, reviewed

REVIEW = ROOT/'prototype/artifacts/robustness-iteration-27'
NAMES = ['all31', 'single22']


def build_cases(path, points):
    v=path['steps'][1]['input']
    if len(points)!=31 or len(v['matches'])!=31:
        raise ValueError('frozen31 review required')
    for n,(point,index,pair) in enumerate(zip(points,v['original_detection_indices'],v['matches'])):
        if (point['id'],point['detection_index'],point['world'],point['review_center'])!=(n,index,pair['world'],pair['xy']):
            raise ValueError('review/pair identity changed')
    pairs=[dict(pair_id=p['id'],detection_index=p['detection_index'],hyg_id=p['hyg_id'],world=m['world'],xy=m['xy']) for p,m in zip(points,v['matches'])]
    baseline=dict(name=NAMES[0],initial=path['steps'][0]['camera'],matches=pairs,
        prior_weight=None,expected=path['steps'][1]['camera'])
    candidate=copy.deepcopy(baseline)
    candidate.update(name=NAMES[1],expected=None,matches=[m for p,m in zip(points,pairs) if p['label']=='visible_source'])
    if len(candidate['matches'])!=22:raise ValueError('frozen22 visual subset required')
    return [baseline,candidate]


def adapted_harness(text):
    old='assert_eq!(input.id, "wm_r_143159342");'
    if text.count(old)!=1:raise ValueError('existing harness frame guard changed')
    return text.replace(old,'assert_eq!(input.id, "'+RID+'");')


def prepare(out):
    _,frame=reviewed(REVIEW);_,path=load_stage()
    cases=build_cases(path,frame['points'])
    sources=next(f['sources'] for f in json.loads(GEOMETRY.read_text())['frames'] if f['id']==RID)
    prepare_workspace(out)
    pipeline=out/'pipeline';target=pipeline/'crates/starglyph-core/src/solve.rs'
    original=ROOT/'prototype/crates/starglyph-core/src/solve.rs'
    target.write_text(original.read_text()+'\n#[cfg(test)]\nmod pair_refit;\n')
    (target.parent/'solve/centroid_origin.rs').unlink()
    (target.parent/'solve/pair_refit.rs').write_text(adapted_harness(Path(__file__).with_name('pair_refit.rs').read_text()))
    write_json(out/'input.json',dict(id=RID,split='development',cases=cases,probe_worlds=[s['world'] for s in sources]))
    cargo=['cargo','test','--offline','--locked','--release','--manifest-path',str(pipeline/'Cargo.toml'),
        '-p','starglyph-core','--features','serde_json/float_roundtrip','--lib','--no-run','--message-format','json']
    command(out,'build',cargo,timeout=600)
    artifacts=[json.loads(s) for s in (out/'build.log').read_text().splitlines() if s.startswith('{')]
    binaries=[a['executable'] for a in artifacts if a.get('reason')=='compiler-artifact' and a.get('executable') and a['target']['name']=='starglyph_core']
    if len(binaries)!=1:raise ValueError('unique core test executable required')
    shutil.copy2(binaries[0],out/'pair-test')
    files=[Path(__file__).resolve(),out/'input.json',out/'pair-test',GEOMETRY,
        REVIEW/'review.json',REVIEW/'protocol.json',REVIEW/'results.json',DROP/'results.json',PRIOR/'replay-plan.json',
        ROOT/'docs/experiments/robustness-iteration-5-geometry.json',
        *(Path(__file__).with_name(n) for n in ['pair_refit.rs','robustness_remaining_sources.py','robustness_pair_report.py',
            'robustness_centroid_origin.py','robustness_centroid_origin_report.py','robustness_candidate_trace.py','collection_run.py','compare_local_wcs.py'])]
    files.extend(ROOT/'data/samples/sky-samples'/n for n in ['manifest.json','robustness-results.json','collection-wcs-review.json'])
    files.append(ROOT/frame['image'])
    for base in [ROOT/'prototype',pipeline]:
        files.extend([base/'Cargo.toml',base/'Cargo.lock'])
        for name in ['starglyph-core','starglyph-cli','simulator-core']:
            files.extend((base/'crates'/name/'src').rglob('*.rs'));files.append(base/'crates'/name/'Cargo.toml')
    write_json(out/'protocol.json',dict(iteration=28,id=RID,split='development',holdout=False,cases=NAMES,
        manual_diagnostic=True,selected_pair_ids=[p['pair_id'] for p in cases[1]['matches']],
        preserved=['common initial camera','original detector centroids','catalogue identities and vectors','retained pair order',
            'production five-parameter LM with natural k1 prior and existing iteration limits','prior35 scoring sources'],
        criteria=['reproduce31-pair camera: focal<1e-8px, k1<1e-10, rotation norm<1e-10',
            'report all35, outside union31 fits, stricter outside all40 detections, every10percent edge group',
            'same native r8/r12 sensitivity as previous iterations',
            'diagnostic favourable only if >=10percent outside-all40 RMS reduction and no group regression>0.5px for all centroid methods',
            'no solve-rate, acceptance, automatic improvement or ground-truth promotion from manual subset'],
        budget='one identical production LM call per case, no restarts or rematches; executable wall limit60s',
        hashes={str(p.relative_to(ROOT)):digest(p) for p in sorted(set(files))}))


def validate(out):
    p=json.loads((out/'protocol.json').read_text())
    if (p['id'],p['split'],p['holdout'],p['cases'])!=(RID,'development',False,NAMES):
        raise ValueError('frozen development protocol required')
    for name,sha in p['hashes'].items():
        if digest(ROOT/name)!=sha:raise ValueError('frozen input changed: '+name)
    return p


def run(out):
    validate(out)
    if (out/'run.json').exists():raise ValueError('fresh run required')
    command(out,'run',[str(out/'pair-test'),'--exact','solve::pair_refit::fixed_pairs','--ignored'],
        {**os.environ,'STARGLYPH_PAIR_INPUT':str(out/'input.json'),'STARGLYPH_PAIR_OUTPUT':str(out/'fits.json')},60)
    validate(out)


def summarize(out):
    validate(out)
    inputs=json.loads((out/'input.json').read_text())['cases']
    fitted=json.loads((out/'fits.json').read_text())
    if fitted['id']!=RID or fitted['split']!='development' or [c['name'] for c in fitted['cases']]!=NAMES:
        raise ValueError('fit cases changed')
    if not fitted['cases'][0]['reproduced_previous_first_fit']:raise ValueError('control not reproduced')
    sources=next(f['sources'] for f in json.loads(GEOMETRY.read_text())['frames'] if f['id']==RID)
    alternate=next(f['sources'] for f in json.loads((ROOT/'docs/experiments/robustness-iteration-5-geometry.json').read_text())['frames'] if f['id']==RID)
    alternate={s['source_id']:s for s in alternate}
    xy=np.array([s['xy'] for s in sources]);radec=np.array([s['radec'] for s in sources])
    positions=dict(external_centroid=xy,**{f'native_r{r}':np.array([alternate[s['source_id']][f'native_r{r}_xy'] for s in sources]) for r in [8,12]})
    original=json.loads((PRIOR/'replay-plan.json').read_text())['inputs']['control']['detections']
    masks=geometry_groups(xy,[np.array([[d['x'],d['y']] for d in original])],3648,5472,12)
    masks['outside_all40_inputs']=masks.pop('outside_current_inputs')
    masks['outside_union31_fits']=geometry_groups(xy,[np.array([m['xy'] for m in inputs[0]['matches']])],3648,5472,12)['outside_current_inputs']
    results=[]
    for case,inp in zip(fitted['cases'],inputs):
        if case['matches']!=len(inp['matches']) or case['prior_scale']!=1.:raise ValueError('fit objective changed')
        cam=case['camera'];worlds=np.array([s['world'] for s in sources]);rotated=worlds@np.asarray(cam['world_to_camera']).T
        radius2=np.sum((rotated[:,:2]/rotated[:,2,None])**2,axis=1)
        if cam['k1']<0 and np.any(radius2>=-1/(3*cam['k1'])):raise ValueError('non-monotone probe projection')
        predicted=project(cam,radec)
        geometry={centroid:{name:stats(np.linalg.norm(predicted[mask]-pos[mask],axis=1)) if mask.any() else dict(count=0)
            for name,mask in masks.items()} for centroid,pos in positions.items()}
        residual=np.array(case['residuals'])[:-1].reshape(-1,2)
        # Score both pair sets on both cameras; avoid comparing different training sets alone.
        pair_metrics={}
        for subset in inputs:
            w=np.array([m['world'] for m in subset['matches']])
            angles=np.column_stack((np.rad2deg(np.arctan2(w[:,1],w[:,0])),np.rad2deg(np.arcsin(w[:,2]))))
            e=np.linalg.norm(project(cam,angles)-np.array([m['xy'] for m in subset['matches']]),axis=1)
            pair_metrics[subset['name']]=stats(e)
        results.append(dict(name=case['name'],matches=case['matches'],camera=cam,elapsed_ms=case['elapsed_ms'],
            prior_weight=case['prior_weight'],prior_scale=case['prior_scale'],
            reproduced_control=case['reproduced_previous_first_fit'],fit_rms_px=float(np.sqrt(np.mean(np.sum(residual**2,axis=1)))),
            pair_metrics=pair_metrics,conditional_geometry=geometry,sensitivity=sensitivity(case['jacobian'],case['probe_jacobian']),
            sources=[dict(source_id=s['source_id'],hyg_id=s['hyg_id'],xy=s['xy'],predicted_xy=pred.tolist(),
                outside_all40_inputs=bool(masks['outside_all40_inputs'][i]),outside_union31_fits=bool(masks['outside_union31_fits'][i])) for i,(s,pred) in enumerate(zip(sources,predicted))]))
    regressions=[]
    for centroid in positions:
        for name in masks:
            b=results[0]['conditional_geometry'][centroid][name];a=results[1]['conditional_geometry'][centroid][name]
            if b['count'] and a['rms_px']-b['rms_px']>.5:regressions.append(dict(centroid=centroid,group=name,rms_delta_px=a['rms_px']-b['rms_px']))
    ratios={centroid:results[1]['conditional_geometry'][centroid]['outside_all40_inputs']['rms_px']/results[0]['conditional_geometry'][centroid]['outside_all40_inputs']['rms_px'] for centroid in positions}
    write_json(out/'results.json',dict(iteration=28,id=RID,holdout=False,manual_diagnostic=True,
        protocol_sha256=digest(out/'protocol.json'),cases=results,regressions=regressions,outside_all40_rms_ratios=ratios,
        diagnostic_criteria_passed=not regressions and all(r<=.9 for r in ratios.values()),automatic_improvement_confirmed=False,
        independent_ground_truth=False,new_searches=0,refinements=2,rematches=0,
        limitations=['manual visual subset, no automatic source selector','conditional identities and no accepted WCS',
            'same-field development diagnostic; not a solve-rate or negative-acceptance trial']))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['prepare','run','summarize','validate']);parser.add_argument('--out-dir',type=Path,required=True)
    args=parser.parse_args();globals()[args.stage](args.out_dir.resolve())
