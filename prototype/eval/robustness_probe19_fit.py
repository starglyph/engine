#!/usr/bin/env python3
"""Check the frozen tangent prediction with the existing production LM on19 probes."""
import argparse
import copy
import os
from pathlib import Path
import shutil

import numpy as np

from collection_run import ROOT, digest, write_json
from compare_local_wcs import project, stats
from robustness_candidate_trace import command
from robustness_coverage_fit import METHODS, ARMS, inputs, validate as validate30
from robustness_single22_residuals import RID, prior, read
from robustness_tangent_fit import validate as validate31
from robustness_geometry import selected

FIT30 = ROOT/'prototype/artifacts/robustness-iteration-30'
LINEAR31 = ROOT/'prototype/artifacts/robustness-iteration-31'
NAMES = [arm+'_'+m for m in METHODS for arm in ARMS]


def build_cases(saved, old_inputs, probes, alternate, ids):
    old = {c['name']:c for c in old_inputs['cases']}
    lookup = {p['source_id']:p for p in probes}
    if len(ids)!=19 or len(set(ids))!=19 or any(i not in lookup for i in ids):
        raise ValueError('frozen19 distinct probes required')
    used = {m['hyg_id'] for name in NAMES for m in old[name]['matches']}
    if any(lookup[i]['hyg_id'] in used for i in ids):
        raise ValueError('probe19 overlaps former22 fits')
    result=[]
    for case in saved['cases']:
        name=case['name']
        if name not in NAMES:continue
        method=next(m for m in METHODS if name.endswith('_'+m))
        control=copy.deepcopy(old[name])
        control.update(name='control_'+name,expected=case['camera'],prior_weight=case['prior_weight'])
        matches=[dict(source_id=i,hyg_id=lookup[i]['hyg_id'],world=lookup[i]['world'],
            xy=lookup[i]['xy'] if method=='external' else alternate[i][method+'_xy']) for i in ids]
        result.extend([control,dict(name='probe19_'+name,initial=case['camera'],matches=matches,
            prior_weight=case['prior_weight'],expected=None)])
    if [c['name'] for c in result]!=[prefix+n for n in NAMES for prefix in ('control_','probe19_')]:
        raise ValueError('six saved starts required')
    for i in range(0,len(NAMES),2):
        a,b=result[2*i+1],result[2*(i+1)+1]
        if a['matches']!=b['matches'] or a['prior_weight']!=b['prior_weight']:
            raise ValueError('different objectives across initial cameras')
    return dict(id=RID,split='development',cases=result,probe_worlds=[p['world'] for p in probes])


def prepare(out):
    previous=validate30(FIT30);validate31(LINEAR31)
    checks=read(prior('31-checks'))
    for name in ('input','results'):
        p=LINEAR31/(name+'.json')
        if digest(p)!=checks['artifact_hashes'][str(p.relative_to(ROOT))]:
            raise ValueError('linear prediction changed')
    _,probes,alternate=inputs()
    ids=read(LINEAR31/'input.json')['groups']['common_unused']
    if ids!=previous['group_probe_ids']['common_unused']:
        raise ValueError('common19 group changed')
    raw=read(FIT30/'fits.json')
    if digest(FIT30/'fits.json')!=read(prior('30-checks'))['artifacts'][str((FIT30/'fits.json').relative_to(ROOT))]:
        raise ValueError('saved nonlinear starts changed')
    inp=build_cases(raw,read(FIT30/'input.json'),probes,alternate,ids)
    if out.exists():raise ValueError('fresh output required')
    out.mkdir(parents=True)
    shutil.copy2(FIT30/'pair-test',out/'pair-test')
    write_json(out/'input.json',inp)
    files=[Path(__file__).resolve(),out/'input.json',out/'pair-test',FIT30/'protocol.json',FIT30/'input.json',FIT30/'fits.json',
        LINEAR31/'protocol.json',LINEAR31/'input.json',LINEAR31/'results.json',
        *(prior(n) for n in ['30-checks','31-checks','13-geometry','5-geometry']),
        *(Path(__file__).with_name(n) for n in ['pair_refit.rs','robustness_coverage_fit.py','robustness_tangent_fit.py',
            'robustness_single22_residuals.py','compare_local_wcs.py','collection_run.py','robustness_candidate_trace.py'])]
    files += [ROOT/'data/samples/sky-samples'/n for n in ['manifest.json','robustness-results.json','collection-wcs-review.json']]
    write_json(out/'protocol.json',dict(iteration=32,id=RID,split='development',holdout=False,manual_diagnostic=True,
        names=NAMES,case_names=[c['name'] for c in inp['cases']],fit_source_ids=ids,
        former_groups=previous['group_probe_ids'],
        budget='one unchanged production5parameter LM call per case;6 exact controls+6 fits; no restarts/rematches; process limit60s',
        prior='explicit saved weight per case, identical across paired starts; same target k1=0',
        criteria=['six original22 controls reproduce saved cameras with existing harness tolerances',
            'linear prediction agrees if vector discrepancy RMS<=0.1px and max<=0.5px on both19 and former22, in all6 cases',
            'paired initial cameras converge if max projection separation on all35<=0.01px for each centroid method',
            'report backward transfer to unfitted former22;19 and all its legacy groups are now in-sample',
            'no nonlinear LOO, additional parameter, automatic improvement, solver acceptance or ground-truth claim'],
        hashes={str(p.relative_to(ROOT)):digest(p) for p in files}))


def validate(out):
    p=read(out/'protocol.json')
    if (p['id'],p['split'],p['holdout'])!=(RID,'development',False):
        raise ValueError('frozen development required')
    selected([RID]);validate30(FIT30)
    for name,sha in p['hashes'].items():
        if digest(ROOT/name)!=sha:raise ValueError('frozen input changed: '+name)
    return p


def run(out):
    validate(out)
    if (out/'run.json').exists():raise ValueError('fresh run required')
    command(out,'run',[str(out/'pair-test'),'--exact','solve::pair_refit::fixed_pairs','--ignored'],
        {**os.environ,'STARGLYPH_PAIR_INPUT':str(out/'input.json'),'STARGLYPH_PAIR_OUTPUT':str(out/'fits.json')},60)
    validate(out)


def residual_stats(vectors):
    return stats(np.linalg.norm(vectors,axis=1))


def pair_residuals(camera, matches):
    worlds=np.array([p['world'] for p in matches])
    angles=np.column_stack((np.rad2deg(np.arctan2(worlds[:,1],worlds[:,0])),np.rad2deg(np.arcsin(worlds[:,2]))))
    return project(camera,angles)-[p['xy'] for p in matches]


def summarize(out):
    protocol=validate(out);fitted=read(out/'fits.json')
    if fitted['id']!=RID or fitted['split']!='development' or [c['name'] for c in fitted['cases']]!=protocol['case_names']:
        raise ValueError('fit scope changed')
    cases={c['name']:c for c in fitted['cases']}
    old={c['name']:c for c in read(FIT30/'fits.json')['cases']}
    linear={c['name']:c for c in read(LINEAR31/'results.json')['cases']}
    old_inputs={c['name']:c for c in read(FIT30/'input.json')['cases']}
    _,probes,alternate=inputs()
    ids=protocol['fit_source_ids'];indices=[next(i for i,p in enumerate(probes) if p['source_id']==s) for s in ids]
    rows=[];predictions={}
    for name in NAMES:
        control=cases['control_'+name];case=cases['probe19_'+name];method=next(m for m in METHODS if name.endswith('_'+m))
        if not control['reproduced_previous_first_fit'] or control['matches']!=22 or case['matches']!=19:
            raise ValueError('control or pair count mismatch')
        if case['prior_weight']!=old[name]['prior_weight']:
            raise ValueError('prior changed')
        cam=case['camera'];worlds=np.array([p['world'] for p in probes]);rotated=worlds@np.array(cam['world_to_camera']).T
        r2=np.sum((rotated[:,:2]/rotated[:,2,None])**2,axis=1)
        if np.any(rotated[:,2]<=0) or (cam['k1']<0 and np.any(r2>=-1/(3*cam['k1']))):
            raise ValueError('probe outside monotone camera domain')
        pred=project(cam,[p['radec'] for p in probes]);predictions[name]=pred
        positions=np.array([p['xy'] if method=='external' else alternate[p['source_id']][method+'_xy'] for p in probes])
        observed=(pred-positions)[indices]
        if not np.allclose(observed.reshape(-1),np.array(case['residuals'])[:-1],rtol=0,atol=1e-8):
            raise ValueError('fit residual order mismatch')
        lr=linear[name]
        if [s['source_id'] for s in lr['sources']]!=ids:raise ValueError('linear19 order changed')
        forecast=np.array([s['all19_in_sample'] for s in lr['sources']])
        back=pair_residuals(cam,old_inputs[name]['matches'])
        oldr=np.array(old[name]['residuals'])[:-1].reshape(-1,2)
        linear_back=oldr+(np.array(old[name]['jacobian'])[:-1]@np.array(lr['all19']['delta_parameters'])).reshape(-1,2)
        agreement=dict(fit19=residual_stats(observed-forecast),former22=residual_stats(back-linear_back))
        rows.append(dict(name=name,camera=cam,initial=case['initial'],prior_weight=case['prior_weight'],prior_scale=case['prior_scale'],
            elapsed_ms=case['elapsed_ms'],control_elapsed_ms=control['elapsed_ms'],control_reproduced=True,
            fit19_rms_px=residual_stats(observed)['rms_px'],linear19_rms_px=residual_stats(forecast)['rms_px'],
            former22=dict(before=residual_stats(oldr),linear=residual_stats(linear_back),nonlinear=residual_stats(back)),
            linear_agreement=agreement,linear_agreement_passed=all(g['rms_px']<=.1 and g['max_px']<=.5 for g in agreement.values()),
            in_sample_groups={k:residual_stats(observed[[i for i,s in enumerate(ids) if s in subset]]) if subset else dict(count=0)
                for k,subset in protocol['former_groups'].items()},
            sources=[dict(source_id=p['source_id'],hyg_id=p['hyg_id'],in_fit=p['source_id'] in ids,
                predicted_xy=xy.tolist(),residual_xy_px=(xy-pos).tolist()) for p,xy,pos in zip(probes,pred,positions)],
            former22_residual_xy_px=back.tolist()))
    convergence={m:residual_stats(predictions[ARMS[0]+'_'+m]-predictions[ARMS[1]+'_'+m]) for m in METHODS}
    write_json(out/'results.json',dict(iteration=32,id=RID,split='development',holdout=False,manual_diagnostic=True,
        protocol_sha256=digest(out/'protocol.json'),cases=rows,paired_start_projection_difference=convergence,
        starts_converged=all(x['max_px']<=.01 for x in convergence.values()),
        linear_agreement_confirmed=all(r['linear_agreement_passed'] for r in rows),new_refinements=12,new_searches=0,
        nonlinear_loo=False,automatic_improvement_confirmed=False,independent_ground_truth=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['prepare','run','summarize','validate']);parser.add_argument('--out-dir',type=Path,required=True)
    args=parser.parse_args();globals()[args.stage](args.out_dir.resolve())
