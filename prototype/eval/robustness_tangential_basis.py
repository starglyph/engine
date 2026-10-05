#!/usr/bin/env python3
"""Frozen fit22/probe19 test of two tangential-distortion columns; no nonlinear fit."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from collection_run import ROOT, digest, write_json
from compare_local_wcs import stats
from robustness_geometry import selected
from robustness_single22_residuals import RID, prior, read
from robustness_tangent_fit import correction, rms
from robustness_observability import linear_system, RCOND

NAMES = [a+'_'+m for m in ('external','native_r8','native_r12') for a in ('original22','coverage22')]
PARAMETERS = ['ra_deg','dec_deg','roll_deg','focal_px','k1','p1','p2']
FORMULA_URL = 'https://docs.opencv.org/4.13.0/d9/d0c/group__calib3d.html'


def tangential_columns(camera, worlds):
    """p1/p2 derivatives at zero, using undistorted image-oriented x/y (y down)."""
    worlds=np.asarray(worlds,float)
    rotation=np.asarray(camera['world_to_camera'],float)
    if worlds.ndim!=2 or worlds.shape[1]!=3 or rotation.shape!=(3,3) or not np.isfinite(worlds).all():
        raise ValueError('finite Nx3 directions and camera rotation required')
    cam=worlds@rotation.T
    if not np.isfinite(cam).all() or np.any(cam[:,2]<=.05):raise ValueError('direction outside camera')
    x=cam[:,0]/cam[:,2];y=-cam[:,1]/cam[:,2];r2=x*x+y*y
    jac=np.empty((len(worlds),2,2))
    jac[:,0,0]=2*x*y;jac[:,0,1]=r2+2*x*x
    jac[:,1,0]=r2+2*y*y;jac[:,1,1]=2*x*y
    jac*=camera['focal_px']
    if not np.isfinite(jac).all():raise ValueError('nonfinite tangential columns')
    return jac.reshape(-1,2)


def fit_step(jac, residuals, extra=None):
    """Only training data and its unchanged k1 prior enter the correction."""
    jac,residuals=np.asarray(jac,float),np.asarray(residuals,float)
    if jac.shape!=(45,5) or residuals.shape!=(45,):raise ValueError('fixed22 plus one prior required')
    if extra is not None:
        extra=np.asarray(extra,float)
        if extra.shape!=(44,2):raise ValueError('two columns for22 pairs required')
        jac=np.column_stack((jac,np.vstack((extra,np.zeros((1,2))))))
    delta,information=correction(jac[:-1],residuals[:-1],jac[-1],residuals[-1])
    # Record data-only conditioning too; correction() includes the prior in scaling.
    data_info,_=linear_system(jac,44)
    remaining=residuals+jac@delta
    return dict(delta_parameters=delta.tolist(),information=information,data_information=data_info,
        fit22_rms_px=rms(remaining[:-1].reshape(-1,2)),fit22_residual_xy_px=remaining[:-1].reshape(-1,2).tolist(),
        prior_residual=float(remaining[-1]),objective=float(remaining@remaining))


def geometry(residuals, source_ids, groups):
    residuals=np.asarray(residuals)
    return {k:stats(np.linalg.norm(residuals[[source_ids.index(i) for i in ids]],axis=1)) if ids else dict(count=0) for k,ids in groups.items()}


def transfer_verdict(before, after):
    ratio=after['strict_unused']['rms_px']/before['strict_unused']['rms_px']
    regressions={k:after[k]['rms_px']-v['rms_px'] for k,v in before.items() if v['count'] and after[k]['rms_px']-v['rms_px']>.5}
    return dict(strict15_rms_ratio=ratio,regressions=regressions,passed=ratio<=.9 and not regressions)


def prepare(out):
    selected([RID])
    if out.exists():raise ValueError('fresh output required')
    checks=read(prior('31-checks'))
    for n in ('input','results','protocol'):
        path=prior('31-'+n)
        if digest(path)!=checks['artifact_hashes'][str(path.relative_to(ROOT))]:raise ValueError('published31 changed')
    for name,sha in checks['protected_hashes'].items():
        if digest(ROOT/name)!=sha:raise ValueError('protected input changed')
    frozen=read(prior('31-input'));old=read(prior('30-input'))
    if frozen['id']!=RID or frozen['split']!='development' or [c['name'] for c in frozen['cases']]!=NAMES:
        raise ValueError('six development cameras required')
    # Original31 preparation pinned these bytes before computing its matrices.
    canonical=(json.dumps(old,indent=2,ensure_ascii=False,allow_nan=False)+'\n').encode()
    expected=read(prior('31-protocol'))['hashes']['prototype/artifacts/robustness-iteration-30/input.json']
    if hashlib.sha256(canonical).hexdigest()!=expected:raise ValueError('original30 input changed')
    probes=next(f['sources'] for f in read(prior('13-geometry'))['frames'] if f['id']==RID)
    if digest(prior('13-geometry'))!=read(prior('31-protocol'))['hashes'][str(prior('13-geometry').relative_to(ROOT))]:raise ValueError('probe worlds changed')
    if [p['source_id'] for p in probes]!=frozen['source_ids']:raise ValueError('probe order changed')
    ids=frozen['groups']['common_unused']
    if len(ids)!=19 or len(frozen['groups']['strict_unused'])!=15:raise ValueError('fixed19/15 required')
    lookup={c['name']:c for c in old['cases']};bundle=[]
    for c in frozen['cases']:
        pairs=lookup[c['name']]['matches']
        if len(pairs)!=22 or any(p['hyg_id'] in {m['hyg_id'] for m in pairs} for p in probes if p['source_id'] in ids):
            raise ValueError('fit22/probe19 disjoint identities required')
        bundle.append(dict(**c,fit_pairs=pairs,fit_tangential_columns=tangential_columns(c['camera'],[p['world'] for p in pairs]).tolist(),
            probe_tangential_columns=tangential_columns(c['camera'],[p['world'] for p in probes]).tolist()))
    out.mkdir(parents=True)
    write_json(out/'input.json',dict(id=RID,split='development',cases=bundle,source_ids=frozen['source_ids'],groups=frozen['groups'],probes=probes))
    paths=[Path(__file__).resolve(),out/'input.json',*(prior(n) for n in ('31-input','31-results','31-protocol','31-checks','30-input','13-geometry')),
        *(Path(__file__).with_name(n) for n in ('robustness_tangent_fit.py','robustness_observability.py','robustness_geometry.py','robustness_single22_residuals.py','collection_run.py','compare_local_wcs.py')),
        *(ROOT/p for p in checks['protected_hashes'])]
    write_json(out/'protocol.json',dict(iteration=34,id=RID,split='development',holdout=False,names=NAMES,parameter_order=PARAMETERS,
        formula_source=FORMULA_URL,coordinates='x=Xc/Zc, y=-Yc/Zc; y down; undistorted normalized camera plane',
        added_basis='dx=f*(2*p1*x*y+p2*(r2+2*x*x)); dy=f*(p1*(r2+2*y*y)+2*p2*x*y)',
        estimation='one linear correction on each frozen22; reuse production5 Jacobian, add analytic p1/p2 at zero, preserve nonzero k1 prior residual; no prior on p1/p2',
        budget='12 small SVD corrections:6 five-parameter controls and6 seven-parameter variants; no nonlinear iterations, LOO, rematches or threshold search',
        scoring='same frozen19/strict15 and all prior spatial groups; neither values nor residuals enter estimation; report all35 with in_fit labels',
        rcond=RCOND,criteria=['five-parameter controls reproduce iteration31 stationary step and geometry to1e-8px',
            'all six data/augmented matrices have full rank7',
            'strict15 RMS decreases at least10percent against five-parameter control and no nonempty group RMS increases more than0.5px in each of six cases',
            'retain both left-edge sources and empty bottom group; descriptive tangent prediction only, not physical camera or automatic improvement'],
        hashes={str(p.relative_to(ROOT)):digest(p) for p in paths}))


def validate(out):
    p=read(out/'protocol.json')
    if (p['id'],p['split'],p['holdout'],p['names'])!=(RID,'development',False,NAMES):raise ValueError('frozen development required')
    selected([RID])
    for name,sha in p['hashes'].items():
        if digest(ROOT/name)!=sha:raise ValueError('frozen input changed: '+name)
    return p


def measure(out):
    validate(out);inp=read(out/'input.json');previous={c['name']:c for c in read(prior('31-results'))['cases']}
    if inp['id']!=RID or inp['split']!='development' or [c['name'] for c in inp['cases']]!=NAMES:raise ValueError('input scope changed')
    rows=[]
    for c in inp['cases']:
        j=np.array(c['jacobian']);r=np.array(c['residuals']);jp=np.array(c['probe_jacobian']);rp=np.array(c['probe_residuals'])
        five=fit_step(j,r);seven=fit_step(j,r,c['fit_tangential_columns'])
        if not np.allclose(five['delta_parameters'],previous[c['name']]['stationarity']['delta_parameters'],rtol=0,atol=1e-8):raise ValueError('control step changed')
        five_r=rp+(jp@five['delta_parameters']).reshape(-1,2)
        seven_j=np.column_stack((jp,c['probe_tangential_columns']))
        seven_r=rp+(seven_j@seven['delta_parameters']).reshape(-1,2)
        metrics={stage:geometry(v,inp['source_ids'],inp['groups']) for stage,v in [('baseline',rp),('five',five_r),('seven',seven_r)]}
        for k,v in metrics['five'].items():
            if v['count'] and abs(v['rms_px']-previous[c['name']]['geometry']['after_training_step'][k]['rms_px'])>1e-8:raise ValueError('control geometry changed')
        verdict=transfer_verdict(metrics['five'],metrics['seven'])
        full_rank=seven['data_information']['data_rank']==7 and seven['information']['augmented_rank']==7
        rows.append(dict(name=c['name'],five=five,seven=seven,geometry=metrics,transfer=verdict,full_rank=full_rank,
            descriptive_passed=full_rank and verdict['passed'],
            max_probe_prediction_change_px=float(np.linalg.norm(seven_r-five_r,axis=1).max()),
            sources=[dict(source_id=p['source_id'],hyg_id=p['hyg_id'],in_fit=p['hyg_id'] in {m['hyg_id'] for m in c['fit_pairs']},
                in_common19=p['source_id'] in inp['groups']['common_unused'],baseline_xy_px=rp[i].tolist(),five_xy_px=five_r[i].tolist(),seven_xy_px=seven_r[i].tolist()) for i,p in enumerate(inp['probes'])]))
    write_json(out/'results.json',dict(iteration=34,id=RID,split='development',holdout=False,protocol_sha256=digest(out/'protocol.json'),cases=rows,
        controls_reproduced=True,descriptive_transfer_passed=all(r['descriptive_passed'] for r in rows),linear_systems=12,
        nonlinear_fits=0,new_searches=0,production_changed=False,automatic_improvement_confirmed=False,independent_ground_truth=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=('prepare','measure','validate'));parser.add_argument('--out-dir',type=Path,required=True)
    args=parser.parse_args();globals()[args.stage](args.out_dir.resolve())
