#!/usr/bin/env python3
"""Current-camera tangent-space diagnostics on frozen development correspondences."""
import argparse
from pathlib import Path

import numpy as np

from collection_run import ROOT, digest, write_json
from compare_local_wcs import project, stats
from robustness_coverage_fit import METHODS, ARMS, inputs, validate as validate_coverage
from robustness_single22_residuals import RID, prior, read
from robustness_geometry import selected
from robustness_observability import linear_system, RCOND

OLD = ROOT/'prototype/artifacts/robustness-iteration-30'


def correction(jac, residual, prior_row, prior_residual):
    """Minimize ||r+J delta|| including the unchanged soft prior, using existing SVD."""
    jac,residual = np.asarray(jac,float),np.asarray(residual,float)
    if jac.ndim!=2 or residual.shape!=(len(jac),) or not np.isfinite(residual).all():
        raise ValueError('invalid linear residual system')
    augmented = np.vstack((jac,prior_row))
    target = np.r_[residual,prior_residual]
    if not np.isfinite(target).all():raise ValueError('non-finite prior residual')
    # Include the prior column of the response: its residual is not generally zero.
    info,response = linear_system(augmented,len(augmented))
    if response is None:
        raise ValueError('rank deficient augmented tangent system')
    delta = -response@target
    return delta,info


def leave_one_probe_out(jac, residual, prior_row, prior_residual):
    jac,residual = np.asarray(jac,float),np.asarray(residual,float)
    if residual.ndim!=2 or residual.shape[1]!=2 or jac.shape!=(2*len(residual),5):
        raise ValueError('paired five-parameter probe system required')
    rows=[]
    for i in range(len(residual)):
        keep=np.repeat(np.arange(len(residual))!=i,2)
        delta,info=correction(jac[keep],residual.reshape(-1)[keep],prior_row,prior_residual)
        change=jac[2*i:2*i+2]@delta
        rows.append(dict(excluded_index=i,delta_parameters=delta.tolist(),information=info,
            predicted_change_xy_px=change.tolist(),residual_xy_px=(residual[i]+change).tolist()))
    return rows


def prepare(out):
    previous=validate_coverage(OLD)
    _,probes,alternate=inputs()
    raw=read(OLD/'fits.json')
    published=read(prior('30-results'))
    checks=read(prior('30-checks'))
    for path in (OLD/'fits.json',OLD/'input.json',OLD/'results.json'):
        if digest(path)!=checks['artifacts'][str(path.relative_to(ROOT))]:
            raise ValueError('saved fit30 export changed')
    names=[arm+'_'+m for m in METHODS for arm in ARMS]
    cases=[c for c in raw['cases'] if c['name'] in names]
    if raw['id']!=RID or raw['split']!='development' or [c['name'] for c in cases]!=names:
        raise ValueError('six frozen development fits required')
    scoring=previous['group_probe_ids']
    if len(scoring['common_unused'])!=19 or len(scoring['strict_unused'])!=15:
        raise ValueError('unchanged19/15 probe groups required')
    case_inputs={c['name']:c for c in read(OLD/'input.json')['cases']}
    old_results={c['name']:c for c in published['cases']}
    bundle=[]
    for case in cases:
        name=case['name'];cam=case['camera']
        method=next(m for m in METHODS if name.endswith('_'+m))
        measured=np.array([p['xy'] if method=='external' else alternate[p['source_id']][method+'_xy'] for p in probes])
        prediction=project(cam,[p['radec'] for p in probes])
        if not np.array_equal(prediction,[s['predicted_xy'] for s in old_results[name]['sources']]):
            raise ValueError('prior projections not reproduced')
        jac=np.array(case['jacobian']);pj=np.array(case['probe_jacobian']);res=np.array(case['residuals'])
        if jac.shape!=(45,5) or pj.shape!=(71,5) or res.shape!=(45,) or not np.allclose(pj[-1],0,rtol=0,atol=0):
            raise ValueError('unexpected production derivative shape')
        pairs=case_inputs[name]['matches'];world=np.array([p['world'] for p in pairs])
        angles=np.column_stack((np.rad2deg(np.arctan2(world[:,1],world[:,0])),np.rad2deg(np.arcsin(world[:,2]))))
        pair_res=project(cam,angles)-[p['xy'] for p in pairs]
        if not np.allclose(pair_res.reshape(-1),res[:-1],rtol=0,atol=1e-8):
            raise ValueError('production residual sign or order mismatch')
        bundle.append(dict(name=name,method=method,camera=cam,jacobian=case['jacobian'],residuals=case['residuals'],
            probe_jacobian=pj[:-1].tolist(),probe_residuals=(prediction-measured).tolist()))
    if out.exists():raise ValueError('fresh output required')
    out.mkdir(parents=True)
    write_json(out/'input.json',dict(id=RID,split='development',cases=bundle,source_ids=[p['source_id'] for p in probes],groups=scoring))
    files=[Path(__file__).resolve(),out/'input.json',OLD/'fits.json',OLD/'protocol.json',OLD/'input.json',
        *(prior(n) for n in ['30-checks','30-results','30-protocol','13-geometry','5-geometry']),
        *(Path(__file__).with_name(n) for n in ['robustness_observability.py','robustness_coverage_fit.py','robustness_single22_residuals.py',
            'collection_run.py','compare_local_wcs.py'])]
    files += [ROOT/'data/samples/sky-samples'/n for n in ['manifest.json','robustness-results.json','collection-wcs-review.json']]
    write_json(out/'protocol.json',dict(iteration=31,id=RID,split='development',holdout=False,case_names=names,
        hashes={str(p.relative_to(ROOT)):digest(p) for p in files},rcond=RCOND,
        parameter_order=['ra_deg','dec_deg','roll_deg','focal_px','k1'],
        analyses=['one Gauss-Newton step on original22 residuals with original nonzero prior residual',
            'optimistic in-sample tangent correction on common19; report transfer back to original22',
            '19 leave-one-probe-out corrections on18 common probes; evaluate excluded x/y together'],
        criteria=['stationary diagnostic if additional original22 RMS gain<0.01px in every case',
            'LOO descriptive transfer passes only with>=10percent strict15 RMS decrease and no group RMS increase>0.5px in all6 cases',
            'LOO metrics pool19 different linear corrections, never claim one improved physical camera',
            'no nonlinear extrapolation, new parameter, rematch, acceptance change or ground-truth promotion'],
        limitations=['local tangent space only, not global model adequacy','prior camera trained on different22; common19 excluded from all previous inputs to fit30',
            'same-field conditional identities; no collection holdout'],nonlinear_fits=0,new_searches=0))


def validate(out):
    p=read(out/'protocol.json')
    if (p['id'],p['split'],p['holdout'])!=(RID,'development',False):
        raise ValueError('frozen development protocol required')
    selected([RID])
    for name,sha in p['hashes'].items():
        if digest(ROOT/name)!=sha:raise ValueError('frozen input changed: '+name)
    return p


def rms(vectors):
    return float(np.sqrt(np.mean(np.sum(np.asarray(vectors)**2,axis=1))))


def measure(out):
    protocol=validate(out);inp=read(out/'input.json')
    if inp['id']!=RID or inp['split']!='development' or [c['name'] for c in inp['cases']]!=protocol['case_names']:
        raise ValueError('input scope changed')
    source_ids=inp['source_ids'];common=inp['groups']['common_unused']
    indices=np.array([source_ids.index(i) for i in common])
    masks={k:np.array([i in ids for i in common]) for k,ids in inp['groups'].items()}
    rows=[]
    for case in inp['cases']:
        j=np.array(case['jacobian']);r=np.array(case['residuals']);jp=np.array(case['probe_jacobian'])
        rp=np.array(case['probe_residuals']);prior_row=j[-1];prior_res=r[-1]
        delta,info=correction(j[:-1],r[:-1],prior_row,prior_res)
        remaining=r+j@delta
        train=r[:-1].reshape(-1,2)
        stationarity=dict(delta_parameters=delta.tolist(),information=info,baseline_rms_px=rms(train),
            linear_rms_px=rms(remaining[:-1].reshape(-1,2)),
            objective_before=float(r@r),objective_after=float(remaining@remaining),
            explainable_objective_fraction=float(1-(remaining@remaining)/(r@r)),
            max_probe_change_px=float(np.linalg.norm((jp@delta).reshape(-1,2),axis=1).max()))
        pjac=jp.reshape(-1,2,5)[indices].reshape(-1,5);pres=rp[indices]
        step,full_info=correction(pjac,pres.reshape(-1),prior_row,prior_res)
        full=(pres.reshape(-1)+pjac@step).reshape(-1,2)
        loo=leave_one_probe_out(pjac,pres,prior_row,prior_res)
        transferred=np.array([x['residual_xy_px'] for x in loo])
        systems=dict(baseline=pres,after_training_step=(rp+(jp@delta).reshape(-1,2))[indices],
                     all19_in_sample=full,leave_one_probe_out=transferred)
        geometry={stage:{k:stats(np.linalg.norm(v[m],axis=1)) if m.any() else dict(count=0) for k,m in masks.items()} for stage,v in systems.items()}
        baseline=geometry['baseline'];after=geometry['leave_one_probe_out']
        ratio=after['strict_unused']['rms_px']/baseline['strict_unused']['rms_px']
        regressions={k:after[k]['rms_px']-baseline[k]['rms_px'] for k in masks if baseline[k]['count'] and after[k]['rms_px']-baseline[k]['rms_px']>.5}
        rows.append(dict(name=case['name'],stationarity=stationarity,geometry=geometry,
            all19=dict(delta_parameters=step.tolist(),information=full_info,
                transferred_training22_rms_px=rms((r[:-1]+j[:-1]@step).reshape(-1,2))),
            leave_one_probe_out=[dict(**entry,source_id=common[i]) for i,entry in enumerate(loo)],
            sources=[dict(source_id=s,**{stage:v[i].tolist() for stage,v in systems.items()}) for i,s in enumerate(common)],
            strict_loo_rms_ratio=ratio,regressions=regressions,descriptive_transfer_passed=ratio<=.9 and not regressions))
    write_json(out/'results.json',dict(iteration=31,id=RID,split='development',holdout=False,
        protocol_sha256=digest(out/'protocol.json'),cases=rows,
        all_stationary=all(r['stationarity']['baseline_rms_px']-r['stationarity']['linear_rms_px']<.01 for r in rows),
        descriptive_transfer_passed=all(r['descriptive_transfer_passed'] for r in rows),
        nonlinear_fits=0,new_searches=0,automatic_improvement_confirmed=False,independent_ground_truth=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['prepare','measure','validate']);parser.add_argument('--out-dir',type=Path,required=True)
    args=parser.parse_args();globals()[args.stage](args.out_dir.resolve())
