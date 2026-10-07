#!/usr/bin/env python3
"""Fixed16 training-only p1/p2 tangent prediction on Orion; no nonlinear fit."""
import argparse
from pathlib import Path
import time

import numpy as np

from collection_run import ROOT, digest, write_json
from compare_local_wcs import project
from robustness_orion_fit import METHODS, WIDTH, HEIGHT, RID, previous, read, working_xy
from robustness_reference_audit import selected
from robustness_tangential_basis import tangential_columns, geometry, PARAMETERS
from robustness_tangent_fit import correction
from robustness_observability import linear_system, RCOND


def fit16(jacobian, residuals, extra=None):
    """Adapt fixed row count only; reuse prior-aware correction and rank checks."""
    j,r=np.asarray(jacobian,float),np.asarray(residuals,float)
    if j.shape!=(33,5) or r.shape!=(33,):
        raise ValueError('fixed16 plus one prior required')
    if extra is not None:
        extra=np.asarray(extra,float)
        if extra.shape!=(32,2):
            raise ValueError('two tangential columns for16 pairs required')
        j=np.column_stack((j,np.vstack((extra,np.zeros((1,2))))))
    started=time.perf_counter()
    delta,info=correction(j[:-1],r[:-1],j[-1],r[-1])
    elapsed_ms=(time.perf_counter()-started)*1000
    data_info,_=linear_system(j,32)
    remaining=r+j@delta
    return dict(delta_parameters=delta.tolist(),information=info,data_information=data_info,
        residuals_working_px=remaining.tolist(),objective=float(remaining@remaining),elapsed_ms=elapsed_ms)


def transfer(before,after):
    if set(before)!=set(after) or any(v['count']!=after[k]['count'] for k,v in before.items()):
        raise ValueError('check group membership changed')
    ratio=after['check31/all_reviewed']['rms_px']/before['check31/all_reviewed']['rms_px']
    regressions={k:after[k]['rms_px']-v['rms_px'] for k,v in before.items()
        if k.startswith('check31/') and v['count'] and after[k]['rms_px']-v['rms_px']>.5}
    return dict(check31_rms_ratio=ratio,regressions=regressions,passed=ratio<=.9 and not regressions)


def prepare(out):
    selected()
    paths=[previous(n) for n in ['40-results','40-checks','43-input','43-fits','43-results','43-checks','44-input','44-checks']]
    for n in ['40-results','43-input','43-fits','43-results','44-input']:
        p=previous(n);checks=read(previous(n[:2]+'-checks'))
        if digest(p)!=checks['artifact_hashes'][str(p.relative_to(ROOT))]:
            raise ValueError('published evidence changed: '+n)
    for name,sha in read(previous('44-checks'))['protected_hashes'].items():
        if digest(ROOT/name)!=sha:raise ValueError('protected data changed')
        paths.append(ROOT/name)
    inp,raw,saved,selection=(read(previous(n)) for n in ['43-input','43-fits','43-results','44-input'])
    if any((p['id'],p['split'])!=(RID,'development') for p in (inp,raw,saved,selection)):
        raise ValueError('development only')
    sources=read(previous('40-results'))['sources'];ids=[s['id'] for s in sources]
    if ids!=selection['source_ids'] or selection['selection']!=saved['selection']:
        raise ValueError('selection changed')
    fit_ids=selection['selection']['fit_ids'];check_ids=selection['selection']['check_ids']
    if len(fit_ids)!=16 or len(check_ids)!=31 or set(fit_ids)&set(check_ids) or set(fit_ids+check_ids)!=set(ids):
        raise ValueError('disjoint16/31 required')
    cases=[]
    for method in METHODS:
        name='replay_'+method
        c=next(c for c in raw['cases'] if c['name']==name)
        pairs=next(c for c in inp['cases'] if c['name']==name)['matches']
        old=next(c for c in saved['cases'] if c['name']==name)
        if {m['source_id'] for m in pairs}!=set(fit_ids) or len(pairs)!=16:
            raise ValueError('fit membership changed')
        for m in pairs:
            source=next(s for s in sources if s['id']==m['source_id'])
            if m['hyg_id']!=source['hyg_id'] or m['world']!=inp['probe_worlds'][ids.index(m['source_id'])] or m['xy']!=working_xy(source['coordinates'][method],c['camera']):
                raise ValueError('pair identity or measurement changed')
        if {m['hyg_id'] for m in pairs}&{s['hyg_id'] for s in sources if s['id'] in check_ids}:
            raise ValueError('check identity overlaps training')
        cam=c['camera'];pred=project(cam,[s['radec'] for s in sources])
        residual=pred-working_xy([s['coordinates'][method] for s in sources],cam)
        fit_indices=[ids.index(m['source_id']) for m in pairs]
        if not np.allclose(residual[fit_indices].ravel(),c['residuals'][:-1],rtol=0,atol=1e-8):
            raise ValueError('pair residual sign or order mismatch')
        if np.shape(c['probe_jacobian'])!=(95,5) or np.any(np.array(c['probe_jacobian'])[-1]!=0):
            raise ValueError('saved47 probe derivative shape changed')
        cases.append(dict(method=method,camera=cam,fit_pairs=pairs,prior_weight=c['prior_weight'],
            jacobian=c['jacobian'],residuals=c['residuals'],probe_jacobian=c['probe_jacobian'][:-1],
            probe_residuals=residual.tolist(),control_delta=old['stationarity']['delta_parameters'],
            fit_tangential_columns=tangential_columns(cam,[m['world'] for m in pairs]).tolist(),
            probe_tangential_columns=tangential_columns(cam,inp['probe_worlds']).tolist()))
    if out.exists():raise ValueError('fresh output required')
    out.mkdir(parents=True)
    write_json(out/'input.json',dict(id=RID,split='development',cases=cases,selection=selection['selection'],
        source_ids=ids,sources=[dict(id=s['id'],hyg_id=s['hyg_id'],name=s['name']) for s in sources],groups=selection['groups']))
    paths += [Path(__file__).resolve(),out/'input.json',*(Path(__file__).with_name(n) for n in [
        'robustness_tangential_basis.py','robustness_tangent_fit.py','robustness_observability.py',
        'robustness_orion_fit.py','robustness_reference_audit.py','collection_run.py','compare_local_wcs.py'])]
    write_json(out/'protocol.json',dict(iteration=45,id=RID,split='development',holdout=False,
        parameter_order=PARAMETERS,methods=list(METHODS),rcond=RCOND,
        model='reuse existing tangential_columns at p1=p2=0; production5 Jacobian plus2 analytic columns; unchanged nonzero k1 prior, zero added prior columns',
        budget='six SVD corrections:3 five-parameter controls+3 seven-parameter variants; no nonlinear fit, LOO, rematch or parameter sweep',
        scoring='unchanged44 groups, all47/fit16/check31; check coordinates and residuals do not enter fit16 correction',
        criteria=['reproduce five-parameter43 stationary delta to1e-10 and fit/probe residual consistency to1e-8px',
            'all three data and augmented seven-parameter systems have full rank7',
            'check31 RMS falls>=10percent vs five-parameter tangent control and no nonempty check31 group grows>0.5 original px, for each of3 centroid methods',
            'report all47 residual vectors and all groups including empty; no source removal',
            'linear prediction is not a physically evaluated camera, nonlinear confirmation and safe projection domain remain required'],
        hashes={str(p.relative_to(ROOT)):digest(p) for p in paths}))


def validate(out):
    p=read(out/'protocol.json')
    if (p['id'],p['split'],p['holdout'],p['methods'])!=(RID,'development',False,list(METHODS)):
        raise ValueError('frozen development required')
    selected()
    for name,sha in p['hashes'].items():
        if digest(ROOT/name)!=sha:raise ValueError('frozen input changed: '+name)
    return p


def measure(out):
    validate(out);inp=read(out/'input.json');rows=[];started=time.monotonic()
    for c in inp['cases']:
        five=fit16(c['jacobian'],c['residuals']);seven=fit16(c['jacobian'],c['residuals'],c['fit_tangential_columns'])
        if not np.allclose(five['delta_parameters'],c['control_delta'],rtol=0,atol=1e-10):
            raise ValueError('five-parameter stationary step changed')
        jp=np.array(c['probe_jacobian']);rp=np.array(c['probe_residuals'])
        scale=np.array([WIDTH/c['camera']['width'],HEIGHT/c['camera']['height']])
        residuals=dict(baseline=rp*scale,five=(rp+(jp@five['delta_parameters']).reshape(-1,2))*scale,
            seven=(rp+(np.column_stack((jp,c['probe_tangential_columns']))@seven['delta_parameters']).reshape(-1,2))*scale)
        fit_indices=[inp['source_ids'].index(m['source_id']) for m in c['fit_pairs']]
        for result,name in [(five,'five'),(seven,'seven')]:
            if not np.allclose(residuals[name][fit_indices],np.array(result['residuals_working_px'][:-1]).reshape(-1,2)*scale,rtol=0,atol=1e-8):
                raise ValueError('fit/probe tangent predictions disagree')
        metrics={name:geometry(r,inp['source_ids'],inp['groups']) for name,r in residuals.items()}
        verdict=transfer(metrics['five'],metrics['seven'])
        full_rank=seven['data_information']['data_rank']==7 and seven['information']['augmented_rank']==7
        rows.append(dict(method=c['method'],five=five,seven=seven,geometry=metrics,transfer=verdict,full_rank=full_rank,
            descriptive_passed=full_rank and verdict['passed'],
            max_prediction_change_original_px=float(np.linalg.norm(residuals['seven']-residuals['five'],axis=1).max()),
            residual_xy_original_px={k:v.tolist() for k,v in residuals.items()}))
    write_json(out/'results.json',dict(iteration=45,id=RID,split='development',holdout=False,
        protocol_sha256=digest(out/'protocol.json'),selection=inp['selection'],sources=inp['sources'],group_ids=inp['groups'],cases=rows,
        controls_reproduced=True,descriptive_transfer_passed=all(c['descriptive_passed'] for c in rows),elapsed_s=time.monotonic()-started,
        linear_systems=6,nonlinear_fits=0,new_solver_calls=0,new_wcs_calls=0,production_changed=False,
        independent_ground_truth=False,automatic_improvement_confirmed=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['prepare','measure','validate']);p.add_argument('--out-dir',type=Path,required=True)
    a=p.parse_args();globals()[a.stage](a.out_dir.resolve())
