#!/usr/bin/env python3
"""Frozen coherent cell shifts: nonlinear fits versus prior local prediction."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np

from collection_run import ROOT, digest, write_json
from robustness_geometry import selected
from robustness_observability import linear_system, validate as validate_prior
from robustness_radial_fit import IDS, radial_sensor_coverage
from robustness_residual_fields import decompose, groups

PRIOR=ROOT/'prototype/artifacts/robustness-iteration-14/fixed'
ORIGINAL=ROOT/'docs/experiments/robustness-iteration-13-input.json'
DELTAS=[[.1,0.],[-.1,0.],[0.,.1],[0.,-.1],[1.,0.],[-1.,0.],[0.,1.],[0.,-1.]]


def cell_selection(frame):
    xy=np.array([s['xy'] for s in frame['sources']]);training=np.array([not s['outside_both_inputs'] for s in frame['sources']])
    w,h=frame['initial']['width'],frame['initial']['height']
    rho,_,_,_=decompose(xy,np.zeros_like(xy),w,h)
    masks=groups(xy,~training,w,h,rho)
    return [dict(name=k,training_indices=np.flatnonzero(m[training]).tolist()) for k,m in masks.items() if k.startswith('cell_')]


def perturbation_vector(count, indices, delta):
    if len(set(indices))!=len(indices) or any(i<0 or i>=count for i in indices):raise ValueError('invalid cell membership')
    result=np.zeros((count,2));result[indices]=delta
    return result.ravel()


def agreement_gate(metrics):
    if metrics['invalid_count'] or not metrics['count']:return False
    return metrics['mismatch_rms_px']<=max(.01,.1*metrics['motion_rms_px'])


def prepare(out):
    selected(IDS);validate_prior(PRIOR)
    old=json.loads((ROOT/'docs/experiments/robustness-iteration-14-results.json').read_text())
    if digest(PRIOR/'jacobians.json')!=old['jacobians_sha256'] or digest(PRIOR/'protocol.json')!=old['protocol_sha256']:
        raise ValueError('prior derivative export changed')
    source=json.loads(ORIGINAL.read_text());fixed=json.loads((PRIOR/'input.json').read_text())
    if source['split']!='development' or [f['id'] for f in source['frames']]!=IDS:raise ValueError('fixed development required')
    for f,g in zip(source['frames'],fixed['frames']):
        if f['id']!=g['id'] or f['sources']!=g['sources']:raise ValueError('prior source identities changed')
        f['cells']=cell_selection(f)
        f['expected']=[{k:c[k] for k in ['name','params','projected_xy']} for c in g['cases']]
    source['deltas']=DELTAS
    if out.exists():raise ValueError('fresh output directory required')
    workspace=out/'pipeline';workspace.mkdir(parents=True)
    for name in ['crates','tetra3']:shutil.copytree(PRIOR/'pipeline'/name,workspace/name)
    for name in ['Cargo.toml','Cargo.lock']:shutil.copyfile(PRIOR/'pipeline'/name,workspace/name)
    parent=workspace/'crates/starglyph-core/src/solve/radial_refit.rs'
    parent.write_text(parent.read_text()+'\n#[path = "radial_perturbation.rs"]\nmod perturbation;\n')
    rust=Path(__file__).with_name('radial_perturbation.rs');shutil.copyfile(rust,parent.with_name(rust.name))
    write_json(out/'input.json',source)
    files=[Path(__file__).resolve(),rust,ORIGINAL,PRIOR/'protocol.json',PRIOR/'input.json',PRIOR/'jacobians.json',
        ROOT/'docs/experiments/robustness-iteration-14-results.json',out/'input.json',ROOT/'data/samples/sky-samples/manifest.json']
    files += [ROOT/'prototype/eval'/n for n in ['collection_run.py','robustness_geometry.py','robustness_observability.py','robustness_radial_fit.py','robustness_residual_fields.py']]
    files += list((workspace/'crates').glob('*/src/**/*.rs'))+list((workspace/'tetra3/src').rglob('*.rs'))
    files += list((workspace/'crates').glob('*/Cargo.toml'))+[workspace/'Cargo.toml',workspace/'Cargo.lock',workspace/'tetra3/Cargo.toml']
    write_json(out/'protocol.json',dict(iteration=15,split='development',ids=IDS,holdout=False,
        hashes={str(p.resolve()):digest(p) for p in files},deltas_original_px=DELTAS,cells='all nine fixed 3x3 cells; empty cells explicitly skipped',
        initial='same original iteration-13 footprint camera for every fit; k2 starts at zero',
        unchanged=['pair identities/order/count','probe coordinates','initial camera and prior weight','30 LM iterations, 10 damping trials','coefficient bounds and monotone guard'],
        criteria=['six zero-shift controls reproduce iteration 13 exactly',
            'for outside-input and every nonempty edge group compare mismatch RMS <= max(0.01 px, 10% actual motion RMS)',
            'report all shifts, parameter deltas, budgets, bound hits, invalid projections, sensor coverage and signed symmetry',
            'no selecting a favorable shift or promoting shifted coordinates as ground truth'],
        interpretation='coordinate sensitivity diagnostic, not an algorithm improvement',wall_timeout_s=240))


def validate(out):
    p=json.loads((out/'protocol.json').read_text())
    if p['split']!='development' or p['holdout'] or p['ids']!=IDS:raise ValueError('fixed development required')
    selected(p['ids'])
    for path,sha in p['hashes'].items():
        if digest(Path(path))!=sha:raise ValueError('frozen input changed: '+path)
    return p


def run(out):
    validate(out);workspace=out/'pipeline'
    cmd=['cargo','test','--offline','--locked','--release','--manifest-path',str(workspace/'Cargo.toml'),'-p','starglyph-core',
         'solve::radial_refit::perturbation::','--','--include-ignored','--nocapture']
    env={**os.environ,'CARGO_TARGET_DIR':str(workspace/'target'),'STARGLYPH_PERTURB_INPUT':str(out/'input.json'),'STARGLYPH_PERTURB_OUTPUT':str(out/'fits.json')}
    started=time.monotonic()
    with (out/'run.log').open('w') as log:
        try:
            proc=subprocess.run(cmd,cwd=ROOT/'prototype',env=env,stdout=log,stderr=subprocess.STDOUT,timeout=240,check=False)
            status=dict(status='completed' if proc.returncode==0 else 'failed',exit_code=proc.returncode)
        except subprocess.TimeoutExpired:status=dict(status='timeout',exit_code=None)
    write_json(out/'run.json',dict(status,command=cmd,elapsed_s=time.monotonic()-started))
    if status['status']!='completed':raise RuntimeError('perturbation run failed: see run.log')


def projection_array(values):
    return np.array([[np.nan,np.nan] if v is None else v for v in values])


def trial_metrics(actual, prediction, errors, mask):
    valid=np.isfinite(actual).all(axis=1)&np.isfinite(errors)
    take=mask&valid
    result=dict(count=int(sum(take)),invalid_count=int(sum(mask&~valid)))
    if take.any():
        norm=lambda a:np.linalg.norm(a[take],axis=1)
        rms=lambda a:float(np.sqrt(np.mean(a**2)))
        motion,linear,mismatch=norm(actual),norm(prediction),norm(actual-prediction)
        result.update(motion_rms_px=rms(motion),linear_rms_px=rms(linear),mismatch_rms_px=rms(mismatch),
                      max_motion_px=float(max(motion)),max_mismatch_px=float(max(mismatch)),geometry_rms_px=rms(errors[take]))
    return result


def report(out):
    validate(out);source=json.loads((out/'input.json').read_text());raw=json.loads((out/'fits.json').read_text())
    derivatives=json.loads((PRIOR/'jacobians.json').read_text())
    if raw['split']!='development' or [f['id'] for f in raw['frames']]!=IDS:raise ValueError('fit selection changed')
    frames=[]
    for f,solved,deriv in zip(source['frames'],raw['frames'],derivatives['frames']):
        xy=np.array([s['xy'] for s in f['sources']]);outside=np.array([s['outside_both_inputs'] for s in f['sources']]);n=int(sum(~outside))
        w,h=f['initial']['width'],f['initial']['height'];rho,_,_,_=decompose(xy,np.zeros_like(xy),w,h)
        masks={k:m for k,m in groups(xy,outside,w,h,rho).items() if k in ['all_reviewed','outside_both_inputs'] or '10pct' in k}
        cells={c['name']:c['training_indices'] for c in f['cells']};cases=[]
        if [c['name'] for c in solved['cases']]!=['k1','k1_k2']:raise ValueError('arms changed')
        for case,jac_case,expected in zip(solved['cases'],deriv['cases'],f['expected']):
            base=case['baseline'];base_xy=projection_array(base['projected_xy'])
            if base['params']!=expected['params'] or base['projected_xy']!=expected['projected_xy']:raise ValueError('zero-shift control differs')
            step=next(s for s in jac_case['steps'] if s['multiplier']==1.)
            _,response=linear_system(step['jacobian'],2*n)
            if response is None:raise ValueError('unobservable prior system')
            probes=np.array(step['probe_jacobian']);trials=[];motions={};expected_order=[]
            for cell,indices in cells.items():
                expected_order += [(cell,tuple(d)) for d in DELTAS] if indices else [(cell,None)]
            if [(t['cell'],tuple(t['delta_xy']) if 'delta_xy' in t else None) for t in case['trials']]!=expected_order:
                raise ValueError('trial selection changed')
            for t in case['trials']:
                if t['status']=='skipped_empty_cell':trials.append(t);continue
                fit=t['fit'];delta=perturbation_vector(n,cells[t['cell']],t['delta_xy'])
                linear_params=response@delta;predicted=(probes@linear_params).reshape(-1,2)
                projected=projection_array(fit['projected_xy']);actual=projected-base_xy
                errors=np.linalg.norm(projected-xy,axis=1);metrics={k:trial_metrics(actual,predicted,errors,m) for k,m in masks.items()}
                failed=[k for k,m in metrics.items() if k!='all_reviewed' and (m['count'] or m['invalid_count']) and not agreement_gate(m)]
                motions[(t['cell'],tuple(t['delta_xy']))]=actual
                trials.append(dict(cell=t['cell'],delta_xy=t['delta_xy'],status=t['status'],training_count=len(cells[t['cell']]),
                    params=fit['params'],linear_parameter_delta=linear_params.tolist(),
                    elapsed_ms=fit['elapsed_ms'],iterations=fit['iterations'],residual_evaluations=fit['residual_evaluations'],
                    stop=fit['stop'],bound_hit=fit['bound_hit'],sensor_coverage=radial_sensor_coverage(fit['camera']),
                    fit_rms_px=fit['fit_rms_px'],objective=fit['objective'],groups=metrics,linear_agreement_failed_groups=failed))
            symmetry=[]
            for (cell,delta),actual in motions.items():
                if max(delta)<=0:continue
                opposite=motions[(cell,tuple(-v for v in delta))]
                value=(actual+opposite)/2
                symmetry.append(dict(cell=cell,positive_delta_xy=list(delta),
                    outside_even_component_rms_px=float(np.sqrt(np.mean(np.sum(value[outside]**2,axis=1)))) if np.isfinite(value[outside]).all() else None))
            cases.append(dict(name=case['name'],baseline={k:v for k,v in base.items() if k!='projected_xy'},trials=trials,signed_symmetry=symmetry))
        frames.append(dict(id=f['id'],training_count=n,probe_count=len(xy),outside_count=int(sum(outside)),
            sources=[{k:s[k] for k in ['source_id','name','xy','outside_both_inputs']} for s in f['sources']],cells=f['cells'],cases=cases))
    result=dict(iteration=15,split='development',protocol_sha256=digest(out/'protocol.json'),fits_sha256=digest(out/'fits.json'),frames=frames,
        interpretation='injected coherent coordinate shifts; retain all outcomes; no measured correction or new ground truth')
    (out/'results.json').write_text(json.dumps(result,separators=(',',':'),allow_nan=False)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['prepare','run','report']);p.add_argument('out',type=Path)
    a=p.parse_args();globals()[a.command](a.out.resolve())
