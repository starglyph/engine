#!/usr/bin/env python3
"""Local information and spatial influence at saved cameras; no refitting."""
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
from robustness_pair_report import sensitivity
from robustness_radial_fit import IDS, validate as validate_previous
from robustness_residual_fields import decompose, groups

PRIOR = ROOT/'prototype/artifacts/robustness-iteration-13/final'
GEOMETRY = ROOT/'docs/experiments/robustness-iteration-13-geometry.json'
RCOND = 1e-12


def prepare(out):
    selected(IDS)
    validate_previous(PRIOR)
    geometry=json.loads(GEOMETRY.read_text())
    if digest(PRIOR/'fits.json')!=geometry['fits_sha256'] or digest(PRIOR/'protocol.json')!=geometry['protocol_sha256']:
        raise ValueError('prior results changed')
    if out.exists():raise ValueError('fresh output directory required')
    workspace=out/'pipeline';workspace.mkdir(parents=True)
    for name in ['crates','tetra3']:
        shutil.copytree(PRIOR/'pipeline'/name,workspace/name)
    for name in ['Cargo.toml','Cargo.lock']:
        shutil.copyfile(PRIOR/'pipeline'/name,workspace/name)
    rust=Path(__file__).with_name('radial_sensitivity.rs')
    parent=workspace/'crates/starglyph-core/src/solve/radial_refit.rs'
    parent.write_text(parent.read_text()+'\n#[path = "radial_sensitivity.rs"]\nmod sensitivity_export;\n')
    shutil.copyfile(rust,parent.with_name(rust.name))
    frames=[]
    for frame in geometry['frames']:
        cases=[]
        for c in frame['cases']:
            camera=c['camera'];keys=['ra_deg','dec_deg','roll_deg','focal_px','k1']+(['k2'] if c['name']=='k1_k2' else [])
            cases.append(dict(name=c['name'],params=[camera[k] for k in keys],prior_weight=c['prior_weight'],
                              projected_xy=c['projected_xy'],fit_residuals=c['fit_residuals']))
        frames.append(dict(id=frame['id'],width=camera['width'],height=camera['height'],sources=frame['sources'],cases=cases))
    if [f['id'] for f in frames]!=IDS:raise ValueError('fixed selection changed')
    write_json(out/'input.json',dict(split='development',frames=frames))
    files=[Path(__file__).resolve(),rust,GEOMETRY,PRIOR/'protocol.json',PRIOR/'fits.json',out/'input.json',
           ROOT/'data/samples/sky-samples/manifest.json']
    files += [ROOT/'prototype/eval'/name for name in ['collection_run.py','robustness_pair_report.py',
        'robustness_geometry.py','robustness_residual_fields.py','robustness_radial_fit.py']]
    files += list((workspace/'crates').glob('*/src/**/*.rs'))+list((workspace/'tetra3/src').rglob('*.rs'))
    files += list((workspace/'crates').glob('*/Cargo.toml'))+[workspace/'Cargo.toml',workspace/'Cargo.lock',workspace/'tetra3/Cargo.toml']
    write_json(out/'protocol.json',dict(iteration=14,split='development',ids=IDS,holdout=False,
        hashes={str(p.resolve()):digest(p) for p in files},step_multipliers=[.5,1.,2.],rcond=RCOND,
        training='same 17/26/19 pairs; outside 18/14/23 are probes only',spatial_partition='unchanged image-coordinate 3x3 cells',
        model='local Gauss-Newton response to unit independent coordinate perturbations; fixed prior; not calibrated uncertainty',
        criteria=['reproduce all saved residuals exactly and projections to 1e-10 px',
            'five-parameter sensitivity agrees with existing one-prior helper',
            'report column-normalized data and augmented ranks/condition, parameter response correlations and probe gains',
            'retain every cell and empty edge; sum cell noise energies to total; no selecting sources by influence',
            'report half/double finite-difference sensitivity; no fit, deletion refit or camera selection'],wall_timeout_s=240))


def validate(out):
    p=json.loads((out/'protocol.json').read_text())
    if p['split']!='development' or p['holdout'] or p['ids']!=IDS:raise ValueError('fixed development required')
    selected(p['ids'])
    for path,sha in p['hashes'].items():
        if digest(Path(path))!=sha:raise ValueError('frozen input changed: '+path)
    return p


def run(out):
    validate(out)
    workspace=out/'pipeline'
    command=['cargo','test','--offline','--locked','--release','--manifest-path',str(workspace/'Cargo.toml'),
             '-p','starglyph-core','solve::radial_refit::sensitivity_export::export','--','--ignored','--nocapture']
    env={**os.environ,'CARGO_TARGET_DIR':str(workspace/'target'),'STARGLYPH_SENSITIVITY_INPUT':str(out/'input.json'),
         'STARGLYPH_SENSITIVITY_OUTPUT':str(out/'jacobians.json')}
    start=time.monotonic()
    with (out/'run.log').open('w') as log:
        try:
            proc=subprocess.run(command,cwd=ROOT/'prototype',env=env,stdout=log,stderr=subprocess.STDOUT,timeout=240,check=False)
            status=dict(status='completed' if proc.returncode==0 else 'failed',exit_code=proc.returncode)
        except subprocess.TimeoutExpired:status=dict(status='timeout',exit_code=None)
    write_json(out/'run.json',dict(status,command=command,elapsed_s=time.monotonic()-start))
    if status['status']!='completed':raise RuntimeError('derivative export failed: see run.log')


def linear_system(jac, data_rows):
    """Generalize the existing one-prior response to any fixed prior rows."""
    jac=np.asarray(jac,dtype=float)
    if jac.ndim!=2 or not 0<data_rows<=len(jac) or not np.isfinite(jac).all():raise ValueError('invalid Jacobian')
    norms=np.linalg.norm(jac[:data_rows],axis=0)
    if np.any(norms==0):raise ValueError('unobserved parameter')
    scaled=jac/norms
    sd=np.linalg.svd(scaled[:data_rows],compute_uv=False)
    sa=np.linalg.svd(scaled,compute_uv=False)
    dim=jac.shape[1]
    rank=lambda s:int(sum(s>s[0]*RCOND))
    info=dict(data_rank=rank(sd),augmented_rank=rank(sa),parameters=dim,
        data_condition=float(sd[0]/sd[-1]) if rank(sd)==dim else None,
        augmented_condition=float(sa[0]/sa[-1]) if rank(sa)==dim else None,
        normalized_data_singular_values=sd.tolist())
    if rank(sa)<dim:return info,None
    response=np.linalg.pinv(scaled,rcond=RCOND)[:,:data_rows]/norms[:,None]
    return info,response


def propagation_summary(propagation, mask):
    selected_values=propagation.reshape(-1,2,propagation.shape[1])[mask]
    if not len(selected_values):return dict(count=0)
    factors=np.sqrt(np.sum(selected_values**2,axis=(1,2)))
    return dict(count=len(factors),rms_gain=float(np.sqrt(np.mean(factors**2))),max_gain=float(np.max(factors)))


def spatial_influence(propagation, probe_mask, cells):
    """Partition coordinate-noise energy, not causal error or removal benefit."""
    values=propagation.reshape(-1,2,propagation.shape[1])[probe_mask]
    total=float(np.sum(values**2))
    result={}
    for name,mask in cells.items():
        energy=float(np.sum(values[:,:,np.repeat(mask,2)]**2))
        result[name]=dict(training_count=int(sum(mask)),noise_energy_share=energy/total if total else None)
    return result


def analyze(jac, probes, training_count, masks, cells):
    jac,probes=np.asarray(jac),np.asarray(probes)
    info,response=linear_system(jac,2*training_count)
    if response is None:return dict(information=info,observable=False)
    propagation=probes@response
    gain=np.linalg.norm(response,axis=1)
    correlation=(response@response.T)/(gain[:,None]*gain[None,:])
    info.update(parameter_gain_per_coordinate_sigma=gain.tolist(),parameter_response_correlation=correlation.tolist())
    return dict(information=info,observable=True,probe_gain=np.sqrt(np.sum(propagation.reshape(-1,2,response.shape[1])**2,axis=(1,2))).tolist(),
        groups={k:propagation_summary(propagation,m) for k,m in masks.items()},
        spatial_noise_shares={k:spatial_influence(propagation,m,cells) for k,m in masks.items() if k=='outside_both_inputs' or '10pct' in k})


def report(out):
    validate(out)
    source=json.loads((out/'input.json').read_text());raw=json.loads((out/'jacobians.json').read_text())
    if raw['split']!='development' or [f['id'] for f in raw['frames']]!=IDS:raise ValueError('export selection changed')
    frames=[]
    for f,exported in zip(source['frames'],raw['frames']):
        xy=np.array([s['xy'] for s in f['sources']]);outside=np.array([s['outside_both_inputs'] for s in f['sources']]);training=~outside
        rho,_,_,_=decompose(xy,np.zeros_like(xy),f['width'],f['height'])
        masks=groups(xy,outside,f['width'],f['height'],rho)
        cells={k:m[training] for k,m in masks.items() if k.startswith('cell_')}
        cases=[]
        if [c['name'] for c in exported['cases']]!=['k1','k1_k2']:raise ValueError('arms changed')
        for c in exported['cases']:
            if not c['fit_residuals_exact'] or c['projection_max_abs_px']>=1e-10:raise ValueError('camera replay differs')
            steps=[]
            for step in c['steps']:
                jac=np.array(step['jacobian']);probes=np.array(step['probe_jacobian'])
                if jac.shape!=(2*int(sum(training))+c['prior_rows'],len(c['params'])) or probes.shape!=(2*len(xy),len(c['params'])):
                    raise ValueError('Jacobian shape changed')
                result=analyze(jac,probes,int(sum(training)),masks,cells)
                if c['name']=='k1':
                    old=sensitivity(jac,np.vstack((probes,np.zeros((1,5)))))
                    np.testing.assert_allclose(result['probe_gain'],old['probe_amplification'],rtol=1e-8,atol=1e-10)
                    np.testing.assert_allclose(result['information']['data_condition'],old['column_normalized_data_condition'],rtol=1e-10)
                steps.append(dict(multiplier=step['multiplier'],**result))
            primary=next(s for s in steps if s['multiplier']==1.)
            differences=[]
            for step in steps:
                if not step['observable'] or not primary['observable']:raise ValueError('unobservable full system')
                differences.append(float(np.max(np.abs(np.array(step['probe_gain'])/primary['probe_gain']-1))))
            cases.append(dict(name=c['name'],params=c['params'],prior_rows=c['prior_rows'],projection_max_abs_px=c['projection_max_abs_px'],
                primary=primary,step_checks=[dict(multiplier=s['multiplier'],data_condition=s['information']['data_condition'],
                    outside_rms_gain=s['groups']['outside_both_inputs']['rms_gain'],max_probe_gain_relative_change=d)
                    for s,d in zip(steps,differences)]))
        frames.append(dict(id=f['id'],sources=[{k:s[k] for k in ['source_id','name','xy','outside_both_inputs']} for s in f['sources']],
                           training_cells={k:int(sum(v)) for k,v in cells.items()},cases=cases))
    write_json(out/'results.json',dict(iteration=14,split='development',protocol_sha256=digest(out/'protocol.json'),
        jacobians_sha256=digest(out/'jacobians.json'),derivative_elapsed_s=raw['elapsed_s'],frames=frames,
        interpretation='local Gauss-Newton influence only; no noise calibration, nonlinear refit, deletion or ground-truth certification'))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['prepare','run','report']);p.add_argument('out',type=Path)
    a=p.parse_args();globals()[a.command](a.out.resolve())
