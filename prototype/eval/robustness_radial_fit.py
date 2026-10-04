#!/usr/bin/env python3
"""Frozen k1 versus k1+k2 diagnostic on existing development correspondences."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np

from collection_run import ROOT, digest, write_json
from compare_local_wcs import project, stats
from robustness_geometry import report_path, selected
from robustness_residual_fields import GEOMETRY, REVIEW, decompose, groups
from tetra3_footprint import prepare as prepare_pipeline

IDS = ['wm_r_132162731', 'wm_r_143159342', 'wm_r_149276071']
PREVIOUS = ROOT/'docs/experiments/robustness-iteration-12-protocol.json'


def optimizer_source(source):
    """Keep the production LM update/stop rules; extend only its parameter vector."""
    start = source.index('fn refine_pose_with_scale(')
    end = source.index('\nfn residuals(', start)
    text = source[start:end]
    replacements = [
        ('fn refine_pose_with_scale(', 'fn fit('),
        ('    scale: f64,', '    extra: bool,\n    budget: &Budget,'),
        (') -> CameraSolution {', ') -> Vec<f64> {'),
        ('let dim = if free_k1 { 5 } else { 4 };', 'let dim = if extra { 6 } else if free_k1 { 5 } else { 4 };'),
        ('k1_reg_weight(initial.fov_x_deg()) * scale', 'k1_reg_weight(initial.fov_x_deg())'),
        ('    let width = initial.width;', '    if extra { params.push(0.); }\n    let width = initial.width;'),
        ('let eval = |p: &[f64]| residuals(p, free_k1, k1_weight, width, height, matches).norm_squared();',
         'let eval = |p: &[f64]| { budget.evaluations.set(budget.evaluations.get()+1);\n'
         '        residuals(p, free_k1, k1_weight, width, height, matches).norm_squared() };'),
        ('    for _ in 0..30 {', '    budget.stop.set("iteration_limit");\n    for _ in 0..30 {\n'
         '        budget.iterations.set(budget.iterations.get()+1);\n'
         '        budget.evaluations.set(budget.evaluations.get()+1+dim);'),
        ('            let new_cost = eval(&trial);',
         '            if extra { trial[5] = trial[5].clamp(-0.5, 0.5); }\n            let new_cost = eval(&trial);'),
        ('                    return to_solution(&params, free_k1, initial);',
         '                    budget.stop.set("small_improvement");\n                    return params;'),
        ('        if !improved {\n            break;',
         '        if !improved {\n            budget.stop.set("no_accepted_step");\n            break;'),
        ('    to_solution(&params, free_k1, initial)', '    params'),
    ]
    for before, after in replacements:
        if text.count(before) != 1:
            raise ValueError('production optimizer changed: '+before)
        text = text.replace(before, after)
    jac = source[source.index('fn numeric_jacobian('):source.index('\nfn to_solution(')]
    return '// Generated from frozen production solve.rs; do not hand edit.\n'+text+'\n'+jac


def build_frames(geometry, reviews, reports):
    ids = [f['id'] for f in geometry['frames']]
    selected(ids)  # Check split before opening any report/image in callers.
    if ids != IDS or geometry['split'] != 'development':
        raise ValueError('fixed development selection required')
    frames = []
    for frame in geometry['frames']:
        rid = frame['id']
        reviewed = next(f for f in reviews['frames'] if f['id'] == rid)
        by_id = {p['id']:p for p in reviewed['points']}
        sources = []
        for s in frame['sources']:
            p = by_id[s['source_id']]
            if p['hyg_id'] != s['hyg_id']:
                raise ValueError('source identity changed')
            ra, dec = np.deg2rad([p['ra_deg'], p['dec_deg']])
            sources.append(dict(source_id=s['source_id'], hyg_id=s['hyg_id'], name=s['name'],
                xy=s['xy'], world=[float(np.cos(dec)*np.cos(ra)), float(np.cos(dec)*np.sin(ra)), float(np.sin(dec))],
                radec=[p['ra_deg'], p['dec_deg']], outside_both_inputs=s['outside_both_inputs']))
        frames.append(dict(id=rid, initial=reports[rid]['camera'], sources=sources))
    if [sum(not s['outside_both_inputs'] for s in f['sources']) for f in frames] != [17,26,19]:
        raise ValueError('frozen training membership changed')
    return frames


def prepare(out, registry):
    records = selected(IDS)
    if out.exists():
        raise ValueError('fresh output directory required')
    previous = json.loads(PREVIOUS.read_text())
    for path, sha in previous['hashes'].items():
        if digest(ROOT/path) != sha:
            raise ValueError('previous frozen input changed: '+path)
    geometry = json.loads(GEOMETRY.read_text())
    reports = {rid:json.loads(report_path(rid, 'footprint').read_text()) for rid in IDS}
    for r in records:
        report = reports[r['id']]
        if report['source_sha256'] != r['clean_sha256'] or [report['width'],report['height']] != [r['width'],r['height']]:
            raise ValueError('saved image identity changed')
    frames = build_frames(geometry, json.loads(REVIEW.read_text()), reports)
    prepare_pipeline(out, registry)
    workspace = out/'pipeline'
    # This diagnostic never invokes tetra3; undo the helper's footprint patch.
    for name in ('solve.rs','track.rs'):
        shutil.copyfile(registry/'src/solver'/name, workspace/'tetra3/src/solver'/name)
    manifest = workspace/'Cargo.toml'
    text = manifest.read_text()
    if text.count('serde_json = "1"') != 1:
        raise ValueError('serde declaration changed')
    manifest.write_text(text.replace('serde_json = "1"', 'serde_json = { version = "1", features = ["float_roundtrip"] }'))
    solve = workspace/'crates/starglyph-core/src/solve.rs'
    generated = solve.parent/'solve/radial_optimizer.rs'
    generated.write_text(optimizer_source(solve.read_text()))
    solve.write_text(solve.read_text()+'\n#[cfg(test)]\nmod radial_refit;\n')
    rust = ROOT/'prototype/eval/radial_refit.rs'
    shutil.copyfile(rust, solve.parent/'solve/radial_refit.rs')
    subprocess.run(['rustfmt','--edition','2021',str(generated)], cwd=ROOT/'prototype',check=True)
    with (out/'lock-resolution.log').open('w') as log:
        subprocess.run(['cargo','update','--offline','--workspace','--manifest-path',str(manifest)],
                       cwd=ROOT/'prototype',stdout=log,stderr=subprocess.STDOUT,check=True)
    write_json(out/'input.json',dict(split='development',frames=frames))
    files = [Path(__file__).resolve(),rust,GEOMETRY,REVIEW,PREVIOUS,out/'input.json',
             ROOT/'prototype/Cargo.lock',ROOT/'data/samples/sky-samples/manifest.json']
    files += [ROOT/'prototype/eval'/n for n in ('collection_run.py','compare_local_wcs.py',
        'robustness_geometry.py','robustness_residual_fields.py','tetra3_footprint.py')]
    files += [report_path(rid,'footprint') for rid in IDS]
    files += list((workspace/'crates').glob('*/src/**/*.rs'))+list((workspace/'tetra3/src').rglob('*.rs'))
    files += list((workspace/'crates').glob('*/Cargo.toml'))+[manifest,workspace/'Cargo.lock',workspace/'tetra3/Cargo.toml']
    write_json(out/'protocol.json',dict(iteration=13,ids=IDS,split='development',holdout=False,
        hashes={str(p.resolve()):digest(p) for p in files},
        arms=['k1','k1_k2'],model='r_distorted = r * (1 + k1*r^2 + k2*r^4); r in tangent-plane units',
        initial='same saved footprint camera in original pixels; k2=0',
        training='all previously reviewed sources not outside_both_inputs; unchanged xy and identities',
        training_counts=[17,26,19],evaluation_counts=[18,14,23],
        optimizer=dict(max_iterations=30,max_damping_trials=10,parameter_bounds=[-.5,.5],
            k1_prior='production natural weight from initial FOV, scale=1',k2_prior='same residual weight as k1',
            finite_difference='unchanged production steps including 1e-4 for k2',wall_timeout_s=240),
        criteria=['five-parameter control exactly equals production refine_pose_with_scale',
                  'primary wm_r_143159342 outside-input RMS must decrease by at least 10%',
                  'no outside-input RMS regression above 0.5 original px on either other frame',
                  'no nonempty 10-percent edge RMS regression above 0.5 original px on any frame',
                  'all 117 sources remain projectable; no coefficient bound hit',
                  'report every source, empty groups, full-sensor monotone coverage and fit budget'],
        interpretation='manual conditional fixed-pair model diagnostic; no rematching or automatic acceptance'))


def validate(out):
    p = json.loads((out/'protocol.json').read_text())
    if p['split'] != 'development' or p['holdout'] or p['ids'] != IDS:
        raise ValueError('fixed development selection required')
    selected(p['ids'])
    for path, sha in p['hashes'].items():
        if digest(Path(path)) != sha:
            raise ValueError('frozen input changed: '+path)
    return p


def run(out):
    validate(out)
    workspace = out/'pipeline'
    command = ['cargo','test','--offline','--locked','--release','--manifest-path',str(workspace/'Cargo.toml'),
        '-p','starglyph-core','solve::radial_refit::','--','--include-ignored','--nocapture']
    env = {**os.environ,'CARGO_TARGET_DIR':str(workspace/'target'),
           'STARGLYPH_RADIAL_INPUT':str(out/'input.json'),'STARGLYPH_RADIAL_OUTPUT':str(out/'fits.json')}
    started = time.monotonic()
    with (out/'run.log').open('w') as log:
        try:
            proc = subprocess.run(command,cwd=ROOT/'prototype',env=env,stdout=log,stderr=subprocess.STDOUT,timeout=240,check=False)
            status = dict(status='completed' if proc.returncode==0 else 'failed',exit_code=proc.returncode)
        except subprocess.TimeoutExpired:
            status = dict(status='timeout',exit_code=None)
    status.update(command=command,elapsed_s=time.monotonic()-started)
    write_json(out/'run.json',status)
    if status['status'] != 'completed':
        raise RuntimeError('diagnostic failed: see run.log')


def project_extended(camera, radec):
    base = project(camera,radec)
    ra,dec = np.deg2rad(np.asarray(radec)).T
    world = np.column_stack((np.cos(dec)*np.cos(ra),np.cos(dec)*np.sin(ra),np.sin(dec)))
    cam = world@np.asarray(camera['world_to_camera']).T
    uv = cam[:,:2]/cam[:,2,None]
    r2 = np.sum(uv**2,axis=1)
    return base+camera['focal_px']*uv*np.array([1.,-1.])*camera['k2']*r2[:,None]**2


def radial_sensor_coverage(camera):
    """Can the central monotone branch reach the farthest sensor corner?"""
    k1,k2 = camera['k1'],camera['k2']
    roots = np.roots([5*k2,3*k1,1.]) if k2 else np.roots([3*k1,1.]) if k1 else []
    positive = [float(r.real) for r in roots if abs(r.imag)<1e-12 and r.real>0]
    if not positive:
        return dict(covers_sensor=True,first_turning_radius=None)
    t = min(positive)
    maximum = np.sqrt(t)*(1+k1*t+k2*t*t)*camera['focal_px']
    needed = np.hypot(camera['width']/2,camera['height']/2)
    return dict(covers_sensor=bool(maximum>needed),first_turning_radius=float(np.sqrt(t)),
                maximum_distorted_radius_px=float(maximum),corner_radius_px=float(needed))


def report(out):
    validate(out)
    inputs = json.loads((out/'input.json').read_text())
    fits = json.loads((out/'fits.json').read_text())
    if fits['split'] != 'development' or [f['id'] for f in fits['frames']] != IDS:
        raise ValueError('fit selection changed')
    frames = []
    for source, fitted in zip(inputs['frames'],fits['frames']):
        xy = np.array([s['xy'] for s in source['sources']])
        outside = np.array([s['outside_both_inputs'] for s in source['sources']])
        initial = source['initial']; w,h = initial['width'],initial['height']
        rho,_,_,_ = decompose(xy,np.zeros_like(xy),w,h)
        masks = groups(xy,outside,w,h,rho)
        masks['training'] = ~outside
        cases = []
        if [c['name'] for c in fitted['cases']] != ['k1','k1_k2'] or not fitted['cases'][0]['control_exact']:
            raise ValueError('control reproduction required')
        for c in fitted['cases']:
            predicted = project_extended(c['camera'],[s['radec'] for s in source['sources']])
            rust = np.array([[np.nan,np.nan] if p is None else p for p in c['projected_xy']])
            valid = np.isfinite(rust).all(axis=1)
            if not np.allclose(predicted[valid],rust[valid],atol=1e-8,rtol=0):
                raise ValueError('Rust/Python projections disagree')
            errors = np.linalg.norm(rust-xy,axis=1)
            def metrics(mask):
                result = stats(errors[mask&valid]) if (mask&valid).any() else dict(count=0)
                result['invalid_count'] = int(sum(mask&~valid))
                return result
            cases.append(dict(c,projection_crosscheck_max_abs_px=float(np.max(np.abs(predicted[valid]-rust[valid]))) if valid.any() else None,
                source_errors_px=[float(e) if np.isfinite(e) else None for e in errors],
                fit_rms_px=float(np.sqrt(np.mean(np.sum(np.asarray(c['fit_residuals']).reshape(-1,2)**2,axis=1)))),
                groups={name:metrics(mask) for name,mask in masks.items()},sensor_coverage=radial_sensor_coverage(c['camera'])))
        a,b=cases
        differences={g:b['groups'][g]['rms_px']-a['groups'][g]['rms_px'] for g in masks
                     if a['groups'][g]['count'] and b['groups'][g]['count']}
        edge_regressions=[g for g in ('left_10pct','right_10pct','top_10pct','bottom_10pct') if differences.get(g,0)>.5]
        bound_hit=any(abs(b['camera'][k])>=.5-1e-12 for k in ('k1','k2'))
        baseline_rms=a['groups']['outside_both_inputs'].get('rms_px')
        variant_rms=b['groups']['outside_both_inputs'].get('rms_px')
        ratio=variant_rms/baseline_rms if baseline_rms and variant_rms is not None else float('inf')
        criteria=dict(primary_improves_10pct=ratio<=.9 if source['id']==IDS[1] else None,
            regression_frame_outside_pass=differences.get('outside_both_inputs',float('inf'))<=.5 if source['id']!=IDS[1] else None,
            edge_regressions=edge_regressions,no_bound_hit=not bound_hit,
            all_projectable=all(c['groups']['all_reviewed']['invalid_count']==0 for c in cases))
        frames.append(dict(id=source['id'],sources=source['sources'],cases=cases,group_rms_change_px=differences,criteria=criteria))
    passed=all(not f['criteria']['edge_regressions'] and f['criteria']['no_bound_hit'] and f['criteria']['all_projectable']
        and f['criteria']['primary_improves_10pct'] is not False and f['criteria']['regression_frame_outside_pass'] is not False for f in frames)
    write_json(out/'geometry.json',dict(iteration=13,split='development',protocol_sha256=digest(out/'protocol.json'),
        fits_sha256=digest(out/'fits.json'),frames=frames,diagnostic_gate_passed=passed,
        limitations=['conditional manual identities and original native extraction coordinates',
                     'training membership derived from prior input proximity, not original solver correspondences',
                     'no rematching, acceptance test, holdout or solve-rate measurement']))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['prepare','run','report'])
    parser.add_argument('out',type=Path)
    parser.add_argument('--registry',type=Path)
    args=parser.parse_args()
    if args.command=='prepare':
        if args.registry is None: parser.error('--registry required')
        prepare(args.out.resolve(),args.registry.resolve())
    elif args.command=='run':run(args.out.resolve())
    else:report(args.out.resolve())
