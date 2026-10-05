#!/usr/bin/env python3
"""Describe the frozen single22 residual field; no camera fitting or selection."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from collection_run import ROOT, digest, write_json
from compare_local_wcs import project
from robustness_geometry import selected
from robustness_residual_fields import decompose, vector_metrics, groups, agreement

RID = 'wm_r_132162731'
PUBLIC = ROOT/'docs/experiments'
ARMS = ('all31', 'single22')


def prior(name):
    return PUBLIC/f'robustness-iteration-{name}.json'


def read(path):
    return json.loads(path.read_text())


def datasets(review, inputs, geometry, alternate, saved):
    """Retain pair order and probe identities; never select by residual magnitude."""
    singles = [p for p in review['points'] if p['label'] == 'visible_source']
    pairs = inputs['cases'][1]['matches']
    if len(singles) != 22 or len(pairs) != 22:
        raise ValueError('frozen22 required')
    for p, m in zip(singles, pairs):
        if (p['id'], p['hyg_id'], p['world'], p['review_center']) != (m['pair_id'], m['hyg_id'], m['world'], m['xy']):
            raise ValueError('fit pair identity changed')
    probes = next(f['sources'] for f in geometry['frames'] if f['id'] == RID)
    alt = {s['source_id']: s for s in next(f['sources'] for f in alternate['frames'] if f['id'] == RID)}
    if len(probes) != 35:
        raise ValueError('frozen35 required')
    for case in saved['cases']:
        if [(s['source_id'], s['hyg_id'], s['xy']) for s in case['sources']] != [(s['source_id'], s['hyg_id'], s['xy']) for s in probes]:
            raise ValueError('probe identity changed')
    for s in probes:
        if alt[s['source_id']]['hyg_id'] != s['hyg_id'] or alt[s['source_id']]['xy'] != s['xy']:
            raise ValueError('alternate centroid identity changed')
    return dict(
        fit22=dict(sources=[dict(id=p['id'], hyg_id=p['hyg_id'], name=p['name']) for p in singles],
            radec=[p['radec'] for p in singles], base='detector',
            positions={kind:[p['positions'][kind] for p in singles] for kind in ('detector','external','native_r8','native_r12')}),
        probes35=dict(sources=[dict(id=s['source_id'], hyg_id=s['hyg_id'], name=s['name']) for s in probes],
            radec=[s['radec'] for s in probes], base='external',
            positions=dict(external=[s['xy'] for s in probes], **{f'native_r{r}':[alt[s['source_id']][f'native_r{r}_xy'] for s in probes] for r in (8,12)})))


def prepare(out):
    selected([RID])
    if out.exists():
        raise ValueError('fresh output required')
    files = [prior(n) for n in ('27-results','28-input','28-results','28-protocol','28-checks','13-geometry','5-geometry')]
    # These published artifacts must match the hashes recorded before/after fit28.
    checks = read(prior('28-checks'))
    for name in ('protocol','input','results'):
        key = f'prototype/artifacts/robustness-iteration-28/{name}.json'
        if digest(prior('28-'+name)) != checks['artifact_hashes'][key]:
            raise ValueError('published fit28 artifact changed')
    old = read(prior('28-protocol'))['hashes']
    for path, expected in checks['protected_hashes'].items():
        if digest(ROOT/path) != expected:
            raise ValueError('protected input changed')
        files.append(ROOT/path)
    for path in (prior('13-geometry'), prior('5-geometry')):
        if digest(path) != old[str(path.relative_to(ROOT))]:
            raise ValueError('prior geometry changed')
    # The published27 JSON is compact; restore the original write_json encoding.
    reviewed_bytes = (json.dumps(read(prior('27-results')),indent=2,ensure_ascii=False,allow_nan=False)+'\n').encode()
    if hashlib.sha256(reviewed_bytes).hexdigest() != old['prototype/artifacts/robustness-iteration-27/results.json']:
        raise ValueError('reviewed centroids changed')
    files += [Path(__file__).resolve(), *(Path(__file__).with_name(n) for n in
        ('robustness_residual_fields.py','compare_local_wcs.py','collection_run.py','robustness_geometry.py'))]
    out.mkdir(parents=True)
    write_json(out/'protocol.json', dict(iteration=29,id=RID,split='development',holdout=False,
        arms=list(ARMS),hashes={str(p.relative_to(ROOT)):digest(p) for p in files},
        residual_sign='predicted minus measured; x right, y down',
        decomposition='reuse iteration12; origin width/2,height/2; clockwise tangential',
        groups='unchanged22 and35; fixed baseline-centroid edges10pct, grid3x3, radius bins0/.35/.70/1; saved outside-all40 and outside-union31 masks',
        centroid_methods='fit22 detector/external/r8/r12; probes35 external/r8/r12; no recentering or refit',
        criteria=['exactly reproduce saved fit28 projections and corresponding RMS',
            'report both camera arms, all sources, empty groups and every centroid method',
            'compare full vector fields and directions for residuals>=1px using existing agreement',
            'descriptive only: no parameter choice, significance or causal claim from agreement alone'],
        plot_arrow_magnification=15.,new_fits=0,new_searches=0))


def validate(out):
    p = read(out/'protocol.json')
    if (p['id'],p['split'],p['holdout'],p['arms']) != (RID,'development',False,list(ARMS)):
        raise ValueError('frozen development protocol required')
    selected([RID])
    for name, sha in p['hashes'].items():
        if digest(ROOT/name) != sha:
            raise ValueError('frozen input changed: '+name)
    return p


def describe(positions, prediction, masks, width, height, base):
    fields = {}
    vectors = {}
    for method, xy in positions.items():
        xy = np.asarray(xy)
        residual = prediction-xy
        vectors[method] = residual
        _, rr, tt, valid = decompose(xy,residual,width,height)
        fields[method] = dict(residual_xy_px=residual.tolist(),
            groups={name:vector_metrics(residual[m],rr[m],tt[m],valid[m],np.hypot(width/2,height/2)) for name,m in masks.items()})
    comparisons = {method:{name:agreement(vectors[base],v,m) for name,m in masks.items()}
                   for method,v in vectors.items() if method != base}
    return dict(methods=fields,centroid_agreement=comparisons)


def measure(out):
    validate(out)
    review, inputs, saved = (read(prior(n)) for n in ('27-results','28-input','28-results'))
    data = datasets(review,inputs,read(prior('13-geometry')),read(prior('5-geometry')),saved)
    cases = {c['name']:c for c in saved['cases']}
    w,h = cases['single22']['camera']['width'],cases['single22']['camera']['height']
    result = {}
    for name, dataset in data.items():
        base = dataset['base']
        xy = np.asarray(dataset['positions'][base])
        rho,*_ = decompose(xy,np.zeros_like(xy),w,h)
        masks = groups(xy,np.zeros(len(xy),dtype=bool),w,h,rho)
        masks.pop('outside_both_inputs')
        if name == 'probes35':
            for key in ('outside_all40_inputs','outside_union31_fits'):
                masks[key] = np.array([s[key] for s in cases['all31']['sources']])
                if masks[key].tolist() != [s[key] for s in cases['single22']['sources']]:
                    raise ValueError('outside group changed between cameras')
        arms = {}
        for arm in ARMS:
            pred = project(cases[arm]['camera'],dataset['radec'])
            if name == 'probes35':
                old = np.array([s['predicted_xy'] for s in cases[arm]['sources']])
                if not np.array_equal(pred,old):
                    raise ValueError('saved projection not reproduced')
            arms[arm] = describe(dataset['positions'],pred,masks,w,h,base)
            actual = arms[arm]['methods'][base]['groups']['all_reviewed']['rms_px']
            expected = cases[arm]['pair_metrics']['single22']['rms_px'] if name == 'fit22' else cases[arm]['conditional_geometry']['external_centroid']['all_reviewed']['rms_px']
            if not np.isclose(actual,expected,rtol=0,atol=1e-8):
                raise ValueError('saved RMS not reproduced')
        result[name] = dict(**dataset,arms=arms,group_indices={k:np.flatnonzero(m).tolist() for k,m in masks.items()},
            camera_agreement={method:{k:agreement(np.array(arms['all31']['methods'][method]['residual_xy_px']),
                np.array(arms['single22']['methods'][method]['residual_xy_px']),m) for k,m in masks.items()} for method in dataset['positions']})
    write_json(out/'results.json',dict(iteration=29,id=RID,split='development',holdout=False,width=w,height=h,
        protocol_sha256=digest(out/'protocol.json'),datasets=result,projections_reproduced=True,
        new_fits=0,new_searches=0,production_changed=False,independent_ground_truth=False,
        limitations=['conditional identities shared by fit and review','same-field diagnostic, not independent calibration',
            'centroid sensitivity does not exclude shared systematic image or catalogue errors','empty lower field unmeasured']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['prepare','measure','validate'])
    parser.add_argument('--out-dir',type=Path,required=True)
    args = parser.parse_args()
    globals()[args.stage](args.out_dir.resolve())
