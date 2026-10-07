#!/usr/bin/env python3
"""Attribute saved Orion pattern errors to identified edges and endpoint residuals."""
import argparse
from pathlib import Path
import time

import numpy as np

from collection_run import ROOT, digest, write_json
from robustness_orion_fit import RID, WIDTH, HEIGHT, previous, read, working_xy
from robustness_reference_audit import selected
from robustness_pattern_edge_geometry import PAIRS, angles, signature, ratio_terms, attribute


def build_input():
    inp=read(previous('50-input'));curves=read(previous('50-curves'));old=read(previous('50-results'))
    sources=read(previous('40-results'))['sources'];saved=read(previous('45-nonlinear-results'))
    if any((x['id'],x['split'],x['holdout'])!=(RID,'development',False) for x in (inp,curves,old,saved)):
        raise ValueError('development only')
    ids=[s['id'] for s in sources]
    if ids!=[s['id'] for s in saved['sources']]:raise ValueError('saved source order changed')
    arm=next(a for a in inp['arms'] if a['name']=='seven')
    camera=arm['camera'];case45=next(c for c in saved['cases'] if c['method']=='external_centroid')['cases']['seven']
    if camera!=case45['camera']:raise ValueError('saved camera changed')
    fov=inp['fovs_rad'][inp['fixed_indices']['seven']]
    # Preserve the f32 pixel scale used by the saved tetra3 run.
    focal=1/float(np.float32(np.tan(fov/2)/800))
    rows=[]
    for spec,raw,prior in zip(inp['cases'],curves['cases'],old['cases']):
        if not spec['name']==raw['name']==prior['name']:raise ValueError('case order changed')
        indices=[ids.index(i) for i in spec['source_ids']]
        points=[sources[i] for i in indices]
        ideal=[arm['projection_checks'][i]['ideal'] for i in indices]
        predicted=np.array([case45['probe_projected_xy'][i] for i in indices])
        if not np.allclose(predicted-[800,521],[arm['projection_checks'][i]['observed'] for i in indices],rtol=0,atol=1e-8):
            raise ValueError('saved forward projection mismatch')
        measured=np.array(spec['xy'])
        if spec['method']!='detector' and not np.array_equal(measured,working_xy([s['coordinates'][spec['method']] for s in points],camera)):
            raise ValueError('centroid coordinates changed')
        rows.append(dict(name=spec['name'],quartet=spec['quartet'],method=spec['method'],
            sources=[dict(id=s['id'],hyg_id=s['hyg_id'],name=s['name'],fit=s['id'] in inp['selection']['fit_ids']) for s in points],
            measured_working_xy=measured.tolist(),predicted_working_xy=predicted.tolist(),
            forward_residual_original_px=((predicted-measured)*[WIDTH/1600,HEIGHT/1042]).tolist(),
            ideal_centered_xy=ideal,catalog_vectors=raw['catalog_vectors'],catalog_ratios=raw['catalog_ratios'],
            arms=[dict(name=a['name'],xy=a['ideal_xy'],expected=prior['arms'][a['name']]['fixed']['seven']) for a in raw['arms']]))
    if len(rows)!=17:raise ValueError('all17 original cases required')
    return dict(id=RID,split='development',holdout=False,fov_rad=fov,focal_px=focal,camera=camera,cases=rows,
        focus=['check_quartet_1','local_control'],selection=inp['selection'])


def prepare(out):
    selected()
    if out.exists():raise ValueError('fresh output required')
    paths=[]
    for name in ['40-results','45-nonlinear-results','50-input','50-curves','50-results']:
        p=previous(name);check=previous(name[:2]+'-checks')
        if digest(p)!=read(check)['artifact_hashes'][str(p.relative_to(ROOT))]:raise ValueError('published evidence changed: '+name)
        paths.extend([p,check])
    protected=read(previous('50-checks'))['protected_hashes']
    for name,sha in protected.items():
        if digest(ROOT/name)!=sha:raise ValueError('protected data changed')
        paths.append(ROOT/name)
    inp=build_input();out.mkdir(parents=True);write_json(out/'input.json',inp)
    paths += [out/'input.json',Path(__file__).resolve(),*(Path(__file__).with_name(n) for n in [
        'robustness_pattern_edge_geometry.py','collection_run.py','robustness_orion_fit.py','robustness_reference_audit.py'])]
    write_json(out/'protocol.json',dict(iteration=51,id=RID,split='development',holdout=False,
        scope='all17 saved50 cases and4 arms at same fixed seven FOV; focus check1 andlocal; no new correction or fit',
        analyses=['six identity-labelled angular edges; sorted-rank comparison and exact numerator/denominator split',
            'ideal projections from saved45 seven camera; fixed catalogue edge ordering for endpoint derivatives',
            'linear contributions per source, radial and transverse in ideal camera plane; no residual cancellation applied to data'],
        criteria=['f64 reconstruction versus saved f32 ratios<=1e-6 with identical pass flags; catalogue ratios<=1e-6',
            'ideal projected ratios versus catalogue<=1e-6; forward saved projection equality<=1e-8 working px',
            'exact numerator+denominator decomposition error<=1e-12',
            'endpoint linear attribution usable only if max nonlinear residual<=1e-4 and derivative step1e-3 versus5e-4 changes<=1e-7',
            'record all order changes and all contributions, including cancellations; no star exclusion, camera change, source-cause or GT assertion'],
        protected_hashes=protected,hashes={str(p.relative_to(ROOT)):digest(p) for p in sorted(set(paths))},
        new_fits=0,new_solver_calls=0,new_wcs_calls=0))


def validate(out):
    p=read(out/'protocol.json')
    if (p['id'],p['split'],p['holdout'])!=(RID,'development',False):raise ValueError('development only')
    selected()
    for name,sha in p['hashes'].items():
        if digest(ROOT/name)!=sha:raise ValueError('frozen input changed: '+name)
    return p


def measure(out):
    validate(out)
    if (out/'results.json').exists():raise ValueError('fresh analysis required')
    started=time.monotonic();inp=read(out/'input.json');rows=[];max_replay=0.;max_ideal=0.
    for c in inp['cases']:
        catalog=angles(c['catalog_vectors']);order=np.argsort(catalog,kind='stable');ratios=catalog[order[:5]]/catalog[order[-1]]
        max_replay=max(max_replay,float(np.max(np.abs(ratios-c['catalog_ratios']))))
        ideal=np.array(c['ideal_centered_xy']);ie,io,ir=signature(ideal,inp['focal_px'])
        max_ideal=max(max_ideal,float(np.max(np.abs(ir-ratios))))
        arms=[]
        for a in c['arms']:
            edges,ranking,sorted_ratios=signature(a['xy'],inp['focal_px'])
            err=float(np.max(np.abs(sorted_ratios-a['expected']['ratios'])))
            max_replay=max(max_replay,err)
            if bool(np.all(np.abs(sorted_ratios-ratios)<.006))!=a['expected']['ratios_pass']:
                raise ValueError('f32 pass flag not reproduced')
            numerator,denominator=ratio_terms(edges,catalog,order)
            identity_delta=edges[order[:5]]/edges[order[-1]]-ratios
            if np.max(np.abs(identity_delta-numerator-denominator))>1e-12:raise ValueError('ratio decomposition failed')
            arms.append(dict(name=a['name'],edges_rad=edges.tolist(),edge_error_arcsec=(3600*np.rad2deg(edges-catalog)).tolist(),
                image_order=ranking.tolist(),same_edge_order=bool(np.array_equal(order,ranking)),sorted_ratio_delta=(sorted_ratios-ratios).tolist(),
                identified_ratio_delta=identity_delta.tolist(),numerator_contribution=numerator.tolist(),denominator_contribution=denominator.tolist(),
                replay_max_error=err,ratios_pass=a['expected']['ratios_pass']))
        corrected=np.array(next(a['xy'] for a in c['arms'] if a['name']=='seven'))
        attribution=attribute(ideal,corrected,inp['focal_px'],order)
        attribution['usable']=attribution['max_linear_error']<=1e-4 and attribution['max_step_change']<=1e-7
        rows.append(dict(name=c['name'],quartet=c['quartet'],method=c['method'],sources=c['sources'],
            forward_residual_original_px=c['forward_residual_original_px'],ideal_centered_xy=c['ideal_centered_xy'],corrected_centered_xy=corrected.tolist(),
            edge_source_ids=[[c['sources'][i]['id'],c['sources'][j]['id']] for i,j in PAIRS],
            catalog_edges_rad=catalog.tolist(),catalog_order=order.tolist(),ideal_ratio_error=(ir-ratios).tolist(),
            arms=arms,attribution=attribution))
    if max_replay>1e-6 or max_ideal>1e-6:raise ValueError('saved geometry replay failed')
    write_json(out/'results.json',dict(iteration=51,id=RID,split='development',holdout=False,
        protocol_sha256=digest(out/'protocol.json'),cases=rows,max_replay_ratio_error=max_replay,max_ideal_ratio_error=max_ideal,
        all_attributions_usable=all(c['attribution']['usable'] for c in rows),elapsed_s=time.monotonic()-started,
        new_fits=0,new_solver_calls=0,new_wcs_calls=0,production_changed=False,independent_ground_truth=False,automatic_improvement_confirmed=False))
    validate(out)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['prepare','measure','validate']);p.add_argument('--out-dir',type=Path,required=True)
    a=p.parse_args();globals()[a.stage](a.out_dir.resolve())
