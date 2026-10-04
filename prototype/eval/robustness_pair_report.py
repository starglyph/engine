#!/usr/bin/env python3
"""Evaluate frozen pair fits and linearized sensitivity on unchanged probes."""
import argparse
import json
from pathlib import Path

import numpy as np

from collection_run import ROOT, digest, write_json
from compare_local_wcs import project, stats
from robustness_geometry import selected
from robustness_refinement import RID
from robustness_refinement_report import lift


def sensitivity(jacobian, probe_jacobian):
    jac=np.asarray(jacobian,dtype=float)
    probes=np.asarray(probe_jacobian,dtype=float)[:-1]
    data=jac[:-1]
    norms=np.linalg.norm(data,axis=0)
    if np.any(norms==0) or not np.isfinite(jac).all():
        raise ValueError('degenerate or non-finite Jacobian')
    singular=np.linalg.svd(data/norms,compute_uv=False)
    response=np.linalg.pinv(jac,rcond=1e-12)[:,:-1]
    propagation=probes@response
    factors=np.sqrt(np.sum(propagation.reshape(-1,2,response.shape[1])**2,axis=(1,2)))
    focal,k1=response[3],response[4]
    correlation=np.dot(focal,k1)/(np.linalg.norm(focal)*np.linalg.norm(k1))
    return dict(column_normalized_data_condition=float(singular[0]/singular[-1]),
        focal_k1_response_correlation=float(correlation),probe_amplification=factors.tolist(),
        interpretation='linear response to unit independent coordinate perturbations, including fixed prior; not calibrated uncertainty')


def summarize(out,output):
    protocol=json.loads((out/'protocol.json').read_text())
    if protocol['id']!=RID or protocol['split']!='development':
        raise ValueError('development required')
    selected([RID])
    # Cargo prunes the isolated workspace lock when removing desktop and replacing
    # registry tetra3 by the local copy. Preserve both pre/post hashes in checks.
    mutable_lock=str((out/'pipeline/Cargo.lock').resolve())
    for path,sha in protocol['hashes'].items():
        if path!=mutable_lock and digest(Path(path))!=sha:
            raise ValueError('frozen input changed: '+path)
    prior=json.loads((ROOT/'docs/experiments/robustness-iteration-5-geometry.json').read_text())
    frame=next(f for f in prior['frames'] if f['id']==RID)
    review=json.loads((ROOT/'docs/experiments/robustness-iteration-5-review.json').read_text())
    observations=next(f for f in review['frames'] if f['id']==RID)
    sources=[]
    for s in frame['sources']:
        point=next(p for p in observations['points'] if p['id']==s['source_id'])
        sources.append(dict(id=s['source_id'],hyg_id=s['hyg_id'],name=s['name'],xy=s['xy'],
                            outside_both_inputs=s['outside_both_inputs'],ra_deg=point['ra_deg'],dec_deg=point['dec_deg']))
    xy=np.array([s['xy'] for s in sources]);world=[[s['ra_deg'],s['dec_deg']] for s in sources]
    masks=dict(all_reviewed=np.ones(len(sources),dtype=bool),outside_both_inputs=np.array([s['outside_both_inputs'] for s in sources]),
               left_10pct=xy[:,0]<300,right_10pct=xy[:,0]>2700,top_10pct=xy[:,1]<400,bottom_10pct=xy[:,1]>3600)
    fits=json.loads((out/'fits.json').read_text())
    if fits['id']!=RID or fits['split']!='development' or [c['name'] for c in fits['cases']]!=protocol['cases']:
        raise ValueError('fit selection differs')
    results=[]
    for case in fits['cases']:
        camera=case['camera']
        projected=lift(project(camera,world),camera,3000,4000)
        errors=np.linalg.norm(projected-xy,axis=1)
        r=np.array(case['residuals'])[:-1].reshape(-1,2)
        results.append(dict(name=case['name'],camera=camera,matches=case['matches'],elapsed_ms=case['elapsed_ms'],
            prior_weight=case['prior_weight'],prior_scale=case['prior_scale'],
            reproduced_previous_first_fit=case['reproduced_previous_first_fit'],
            fit_rms_original_px=float(2.5*np.sqrt(np.mean(np.sum(r*r,axis=1)))),
            groups={k:stats(errors[m]) if m.any() else dict(count=0) for k,m in masks.items()},
            source_errors_px=errors.tolist(),sensitivity=sensitivity(case['jacobian'],case['probe_jacobian'])))
    write_json(output,dict(iteration=7,id=RID,split='development',protocol_sha256=digest(out/'protocol.json'),
        review_sha256=protocol['review_sha256'],fits_sha256=digest(out/'fits.json'),sources=sources,cases=results,
        limitations=['first fixed-pair fit only; no matching, rematching, scale-lift refit or acceptance change',
                    'visually rejected foreground correspondence removed only in diagnostic cases',
                    'other catalogue identities remain conditional; possible blend kept explicit',
                    'normalized Jacobian condition and linear response are sensitivity diagnostics, not confidence intervals']))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('out',type=Path);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();summarize(a.out.resolve(),a.output)
