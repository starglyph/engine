#!/usr/bin/env python3
"""Offline identity/epoch/residual audit of the 13 frozen legacy smartphone pairs.

No solving, rematching, camera fitting or modification of acceptance. Linear
residual decompositions are descriptive diagnostics, not camera corrections.
"""
import argparse
import csv
import gzip
import json
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from PIL import Image, ImageDraw, ImageOps

from collection_run import ROOT, digest, write_json
from robustness_candidate_trace import FRAME
from robustness_residual_fields import decompose, vector_metrics

TRACE = ROOT/'docs/experiments/robustness-iteration-19-results.json'
CATALOG = ROOT/'data/catalogs/hyg_v42.csv.gz'
REFERENCE = ROOT/f'prototype/artifacts/smartphone-wcs-compatible/{FRAME}/reference.json'
IMAGE = ROOT/f'data/input/smartphone/{FRAME}.jpg'


def reference_paths():
    ref = json.loads(REFERENCE.read_text())
    paths = {}
    for key in ('wcs', 'correspondences'):
        path = Path(ref[key+'_file'])
        paths[key] = path if path.is_absolute() else ROOT/'prototype'/path
        if digest(paths[key]) != ref[key+'_sha256']:
            raise ValueError('external artifact hash mismatch')
    if digest(IMAGE) != ref['source_sha256']:
        raise ValueError('wrong reference source')
    return ref, paths


def prepare(out):
    if out.exists():
        raise ValueError('fresh output directory required')
    _, paths = reference_paths()
    trace = json.loads(TRACE.read_text())
    if trace['id'] != FRAME or len(trace['first_candidates']['control']['pairs']) != 13:
        raise ValueError('fixed legacy pair set required')
    files = [TRACE, CATALOG, REFERENCE, IMAGE, *paths.values(), Path(__file__).resolve(),
             ROOT/'prototype/eval/collection_run.py', ROOT/'prototype/eval/robustness_candidate_trace.py',
             ROOT/'prototype/eval/robustness_residual_fields.py',
             ROOT/'data/samples/sky-samples/manifest.json',
             ROOT/'data/samples/sky-samples/robustness-results.json',
             ROOT/'data/samples/sky-samples/collection-wcs-review.json']
    out.mkdir(parents=True)
    write_json(out/'protocol.json', dict(iteration=20, id=FRAME, holdout=False,
        hashes={str(p.relative_to(ROOT)):digest(p) for p in files},
        study='exploratory offline diagnosis after iteration19, not an algorithm A/B',
        identity='all 13 pairs; existing pending WCS, all corr weights reported, HYG <=6.8 neighbors',
        neighbor_search=dict(catalog_mag_limit=6.8, cone_deg=60, nearest_count=3),
        decomposition=['translation','translation_rotation','similarity','similarity_radial'],
        evaluation='unweighted linear residual explanation; all pairs and leave-one-pair-out; no exclusion or threshold tuning',
        criteria=['retain every pair and uncertainty, including unresolved catalog components',
                  'cross-check TAN reconstruction against saved tetra3 world_to_pixel',
                  'compare epoch effect to original residuals',
                  'training error decrease alone is not a reliable residual explanation',
                  'no ground truth promotion, camera correction, solve or holdout run']))


def validate(out):
    protocol = json.loads((out/'protocol.json').read_text())
    if protocol['id'] != FRAME or protocol['holdout']:
        raise ValueError('fixed legacy frame only')
    for path, sha in protocol['hashes'].items():
        if digest(ROOT/path) != sha:
            raise ValueError('frozen input changed: '+path)
    return protocol


def units(radec):
    ra, dec = np.radians(np.asarray(radec)).T
    return np.column_stack((np.cos(dec)*np.cos(ra),np.cos(dec)*np.sin(ra),np.sin(dec)))


def linear_diagnostics(xy, errors, focal, size):
    """Describe residual vectors with fixed bases, checking unseen pairs via LOO."""
    xy, errors = np.asarray(xy), np.asarray(errors)
    offset = (xy-np.asarray(size)/2)/focal
    translation = np.tile(np.eye(2), (len(xy),1))
    turn = np.column_stack((-offset[:,1],offset[:,0])).reshape(-1,1)
    scale = offset.reshape(-1,1)
    radial = (offset*np.sum(offset**2,axis=1)[:,None]).reshape(-1,1)
    matrices = dict(translation=translation,
        translation_rotation=np.column_stack((translation,turn)),
        similarity=np.column_stack((translation,turn,scale)),
        similarity_radial=np.column_stack((translation,turn,scale,radial)))
    rows = {}
    target = errors.reshape(-1)
    for name, design in matrices.items():
        beta, _, rank, singular = np.linalg.lstsq(design,target,rcond=None)
        if rank != design.shape[1]:
            raise ValueError('rank-deficient residual design')
        residual = errors-(design@beta).reshape(-1,2)
        loo = []
        for i in range(len(xy)):
            keep = np.repeat(np.arange(len(xy))!=i,2)
            coeff, _, rank_i, _ = np.linalg.lstsq(design[keep],target[keep],rcond=None)
            if rank_i != design.shape[1]:
                raise ValueError('rank-deficient leave-one-out design')
            loo.append(errors[i]-(design@coeff).reshape(-1,2)[i])
        rows[name] = dict(coefficients=beta.tolist(),condition_number=float(singular[0]/singular[-1]),
            training_rms_px=float(np.sqrt(np.mean(np.sum(residual**2,axis=1)))),
            leave_one_out_rms_px=float(np.sqrt(np.mean(np.sum(np.asarray(loo)**2,axis=1)))),
            leave_one_out_residuals_px=np.asarray(loo).tolist())
    return rows


def native_wcs(solution):
    if solution['camera_model']['distortion'] != 'None':
        raise ValueError('TAN replay requires undistorted input')
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = np.asarray(solution['camera_model']['crpix'])+1
    wcs.wcs.crval = np.degrees(solution['crval_rad'])
    wcs.wcs.cd = np.degrees(solution['cd_matrix'])
    wcs.wcs.ctype = ['RA---TAN','DEC--TAN']
    return wcs


def field_metrics(row, catalog):
    pairs = row['pairs']; sol = row['solution']
    xy = np.array([p['xy'] for p in pairs])
    predicted = np.array([p['tetra_centered_xy'] for p in pairs])
    inputs = np.array([p['input_centroid'] for p in pairs])
    wcs = native_wcs(sol)
    replay = wcs.all_world2pix([p['radec'] for p in pairs],0)
    crosscheck = float(np.max(np.abs(replay-predicted)))
    if crosscheck>1e-8:
        raise ValueError('native TAN convention mismatch')
    j2000 = [[float(catalog[p['catalog_id']]['ra'])*15,float(catalog[p['catalog_id']]['dec'])] for p in pairs]
    epoch_shift = wcs.all_world2pix(j2000,0)-replay
    errors = predicted-inputs
    width,height = sol['image_width'],sol['image_height']
    rho,radial,tangential,valid = decompose(xy,errors,width,height)
    metrics = vector_metrics(errors,radial,tangential,valid,np.hypot(width/2,height/2))
    return dict(metrics=metrics,native_crosscheck_max_px=crosscheck,
        max_epoch_projection_change_px=float(np.max(np.linalg.norm(epoch_shift,axis=1))),
        linear=linear_diagnostics(xy,errors,row['camera']['focal_px'],[width,height]),
        pairs=[dict(catalog_id=p['catalog_id'],detection_index=p['detection_index'],
            xy=p['xy'],residual_xy_px=e.tolist(),radius_fraction=float(r),
            radial_px=float(rad),tangential_px=float(tan),epoch_shift_px=epoch.tolist())
            for p,e,r,rad,tan,epoch in zip(pairs,errors,rho,radial,tangential,epoch_shift)])


def identity_rows(row, catalog, ref, paths, settings):
    entries = [p for p in catalog.values() if int(p['id'])!=0 and float(p['mag'])<=settings['catalog_mag_limit']]
    radec = np.array([[float(p['ra'])*15,float(p['dec'])] for p in entries])
    world = units(radec)
    bore = units([np.degrees(row['solution']['crval_rad'])])[0]
    inside = world@bore>np.cos(np.radians(settings['cone_deg']))
    entries = [p for p,keep in zip(entries,inside) if keep]
    radec = radec[inside]
    wcs = WCS(fits.getheader(paths['wcs']))
    projected = wcs.all_world2pix(radec,0,quiet=True)
    corr = fits.getdata(paths['correspondences'])
    corr_world = units(np.column_stack((corr['index_ra'],corr['index_dec'])))
    scale = np.array([ref['width']/row['camera']['width'],ref['height']/row['camera']['height']])
    result = []
    for p in row['pairs']:
        own = next(i for i,s in enumerate(entries) if int(s['id'])==p['catalog_id'])
        observed = np.array(p['xy'])*scale
        distances = np.linalg.norm(projected-observed,axis=1)
        nearest = np.argsort(np.where(np.isfinite(distances),distances,np.inf))[:settings['nearest_count']]
        if not np.isfinite(distances[nearest]).all():
            raise ValueError('insufficient finite neighbors')
        delta = np.degrees(np.arctan2(np.linalg.norm(np.cross(corr_world,p['world']),axis=1),corr_world@p['world']))*3600
        ci = int(np.argmin(delta))
        own_star = entries[own]
        siblings = [s for s in entries if s['comp_primary'] and s['comp_primary']==own_star['comp_primary'] and s['id']!=own_star['id']]
        result.append(dict(catalog_id=p['catalog_id'],detection_index=p['detection_index'],
            name=own_star['proper'] or own_star['bf'] or own_star['gl'] or own_star['hip'],mag=float(own_star['mag']),
            detection_full_xy=observed.tolist(),claimed_wcs_xy=projected[own].tolist(),
            claimed_wcs_residual_full_px=float(distances[own]),
            nearest_catalog=[dict(id=int(entries[i]['id']),name=entries[i]['proper'] or entries[i]['bf'],
                mag=float(entries[i]['mag']),wcs_xy=projected[i].tolist(),distance_full_px=float(distances[i])) for i in nearest],
            components=[dict(id=int(s['id']),component=s['comp'],mag=float(s['mag'])) for s in siblings],
            closest_corr=dict(angle_arcsec=float(delta[ci]),weight=float(corr[ci]['match_weight']),
                radec=[float(corr[ci]['index_ra']),float(corr[ci]['index_dec'])]),
            identity_status='unresolved_components' if siblings else 'conditional_WCS_consistency_only'))
    return result


def preview(out, rows):
    image = ImageOps.exif_transpose(Image.open(IMAGE)).convert('RGB')
    canvas = Image.new('RGB',(1000,880),(20,20,20)); draw=ImageDraw.Draw(canvas)
    for i,p in enumerate(rows):
        x,y=p['detection_full_xy']; left,top=int(x)-100,int(y)-90
        tile=np.asarray(image.crop((left,top,left+200,top+180)))/255.
        tile=Image.fromarray(np.uint8(np.power(tile,.45)*255))
        ox,oy=(i%4)*250,(i//4)*220;canvas.paste(tile,(ox,oy+30))
        draw.text((ox+2,oy+2),f"det {p['detection_index']} HYG {p['catalog_id']}",fill='white')
        for pos,color in [(p['detection_full_xy'],'yellow'),(p['claimed_wcs_xy'],'cyan')]:
            xx,yy=np.array(pos)-[left,top]+[ox,oy+30]
            draw.ellipse((xx-6,yy-6,xx+6,yy+6),outline=color,width=1)
    canvas.save(out/'wcs-pair-crops.png')


def run(out):
    protocol = validate(out); ref,paths=reference_paths()
    with gzip.open(CATALOG,'rt') as f:
        catalog={int(p['id']):p for p in csv.DictReader(f)}
    trace=json.loads(TRACE.read_text())['first_candidates']
    fields={arm:field_metrics(row,catalog) for arm,row in trace.items()}
    identities=identity_rows(trace['control'],catalog,ref,paths,protocol['neighbor_search'])
    result=dict(iteration=20,id=FRAME,protocol_sha256=digest(out/'protocol.json'),
        fields=fields,identities=identities,external_reference_status='pending',
        accepted_ground_truth=False,automatic_improvement_confirmed=False,
        limitations=['WCS consistency and catalog brightness are not independent proof of identity',
            'same 13 training pairs are conditional, not an independent geometry set',
            'linearized residual bases are diagnostic, not nonlinear camera fits or algorithm variants',
            'LOO holds out a pair from regression only; it is not collection holdout'])
    write_json(out/'results.json',result);preview(out,identities)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['prepare','run'])
    parser.add_argument('--out-dir',type=Path,required=True)
    args=parser.parse_args(); globals()[args.stage](args.out_dir.resolve())
