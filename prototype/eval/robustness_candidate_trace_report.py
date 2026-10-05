#!/usr/bin/env python3
"""Measure frozen pre-acceptance projections; never change a solve or WCS review."""
import argparse
import json
from pathlib import Path

import numpy as np
from astropy.io import fits

from collection_run import ROOT, digest, write_json
from robustness_candidate_trace import FRAME, traces, validate


def project(camera, world):
    ra,dec,roll=np.radians([camera[k] for k in ('ra_deg','dec_deg','roll_deg')])
    forward=np.array([np.cos(dec)*np.cos(ra),np.cos(dec)*np.sin(ra),np.sin(dec)])
    up=np.array([0.,1.,0.]) if abs(forward[2])>.99 else np.array([0.,0.,1.])
    right=np.cross(forward,up); right/=np.linalg.norm(right)
    up=np.cross(right,forward); up/=np.linalg.norm(up)
    rot=np.array([right*np.cos(roll)+up*np.sin(roll),-right*np.sin(roll)+up*np.cos(roll),forward])
    cam=np.asarray(world)@rot.T
    if np.any(cam[...,2]<=.05):
        raise ValueError('behind camera')
    return np.column_stack((camera['width']/2+camera['focal_px']*cam[:,0]/cam[:,2],
                            camera['height']/2-camera['focal_px']*cam[:,1]/cam[:,2]))


def rms(values):
    return float(np.sqrt(np.mean(np.square(values))))


def pair_metrics(row):
    pairs=row['pairs']
    if any('world' not in p for p in pairs):
        raise ValueError('incomplete pair projection')
    xy=np.array([p['xy'] for p in pairs]); predicted=np.array([p['adapter_xy'] for p in pairs])
    discrepancy=float(np.max(np.abs(project(row['camera'],[p['world'] for p in pairs])-predicted)))
    if discrepancy>1e-8:
        raise ValueError('projection convention mismatch')
    native=np.array([p['tetra_centered_xy'] for p in pairs])-np.array([p['input_centroid'] for p in pairs])
    return dict(native_pair_rms_px=rms(np.linalg.norm(native,axis=1)),
                adapter_pair_rms_px=rms(np.linalg.norm(predicted-xy,axis=1)),
                projection_replay_max_error_px=discrepancy)


def boundary_rows(control, centered):
    if control['detections']!=centered['detections']:
        raise ValueError('different detection inputs')
    key=lambda row:{(p['catalog_id'],p['detection_index']) for p in row['pairs']}
    if key(control)!=key(centered):
        raise ValueError('different first-candidate pair identities')
    rows=[]
    for hit in control['verification']['matches']:
        xy=np.array(hit['xy'])
        pred={arm:project(row['camera'],[hit['world']])[0] for arm,row in [('control',control),('centered',centered)]}
        rows.append(dict(detection_index=hit['detection_index'],xy=hit['xy'],world=hit['world'],
            residual_px={arm:float(np.linalg.norm(p-xy)) for arm,p in pred.items()},
            projection_shift_px=(pred['centered']-pred['control']).tolist(),
            diagnostic_centered_plus_half_residual_px=float(np.linalg.norm(pred['centered']+.5-xy)),
            diagnostic_control_minus_half_residual_px=float(np.linalg.norm(pred['control']-.5-xy))))
    return rows


def external_pairs(control):
    folder=ROOT/f'prototype/artifacts/smartphone-wcs-compatible/{FRAME}'
    ref=json.loads((folder/'reference.json').read_text())
    if digest(ROOT/f'data/input/smartphone/{FRAME}.jpg')!=ref['source_sha256']:
        raise ValueError('external reference belongs to different image')
    corr_path=Path(ref['correspondences_file'])
    if not corr_path.is_absolute(): corr_path=ROOT/'prototype'/corr_path
    if digest(corr_path)!=ref['correspondences_sha256']:
        raise ValueError('external correspondences changed')
    corr=fits.getdata(corr_path); corr=corr[corr['match_weight']>=.95]
    ra,dec=np.radians(corr['index_ra']),np.radians(corr['index_dec'])
    units=np.column_stack((np.cos(dec)*np.cos(ra),np.cos(dec)*np.sin(ra),np.sin(dec)))
    field=np.column_stack((corr['field_x'],corr['field_y']))-1
    scale=np.array([ref['width']/control['camera']['width'],ref['height']/control['camera']['height']])
    rows=[]
    for p in control['pairs']:
        world=np.array(p['world'])
        angles=np.degrees(np.arctan2(np.linalg.norm(np.cross(units,world),axis=1),units@world))*3600
        index=int(np.argmin(angles))
        rows.append(dict(catalog_id=p['catalog_id'],detection_index=p['detection_index'],
            nearest_external_angle_arcsec=float(angles[index]),
            external_radec=[float(corr[index]['index_ra']),float(corr[index]['index_dec'])],
            external_xy=field[index].tolist(),detection_full_xy=(np.array(p['xy'])*scale).tolist(),
            detection_to_external_px=float(np.linalg.norm(np.array(p['xy'])*scale-field[index])),
            proposed_identity_same_within_30arcsec=bool(angles[index]<30)))
    return dict(status='pending, not ground truth',reference_sha256=digest(folder/'reference.json'),
                correspondences_sha256=digest(corr_path),source_sha256=ref['source_sha256'],pairs=rows)


def review_crops(out, result):
    """Display-only gamma and crops; input pixels and measured centroids stay intact."""
    from PIL import Image, ImageOps, ImageDraw

    first=result['first_candidates']
    a=first['control']; b=first['centered']; lookup={(p['catalog_id'],p['detection_index']):p for p in b['pairs']}
    im=ImageOps.exif_transpose(Image.open(ROOT/f'data/input/smartphone/{FRAME}.jpg')).convert('RGB')
    canvas=Image.new('RGB',(1000,4*220),(20,20,20)); draw=ImageDraw.Draw(canvas)
    scale=np.array(im.size)/[a['camera']['width'],a['camera']['height']]
    for i,p in enumerate(a['pairs']):
     x,y=np.array(p['xy'])*scale; left,top=int(x)-100,int(y)-90
     tile=im.crop((left,top,left+200,top+180))
     arr=np.asarray(tile)/255.; tile=Image.fromarray(np.uint8(np.power(arr,.45)*255))
     ox,oy=(i%4)*250,(i//4)*220
     canvas.paste(tile,(ox,oy+30)); draw.text((ox+3,oy+3),f"det {p['detection_index']} HYG {p['catalog_id']}",fill='white')
     for pos,color in [(p['xy'],'yellow'),(p['adapter_xy'],'cyan'),(lookup[p['catalog_id'],p['detection_index']]['adapter_xy'],'red')]:
      xx,yy=np.array(pos)*scale-[left,top]+[ox,oy+30]
      draw.ellipse((xx-6,yy-6,xx+6,yy+6),outline=color,width=1)
    canvas.save(out/'pair-crops.png')


def summarize(out):
    validate(out)
    raw={arm:traces(out/f'run-{arm}.log') for arm in ('control','centered')}
    first={arm:rows[0] for arm,rows in raw.items()}
    candidates={}
    for arm,rows in raw.items():
        candidates[arm]=[]
        for index,row in enumerate(rows):
            s=row['solution']; v=row['verification']
            candidates[arm].append(dict(index=index,width=s['image_width'],height=s['image_height'],
                matches=s['num_matches'],prob=s['prob'],parity_flip=s['parity_flip'],hits=v['hits'],
                log_odds=v['log_odds'],camera=row['camera'],**pair_metrics(row)))
    boundary=boundary_rows(first['control'],first['centered'])
    reports={arm:json.loads((out/arm/f'solve-reports/{FRAME}.json').read_text())['report'] for arm in raw}
    result=dict(iteration=19,id=FRAME,protocol_sha256=digest(out/'protocol.json'),
        same_first_candidate_detections=True,same_first_candidate_pair_set=True,
        candidate_counts={arm:len(rows) for arm,rows in raw.items()},candidates=candidates,
        first_candidates=first,boundary=boundary,external_pending=external_pairs(first['control']),
        lost_first_candidate_hits=[r['detection_index'] for r in boundary if r['residual_px']['centered']>3],
        outcomes={arm:dict(status=r['status'],quality=r.get('quality'),failure=r.get('failure')) for arm,r in reports.items()},
        log_sha256={arm:digest(out/f'run-{arm}.log') for arm in raw},
        attempt_timeouts={arm:(out/f'run-{arm}.log').read_text().count('err=Timeout') for arm in raw},
        runs=json.loads((out/'runs.json').read_text()),
        limitations=['one legacy regression, not collection solve-rate experiment',
                    'native tetra3 match count and 3px Starglyph verification have different tolerances',
                    'half-pixel counterfactual on frozen pairs is diagnostic only; not a proposed acceptance change',
                    'pending external agreement does not certify identity or field geometry'])
    write_json(out/'results.json',result)
    review_crops(out,result)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--out-dir',type=Path,required=True)
    summarize(p.parse_args().out_dir.resolve())
