#!/usr/bin/env python3
"""Image-only registration of one existing overlapping smartphone neighbor.

Reuses four-anchor registration and aperture centroiding. No catalog, WCS or
solve is used; all disputed sources are excluded from the fit and its controls.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from collection_run import ROOT, digest, write_json
from review_external_stars import registration_check
from robustness_geometry import native_centroid
from sky_statistics_experiment import point_in_polygon

ANNOTATIONS=ROOT/'docs/experiments/robustness-iteration-21-annotations.json'
IDS=['20260831_214709','20260831_214642','20260831_214748']


def image_path(rid):
    if rid not in IDS:
        raise ValueError('only frozen legacy images allowed')
    return ROOT/f'data/input/smartphone/{rid}.jpg'


def prepare(out):
    if out.exists():
        raise ValueError('fresh output directory required')
    ann=json.loads(ANNOTATIONS.read_text())
    if [ann['source'],ann['target'],ann['second_neighbor']['id']]!=IDS:
        raise ValueError('fixed legacy selection required')
    files=[ANNOTATIONS,Path(__file__).resolve(),ROOT/'data/input/smartphone/manifest.json',
        ROOT/'docs/experiments/robustness-iteration-19-results.json',
        ROOT/ann['target_detections_source'],ROOT/ann['second_neighbor']['manual_detections_preview_source']]
    files.extend(image_path(rid) for rid in IDS)
    files.extend(ROOT/'prototype/eval'/name for name in ('review_external_stars.py','robustness_geometry.py',
        'sky_statistics_experiment.py','compare_local_wcs.py','collection_run.py'))
    out.mkdir(parents=True)
    write_json(out/'protocol.json',dict(iteration=21,ids=IDS,holdout=False,
        hashes={str(p.relative_to(ROOT)):digest(p) for p in files},
        source='four visual image anchors; no catalogue or camera pose',
        controls='three visually matched sources excluded from homography fitting; selected after provisional review',
        centroid_radii_px=[8,12],background_annulus_px=[16,22],
        photometry='8-bit luminance; median annulus; scale=max(1,1.4826*MAD); descriptive contrast, not calibrated probability',
        contrast_aperture_px=12,search_aperture_px=20,
        criteria=['retain all four disputed sources, including out-of-frame and extrapolated predictions',
                  'report each control residual and aperture sensitivity without tuning anchor membership',
                  'absence cannot certify a false catalog match; changes in visibility remain possible',
                  'no algorithm change, reference promotion, new images, solve or WCS run']))


def validate(out):
    p=json.loads((out/'protocol.json').read_text())
    if p['ids']!=IDS or p['holdout']:
        raise ValueError('fixed legacy selection required')
    for name,sha in p['hashes'].items():
        if digest(ROOT/name)!=sha:
            raise ValueError('frozen input changed: '+name)
    return p


def project(homography, xy):
    p=np.append(np.asarray(xy,dtype=float),1.)@np.asarray(homography).T
    if abs(p[2])<1e-10:
        raise ValueError('projection at infinity')
    return p[:2]/p[2]


def contrast(gray, xy, aperture):
    x,y=xy
    if not (23<=x<gray.shape[1]-24 and 23<=y<gray.shape[0]-24):
        return dict(status='outside_measurement_margin')
    x0,y0=int(x)-23,int(y)-23
    patch=gray[y0:y0+47,x0:x0+47];yy,xx=np.mgrid[y0:y0+47,x0:x0+47]
    radius=np.hypot(xx-x,yy-y)
    ring=patch[(radius>=16)&(radius<=22)]
    background=float(np.median(ring));noise=max(1.,1.4826*float(np.median(abs(ring-background))))
    values=patch[radius<=aperture]-background
    return dict(status='measured',background=background,robust_scale=noise,
        peak_over_scale=float(values.max()/noise),sum_over_scale_sqrt_n=float(values.sum()/noise/np.sqrt(len(values))))


def fit(ann, gray, radius):
    registration=[];review=[];measured={}
    for i,p in enumerate(ann['points']):
        a=native_centroid(gray[0],p['source_xy'],radius)
        b=native_centroid(gray[1],p['target_xy'],radius)
        registration.append(dict(source_id=i,primary_xy=a,role=p['role']))
        review.append(dict(id=i,x=b[0],y=b[1],label='visible_source'))
        measured[p['id']]=dict(source_xy=a,target_xy=b)
    result=registration_check(dict(points=registration),dict(points=review))
    result['measured']=measured
    for p,row in zip(ann['points'],result['points']):
        row['name']=p['id']
    return result


def render(out, images, cases):
    probes=cases['r12']['probes']
    canvas=Image.new('RGB',(900,len(probes)*200),(20,20,20));draw=ImageDraw.Draw(canvas)
    for j,p in enumerate(probes):
        locations=[(images[0],p['source_xy'],'source'),(images[1],p['predicted_xy'],'registered'),
                   (images[1],p['source_xy'],'same sensor xy')]
        for k,(image,xy,label) in enumerate(locations):
            x,y=xy;ox,oy=k*300,j*200;draw.text((ox,oy),f"det {p['id']}: {label}",fill='white')
            if not (45<x<image.width-45 and 45<y<image.height-45):
                draw.text((ox+10,oy+50),'outside image',fill='orange');continue
            crop=np.asarray(image.crop((int(x)-45,int(y)-45,int(x)+45,int(y)+45)))/255.
            tile=Image.fromarray(np.uint8(crop**.45*255)).resize((180,180))
            canvas.paste(tile,(ox,oy+20));draw.ellipse((ox+84,oy+104,ox+96,oy+116),outline='yellow')
    canvas.save(out/'source-transfer.png')


def run(out):
    protocol=validate(out);ann=json.loads(ANNOTATIONS.read_text())
    images=[ImageOps.exif_transpose(Image.open(image_path(rid))).convert('RGB') for rid in IDS[:2]]
    for image,key in zip(images,('source_size','target_size')):
        if list(image.size)!=ann[key]:raise ValueError('image dimensions changed')
    gray=[np.asarray(im.convert('L'),dtype=float) for im in images]
    cases={}
    for radius in protocol['centroid_radii_px']:
        registration=fit(ann,gray,radius)
        hull=[registration['measured'][name]['source_xy'] for name in ann['anchor_hull_ids']]
        probes=[]
        for p in ann['probes']:
            predicted=project(registration['homography'],p['source_xy'])
            inside=bool(0<=predicted[0]<images[1].width and 0<=predicted[1]<images[1].height)
            probes.append(dict(p,predicted_xy=predicted.tolist(),inside_target=inside,
                inside_anchor_hull=point_in_polygon(*p['source_xy'],hull),
                source_contrast=contrast(gray[0],p['source_xy'],12),
                registered_contrast=contrast(gray[1],predicted,12),
                registered_search_contrast=contrast(gray[1],predicted,20),
                same_sensor_contrast=contrast(gray[1],p['source_xy'],12)))
        cases[f'r{radius}']=dict(registration=registration,probes=probes)
    sensitivities=[dict(id=a['id'],displacement_px=float(np.linalg.norm(np.array(a['predicted_xy'])-b['predicted_xy'])))
        for a,b in zip(cases['r8']['probes'],cases['r12']['probes'])]
    result=dict(iteration=21,ids=IDS,protocol_sha256=digest(out/'protocol.json'),cases=cases,
        centroid_aperture_sensitivity=sensitivities,second_neighbor=ann['second_neighbor'],
        accepted_ground_truth=False,automatic_improvement_confirmed=False,
        limitations=['manual diagnostic registration; no catalog identity certification',
                     'controls not fitted but chosen after provisional alignment, spatial coverage sparse',
                     'contrast is descriptive for correlated processed JPEG noise, not Gaussian significance',
                     'absence at prediction can reflect visibility or model error; not proof of a false source'])
    write_json(out/'results.json',result);render(out,images,cases)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['prepare','run'])
    parser.add_argument('--out-dir',type=Path,required=True)
    args=parser.parse_args();globals()[args.stage](args.out_dir.resolve())
