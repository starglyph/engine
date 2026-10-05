#!/usr/bin/env python3
"""Frozen visual/centroid audit of iteration25's seven changed detections."""
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
from compare_local_wcs import project
from robustness_geometry import selected, native_centroid, validate_review

RID = 'wm_r_132162731'
INDICES = [2, 5, 7, 8, 19, 37, 39]
PRIOR = ROOT/'prototype/artifacts/robustness-iteration-25/fixed'
CATALOG = ROOT/'data/catalogs/hyg_v42.csv.gz'
GEOMETRY = ROOT/'docs/experiments/robustness-iteration-13-geometry.json'


def load_inputs():
    rec = selected([RID])[0]  # Check split before reading pixels or local traces.
    protocol = json.loads((PRIOR/'replay-protocol.json').read_text())
    for name, sha in protocol['hashes'].items():
        if digest(ROOT/name) != sha:
            raise ValueError('changed replay input: '+name)
    inputs = json.loads((PRIOR/'replay-plan.json').read_text())['inputs']
    paths = json.loads((PRIOR/'observed-paths.json').read_text())
    if inputs['control']['detections'] != inputs['huber']['detections']:
        raise ValueError('different detections')
    if any(inputs[a]['epoch_years'] is not None for a in inputs):
        raise ValueError('this catalogue lookup requires frozen epoch-free input')
    return rec, inputs, paths


def stages(inp, path):
    yield 'initial', 3., inp['camera'], inp['verification']
    previous = path[0]['camera']
    for s in path[1:]:
        if s['stage'] == 'rematch':
            yield 'rematch_'+str(s['radius_px']), s['radius_px'], previous, s['input']
        previous = s['camera']


def matched_worlds(inputs, paths):
    worlds = {i:[] for i in INDICES}
    for arm in ['control','huber']:
        for _, _, _, v in stages(inputs[arm],paths[arm]):
            for index, m in zip(v['detection_indices'],v['matches']):
                if index in worlds and m['world'] not in worlds[index]:
                    worlds[index].append(m['world'])
    if any(len(v) != 1 for v in worlds.values()):
        raise ValueError('expected one frozen identity per changed source')
    return {i:v[0] for i,v in worlds.items()}


def prepare(out):
    rec, inputs, paths = load_inputs()
    if out.exists():
        raise ValueError('fresh output required')
    image_path = ROOT/'data/samples/sky-samples'/rec['file']
    if digest(image_path) != rec['clean_sha256']:
        raise ValueError('source pixels changed')
    with gzip.open(CATALOG,'rt') as f:
        stars = [s for s in csv.DictReader(f) if s['mag'] and float(s['mag']) <= 6.8]
    # Match Catalog::parse_coordinates: prefer precise radian columns.
    radec = np.array([np.rad2deg([float(s['rarad']),float(s['decrad'])])
        if s.get('rarad') and s.get('decrad') else [float(s['ra'])*15,float(s['dec'])] for s in stars])
    ra, dec = np.deg2rad(radec).T
    units = np.column_stack((np.cos(dec)*np.cos(ra),np.cos(dec)*np.sin(ra),np.sin(dec)))
    worlds = matched_worlds(inputs,paths)
    probes = next(f['sources'] for f in json.loads(GEOMETRY.read_text())['frames'] if f['id']==RID)
    extraction = ROOT/f'prototype/artifacts/robustness-baseline/wcs-run/{RID}/downsample-2/field.axy'
    data = fits.getdata(extraction)
    header = fits.getheader(extraction)
    if (header['IMAGEW'],header['IMAGEH']) != (rec['width'],rec['height']):
        raise ValueError('extraction coordinates mismatch')
    xy = np.column_stack((data['X'],data['Y'])).astype(float)-1
    reference_path = extraction.parent.parent/'reference.json'
    reference = json.loads(reference_path.read_text())
    wcs_path = ROOT/'prototype'/reference['wcs_file'] if reference.get('wcs_file') else None
    if reference['source_sha256'] != rec['clean_sha256'] or (wcs_path and digest(wcs_path) != reference['wcs_sha256']):
        raise ValueError('external WCS changed')
    points = []
    for index in INDICES:
        det = inputs['control']['detections'][index]
        center = np.array([det['x'],det['y']])
        ids = np.flatnonzero(np.linalg.norm(units-worlds[index],axis=1)<1e-12)
        if len(ids) != 1:
            raise ValueError('ambiguous catalogue lookup')
        row = int(ids[0]); star = stars[row]
        nearest = np.argsort(np.linalg.norm(xy-center,axis=1))[:3]
        angular = np.rad2deg(np.arccos(np.clip(units@worlds[index],-1,1)))
        points.append(dict(id=index,hyg_id=int(star['id']),name=star['proper'] or star['bf'],
            mag=float(star['mag']),world=worlds[index],radec=radec[row].tolist(),detection=det,
            review_center=center.tolist(),choices=[dict(row=int(i),x=float(xy[i,0]),y=float(xy[i,1])) for i in nearest],
            prior_probes=[s for s in probes if np.linalg.norm(np.array(s['xy'])-center)<12],
            catalogue_neighbours=[dict(hyg_id=int(stars[i]['id']),separation_deg=float(angular[i])) for i in np.flatnonzero((angular<.2)&(angular>1e-5))],
            label='pending',selected_row=None))
    out.mkdir(parents=True)
    frame = dict(id=RID,image=str(image_path.relative_to(ROOT)),source_sha256=rec['clean_sha256'],
        width=rec['width'],height=rec['height'],points=points)
    for name in ['candidates.json','review.json']:
        write_json(out/name,dict(split='development',frames=[frame]))
    with Image.open(image_path) as im:
        image = im.convert('RGB')
    overview = image.copy(); overview.thumbnail((900,1200))
    draw = ImageDraw.Draw(overview)
    sheet = Image.new('RGB',(1024,1200)); sd = ImageDraw.Draw(sheet)
    for j,p in enumerate(points):
        x,y = p['review_center']; sx,sy=overview.width/image.width,overview.height/image.height
        draw.ellipse((x*sx-5,y*sy-5,x*sx+5,y*sy+5),outline='cyan')
        draw.text((x*sx+7,y*sy),str(p['id']),fill='yellow')
        left,top=(j%2)*512,(j//2)*300
        for col,radius in enumerate([32,128]):
            crop=image.crop((round(x)-radius,round(y)-radius,round(x)+radius,round(y)+radius))
            crop=ImageOps.autocontrast(crop,cutoff=.5).resize((256,256),Image.Resampling.NEAREST)
            sheet.paste(crop,(left+col*256,top+35))
            for dx in [-16,8]:
                sd.line((left+col*256+128+dx,top+163,left+col*256+136+dx,top+163),fill='cyan')
        sd.text((left+5,top+3),f"det {p['id']} HYG {p['hyg_id']} {p['name']} mag {p['mag']}",fill='white')
        sd.text((left+5,top+18),f"xy {x:.1f},{y:.1f}; left 64px x4 / right 256px x1",fill='white')
    overview.save(out/'overview.png');sheet.save(out/'sources.png')
    files=[Path(__file__).resolve(),image_path,CATALOG,GEOMETRY,extraction,reference_path,
        PRIOR/'replay-protocol.json',PRIOR/'replay-plan.json',PRIOR/'observed-paths.json',
        ROOT/'prototype/crates/starglyph-core/src/catalog.rs',ROOT/'prototype/eval/robustness_geometry.py',ROOT/'prototype/eval/compare_local_wcs.py',ROOT/'prototype/eval/collection_run.py']
    if wcs_path: files.append(wcs_path)
    files.extend(ROOT/'data/samples/sky-samples'/name for name in ['manifest.json','robustness-results.json','collection-wcs-review.json'])
    write_json(out/'protocol.json',dict(iteration=26,id=RID,split='development',holdout=False,
        detection_indices=INDICES,selection='all seven changed detections fixed by iteration25, before visual review',
        hashes={str(p.relative_to(ROOT)):digest(p) for p in files},candidates_sha256=digest(out/'candidates.json'),
        reference=str(wcs_path.relative_to(ROOT)) if wcs_path else None,external_status=reference['status'],
        measurements='existing native_centroid radii 8/12 and annulus 16-22; saved external centroids; no fits',
        criteria='separate physical source, conditional catalogue identity, centroid stability, threshold crossings; no GT promotion or automatic heuristic',
        preview='local autocontrast 0.5 percent, nearest resize only for inspection; original pixels used for measurements'))


def measure(out):
    selected([RID])
    protocol=json.loads((out/'protocol.json').read_text())
    if (protocol['id'],protocol['split'],protocol['holdout'],protocol['detection_indices']) != (RID,'development',False,INDICES):
        raise ValueError('frozen development selection required')
    for name,sha in protocol['hashes'].items():
        if digest(ROOT/name)!=sha: raise ValueError('changed input: '+name)
    if digest(out/'candidates.json')!=protocol['candidates_sha256']: raise ValueError('candidates changed')
    review=json.loads((out/'review.json').read_text())
    validate_review(json.loads((out/'candidates.json').read_text()),review)
    frame=review['frames'][0]
    for p in frame['points']:
        if p['label'] not in ['visible_source','ambiguous','blend','not_visible'] or not p.get('note'):
            raise ValueError('complete image review before measuring')
        if p['selected_row'] is not None and p['selected_row'] not in [c['row'] for c in p['choices']]:
            raise ValueError('external source not in frozen choices')
    _,inputs,paths=load_inputs()
    image=Image.open(ROOT/frame['image']).convert('RGB')
    gray=np.asarray(image,dtype=np.float32)@np.array([.2126,.7152,.0722],dtype=np.float32)
    wcs=WCS(fits.getheader(ROOT/protocol['reference'])) if protocol['reference'] else None
    rows=[]
    for p in frame['points']:
        center=np.array(p['review_center']); radec=[p['radec']]
        native={f'r{r}':native_centroid(gray,center,r) for r in [8,12]}
        external=next(([c['x'],c['y']] for c in p['choices'] if c['row']==p['selected_row']),None)
        stage_rows=[]
        for arm in ['control','huber']:
            for name,radius,camera,v in stages(inputs[arm],paths[arm]):
                pred=project(camera,radec)[0]
                stage_rows.append(dict(arm=arm,stage=name,radius_px=radius,matched=p['id'] in v['detection_indices'],
                    predicted_xy=pred.tolist(),detector_error_px=float(np.linalg.norm(pred-center))))
        ext_wcs=wcs.all_world2pix(np.array(radec),0)[0] if wcs else None
        rows.append(dict(**p,native_centroids=native,external_centroid=external,
            native_offset_px={k:float(np.linalg.norm(np.array(xy)-center)) for k,xy in native.items()},
            native_r8_r12_distance_px=float(np.linalg.norm(np.array(native['r8'])-native['r12'])),
            external_offset_px=None if external is None else float(np.linalg.norm(np.array(external)-center)),
            pending_wcs_prediction=ext_wcs.tolist() if ext_wcs is not None else None,
            pending_wcs_detector_error_px=float(np.linalg.norm(ext_wcs-center)) if ext_wcs is not None else None,
            stages=stage_rows))
    write_json(out/'results.json',dict(iteration=26,id=RID,holdout=False,independent_ground_truth=False,
        protocol_sha256=digest(out/'protocol.json'),review_sha256=digest(out/'review.json'),points=rows,
        algorithm_change=False,new_searches=0,new_refinements=0))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['prepare','measure']);parser.add_argument('--out-dir',type=Path,required=True)
    args=parser.parse_args();globals()[args.stage](args.out_dir.resolve())
