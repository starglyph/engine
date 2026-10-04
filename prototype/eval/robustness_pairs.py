#!/usr/bin/env python3
"""Prepare immutable visual review of the eleven development fit pairs."""
import argparse
import csv
import gzip
import json
from pathlib import Path

import numpy as np
from astropy.io import fits
from PIL import Image, ImageDraw, ImageEnhance, ImageOps

from collection_run import ROOT, digest, write_json
from robustness_geometry import native_centroid, selected, validate_review
from robustness_refinement import RID

PRIOR = ROOT/'prototype/artifacts/robustness-iteration-6/isolated'
REVIEW = ROOT/'docs/experiments/robustness-iteration-5-review.json'


def load_traces():
    selected([RID])
    published = json.loads((ROOT/'docs/experiments/robustness-iteration-6-geometry.json').read_text())
    traces = {}
    for arm in ('control','footprint'):
        path = PRIOR/arm/'trace.json'
        if digest(path) != published['trace_sha256'][arm]:
            raise ValueError('iteration-6 trace changed')
        traces[arm] = json.loads(path.read_text())
        if traces[arm]['id'] != RID or traces[arm]['split'] != 'development':
            raise ValueError('development trace required')
    return traces


def initial_candidate(traces, arm):
    tier = 'deep' if arm == 'control' else 'default'
    return next(c for t in traces[arm]['tiers'] if t['tier']==tier for c in t['candidates'] if c['chosen'])


def prepare(out):
    traces = load_traces()
    if out.exists():
        raise ValueError('fresh output directory required')
    frame = next(f for f in json.loads(REVIEW.read_text())['frames'] if f['id']==RID)
    image_path = ROOT/frame['image']
    if digest(image_path) != frame['source_sha256']:
        raise ValueError('image changed')
    catalog_path = ROOT/'data/catalogs/hyg_v42.csv.gz'
    with gzip.open(catalog_path,'rt') as stream:
        catalog = {int(s['id']):s for s in csv.DictReader(stream)}
    extraction = ROOT/frame['extraction']
    axy = fits.getdata(extraction)
    xy = np.column_stack((axy['X'],axy['Y'])).astype(float)-1
    with Image.open(image_path) as image:
        rgb = np.asarray(image.convert('RGB'),dtype=np.float32)
        gray = rgb @ np.array([.2126,.7152,.0722],dtype=np.float32)
        bright = ImageEnhance.Brightness(image.convert('RGB')).enhance(1.1)
    stage = initial_candidate(traces,'footprint')['stages'][0]
    scale = np.array([frame['width']/stage['camera']['width'],frame['height']/stage['camera']['height']])
    points = []
    for i,pair in enumerate(stage['pairs']):
        if len(pair['hyg_ids']) != 1:
            raise ValueError('ambiguous input catalogue ID')
        star = catalog[pair['hyg_ids'][0]]
        center = (np.array(pair['xy'])+.5)*scale-.5
        nearest = np.argsort(np.linalg.norm(xy-center,axis=1))[:3]
        points.append(dict(id=i,hyg_id=int(star['id']),name=star['proper'] or star['bf'],mag=float(star['mag']),
            ra_deg=float(star['ra'])*15,dec_deg=float(star['dec']),world=pair['world'],
            detector_xy_original=center.tolist(),detector_xy_working=pair['xy'],
            native_r8_xy=native_centroid(gray,center,8),native_r12_xy=native_centroid(gray,center,12),
            choices=[dict(row=int(n),x=float(xy[n,0]),y=float(xy[n,1]),distance_px=float(np.linalg.norm(xy[n]-center))) for n in nearest],
            label='pending',selected_row=None))
    out.mkdir(parents=True)
    template=dict(split='development',frames=[dict(id=RID,image=frame['image'],source_sha256=frame['source_sha256'],
        width=frame['width'],height=frame['height'],points=points)])
    write_json(out/'candidates.json',template)
    write_json(out/'review.json',template)
    sheet=Image.new('RGB',(1200,1200))
    draw=ImageDraw.Draw(sheet)
    for point in points:
        i=point['id']; x,y=point['detector_xy_original']; left,top=(i%4)*300,(i//4)*400
        tile=bright.crop((round(x)-75,round(y)-75,round(x)+75,round(y)+75)).resize((300,300))
        tile=ImageOps.autocontrast(tile,cutoff=.5)  # Display only; measurements use untouched pixels.
        sheet.paste(tile,(left,top))
        # Keep the core unobscured: marks lie outside the central source.
        draw.line((left+140,top+150,left+130,top+150),fill='cyan')
        draw.line((left+160,top+150,left+170,top+150),fill='cyan')
        draw.text((left+8,top+307),f"{i}: HYG {point['hyg_id']} {point['name']}",fill='white')
        draw.text((left+8,top+328),f"mag {point['mag']}; axy offset {point['choices'][0]['distance_px']:.2f}px",fill='white')
    sheet.save(out/'pairs.jpg')
    overview=bright.copy();overview.thumbnail((1000,1400));draw=ImageDraw.Draw(overview)
    for point in points:
        x,y=np.array(point['detector_xy_original'])*[overview.width/frame['width'],overview.height/frame['height']]
        draw.ellipse((x-6,y-6,x+6,y+6),outline='cyan');draw.text((x+8,y),str(point['id']),fill='yellow')
    overview.save(out/'overview.jpg')
    files=[image_path,catalog_path,extraction,REVIEW,Path(__file__),ROOT/'prototype/eval/robustness_geometry.py',
           ROOT/'docs/experiments/robustness-iteration-6-geometry.json']
    write_json(out/'review-protocol.json',dict(id=RID,split='development',candidates_sha256=digest(out/'candidates.json'),
        hashes={str(p.relative_to(ROOT)):digest(p) for p in files},
        selection='all eleven initial footprint fit pairs; no selection by leave-one-out outcome',
        identity_limit='visual source/pattern review conditional on prior camera proposals; no absolute GT'))


def reviewed(out):
    protocol=json.loads((out/'review-protocol.json').read_text())
    if protocol['id'] != RID or protocol['split'] != 'development':
        raise ValueError('development required')
    selected([RID])
    for path,sha in protocol['hashes'].items():
        if digest(ROOT/path) != sha:
            raise ValueError('review input changed')
    if digest(out/'candidates.json') != protocol['candidates_sha256']:
        raise ValueError('review candidates changed')
    review=json.loads((out/'review.json').read_text())
    validate_review(json.loads((out/'candidates.json').read_text()),review)
    points=review['frames'][0]['points']
    used=set()
    for point in points:
        if point['label'] not in ('visible_source','ambiguous','blend','not_visible'):
            raise ValueError('complete visual review')
        if point['label']=='visible_source':
            row=point['selected_row']
            if row not in [c['row'] for c in point['choices']] or row in used:
                raise ValueError('invalid or reused external centroid')
            used.add(row)
    return points


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('out',type=Path)
    a=p.parse_args()
    prepare(a.out.resolve())
