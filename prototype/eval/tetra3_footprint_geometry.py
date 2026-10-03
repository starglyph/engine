#!/usr/bin/env python3
"""Conditional geometry check on reviewed sources withheld from the solver list."""
import argparse
import json
from pathlib import Path

import numpy as np
from astropy.io import fits
from PIL import Image, ImageDraw

from collection_run import ROOT, digest, select_records, write_json
from compare_local_wcs import project, stats

RID = 'wm_r_161606532'


def review(out, output):
    manifest = ROOT / 'data/samples/sky-samples/manifest.json'
    record = select_records(json.loads(manifest.read_text()), 'development', [RID])[0]
    image = manifest.parent / record['file']
    if digest(image) != record['clean_sha256']:
        raise ValueError('image changed')
    path = out / f'development/footprint/{RID}/solve-reports/{RID}.json'
    artifact = json.loads(path.read_text())
    if artifact['source_sha256'] != digest(image) or artifact['report']['status'] != 'solved':
        raise ValueError('no matching solved development artifact')
    prior = ROOT / 'prototype/artifacts/robustness-iteration-3'
    associations = json.loads((prior / 'sources.json').read_text())['associations']
    anchors = [a for a in associations if a['selected_probe']]
    refpath = ROOT / f'prototype/artifacts/robustness-baseline/wcs-run/{RID}/reference.json'
    reference = json.loads(refpath.read_text())
    corrpath = ROOT / 'prototype' / reference['correspondences_file']
    if digest(corrpath) != reference['correspondences_sha256']:
        raise ValueError('external correspondence artifact changed')
    corr = fits.getdata(corrpath)[[a['corr_row'] for a in anchors]]
    measured = np.column_stack((corr['field_x']-1, corr['field_y']-1))
    predicted = project(artifact['camera'], np.column_stack((corr['index_ra'], corr['index_dec'])))
    distance = np.linalg.norm(measured-predicted, axis=1)
    traces = json.loads((prior / 'expanded/traces.json').read_text())['tiers']
    trace = next(t for t in traces if t['width'] == 4080 and t['tier'] == 'default'
                 and not t['quantile'] and not t['blob_concentration'])
    probes = trace['probes'][:len(anchors)]
    withheld = np.array([p['component']['outcome'] != 'selected' for p in probes])
    # Cross-check the exclusion against the actual final solver input, not just
    # trace labels: non-selected probe positions must not name a fitted centroid.
    detections = np.array([[d['x'],d['y']] for d in artifact['report']['detections']])
    nearest = np.min(np.linalg.norm(measured[:,None,:]-detections[None,:,:],axis=2),axis=1)
    if np.any(nearest[withheld] < 12):
        raise ValueError('a withheld probe is close to a solver detection')
    groups = {'all_reviewed_56': np.ones(len(anchors),dtype=bool), 'outside_solver_input':withheld,
              'left_10pct':measured[:,0]<408, 'right_10pct':measured[:,0]>3672,
              'top_10pct':measured[:,1]<307.2}
    result = dict(id=RID, split='development', external_status='pending', report_sha256=digest(path),
        corr_sha256=digest(corrpath), geometry='conditional on reviewed stellar identities; no global ground truth',
        groups={k:stats(distance[v]) for k,v in groups.items()},
        withheld_extent_fraction=(np.ptp(measured[withheld],axis=0)/[4080,3072]).tolist(),
        sources=[dict(hyg_id=a['hyg_id'], corr_row=a['corr_row'], measured_xy=m.tolist(),
            predicted_xy=p.tolist(), error_px=float(e), withheld_from_solver=bool(w),
            nearest_solver_detection_px=float(n)) for a,m,p,e,w,n in zip(anchors,measured,predicted,distance,withheld,nearest)],
        limitations=['sky covers upper portion only; lower field occluded',
                     'external centroids and identities are reviewed candidates, not accepted ground truth'])
    write_json(output, result)
    choices = np.flatnonzero(withheld)
    indices = sorted(set([int(choices[np.argmin(measured[choices,0])]), int(choices[np.argmax(measured[choices,0])]),
                          int(choices[np.argmin(measured[choices,1])]), int(choices[np.argmax(measured[choices,1])]),
                          int(choices[len(choices)//3]), int(choices[2*len(choices)//3])]))
    with Image.open(image) as im:
        sheet = Image.new('RGB',(len(indices)*256,284))
        draw = ImageDraw.Draw(sheet)
        for col,i in enumerate(indices):
            x,y=measured[i];x0,y0=round(x)-64,round(y)-64
            sheet.paste(im.crop((x0,y0,x0+128,y0+128)).resize((256,256)),(col*256,28))
            draw.text((col*256+3,4),f'HYG {anchors[i]["hyg_id"]}; {distance[i]:.2f}px',fill='white')
            for (px,py),color in [(measured[i],'cyan'),(predicted[i],'yellow')]:
                px=(px-x0)*2+col*256;py=(py-y0)*2+28
                draw.ellipse((px-7,py-7,px+7,py+7),outline=color,width=1)
        sheet.save(out / 'withheld-geometry.jpg')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('out',type=Path)
    p.add_argument('report',type=Path)
    a=p.parse_args();review(a.out,a.report)


if __name__ == '__main__':
    main()
