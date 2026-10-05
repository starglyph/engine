#!/usr/bin/env python3
"""Frozen visual probes through the existing detector trace; development only."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance

from collection_run import ROOT, digest, select_records, write_json
from robustness_candidate_trace import command

RID = 'wm_r_134291495'
MANIFEST = ROOT / 'data/samples/sky-samples/manifest.json'
OLD = ROOT / 'prototype/artifacts/robustness-iteration-35'
BINARY = OLD / 'tangential-test'
DIAG = ROOT / f'prototype/artifacts/robustness-iteration-1/diagnosis/{RID}/solve-reports/{RID}.json'
SEEDS = [[3907,412],[2144,468],[2924,484],[3014,562],[3084,950],[3632,956],
         [3766,1008],[3694,1278],[3660,1368],[2070,1122],[2542,1654],[4152,1570],[4308,1202]]


def read(p):
    return json.loads(p.read_text())


def selected():
    rows = select_records(read(MANIFEST), 'development', [RID])
    if len(rows) != 1 or rows[0]['id'] != RID:
        raise ValueError('one development record required')
    return rows[0]


def make_probes(im):
    rows = []
    for i, (x, y) in enumerate(SEEDS):
        patch = np.asarray(im.crop((x-10,y-10,x+11,y+11)).convert('L'))
        yy, xx = np.unravel_index(patch.argmax(), patch.shape)
        kind = 'bright_compact'
        if i == 0: kind = 'selected_positive_control'
        if i in (2, 11, 12): kind = 'faint_ambiguous'
        if i == 8: kind = 'bright_complex'
        rows.append(dict(id=f'P{i:02d}', seed_xy=[x,y], xy=[x-10+int(xx),y-10+int(yy)], visual_class=kind))
    return rows


def prepare(out):
    rec = selected()
    if out.exists(): raise ValueError('fresh output directory required')
    image = MANIFEST.parent / rec['file']
    if digest(image) != rec['clean_sha256']: raise ValueError('image changed')
    binary = read(OLD/'binary.json')
    checks = read(ROOT/'docs/experiments/robustness-iteration-35-checks.json')
    for p in [OLD/'binary.json', OLD/'protocol.json']:
        if digest(p) != checks['artifact_hashes'][str(p.relative_to(ROOT))]: raise ValueError('saved binary provenance changed')
    if digest(BINARY) != binary['sha256'] or digest(OLD/'protocol.json') != binary['protocol_sha256']:
        raise ValueError('saved binary changed')
    old = read(OLD/'protocol.json')
    # The saved binary includes unused test-only LM variants. Check every
    # original core source against its build protocol; source_traces calls only
    # the unchanged production detector, tier configs and image loader.
    core = ROOT/'prototype/crates/starglyph-core/src'
    sources = list(core.rglob('*.rs'))
    for p in sources:
        if digest(p) != old['hashes'][str(p.relative_to(ROOT))]: raise ValueError('production source changed')
    with Image.open(image) as im: probes = make_probes(im.convert('RGB'))
    out.mkdir(parents=True)
    write_json(out/'input.json', dict(id=RID, split='development', image=str(image), image_sha256=digest(image),
        probes=[p['xy'] for p in probes], radius_original_px=12, review=probes,
        provenance='manual central-field inspection; brightest 8-bit luma pixel in fixed +/-10px seed box; no catalogue/WCS identities'))
    paths = [Path(__file__).resolve(), Path(__file__).with_name('collection_run.py'),
        Path(__file__).with_name('robustness_candidate_trace.py'), out/'input.json', MANIFEST, image, DIAG,
        OLD/'binary.json', OLD/'protocol.json', BINARY, *sources,
        *(ROOT/'data/samples/sky-samples'/n for n in ['robustness-results.json','collection-wcs-review.json'])]
    write_json(out/'protocol.json', dict(iteration=37,id=RID,split='development',holdout=False,
        binary=str(BINARY.relative_to(ROOT)), scope='existing source_traces: control, quantile and blob_concentration; four scale/tier lists each',
        probe_count=13, radius_original_px=12, wall_limit_s=180,
        criteria=['all 12 trace passes exactly equal uninstrumented detector, enforced by existing Rust test',
                  'four control lists exactly reproduce saved iteration1 input detections',
                  'retain all probe outcomes, measurements, rank, association distance and centroid offset',
                  'do not equate nearest component association or catalogue-free morphology with star identity',
                  'no automatic improvement or independent WCS claim'],
        hashes={str(p.relative_to(ROOT)):digest(p) for p in paths}))


def validate(out):
    p = read(out/'protocol.json')
    if (p['id'],p['split'],p['holdout']) != (RID,'development',False): raise ValueError('development only')
    selected()
    for name, sha in p['hashes'].items():
        if digest(ROOT/name) != sha: raise ValueError('frozen input changed: '+name)
    return p


def run(out):
    validate(out)
    if (out/'run.json').exists(): raise ValueError('fresh run required')
    command(out,'run',[str(BINARY),'--exact','solve::replay::source_traces','--ignored'],
        {**os.environ,'STARGLYPH_TRACE_INPUT':str(out/'input.json'),'STARGLYPH_TRACE_OUTPUT':str(out/'trace.json')},180)


def probe_row(meta, trace, sx, sy):
    component = trace['component']
    row = dict(id=meta['id'],visual_class=meta['visual_class'],xy=meta['xy'],
        working_xy=trace['point'],distance_to_component_working_px=trace['distance_to_component'],component=component,
        outcome=component['outcome'] if component else 'no_component_in_radius')
    if component and component['centroid'] is not None:
        xy = [(component['centroid'][0]+.5)/sx-.5,(component['centroid'][1]+.5)/sy-.5]
        row['centroid_original_xy'] = xy
        row['centroid_offset_original_px'] = float(np.linalg.norm(np.array(xy)-meta['xy']))
    return row


def summarize(out):
    validate(out); inp=read(out/'input.json'); raw=read(out/'trace.json')['tiers']; old=read(DIAG)['detection_diagnostics']
    if len(raw)!=12: raise ValueError('12 trace tiers required')
    tiers=[];exact=0
    for t in raw:
        mode = 'quantile' if t['quantile'] else 'blob_concentration' if t['blob_concentration'] else 'control'
        if len(t['probes']) != 13: raise ValueError('all probes required')
        if mode=='control':
            d=next(d for d in old if (d['width'],d['tier'])==(t['width'],t['tier']))
            if t['result']['detections'] != d['result']['detections'][:d['max_detections']]:
                raise ValueError('control input mismatch')
            exact+=1
        rows=[probe_row(p,tr,t['width']/6016,t['height']/3760) for p,tr in zip(inp['review'],t['probes'])]
        tiers.append(dict(mode=mode,width=t['width'],height=t['height'],tier=t['tier'],
            detections=len(t['result']['detections']), stats=t['result']['stats'],
            outcomes=dict(Counter(r['outcome'] for r in rows)),probes=rows))
    if exact!=4 or len({(t['mode'],t['width'],t['tier'])for t in tiers})!=12:
        raise ValueError('trace scope changed')
    write_json(out/'results.json',dict(iteration=37,id=RID,split='development',holdout=False,
        protocol_sha256=digest(out/'protocol.json'),unchanged_trace_passes=12,exact_saved_control_lists=exact,
        elapsed_s=read(out/'run.json')['elapsed_s'],tiers=tiers,independent_ground_truth=False,
        production_changed=False,automatic_improvement_confirmed=False))


def figure(out):
    validate(out);inp=read(out/'input.json')
    with Image.open(inp['image']) as source: im=source.convert('RGB')
    sheet=Image.new('RGB',(768,880),(15,15,15));draw=ImageDraw.Draw(sheet)
    for i,p in enumerate(inp['review']):
        cx,cy=p['xy'];crop=ImageEnhance.Brightness(im.crop((cx-64,cy-64,cx+64,cy+64))).enhance(1.5).resize((192,192))
        d=ImageDraw.Draw(crop);d.ellipse((82,82,110,110),outline='yellow',width=1)
        bx=i%4*192;by=i//4*220;sheet.paste(crop,(bx,by));draw.text((bx+4,by+196),f"{p['id']} [{cx},{cy}]",fill='white')
    sheet.save(out/'sources.jpg',quality=90)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['prepare','validate','run','summarize','figure']);p.add_argument('--out-dir',type=Path,required=True)
    a=p.parse_args();globals()[a.stage](a.out_dir.resolve())
