#!/usr/bin/env python3
"""Trace eight frozen development probes with the existing detector harness."""
import argparse
from collections import Counter
import os
from pathlib import Path

from PIL import Image, ImageDraw

from collection_run import ROOT, digest, select_records, write_json
from robustness_candidate_trace import command
from robustness_central_sources import probe_row, read

RID = 'wm_r_16053617'
MANIFEST = ROOT/'data/samples/sky-samples/manifest.json'
OLD = ROOT/'prototype/artifacts/robustness-iteration-54'
BINARY = OLD/'replay-test'
SELECTION = ROOT/'docs/experiments/robustness-iteration-55-input.json'
DIAG = ROOT/f'prototype/artifacts/robustness-iteration-1/diagnosis/{RID}/solve-reports/{RID}.json'
POSITIONS = (0, 1, 2, 18, 19, 20, 22, 23)
EXPECTED_RANKS = (0, 2, 3, 34, 35, 37, 40, 41)
CASE = '2772-default-manual_sky'


def selected():
    records = select_records(read(MANIFEST), 'development', [RID])
    if len(records) != 1 or records[0]['id'] != RID or records[0]['track'] != 'solver':
        raise ValueError('one fixed development solver required')
    return records[0]


def make_probes(saved, diagnostics):
    case = next(c for c in saved['cases'] if c['name'] == CASE)
    selection = next(c for c in saved['selections'] if c['name'] == CASE)
    native = next(d for d in diagnostics if (d['width'], d['tier']) == (2772, 'default'))
    if (saved['split'], case['id'], case['height']) != ('development', RID, 1935):
        raise ValueError('unexpected probe provenance')
    probes = []
    for position, expected in zip(POSITIONS, EXPECTED_RANKS):
        source = case['search'][position]
        if selection['original_indices'][position] != expected or source != native['result']['detections'][expected]:
            raise ValueError('saved source identity changed')
        kind = 'compact_control' if position in POSITIONS[:4] else 'compact_on_bright_background'
        if position == 20:
            kind = 'on_linear_trail_identity_ambiguous'
        probes.append(dict(id=f'P{position:02d}', manual_position=position, native_default_rank=expected,
            xy=[source['x'], source['y']], visual_class=kind, saved_detection=source))
    return probes


def prepare(out):
    record = selected()
    if out.exists():
        raise ValueError('fresh output directory required')
    image = MANIFEST.parent/record['file']
    if digest(image) != record['clean_sha256']:
        raise ValueError('image changed')
    checks = read(ROOT/'docs/experiments/robustness-iteration-54-checks.json')
    for p in [OLD/'binary.json', OLD/'protocol.json']:
        if digest(p) != checks['artifact_hashes'][str(p.relative_to(ROOT))]:
            raise ValueError('saved binary provenance changed')
    binary = read(OLD/'binary.json'); old = read(OLD/'protocol.json')
    if digest(BINARY) != binary['sha256'] or digest(OLD/'protocol.json') != binary['protocol_sha256']:
        raise ValueError('saved binary changed')
    sources = sorted((ROOT/'prototype/crates/starglyph-core/src').rglob('*.rs'))
    for p in sources:
        if digest(p) != old['hashes'][str(p.relative_to(ROOT))]:
            raise ValueError('production source changed')
    checks55 = read(ROOT/'docs/experiments/robustness-iteration-55-checks.json')
    if digest(SELECTION) != checks55['artifact_hashes'][str(SELECTION.relative_to(ROOT))]:
        raise ValueError('saved selection changed')
    probes = make_probes(read(SELECTION), read(DIAG)['detection_diagnostics'])
    out.mkdir(parents=True)
    write_json(out/'input.json', dict(id=RID, split='development', image=str(image),
        width=2772, height=1935, probes=[p['xy'] for p in probes], radius_original_px=12, review=probes,
        provenance='exact iteration55 centroids; neutral square crops reviewed before trace; no catalogue identities'))
    paths = [Path(__file__).resolve(), Path(__file__).with_name('test_robustness_snow_sources.py'),
        *(Path(__file__).with_name(n) for n in ['collection_run.py', 'robustness_candidate_trace.py', 'robustness_central_sources.py']),
        MANIFEST, image, SELECTION, DIAG, BINARY, OLD/'binary.json', OLD/'protocol.json', out/'input.json', *sources,
        *(ROOT/'data/samples/sky-samples'/n for n in ['robustness-results.json', 'collection-wcs-review.json'])]
    write_json(out/'protocol.json', dict(iteration=56, id=RID, split='development', holdout=False,
        binary=str(BINARY.relative_to(ROOT)), probe_count=8, radius_original_px=12, wall_limit_s=180,
        modes=['control', 'quantile', 'blob_concentration'], widths=[1600,2772], tiers=['default','deep'],
        criteria=['12 trace passes equal uninstrumented detector, asserted by saved Rust harness',
            'four control lists exactly reproduce iteration1 top-K detections',
            'all eight native default control associations must reproduce saved centroid and rank',
            'retain every association, outcome, measurement and offset; no nearest-component identity assumption',
            'no parameter selection, matcher, WCS, holdout, production change or solve-rate claim'],
        hashes={str(p.relative_to(ROOT)):digest(p) for p in paths}))


def validate(out):
    protocol = read(out/'protocol.json')
    if (protocol['id'], protocol['split'], protocol['holdout']) != (RID, 'development', False):
        raise ValueError('development only')
    selected()
    for name, sha in protocol['hashes'].items():
        if digest(ROOT/name) != sha:
            raise ValueError('frozen input changed: '+name)
    return protocol


def run(out):
    validate(out)
    if (out/'run.json').exists():
        raise ValueError('fresh run required')
    command(out, 'run', [str(BINARY), '--exact', 'solve::replay::source_traces', '--ignored'],
        {**os.environ, 'STARGLYPH_TRACE_INPUT':str(out/'input.json'), 'STARGLYPH_TRACE_OUTPUT':str(out/'trace.json')}, 180)


def associated_source(meta, row):
    component = row['component']
    return (component is not None and row.get('centroid_offset_original_px', float('inf')) < 1e-8
            and component['rank_before_top_k'] == meta['native_default_rank'])


def summarize(out):
    validate(out)
    inp = read(out/'input.json'); raw = read(out/'trace.json')['tiers']
    old = read(DIAG)['detection_diagnostics']; tiers = []; exact = 0; associations = 0
    expected = {(mode, width, tier) for mode in ['control','quantile','blob_concentration']
                for width in [1600,2772] for tier in ['default','deep']}
    seen = set()
    for t in raw:
        mode = 'quantile' if t['quantile'] else 'blob_concentration' if t['blob_concentration'] else 'control'
        key = (mode, t['width'], t['tier'])
        if key not in expected or key in seen or len(t['probes']) != 8:
            raise ValueError('trace scope changed')
        seen.add(key)
        if t['height'] != (1117 if t['width'] == 1600 else 1935):
            raise ValueError('trace dimensions changed')
        if mode == 'control':
            d = next(d for d in old if (d['width'],d['tier']) == (t['width'],t['tier']))
            if t['result']['detections'] != d['result']['detections'][:d['max_detections']]:
                raise ValueError('control input mismatch')
            exact += 1
        rows = [probe_row(p, tr, t['width']/inp['width'], t['height']/inp['height'])
                for p, tr in zip(inp['review'], t['probes'])]
        for meta, row in zip(inp['review'], rows):
            row['native_default_rank'] = meta['native_default_rank']
            if key == ('control',2772,'default'):
                row['saved_source_reproduced'] = associated_source(meta, row)
                associations += int(row['saved_source_reproduced'])
        tiers.append(dict(mode=mode,width=t['width'],height=t['height'],tier=t['tier'],
            detections=len(t['result']['detections']),stats=t['result']['stats'],
            outcomes=dict(Counter(r['outcome'] for r in rows)),probes=rows))
    if seen != expected or exact != 4:
        raise ValueError('incomplete trace')
    write_json(out/'results.json', dict(iteration=56,id=RID,split='development',holdout=False,
        protocol_sha256=digest(out/'protocol.json'), unchanged_trace_passes=12,exact_saved_control_lists=exact,
        exact_native_probe_associations=associations, comparison_valid=associations == 8,
        elapsed_s=read(out/'run.json')['elapsed_s'],tiers=tiers,independent_ground_truth=False,
        production_changed=False,automatic_improvement_confirmed=False,new_matcher_calls=0,new_wcs_calls=0))
    if associations != 8:
        raise ValueError('ambiguous native source association; preserved in results')


def figure(out):
    validate(out); inp = read(out/'input.json')
    with Image.open(inp['image']) as source:
        im = source.convert('RGB')
    sheet = Image.new('RGB',(1024,568)); draw = ImageDraw.Draw(sheet)
    for j,p in enumerate(inp['review']):
        x,y = (round(v) for v in p['xy']); bx=j%4*256; by=j//4*284
        # Square crop, no brightness adjustment; black padding outside the photo.
        patch=im.crop((x-64,y-64,x+64,y+64)).resize((256,256))
        sheet.paste(patch,(bx,by)); draw.ellipse((bx+118,by+118,bx+138,by+138),outline='yellow')
        draw.text((bx+4,by+258),f"{p['id']} original rank {p['native_default_rank']}",fill='white')
    sheet.save(out/'sources.jpg',quality=95)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['prepare','validate','run','summarize','figure'])
    parser.add_argument('--out-dir',type=Path,required=True)
    args=parser.parse_args(); globals()[args.stage](args.out_dir.resolve())
