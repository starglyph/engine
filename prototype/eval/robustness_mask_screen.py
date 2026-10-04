#!/usr/bin/env python3
"""Development-only screen of the unchanged gradient mask; no detection or solve."""
import argparse
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
import platform
import time

import numpy as np
from PIL import Image, ImageDraw, ImageOps, __version__ as pillow_version

from collection_run import ROOT, digest, select_records, write_json
from gradient_sky_mask import GradientConfig, estimate_gradient
from sky_mask_experiment import in_sky

MANIFEST = ROOT/'data/samples/sky-samples/manifest.json'
GEOMETRY = ROOT/'docs/experiments/robustness-iteration-5-geometry.json'
PAIRS = ROOT/'docs/experiments/robustness-iteration-7-review.json'
BASELINE = ROOT/'prototype/artifacts/robustness-baseline/starglyph'


def records(ids=None):
    return select_records(json.loads(MANIFEST.read_text()), 'development', ids)


def prepare(out):
    selected = records()
    if out.exists():
        raise ValueError('choose a fresh output directory')
    paths = [MANIFEST, GEOMETRY, PAIRS, Path(__file__).resolve(),
             ROOT/'data/samples/sky-samples/robustness-results.json']
    paths += [ROOT/'prototype/eval'/name for name in
              ('gradient_sky_mask.py', 'automatic_sky_mask.py', 'sky_mask_experiment.py',
               'collection_run.py', 'smartphone_gate.py')]
    reports = {}
    for rec in selected:
        source = MANIFEST.parent/rec['file']
        if digest(source) != rec['clean_sha256']:
            raise ValueError('image changed: '+rec['id'])
        paths.append(source)
        report = BASELINE/rec['id']/'solve-reports'/(rec['id']+'.json')
        reports[rec['id']] = str(report.relative_to(ROOT)) if report.exists() else None
        if report.exists():
            paths.append(report)
    out.mkdir(parents=True)
    write_json(out/'protocol.json', dict(iteration=9, split='development',
        ids=[r['id'] for r in selected], algorithm='rgb_gradient_v2', config=asdict(GradientConfig()),
        reports=reports, hashes={str(p.relative_to(ROOT)):digest(p) for p in paths},
        environment=dict(python=platform.python_version(), numpy=np.__version__, pillow=pillow_version),
        criteria=['exclude the confirmed leaf detection', 'retain all 117 fixed iteration-5 source probes',
                  'report each edge group and every removed saved detection/inlier',
                  'abstention means unchanged unmasked path, not a negative classification'],
        scope='mask membership only; no redetection, sky statistics/fill, acceptance or geometry refit',
        followup='full pipeline A/B only if leaf exclusion and all reviewed probes pass; no tuning',
        holdout=False))


def validate(out):
    protocol = json.loads((out/'protocol.json').read_text())
    if protocol['split'] != 'development' or protocol['holdout']:
        raise ValueError('development required')
    selected = records(protocol['ids'])  # Reject holdout before reading photographs.
    if [r['id'] for r in selected] != protocol['ids']:
        raise ValueError('selection changed')
    if protocol['config'] != asdict(GradientConfig()) or protocol['algorithm'] != 'rgb_gradient_v2':
        raise ValueError('fixed gradient configuration changed')
    for path, sha in protocol['hashes'].items():
        if digest(ROOT/path) != sha:
            raise ValueError('frozen input changed: '+path)
    return protocol, selected


def retained(xy, mask):
    return mask is None or in_sky(dict(x=xy[0], y=xy[1]), mask)


def probe_metrics(sources, mask, width, height):
    rows = [dict(source_id=s['source_id'], hyg_id=s['hyg_id'], name=s['name'], xy=s['xy'],
                 outside_both_inputs=s['outside_both_inputs'], retained=retained(s['xy'], mask))
            for s in sources]
    groups = dict(all_reviewed=rows, outside_both_inputs=[r for r in rows if r['outside_both_inputs']],
        left_10pct=[r for r in rows if r['xy'][0] < width*.1],
        right_10pct=[r for r in rows if r['xy'][0] > width*.9],
        top_10pct=[r for r in rows if r['xy'][1] < height*.1],
        bottom_10pct=[r for r in rows if r['xy'][1] > height*.9])
    return dict(sources=rows, groups={k:dict(total=len(v), retained=sum(r['retained'] for r in v))
                                    for k, v in groups.items()})


def saved_detections(path, rec, mask):
    if path is None:
        return dict(status='missing_saved_report', total=None, retained=None)
    report = json.loads((ROOT/path).read_text())
    if (report['id'] != rec['id'] or report['source_sha256'] != rec['clean_sha256']
            or [report['width'], report['height']] != [rec['width'], rec['height']]):
        raise ValueError('saved detector source mismatch')
    detections = report['report']['detections']
    kept = [retained([d['x'], d['y']], mask) for d in detections]
    inliers = [i for i, d in enumerate(detections) if d['inlier']]
    return dict(status='measured', prior_solve_status=report['report']['status'], total=len(kept),
        retained=sum(kept), retained_by_index=kept, inliers_total=len(inliers),
        inliers_retained=sum(kept[i] for i in inliers),
        removed=[dict(index=i, xy=[d['x'], d['y']], inlier=d['inlier'])
                 for i, d in enumerate(detections) if not kept[i]])


def preview(image, rec, mask, probes, pairs, out):
    image = image.copy()
    image.thumbnail((600, 600))
    draw = ImageDraw.Draw(image)
    if mask is not None:
        polygon = [(x*image.width, y*image.height) for x, y in mask['sky_polygon']]
        draw.line(polygon+polygon[:1], fill='#00ffff', width=2)
    for source in probes + pairs:
        x, y = source['xy']
        x = (x+.5)*image.width/rec['width']-.5
        y = (y+.5)*image.height/rec['height']-.5
        color = '#00ff00' if source['retained'] else '#ff4040'
        draw.ellipse((x-3, y-3, x+3, y+3), outline=color, width=1)
    image.save(out/(rec['id']+'.jpg'))


def measure(out, repeat=False):
    protocol, selected = validate(out)
    geometry = json.loads(GEOMETRY.read_text())
    review = json.loads(PAIRS.read_text())
    if geometry['split'] != 'development' or review['split'] != 'development':
        raise ValueError('development annotations required')
    allowed = set(protocol['ids'])
    if any(f['id'] not in allowed for f in geometry['frames']+review['frames']):
        raise ValueError('annotation outside development')
    probes_by_id = {f['id']:f['sources'] for f in geometry['frames']}
    pairs_by_id = {f['id']:f['points'] for f in review['frames']}
    frames, masks = [], []
    previews = out/'previews'
    if not repeat:
        previews.mkdir(exist_ok=False)
    for rec in selected:
        started = time.monotonic()
        with Image.open(MANIFEST.parent/rec['file']) as original:
            image = ImageOps.exif_transpose(original).convert('RGB')
        if list(image.size) != [rec['width'], rec['height']]:
            raise ValueError('normalized image dimensions differ')
        polygon, diagnostic = estimate_gradient(image, GradientConfig())
        elapsed = (time.monotonic()-started)*1000
        mask = None if polygon is None else dict(id=rec['id'], source_sha256=rec['clean_sha256'],
            width=image.width, height=image.height, sky_polygon=polygon)
        if mask is not None:
            masks.append(mask)
        probes = probe_metrics(probes_by_id.get(rec['id'], []), mask, image.width, image.height)
        pairs = [dict(pair_id=p['id'], hyg_id=p['hyg_id'], label=p['label'],
                      xy=p['detector_xy_original'], retained=retained(p['detector_xy_original'], mask))
                 for p in pairs_by_id.get(rec['id'], [])]
        frames.append(dict(id=rec['id'], track=rec['track'], negative=rec['research']['negative'],
            width=image.width, height=image.height, diagnostic=diagnostic, generation_ms=elapsed,
            effective_selected_fraction=1. if mask is None else diagnostic['selected_fraction'],
            reviewed=probes, initial_pairs=pairs,
            saved_detections=saved_detections(protocol['reports'][rec['id']], rec, mask)))
        if not repeat:
            preview(image, rec, mask, probes['sources'], pairs, previews)
    mask_data = dict(schema_version=1, coordinates='exif_oriented_normalized_image_edges',
        description='Fixed rgb_gradient_v2 development diagnostic. Per-image credits in collection manifest.', masks=masks)
    total = sum(f['reviewed']['groups']['all_reviewed']['total'] for f in frames)
    kept = sum(f['reviewed']['groups']['all_reviewed']['retained'] for f in frames)
    leaf = [p for f in frames for p in f['initial_pairs'] if p['hyg_id'] == 80761]
    if total != 117 or len(leaf) != 1 or leaf[0]['label'] != 'not_visible':
        raise ValueError('fixed review selection changed')
    result = dict(iteration=9, split='development', protocol_sha256=digest(out/'protocol.json'), frames=frames,
        summary=dict(frames=len(frames), decisions=dict(Counter(f['diagnostic']['decision'] for f in frames)),
                     reviewed_total=total, reviewed_retained=kept, leaf_excluded=not leaf[0]['retained'],
                     preliminary_gate_passed=kept == total and not leaf[0]['retained']),
        interpretation='membership of existing sources, not a new detector/solver result; no ground-truth promotion')
    if repeat:
        previous = json.loads((out/'results.json').read_text())
        for data in (result, previous):
            for frame in data['frames']:
                frame.pop('generation_ms')
        if result != previous or mask_data != json.loads((out/'masks.json').read_text()):
            raise ValueError('mask screen reproduction differs')
        write_json(out/'repeat.json', dict(status='passed', masks_exact=True, metrics_exact_except_time=True,
            results_sha256=digest(out/'results.json'), masks_sha256=digest(out/'masks.json')))
    else:
        write_json(out/'masks.json', mask_data)
        write_json(out/'results.json', result)
        write_json(previews/'credits.json', [dict(id=r['id'], author=r['author'], license=r['license'],
            page_url=r['page_url'], modifications='Resized preview with mask boundary and retained/excluded source markers.')
            for r in selected])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'measure', 'repeat'])
    parser.add_argument('out', type=Path)
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare(args.out.resolve())
    else:
        measure(args.out.resolve(), repeat=args.command == 'repeat')
