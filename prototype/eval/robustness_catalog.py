#!/usr/bin/env python3
"""Development-only catalogue/pattern diagnosis; external identities stay pending.

Requires numpy, astropy and Pillow. Uses cached WCS and iteration-2 source traces.
No WCS solver, catalogue generation or collection evaluation is launched.
"""
import argparse
import csv
import gzip
import json
from pathlib import Path

from collection_run import ROOT, bounded, digest, select_records, write_json

RID = 'wm_r_161606532'
MANIFEST = ROOT / 'data/samples/sky-samples/manifest.json'
TRACES = ROOT / 'prototype/artifacts/robustness-iteration-2/diagnosis-final'
CATALOG = ROOT / 'data/catalogs/hyg_v42.csv.gz'
DATABASE_NAMES = ['bootstrap-10-70', 'dense-16-30', 'dense-30-54', 'dense-48-88']


def prepare(out):
    import numpy as np
    from astropy.io import fits
    from astropy.wcs import WCS
    from PIL import Image, ImageDraw

    records = {r['id']: r for r in select_records(json.loads(MANIFEST.read_text()), 'development')}
    record = records[RID]  # Guard the split before accessing any image or WCS.
    source = MANIFEST.parent / record['file']
    if digest(source) != record['clean_sha256']:
        raise ValueError('image hash mismatch')
    refpath = ROOT / f'prototype/artifacts/robustness-baseline/wcs-run/{RID}/reference.json'
    ref = json.loads(refpath.read_text())
    corrpath = ROOT / 'prototype' / ref['correspondences_file']
    wcspath = ROOT / 'prototype' / ref['wcs_file']
    if (ref['source_sha256'] != digest(source) or digest(corrpath) != ref['correspondences_sha256']
            or digest(wcspath) != ref['wcs_sha256']):
        raise ValueError('external reference hash mismatch')
    probes = json.loads((TRACES / f'{RID}-trace-input.json').read_text())
    if probes['image_sha256'] != digest(source) or probes['split'] != 'development':
        raise ValueError('trace input mismatch')
    rows = probes['provenance']['corr_rows']
    traces = json.loads((TRACES / f'{RID}-traces.json').read_text())['tiers']
    with gzip.open(CATALOG, 'rt') as stream:
        stars = [s for s in csv.DictReader(stream) if s['mag'] and float(s['mag']) <= 6.8 and int(s['id']) > 0]
    radec = np.array([[float(s['ra'])*15, float(s['dec'])] for s in stars])
    radians = np.deg2rad(radec)
    units = np.column_stack((np.cos(radians[:, 1])*np.cos(radians[:, 0]),
                            np.cos(radians[:, 1])*np.sin(radians[:, 0]), np.sin(radians[:, 1])))
    corr = fits.getdata(corrpath)
    associations = []
    for i, c in enumerate(corr):
        ra, dec = np.deg2rad([c['index_ra'], c['index_dec']])
        u = np.array([np.cos(dec)*np.cos(ra), np.cos(dec)*np.sin(ra), np.sin(dec)])
        distances = np.linalg.norm(units-u, axis=1)
        index = int(np.argmin(distances))
        star = stars[index]
        separation = float(np.rad2deg(2*np.arcsin(distances[index]/2))*3600)
        if separation > 5:
            raise ValueError('catalogue association exceeds frozen 5 arcsec radius')
        associations.append(dict(corr_row=i, hyg_id=int(star['id']), hip=star['hip'],
            name=star['proper'] or star['bf'] or star['con'], mag=float(star['mag']),
            catalog_separation_arcsec=separation, xy=[float(c['field_x']-1), float(c['field_y']-1)],
            weight=float(c['match_weight']), selected_probe=i in rows))
    anchors = [associations[i] for i in rows]
    cases, ranks = [], []
    for width, tier in [(1600, 'default'), (1600, 'deep'), (4080, 'default'), (4080, 'deep')]:
        trace = next(t for t in traces if t['width'] == width and t['tier'] == tier
                     and not t['quantile'] and not t['blob_concentration'])
        detections = trace['result']['detections']
        selected = []
        recovered = []
        mapping = []
        for anchor, probe in zip(anchors, trace['probes']):
            component = probe['component']
            rank = component['rank_before_top_k'] if component else None
            mapping.append(dict(hyg_id=anchor['hyg_id'], mag=anchor['mag'],
                outcome=component['outcome'] if component else 'no_component', rank=rank))
            if component and component['outcome'] == 'selected':
                # Trace rank addresses the actual complete top-K list, not corr order.
                point = detections[rank]
                assert np.linalg.norm(np.array(component['centroid'])-[point['x'], point['y']]) < 1e-6
                selected.append((rank, anchor, point))
            if component and component['centroid'] is not None:
                recovered.append((anchor, dict(x=component['centroid'][0], y=component['centroid'][1],
                                               flux=component['photometry_flux'])))
        selected.sort(key=lambda p: p[0])
        def add(name, points, ids):
            cases.append(dict(name=f'{width}-{tier}-{name}', width=width, height=trace['height'],
                              search=points, search_ids=ids, verification=detections))
        known = {rank: anchor['hyg_id'] for rank, anchor, _ in selected}
        add('control', detections, [known.get(i) for i in range(len(detections))])
        add('visible-only', [d for _, _, d in selected], [a['hyg_id'] for _, a, _ in selected])
        ordered = sorted(selected, key=lambda p: (p[1]['mag'], p[1]['hyg_id']))
        add('visible-catalog-order', [dict(d, flux=float(len(ordered)-i)) for i, (_, _, d) in enumerate(ordered)],
            [a['hyg_id'] for _, a, _ in ordered])
        recovered.sort(key=lambda p: (p[0]['mag'], p[0]['hyg_id']))
        add('recovered-catalog-order', [dict(d, flux=float(len(recovered)-i)) for i, (_, d) in enumerate(recovered)],
            [a['hyg_id'] for a, _ in recovered])
        ranks.append(dict(width=width, tier=tier, detections=len(detections), sources=mapping))
    databases = [ROOT / f'prototype/artifacts/cache/tetra3-{n}-mag65-v0.8.0.bin' for n in DATABASE_NAMES]
    provenance = {str(p.relative_to(ROOT)): digest(p) for p in [MANIFEST, source, CATALOG, refpath,
        corrpath, wcspath, TRACES / f'{RID}-traces.json', TRACES / f'{RID}-trace-input.json', *databases]}
    write_json(out / 'input.json', dict(image=str(source), catalog=str(CATALOG), databases=list(map(str, databases)),
        probe_ids=[a['hyg_id'] for a in anchors], cases=cases,
        protocol=dict(id=RID, split='development', external_status='pending', provenance=provenance,
            search='unchanged blind attempts, prefixes <=30 and 2500ms per attempt',
            verification='unchanged full original tier detections for every intervention',
            interpretation='catalogue-informed diagnostic only, before refinement; no automatic improvement',
            acceptance='unchanged hard matches/probability/hits or soft log-odds thresholds')))
    write_json(out / 'sources.json', dict(id=RID, associations=associations, tiers=ranks))
    # Review stars not used in .corr. Their prediction is a conditional check of
    # the candidate, not a second independent astrometric truth source.
    wcs = WCS(fits.getheader(wcspath))
    projected = wcs.all_world2pix(radec, 0, quiet=True)
    used = {a['hyg_id'] for a in associations}
    withheld = [dict(hyg_id=int(s['id']), mag=float(s['mag']), xy=list(map(float, xy)))
                for s, xy in zip(stars, projected) if int(s['id']) not in used
                and 24 < xy[0] < 4056 and 24 < xy[1] < 900]
    write_json(out / 'noncorr-predictions.json', withheld)
    with Image.open(source) as im:
        overview = im.convert('RGB').resize((1360, 1024))
        draw = ImageDraw.Draw(overview)
        for a in anchors:
            x, y = [p/3 for p in a['xy']]
            draw.ellipse((x-4, y-4, x+4, y+4), outline='red')
            draw.text((x+5, y), str(a['hyg_id']), fill='yellow')
        overview.save(out / 'identities.jpg')
        tiles = [dict(a, label=f"row {a['corr_row']} HYG {a['hyg_id']} {a['name']} m{a['mag']}")
                 for a in associations if a['corr_row'] in (0, 2, 3, 4, 5, 6, 7, 23, 45, 53, 54, 57, 61, 64, 66)]
        tiles += [dict(a, label=f"noncorr HYG {a['hyg_id']} m{a['mag']}") for a in withheld]
        sheet = Image.new('RGB', (5*260, ((len(tiles)+4)//5)*280))
        draw = ImageDraw.Draw(sheet)
        for i, a in enumerate(tiles):
            x, y = a['xy']; left, top = i % 5*260, i//5*280
            tile = im.crop((round(x)-64, round(y)-64, round(x)+64, round(y)+64)).resize((256, 256))
            sheet.paste(tile, (left, top+24))
            draw.text((left+2, top+4), a['label'], fill='white')
            draw.ellipse((left+120, top+144, left+136, top+160), outline='red')
        sheet.save(out / 'identity-crops.jpg')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    parser.add_argument('--expand-from', type=Path,
                        help='extend identities with visually reviewed non-corr sky predictions')
    parser.add_argument('--restore-key-from', type=Path,
                        help='diagnostic restoration of Dschubba from its reduced-image centroid')
    parser.add_argument('--run-binary', type=Path,
                        help='run an already prepared input with a compiled Rust test executable')
    parser.add_argument('--prepare-support', action='store_true',
                        help='prepare expanded source trace and positive regression inputs')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.prepare_support:
        prepare_support(args.out)
    elif args.run_binary:
        run(args.out, args.run_binary)
    elif args.restore_key_from:
        restore_key(args.restore_key_from, args.out)
    elif args.expand_from:
        expand(args.expand_from, args.out)
    else:
        prepare(args.out)


def expand(initial, out):
    """Second diagnostic stage: .corr is not a complete list of visible stars.

    Annotate actual detector lists, including non-corr stars reviewed on crops.
    No new detections, coordinates, search budget or acceptance rules.
    """
    import copy
    import math
    original = json.loads((initial / 'input.json').read_text())
    if original['protocol']['split'] != 'development' or original['protocol']['id'] != RID:
        raise ValueError('only the selected development frame is allowed')
    # Recheck all original inputs, including the frozen split and cached DBs.
    for path, sha in original['protocol']['provenance'].items():
        if digest(ROOT / path) != sha:
            raise ValueError(f'input changed: {path}')
    sources = json.loads((initial / 'sources.json').read_text())
    predictions = json.loads((initial / 'noncorr-predictions.json').read_text())
    anchors = [a for a in sources['associations'] if a['selected_probe']] + predictions
    mags = {a['hyg_id']: a['mag'] for a in anchors}
    cases, associations = [], []
    for control in original['cases']:
        if not control['name'].endswith('-control'):
            continue
        control = copy.deepcopy(control)
        prefix = control['name'].removesuffix('-control')
        ids = []
        for i, d in enumerate(control['search']):
            xy = [(d['x']+.5)*4080/control['width']-.5, (d['y']+.5)*3072/control['height']-.5]
            nearby = sorted([(a['mag'], math.dist(xy, a['xy']), a['hyg_id']) for a in anchors
                             if math.dist(xy, a['xy']) <= 12])
            # Existing component trace identity has priority. Blends without one
            # are labelled by their brightest possible primary, not two stars.
            identity = control['search_ids'][i] or (nearby[0][2] if nearby else None)
            ids.append(identity)
            associations.append(dict(tier=prefix, rank=i, hyg_id=identity,
                candidates=[dict(hyg_id=c[2], distance_original_px=c[1]) for c in nearby],
                inferred_identity='conditional on pending WCS; blends may be ambiguous'))
        if len([i for i in ids if i]) != len({i for i in ids if i}):
            raise ValueError('multiple detections assigned to one catalogue star')
        control['search_ids'] = ids
        cases.append(control)
        selected = [(d, i) for d, i in zip(control['search'], ids) if i]
        for name, ordered in [('visible-only', selected),
                              ('visible-catalog-order', sorted(selected, key=lambda p: (mags[p[1]], p[1])))]:
            points = [dict(d, flux=float(len(ordered)-n)) if name.endswith('catalog-order') else d
                      for n, (d, _) in enumerate(ordered)]
            cases.append(dict(name=f'{prefix}-{name}', width=control['width'], height=control['height'],
                              search=points, search_ids=[i for _, i in ordered], verification=control['verification']))
    original.update(cases=cases, probe_ids=sorted({a['hyg_id'] for a in anchors}))
    original['protocol']['extension'] = dict(initial_input_sha256=digest(initial / 'input.json'),
        prediction_sha256=digest(initial / 'noncorr-predictions.json'), radius_original_px=12,
        reason='corr subset omitted visible bright stars; extend annotations before claiming pattern absence')
    write_json(out / 'input.json', original)
    write_json(out / 'associations.json', associations)


def run(out, binary):
    """Freeze inputs before the diagnostic run; bound total wall time as well."""
    import os
    source = out / 'input.json'
    data = json.loads(source.read_text())
    protocol = data.get('protocol', {})
    if protocol.get('split') != 'development' or protocol.get('id') != RID:
        raise ValueError('only the selected development frame is allowed')
    for path, sha in protocol['provenance'].items():
        if digest(ROOT / path) != sha:
            raise ValueError(f'input changed: {path}')
    files = [source, binary, Path(__file__),
             ROOT / 'prototype/crates/starglyph-core/src/solve.rs',
             ROOT / 'prototype/crates/starglyph-core/src/solve/catalog_replay.rs',
             ROOT / 'prototype/crates/starglyph-core/src/engine.rs', ROOT / 'prototype/Cargo.lock']
    write_json(out / 'freeze-final.json', dict(split='development',
        wall_timeout_s=240, hashes={str(p.resolve()): digest(p) for p in files}))
    os.environ['STARGLYPH_CATALOG_INPUT'] = str(source.resolve())
    os.environ['STARGLYPH_CATALOG_OUTPUT'] = str((out / 'replay-final.json').resolve())
    status = bounded([str(binary.resolve()), '--ignored', '--exact',
        'solve::catalog_replay::catalog_patterns_and_replay', '--nocapture'], out / 'replay-final.log', 240)
    write_json(out / 'run-final.json', status)
    if status['status'] != 'completed':
        raise RuntimeError(f'diagnostic replay failed: {status}')


def restore_key(expanded, out):
    """Oracle ablation, not an automatic detection algorithm or ground truth."""
    import copy
    original = json.loads((expanded / 'input.json').read_text())
    if original['protocol']['split'] != 'development' or original['protocol']['id'] != RID:
        raise ValueError('only the selected development frame is allowed')
    probes = json.loads((expanded / 'trace-input.json').read_text())
    traces = json.loads((expanded / 'traces.json').read_text())['tiers']
    trace = next(t for t in traces if t['width'] == 1600 and t['tier'] == 'default'
                 and not t['quantile'] and not t['blob_concentration'])
    index = probes['hyg_ids'].index(78165)
    component = trace['probes'][index]['component']
    assert component['outcome'] == 'concentration'
    x, y = component['centroid']
    cases = []
    for control in original['cases']:
        if not control['name'].startswith('4080-') or not control['name'].endswith('-control'):
            continue
        cases.append(control)
        restored = copy.deepcopy(control)
        restored['name'] = restored['name'].removesuffix('-control') + '-restore-dschubba'
        restored['search'].insert(0, dict(x=(x+.5)*4080/trace['width']-.5,
            y=(y+.5)*3072/trace['height']-.5, flux=max(d['flux'] for d in control['search'])+1))
        restored['search_ids'].insert(0, 78165)
        cases.append(restored)
    original['cases'] = cases
    original['protocol']['restoration'] = dict(hyg_id=78165, trace_sha256=digest(expanded / 'traces.json'),
        position='measured reduced default component centroid, half-pixel rescaled to native image',
        mass='forced first; diagnostic composition plus priority intervention',
        verification='original full detections, restored point excluded')
    write_json(out / 'input.json', original)


def prepare_support(out):
    original = json.loads((out / 'input.json').read_text())
    if original['protocol']['split'] != 'development' or original['protocol']['id'] != RID:
        raise ValueError('only the selected development frame is allowed')
    sources = json.loads((out / 'sources.json').read_text())
    anchors = [a for a in sources['associations'] if a['selected_probe']]
    anchors += json.loads((out / 'noncorr-predictions.json').read_text())
    write_json(out / 'expanded/trace-input.json', dict(image=original['image'],
        probes=[a['xy'] for a in anchors], hyg_ids=[a['hyg_id'] for a in anchors],
        radius_original_px=12, split='development'))
    records = {r['id']: r for r in select_records(json.loads(MANIFEST.read_text()), 'development')}
    rec = records['wm_r_112929230']
    image = MANIFEST.parent / rec['file']
    if digest(image) != rec['clean_sha256']:
        raise ValueError('positive fixture image changed')
    saved = json.loads((TRACES / 'wm_r_112929230-1600-default-replay-input.json').read_text())
    if Path(saved['image']).resolve() != image.resolve():
        raise ValueError('positive replay image mismatch')
    control = next(c for c in saved['cases'] if c['name'] == 'control')
    write_json(out / 'positive-input.json', dict(image=str(image), catalog=str(CATALOG),
        databases=original['databases'], probe_ids=[], cases=[dict(name='positive-control',
            width=saved['width'], height=saved['height'], search=control['detections'],
            verification=control['detections'])]))


if __name__ == '__main__':
    main()
