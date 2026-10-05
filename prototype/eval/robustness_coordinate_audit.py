#!/usr/bin/env python3
"""Audit existing development pixels and coordinate conventions; never fit a camera."""
import argparse
import io
import json
import math
from pathlib import Path
import platform
import sys
import time
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image, ExifTags, features

from collection_run import ROOT, digest, select_records, write_json

SAMPLES = ROOT / 'data/samples/sky-samples'
sys.path.insert(0, str(SAMPLES))
from fetch_sample import normalize, pixel_sha256  # noqa: E402

IDS = ['wm_r_132162731', 'wm_r_143159342', 'wm_r_149276071']
MANIFEST = SAMPLES / 'manifest.json'
GEOMETRY = ROOT / 'docs/experiments/robustness-iteration-13-geometry.json'
RAW = ROOT / 'prototype/artifacts/robustness-collection/raw'
CORE = ROOT / 'prototype/crates/starglyph-core/src'
EXIF_FIELDS = {'Make', 'Model', 'Software', 'Orientation', 'ExifImageWidth',
               'ExifImageHeight', 'FocalLength', 'FocalLengthIn35mmFilm',
               'DigitalZoomRatio', 'DefaultCropOrigin', 'DefaultCropSize', 'ActiveArea'}
XMP_FIELDS = {'CropTop', 'CropLeft', 'CropBottom', 'CropRight', 'CropAngle',
              'HasCrop', 'CroppedAreaImageWidthPixels', 'CroppedAreaImageHeightPixels',
              'FullPanoWidthPixels', 'FullPanoHeightPixels',
              'DistortionCorrectionAlreadyApplied', 'LensProfileEnable',
              'LensProfileDistortionScale', 'PerspectiveHorizontal', 'PerspectiveVertical'}


def selected(ids):
    records = select_records(json.loads(MANIFEST.read_text()), 'development', ids)
    if ids != IDS or [r['id'] for r in records] != IDS:
        raise ValueError('fixed development selection required')
    return records


def prepare(out, ids):
    records = selected(ids)  # Before opening photographs or experiment results.
    if out.exists():
        raise ValueError('fresh output directory required')
    files = [Path(__file__).resolve(), Path(__file__).with_name('collection_run.py'),
             SAMPLES / 'fetch_sample.py', MANIFEST, GEOMETRY,
             SAMPLES / 'robustness-results.json', SAMPLES / 'collection-wcs-review.json',
             CORE / 'geom.rs', CORE / 'solve.rs', CORE / 'image_input.rs']
    for rec in records:
        files.extend([RAW / rec['id'].removeprefix('wm_r_'), SAMPLES / rec['file']])
    out.mkdir(parents=True)
    write_json(out / 'protocol.json', dict(
        iteration=16, split='development', ids=ids, holdout=False,
        hashes={str(p.relative_to(ROOT)): digest(p) for p in files},
        environment=dict(python=platform.python_version(), pillow=Image.__version__,
                         jpeg=features.version('jpg'), numpy=np.__version__),
        working_max_edge=1600, fixed_projection_shift_px=[-0.5, -0.5],
        criteria=[
            'replay existing normalization; compare raw/file/pixel hashes and dimensions',
            'export only whitelisted geometric metadata; absent crop tags do not prove no crop',
            'verify edge and geometric-centre identities for half-pixel resize mapping',
            'quantify rounded-size anisotropy and fixed-principal-point seed offset before refit',
            'evaluate one fixed -0.5px principal-point shift on all six saved cameras and all probes',
            'keep cameras, pairs, source identities and edge groups fixed; no fitting or solver run',
            'direct projection difference is not a bound on rematching or refitted solve outcomes'],
        unknowns=['physical optical centre', 'crop and resampling before source upload',
                  'full upstream lens/editor processing', 'certified astrometric ground truth']))


def validate(out):
    protocol = json.loads((out / 'protocol.json').read_text())
    if protocol['split'] != 'development' or protocol['holdout']:
        raise ValueError('development required')
    records = selected(protocol['ids'])
    for path, sha in protocol['hashes'].items():
        if digest(ROOT / path) != sha:
            raise ValueError('frozen input changed: ' + path)
    return protocol, records


def xmp_geometry(payload):
    """Return geometry only; never publish location, owner or camera serial fields."""
    if not payload:
        return dict(status='absent', fields={})
    try:
        tree = ET.fromstring(payload)
    except ET.ParseError:
        return dict(status='unparsed', fields={})
    values = {}
    for node in tree.iter():
        for key, value in [*node.attrib.items(), (node.tag, node.text)]:
            if key.rsplit('}', 1)[-1] in XMP_FIELDS and value and value.strip():
                values.setdefault(key, []).append(value.strip())
    return dict(status='parsed', fields=values)


def normalization_audit(record):
    raw_path = RAW / record['id'].removeprefix('wm_r_')
    clean_path = SAMPLES / record['file']
    raw = raw_path.read_bytes()
    if digest(raw_path) != record['orig_sha256'] or digest(clean_path) != record['clean_sha256']:
        raise ValueError('image differs from manifest')
    encoded, pixels, size = normalize(raw)
    with Image.open(io.BytesIO(raw)) as source:
        exif = source.getexif()
        maps = [exif, exif.get_ifd(34665)] if 34665 in exif else [exif]
        fields = {ExifTags.TAGS[k]: str(v) for tags in maps for k, v in tags.items()
                  if ExifTags.TAGS.get(k) in EXIF_FIELDS}
        metadata = dict(exif=fields, xmp=xmp_geometry(source.info.get('xmp')),
                        raw_size=list(source.size), raw_mode=source.mode,
                        orientation=int(exif.get(274, 1)))
    with Image.open(io.BytesIO(encoded)) as replay, Image.open(clean_path) as clean:
        same = (replay.mode == clean.mode and replay.size == clean.size
                and replay.tobytes() == clean.tobytes())
        metadata.update(normalized_size=list(clean.size), normalized_mode=clean.mode,
                        normalized_orientation=clean.getexif().get(274),
                        normalized_pixel_sha256=pixel_sha256(clean))
    checks = dict(raw_file_hash=True, normalized_file_hash=True,
                  replay_pixels_identical=same,
                  replay_pixel_hash_matches_manifest=pixels == record['processing']['pixel_sha256'],
                  dimensions_match_manifest=list(size) == [record['width'], record['height']],
                  orientation_matches_manifest=metadata['orientation'] == record['processing']['orientation_applied'],
                  raw_dimensions_match_manifest=metadata['raw_size'] == [record['orig_width'], record['orig_height']])
    if not all(checks.values()):
        raise ValueError('normalization invariant failed: ' + record['id'])
    return dict(checks=checks, metadata=metadata, physical_principal_point=None,
                pre_upload_crop='unknown', pre_upload_resampling='unknown')


def resize_audit(width, height, max_edge=1600):
    """Algebraic audit of the existing resize/lift, not an image resampling test."""
    if min(width, height, max_edge) <= 0:
        raise ValueError('positive dimensions required')
    ratio = min(1., max_edge / max(width, height))
    working = [max(1, math.floor(d * ratio + .5)) for d in (width, height)]
    scales = np.array([width, height]) / working
    geometric = (np.array(working) - 1) / 2
    lifted = (geometric + .5) * scales - .5
    expected = (np.array([width, height]) - 1) / 2
    edge_error = max(np.max(np.abs((np.array([-.5, -.5]) + .5) * scales - .5 + .5)),
                     np.max(np.abs((np.array(working) - .5 + .5) * scales - .5
                                   - (np.array([width, height]) - .5))))
    # Starglyph's W/2 centre, unlike the geometric pixel-centre midpoint,
    # does not commute with a pixel-centred resize. This is a seed difference:
    # production explicitly refits the lifted correspondences afterwards.
    return dict(working_size=working, scales=scales.tolist(),
                geometric_centre_roundtrip_max_abs_px=float(np.max(np.abs(lifted - expected))),
                pixel_area_boundary_max_abs_px=float(edge_error),
                fixed_centre_seed_offset_px=((scales - 1) / 2).tolist(),
                relative_axis_scale_difference=float(scales[1] / scales[0] - 1),
                focal_x_only_max_vertical_seed_error_px=float(abs(scales[1] - scales[0]) * working[1] / 2),
                production_refits_after_lift=True)


def shift_metrics(sources, case, width, height):
    xy = np.array([s['xy'] for s in sources])
    projected = np.array(case['projected_xy'])
    if projected.shape != xy.shape or not np.isfinite(projected).all():
        raise ValueError('invalid saved projections')
    residual = projected - xy
    shifted = residual - .5
    x, y = xy.T
    masks = dict(all_reviewed=np.ones(len(x), dtype=bool),
                 outside_both_inputs=np.array([s['outside_both_inputs'] for s in sources]),
                 left_10pct=x < width * .1, right_10pct=x > width * .9,
                 top_10pct=y < height * .1, bottom_10pct=y > height * .9)
    rms = lambda r: float(np.sqrt(np.mean(np.sum(r * r, axis=1))))
    groups = {}
    for name, mask in masks.items():
        groups[name] = dict(count=int(sum(mask)))
        if mask.any():
            before, after = rms(residual[mask]), rms(shifted[mask])
            groups[name].update(before_rms_px=before, shifted_rms_px=after,
                                change_px=after-before)
            if abs(after-before) > math.sqrt(.5) + 1e-10:
                raise ValueError('translation bound violated')
    return dict(name=case['name'], groups=groups, fit_or_rematch=False,
                motion_per_probe_px=math.sqrt(.5))


def measure(out):
    protocol, records = validate(out)
    geometry = json.loads(GEOMETRY.read_text())
    if geometry['split'] != 'development' or [f['id'] for f in geometry['frames']] != IDS:
        raise ValueError('saved development geometry required')
    started = time.monotonic()
    frames = []
    for record, saved in zip(records, geometry['frames']):
        start = time.monotonic()
        width, height = record['width'], record['height']
        if len(saved['cases']) != 2 or [c['name'] for c in saved['cases']] != ['k1', 'k1_k2']:
            raise ValueError('saved camera selection changed')
        for case in saved['cases']:
            if [case['camera']['width'], case['camera']['height']] != [width, height]:
                raise ValueError('saved camera dimensions changed')
        frames.append(dict(id=record['id'], normalization=normalization_audit(record),
                           resize=resize_audit(width, height, protocol['working_max_edge']),
                           fixed_camera_shift=[shift_metrics(saved['sources'], c, width, height) for c in saved['cases']],
                           status='completed', elapsed_s=time.monotonic()-start))
    write_json(out / 'results.json', dict(iteration=16, split='development',
        protocol_sha256=digest(out / 'protocol.json'), frames=frames,
        elapsed_s=time.monotonic()-started, solve_runs=0, fit_runs=0,
        acceptance_changed=False, physical_centre_hypothesis='unresolved',
        normalization_geometry_change=False,
        limitations=protocol['unknowns'] + [protocol['criteria'][-1]]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'measure'])
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--ids', default=','.join(IDS))
    args = parser.parse_args()
    if args.stage == 'prepare':
        prepare(args.out_dir.resolve(), args.ids.split(','))
    else:
        if args.ids.split(',') != IDS:
            raise ValueError('measurement uses frozen development selection')
        measure(args.out_dir.resolve())


if __name__ == '__main__':
    main()
