#!/usr/bin/env python3
"""Measure frozen external/visually transferred correspondences without fitting.

These sparse diagnostic points never promote an image to full-field ground truth.
Source coordinates come from Astrometry.net, not the engine detector.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageOps

from compare_local_wcs import project, sha256, stats


def registration_check(registration, reviewed):
    """Fit only the fixed image anchors; no camera or catalog coordinates used."""
    sources = {p['id']: p for p in reviewed['points']}
    rows = registration['points']
    if len({p['source_id'] for p in rows}) != len(rows):
        raise ValueError('duplicate registration source')
    if any(p['role'] not in ('anchor', 'held_out') for p in rows):
        raise ValueError('unsupported registration role')
    if any(sources[p['source_id']]['label'] != 'visible_source' for p in rows):
        raise ValueError('registration source not visually reviewed')
    source = np.array([p['primary_xy'] for p in rows], dtype=float)
    target = np.array([[sources[p['source_id']]['x'], sources[p['source_id']]['y']] for p in rows])
    anchors = np.array([p['role'] == 'anchor' for p in rows])
    if (anchors.sum() != 4 or (~anchors).sum() < 1 or source.shape != target.shape
            or not np.isfinite(source).all() or not np.isfinite(target).all()):
        raise ValueError('registration requires four finite anchors and held-out points')

    def normalize(points):
        center = points.mean(axis=0)
        extent = np.sqrt(np.mean(np.sum((points-center)**2, axis=1)))
        if extent <= 0:
            raise ValueError('degenerate registration anchors')
        scale = np.sqrt(2)/extent
        transform = np.array([[scale, 0, -scale*center[0]], [0, scale, -scale*center[1]], [0, 0, 1]])
        return (points-center)*scale, transform

    a, ta = normalize(source[anchors]); b, tb = normalize(target[anchors])
    design = []
    for (x, y), (u, v) in zip(a, b):
        design.extend([[-x, -y, -1, 0, 0, 0, u*x, u*y, u],
                       [0, 0, 0, -x, -y, -1, v*x, v*y, v]])
    _, singular, vectors = np.linalg.svd(design)
    if singular[-1] < singular[0]*1e-10:
        raise ValueError('degenerate registration anchors')
    homography = np.linalg.inv(tb) @ vectors[-1].reshape(3, 3) @ ta
    projected = np.column_stack((source, np.ones(len(source)))) @ homography.T
    if np.any(np.abs(projected[:, 2]) < 1e-12):
        raise ValueError('registration projects to infinity')
    projected = projected[:, :2]/projected[:, 2, None]
    errors = np.linalg.norm(projected-target, axis=1)
    return {'homography': homography.tolist(), 'held_out_error': stats(errors[~anchors]),
            'points': [{**row, 'predicted_xy': xy.tolist(), 'error_px': float(error)}
                       for row, xy, error in zip(rows, projected, errors)]}


def compare_frame(frame, reviewed, stars, artifact):
    for key in ('source_sha256', 'width', 'height'):
        if frame[key] != reviewed[key] or frame[key] != artifact[key]:
            raise ValueError(f'{key} mismatch')
    if artifact['pixel_convention'] != 'top_left_zero_based':
        raise ValueError('unsupported report pixel convention')
    sources = {p['id']: p for p in reviewed['points']}
    if len(sources) != len(reviewed['points']):
        raise ValueError('duplicate reviewed source ID')
    seen_sources, seen_stars = set(), set()
    expected, world, records = [], [], []
    for match in frame['matches']:
        source_id, star_id = match['source_id'], match['star_id']
        if source_id in seen_sources or star_id in seen_stars:
            raise ValueError('correspondences must be one-to-one')
        seen_sources.add(source_id); seen_stars.add(star_id)
        point, star = sources[source_id], stars[star_id]
        x, y = point['x'], point['y']
        ra, dec = star['ra_deg'], star['dec_deg']
        if (point['label'] != 'visible_source' or not np.isfinite([x, y, ra, dec]).all()
                or not (0 <= x < frame['width'] and 0 <= y < frame['height'])
                or not (0 <= ra < 360 and -90 <= dec <= 90)):
            raise ValueError('invalid reviewed correspondence')
        if match['method'] not in ('blind_external_correspondence', 'visual_pattern_transfer', 'image_registration_transfer'):
            raise ValueError('unsupported correspondence method')
        records.append({**match, 'x': x, 'y': y, 'ra_deg': ra, 'dec_deg': dec})
        expected.append([x, y]); world.append([ra, dec])
    result = {'id': frame['id'], 'solver_status': artifact['report']['status'],
              'review_status': 'diagnostic_only', 'points': records}
    if not records:
        result['reason'] = 'no independently identified stars'
        return result
    expected = np.asarray(expected)
    result['coverage'] = {'x_min': float(expected[:, 0].min()), 'x_max': float(expected[:, 0].max()),
                          'y_min': float(expected[:, 1].min()), 'y_max': float(expected[:, 1].max()),
                          'width_fraction': float(np.ptp(expected[:, 0]) / frame['width']),
                          'height_fraction': float(np.ptp(expected[:, 1]) / frame['height'])}
    if artifact['report']['status'] != 'solved':
        return result
    camera = artifact['camera']
    if (camera['width'], camera['height']) != (frame['width'], frame['height']):
        raise ValueError('camera dimensions mismatch')
    predicted = project(camera, world)
    errors = np.linalg.norm(predicted - expected, axis=1)
    result['external_star_error'] = stats(errors)
    for row, xy, error in zip(records, predicted, errors):
        row.update(predicted_x=float(xy[0]), predicted_y=float(xy[1]), error_px=float(error))
    return result


def preview(image, reviewed, result, out_dir):
    bright = ImageEnhance.Brightness(image.convert('RGB')).enhance(3)
    sheet = Image.new('RGB', (1000, ((len(reviewed['points']) + 4) // 5) * 120), (25, 25, 25))
    draw = ImageDraw.Draw(sheet)
    for i, point in enumerate(reviewed['points']):
        x, y = round(point['x']), round(point['y'])
        left, top = i % 5 * 200, i // 5 * 120
        crop = bright.crop((x - 25, y - 25, x + 25, y + 25)).resize((100, 100))
        sheet.paste(crop, (left, top + 20))
        color = 'cyan' if point['label'] == 'visible_source' else 'orange'
        draw.text((left, top), f"{point['id']:02d} {point['label']}", fill=color)
    sheet.save(out_dir / f"{reviewed['id']}-sources.jpg")
    bright.thumbnail((1200, 1200))
    sx, sy = bright.width / image.width, bright.height / image.height
    draw = ImageDraw.Draw(bright)
    for point in result['points']:
        x, y = (point['x'] + .5) * sx - .5, (point['y'] + .5) * sy - .5
        draw.ellipse((x-5, y-5, x+5, y+5), outline='cyan', width=1)
        draw.text((x+7, y+4), f"S{point['source_id']:02d}", fill='cyan')
        if 'predicted_x' in point:
            px, py = (point['predicted_x']+.5)*sx-.5, (point['predicted_y']+.5)*sy-.5
            draw.line((px-3, py, px+3, py), fill='magenta')
            draw.line((px, py-3, px, py+3), fill='magenta')
    bright.save(out_dir / f"{reviewed['id']}-matches.jpg")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-review', type=Path, required=True)
    parser.add_argument('--correspondences', type=Path, required=True)
    parser.add_argument('--reports-dir', type=Path, required=True)
    parser.add_argument('--input-dir', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    args = parser.parse_args()
    review = json.loads(args.source_review.read_text())
    correspondences = json.loads(args.correspondences.read_text())
    for data in (review, correspondences):
        if data['schema_version'] != 1 or data['coordinates'] != 'exif_oriented_top_left_zero_based':
            raise ValueError('unsupported annotation format')
    if sha256(args.source_review) != correspondences['source_review_sha256']:
        raise ValueError('source review SHA-256 mismatch')
    frames = {f['id']: f for f in review['frames']}
    ids = [f['id'] for f in correspondences['frames']]
    if len(frames) != len(review['frames']) or len(set(ids)) != len(ids):
        raise ValueError('duplicate frame ID')
    args.out_dir.mkdir(parents=True, exist_ok=True)
    results, report_hashes = [], {}
    for frame in correspondences['frames']:
        frame_id = frame['id']
        if Path(frame_id).name != frame_id:
            raise ValueError('invalid frame ID')
        source = args.input_dir / f'{frame_id}.jpg'
        report = args.reports_dir / f'{frame_id}.json'
        if sha256(source) != frame['source_sha256']:
            raise ValueError('source JPEG SHA-256 mismatch')
        artifact = json.loads(report.read_text())
        result = compare_frame(frame, frames[frame_id], correspondences['stars'], artifact)
        registration = correspondences.get('registration')
        if registration and registration['target_frame'] == frame_id:
            primary_id = registration['primary_frame']
            if Path(primary_id).name != primary_id:
                raise ValueError('invalid primary frame ID')
            primary = args.input_dir / f'{primary_id}.jpg'
            metadata = correspondences['primary_reference']
            if sha256(primary) != metadata['source_sha256']:
                raise ValueError('primary JPEG SHA-256 mismatch')
            with Image.open(primary) as raw_primary:
                if ImageOps.exif_transpose(raw_primary).size != (metadata['width'], metadata['height']):
                    raise ValueError('primary oriented dimensions mismatch')
            for point in registration['points']:
                x, y = point['primary_xy']
                if not (0 <= x < metadata['width'] and 0 <= y < metadata['height']):
                    raise ValueError('primary registration pixel outside image')
            result['image_registration'] = registration_check(registration, frames[frame_id])
        with Image.open(source) as raw:
            image = ImageOps.exif_transpose(raw)
            if image.size != (frame['width'], frame['height']):
                raise ValueError('oriented image dimensions mismatch')
            preview(image, frames[frame_id], result, args.out_dir)
        results.append(result); report_hashes[frame_id] = sha256(report)
    output = {'schema_version': 1, 'review_status': 'diagnostic_only',
              'source_review_sha256': sha256(args.source_review),
              'correspondences_sha256': sha256(args.correspondences),
              'report_sha256': report_hashes, 'frames': results}
    (args.out_dir / 'comparison.json').write_text(json.dumps(output, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
