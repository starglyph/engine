#!/usr/bin/env python3
"""Independent Astrometry.net re-solve using its own extracted points and manual sky masks.

Requires completed solve_local_wcs.py outputs. No engine detections, pose/FOV
hints or catalog correspondences are supplied to Astrometry.net.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from sky_mask_experiment import in_sky


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def restore_full_frame(wcs_path, corr_path, roi, width, height):
    """Translate crop-local FITS pixels; SIP coefficients stay relative to CRPIX."""
    header = fits.getheader(wcs_path)
    header['CRPIX1'] += roi[0]
    header['CRPIX2'] += roi[1]
    header['IMAGEW'], header['IMAGEH'] = width, height
    restored_wcs = wcs_path.with_name('full-frame.wcs')
    fits.PrimaryHDU(header=header).writeto(restored_wcs, overwrite=True)
    restored_corr = corr_path.with_name('full-frame.corr')
    if corr_path.exists():
        with fits.open(corr_path) as hdus:
            for axis, offset in (('x', roi[0]), ('y', roi[1])):
                for prefix in ('field', 'index'):
                    hdus[1].data[f'{prefix}_{axis}'] += offset
            hdus.writeto(restored_corr, overwrite=True)
    else:
        restored_corr.unlink(missing_ok=True)
    return restored_wcs, restored_corr


def reviewed_selection(data, review, reference, extraction_sha256, sky_selection):
    """Select frozen, visually reviewed external rows without changing centroids."""
    for key in ('source_sha256', 'width', 'height'):
        if review[key] != reference[key]:
            raise ValueError(f'source review {key} mismatch')
    if review['extraction_sha256'] != extraction_sha256:
        raise ValueError('source review extraction SHA-256 mismatch')
    selected = np.zeros(len(data), dtype=bool)
    seen = set()
    for point in review['points']:
        row = point['axy_row']
        if type(row) is not int or row < 0 or row >= len(data) or row in seen:
            raise ValueError('invalid or duplicate source review row')
        seen.add(row)
        if point['label'] not in ('visible_source', 'ambiguous_texture', 'foreground', 'long_trail'):
            raise ValueError('unsupported source review label')
        expected = np.array([data[row]['X'], data[row]['Y']], dtype=float) - 1
        if not np.allclose([point['x'], point['y']], expected, rtol=0, atol=1e-6):
            raise ValueError('source review coordinates mismatch')
        if point['label'] == 'visible_source':
            if not sky_selection[row]:
                raise ValueError('reviewed source outside sky mask')
            selected[row] = True
    return selected


def verify(reference, mask, raw_dir, out_dir, args):
    source = Path(reference['source'])
    frame_id = source.stem
    if reference['pixel_convention'] != 'top_left_zero_based':
        raise ValueError('unsupported pixel convention')
    if digest(source) != reference['source_sha256'] or mask['source_sha256'] != reference['source_sha256']:
        raise ValueError('source SHA-256 mismatch')
    if (mask['width'],mask['height']) != (reference['width'],reference['height']):
        raise ValueError('oriented mask dimensions mismatch')
    if digest(raw_dir/frame_id/'input.fits') != reference['input_fits_sha256']:
        raise ValueError('normalized FITS SHA-256 mismatch')
    result = {k:reference[k] for k in ('source','source_sha256','width','height','pixel_convention')}
    result.update(status='unresolved', attempts=[])
    roi = getattr(args, 'roi', None)
    if roi is not None:
        if not (0 <= roi[0] < roi[2] <= mask['width'] and 0 <= roi[1] < roi[3] <= mask['height']):
            raise ValueError('ROI outside oriented image or empty')
        result['solve_region_xyxy'] = list(roi)
    solve_width = roi[2] - roi[0] if roi else mask['width']
    solve_height = roi[3] - roi[1] if roi else mask['height']
    review = getattr(args, 'reviews', {}).get(frame_id)
    factors = (review['downsample'],) if review else (4,2)
    if any(factor not in (2, 4) for factor in factors):
        raise ValueError('unsupported reviewed extraction downsample')
    for factor in factors:
        extraction = raw_dir/frame_id/f'downsample-{factor}'/'field.axy'
        if not extraction.exists():
            if review:
                raise ValueError('reviewed extraction missing')
            continue
        with fits.open(extraction) as hdus:
            if (hdus[0].header['IMAGEW'],hdus[0].header['IMAGEH']) != (mask['width'],mask['height']):
                raise ValueError('external extraction geometry mismatch')
            data = hdus[1].data
            keep = np.array([in_sky({'x':float(d['X'])-1,'y':float(d['Y'])-1},mask) for d in data], dtype=bool)
            n_in_sky = int(keep.sum())
            if review:
                keep = reviewed_selection(data, review, reference, digest(extraction), keep)
            if roi:
                keep &= ((data['X'] - 1 >= roi[0]) & (data['X'] - 1 < roi[2])
                         & (data['Y'] - 1 >= roi[1]) & (data['Y'] - 1 < roi[3]))
            selected = data[keep].copy()
            if roi:
                selected['X'] -= roi[0]
                selected['Y'] -= roi[1]
            destination = out_dir/frame_id/f'downsample-{factor}'
            destination.mkdir(parents=True,exist_ok=True)
            xy = destination/'sky.xyls'
            fits.HDUList([fits.PrimaryHDU(),fits.BinTableHDU(selected)]).writeto(xy,overwrite=True)
        for name in ('field.solved','field.wcs','field.corr','full-frame.wcs','full-frame.corr'):
            (destination/name).unlink(missing_ok=True)
        command = [args.solve_field,str(xy),'--width',str(solve_width),'--height',str(solve_height),
                   '--x-column','X','--y-column','Y','--sort-column','FLUX','--dir',str(destination),
                   '--out','field','--overwrite','--no-plots','--no-verify','--new-fits','none',
                   '--uniformize','0','--resort','--pixel-error',str(getattr(args, 'pixel_error', 4)),
                   '--code-tolerance',str(getattr(args, 'code_tolerance', 0.01)),
                   '--depth','10,20,30,50','--cpulimit',str(args.cpu_limit),
                   '--tweak-order',str(getattr(args, 'tweak_order', 2)),
                   '--crpix-center','--config',str(getattr(args, 'solver_config', None) or args.config)]
        attempt = {'extraction_sha256':digest(extraction),'selected_xy_sha256':digest(xy),
                   'n_extracted':len(data),'n_in_sky':n_in_sky,'n_selected':len(selected),'command':command}
        if review:
            attempt['reviewed_rows'] = np.flatnonzero(keep).tolist()
        with (destination/'solve.log').open('w') as log:
            with subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True) as process:
                try:
                    attempt['exit_code'] = process.wait(timeout=args.cpu_limit+30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid,signal.SIGKILL);process.wait();attempt['wall_timeout'] = True
        result['attempts'].append(attempt)
        marker, wcs_path = destination/'field.solved', destination/'field.wcs'
        if attempt.get('exit_code') != 0 or not marker.exists() or marker.read_bytes() != b'\x01' or not wcs_path.exists():
            continue
        corr = destination/'field.corr'
        if roi:
            result['native_crop_wcs_sha256'] = digest(wcs_path)
            wcs_path, corr = restore_full_frame(wcs_path, corr, roi, mask['width'], mask['height'])
        points = np.array([(x,y) for y in np.linspace(0,mask['height']-1,9) for x in np.linspace(0,mask['width']-1,9)])
        wcs = WCS(fits.getheader(wcs_path));world = wcs.all_pix2world(points,0)
        if not np.isfinite(world).all():
            raise ValueError('non-finite WCS')
        center = wcs.all_pix2world([[mask['width']/2,mask['height']/2]],0)[0]
        result.update(status='solved_candidate',review_status='pending',wcs_file=str(wcs_path),wcs_sha256=digest(wcs_path),
                      center_ra_deg=float(center[0]),center_dec_deg=float(center[1]),
                      reference_points=[{'x':float(x),'y':float(y),'ra_deg':float(ra),'dec_deg':float(dec)}
                                        for (x,y),(ra,dec) in zip(points,world)])
        if corr.exists():
            result.update(correspondences_file=str(corr),correspondences_sha256=digest(corr),n_correspondences=len(fits.getdata(corr)))
        break
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-dir',type=Path,required=True)
    parser.add_argument('--masks',type=Path,required=True)
    parser.add_argument('--out-dir',type=Path,required=True)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--solver-config',type=Path,help='explicit alternative verification catalog config')
    parser.add_argument('--solver-index-dir',type=Path,help='index directory for alternative verification provenance')
    parser.add_argument('--solve-field',default='solve-field')
    parser.add_argument('--cpu-limit',type=int,default=30)
    parser.add_argument('--ids',help='comma-separated IDs; all must have completed raw extraction')
    parser.add_argument('--source-review',type=Path,help='frozen visual labels for external extraction rows')
    parser.add_argument('--pixel-error',type=float,default=4,help='external verification positional error in original pixels')
    parser.add_argument('--tweak-order',type=int,choices=(2,3),default=2,help='external SIP polynomial order')
    parser.add_argument('--code-tolerance',type=float,default=0.01,help='external quad matching distance')
    parser.add_argument('--roi',type=int,nargs=4,metavar=('X0','Y0','X1','Y1'),
                        help='half-open region in original zero-based oriented pixels; one frame only')
    args=parser.parse_args()
    if args.cpu_limit<1:parser.error('--cpu-limit must be positive')
    if bool(args.solver_config) != bool(args.solver_index_dir):
        parser.error('--solver-config and --solver-index-dir must be supplied together')
    if not np.isfinite(args.pixel_error) or args.pixel_error <= 0:parser.error('--pixel-error must be finite and positive')
    if not np.isfinite(args.code_tolerance) or args.code_tolerance <= 0:parser.error('--code-tolerance must be finite and positive')
    args.raw_dir=args.raw_dir.resolve();args.out_dir=args.out_dir.resolve();args.config=args.config.resolve()
    raw=json.loads((args.raw_dir/'summary.json').read_text())
    mask_file=json.loads(args.masks.read_text())
    if mask_file['schema_version'] != 1 or mask_file['coordinates'] != 'exif_oriented_normalized_image_edges':
        parser.error('unsupported mask format')
    if not raw['provenance']['config'] or digest(args.config) != raw['provenance']['config']['sha256']:
        parser.error('configuration differs from raw extraction provenance')
    verification_provenance = raw['provenance']
    if args.solver_config:
        args.solver_config = args.solver_config.resolve()
        indices = sorted(args.solver_index_dir.resolve().glob('index-*.fits'))
        if not indices:parser.error('alternative verification index directory is empty')
        verification_provenance = {
            'config': {'path':str(args.solver_config), 'sha256':digest(args.solver_config)},
            'indices': [{'path':str(path), 'sha256':digest(path)} for path in indices],
            'note':'Alternative verification catalog; source extraction provenance retained separately.'}
    masks={m['id']:m for m in mask_file['masks']}
    references={Path(r['source']).stem:r for r in raw['frames']}
    ids=args.ids.split(',') if args.ids else list(references)
    if args.roi and len(ids) != 1:parser.error('--roi requires exactly one frame ID')
    if not set(ids)<=references.keys() or not set(ids)<=masks.keys():parser.error('missing reference or mask ID')
    review_provenance = {}
    if args.source_review:
        review_file = json.loads(args.source_review.read_text())
        if review_file['schema_version'] != 1 or review_file['coordinates'] != 'exif_oriented_top_left_zero_based':
            parser.error('unsupported source review format')
        args.reviews = {frame['id']:frame for frame in review_file['frames']}
        if len(args.reviews) != len(review_file['frames']) or not set(ids) <= args.reviews.keys():
            parser.error('duplicate or missing source review ID')
        review_provenance = {'source_review_sha256':digest(args.source_review)}
    records=[]
    args.out_dir.mkdir(parents=True,exist_ok=True)
    for frame_id in ids:
        result=verify(references[frame_id],masks[frame_id],args.raw_dir,args.out_dir,args)
        records.append(result)
        (args.out_dir/'summary.json').write_text(json.dumps({'solver':'Astrometry.net with manual sky point selection',
            'version':raw['version'],'raw_summary_sha256':digest(args.raw_dir/'summary.json'),
            'mask_sha256':digest(args.masks),**review_provenance,'provenance':raw['provenance'],
            'verification_provenance':verification_provenance,'frames':records},indent=2)+'\n')
        print(frame_id,result['status'],result.get('n_correspondences',''),flush=True)


if __name__=='__main__':main()
