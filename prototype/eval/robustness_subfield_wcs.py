#!/usr/bin/env python3
"""Reuse saved independent extraction and the existing ROI WCS verifier."""
import argparse
import os
from pathlib import Path
import subprocess
import time
from types import SimpleNamespace

from collection_run import ROOT,digest,write_json
from robustness_subfield_identity import RID,REGIONS,read,validate as validate_local
from verify_masked_wcs import verify

RAW=ROOT/f'prototype/artifacts/robustness-baseline/wcs-run/{RID}'
INDICES=ROOT/'prototype/artifacts/sky-edge-review/hyg-indices'


def environment(tool_root):
    env=os.environ.copy()
    env['PATH']=str(tool_root/'usr/bin')+os.pathsep+env.get('PATH','')
    env['LD_LIBRARY_PATH']=str(tool_root/'usr/lib/x86_64-linux-gnu')
    env['PYTHONPATH']=os.pathsep.join([str(tool_root/'usr/lib/python3/dist-packages'),str(ROOT/'prototype/artifacts/robustness-wcs-python')])
    env['PYTHONDONTWRITEBYTECODE']='1'
    return env


def prepare(out,tool_root):
    validate_local(out)
    if (out/'external-protocol.json').exists():raise ValueError('fresh external experiment required')
    subprocess.run(['/usr/bin/python3','-c','from astrometry.util.removelines import removelines; import numpy, astropy'],env=environment(tool_root),check=True)
    raw=out/'external-input'/RID;raw.mkdir(parents=True)
    (raw/'downsample-4').mkdir()
    (raw/'downsample-4/field.axy').symlink_to(RAW/'downsample-4/field.axy')
    # Reconstruct the deleted, rebuildable FITS with the original exporter and
    # require its original hash; the photograph and reference remain unchanged.
    from astropy.io import fits
    from PIL import Image
    import numpy as np
    ref=read(RAW/'reference.json')
    with Image.open(ref['source']) as image:
        pixels=np.asarray(image.convert('L'),dtype=np.float32)
    fits.writeto(raw/'input.fits',pixels)
    if digest(raw/'input.fits')!=ref['input_fits_sha256']:raise ValueError('reconstructed FITS differs')
    # Downsample2 deliberately absent: exactly one saved extraction per ROI.
    ref=read(RAW/'reference.json')
    cfg=out/'external-backend.cfg'
    cfg.write_text('cpulimit 30\nadd_path '+str(INDICES)+'\n'+''.join(f'index index-hyg-{i}.fits\n'for i in range(10,20)))
    paths=[Path(__file__).resolve(),Path(__file__).with_name('verify_masked_wcs.py'),Path(__file__).with_name('sky_mask_experiment.py'),
        out/'protocol.json',RAW/'reference.json',raw/'input.fits',RAW/'downsample-4/field.axy',cfg,INDICES/'provenance.json',INDICES/'provenance-complete.json',
        *(INDICES/f'index-hyg-{i}.fits'for i in range(10,20)),tool_root/'usr/bin/solve-field',tool_root/'usr/bin/astrometry-engine']
    paths+=list((tool_root/'usr/lib/x86_64-linux-gnu').glob('*.so.*'))
    paths+=sorted(p for p in (tool_root/'usr/lib/python3/dist-packages').rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    write_json(out/'external-protocol.json',dict(iteration=39,id=RID,split='development',holdout=False,regions=REGIONS,
        tool_root=str(tool_root),reference=ref,extraction='existing downsample4; same three ROIs; no field re-extraction',
        catalog='existing HYG mag<=6.5 indices10..19; independent algorithm, shared catalog',
        budget=dict(attempts=3,cpu_per_roi_s=30,wall_per_roi_s=60),
        criteria=['same frozen regions as tetra3 local experiment; no coordinate or parameter tuning',
            'retain each external outcome and timeout; successful WCS remains pending until unused-star review',
            'crop-local solution does not validate whole-frame projection'],
        hashes={str(p):digest(p)for p in paths}))


def validate(out):
    validate_local(out);p=read(out/'external-protocol.json')
    if (p['id'],p['split'],p['holdout'],p['regions'])!=(RID,'development',False,[[n,b]for n,b in REGIONS]):raise ValueError('scope changed')
    for name,sha in p['hashes'].items():
        if digest(Path(name))!=sha:raise ValueError('external freeze changed: '+name)
    return p


def run(out):
    p=validate(out)
    if (out/'external-results.json').exists():raise ValueError('fresh run required')
    root=Path(p['tool_root']);os.environ.update(environment(root))
    ref=p['reference'];mask=dict(width=6016,height=3760,source_sha256=ref['source_sha256'],sky_polygon=[[0,0],[1,0],[1,1],[0,1]])
    rows=[]
    for name,box in REGIONS:
        args=SimpleNamespace(solve_field=str(root/'usr/bin/solve-field'),config=out/'external-backend.cfg',roi=box,cpu_limit=30)
        started=time.monotonic();r=verify(ref,mask,out/'external-input',out/'external'/name,args)
        rows.append(dict(name=name,elapsed_s=time.monotonic()-started,result=r))
        write_json(out/'external-results.json',dict(iteration=39,id=RID,split='development',holdout=False,
            protocol_sha256=digest(out/'external-protocol.json'),completed=len(rows),cases=rows,independent_ground_truth=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['prepare','validate','run']);p.add_argument('--out-dir',type=Path,required=True);p.add_argument('--tool-root',type=Path)
    a=p.parse_args()
    if a.stage=='prepare':
        if a.tool_root is None:p.error('--tool-root required')
        prepare(a.out_dir.resolve(),a.tool_root.resolve())
    else:globals()[a.stage](a.out_dir.resolve())
