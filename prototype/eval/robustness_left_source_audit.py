#!/usr/bin/env python3
"""Audit the two frozen left-edge probes without changing identities or cameras."""
import argparse
import csv
import gzip
from pathlib import Path

import numpy as np
from astropy.io import fits
from PIL import Image

from collection_run import ROOT, digest, write_json
from compare_local_wcs import project
from robustness_geometry import CATALOG, selected, native_centroid
from robustness_single22_residuals import RID, prior, read

IDS = (24, 29)


def neighbours(stars, target):
    angles = np.array([[float(s['ra'])*15, float(s['dec'])] for s in stars])
    ra,dec = np.deg2rad(angles).T
    units = np.column_stack((np.cos(dec)*np.cos(ra),np.cos(dec)*np.sin(ra),np.sin(dec)))
    index = next(i for i,s in enumerate(stars) if int(s['id'])==target)
    distance = np.rad2deg(2*np.arcsin(np.clip(np.linalg.norm(units-units[index],axis=1)/2,0,1)))
    order = sorted((i for i in range(len(stars)) if i!=index),key=lambda i:(distance[i],int(stars[i]['id'])))
    def row(i):
        s=stars[i]
        return dict(hyg_id=int(s['id']),name=s['proper'] or s['bf'],mag=float(s['mag']) if s['mag'] else None,
                    radec=angles[i].tolist(),separation_deg=float(distance[i]),comp=s.get('comp'),comp_primary=s.get('comp_primary'))
    return dict(nearest_all_magnitudes=[row(i) for i in order[:6]],
                pattern=[row(i) for i in order if distance[i]<=3 and stars[i]['mag'] and float(stars[i]['mag'])<=8][:6])


def relative_positions(camera, target_radec, measured_xy, other_radec):
    """Translate the saved projection to the fixed target; no scale/rotation fit."""
    return project(camera,other_radec)-project(camera,[target_radec])[0]+measured_xy


def prepare(out):
    rec=selected([RID])[0]
    if out.exists():raise ValueError('fresh output required')
    frame=next(f for f in read(prior('5-review'))['frames'] if f['id']==RID)
    image=ROOT/frame['image']
    if digest(image)!=rec['clean_sha256']:raise ValueError('image changed')
    old=read(prior('32-protocol'))
    for path in [CATALOG,prior('5-geometry'),prior('13-geometry')]:
        # Catalog is pinned by the original review protocol, geometry by iteration32.
        hashes=read(prior('5-protocol'))['hashes'] if path==CATALOG else old['hashes']
        if digest(path)!=hashes[str(path.relative_to(ROOT))]:raise ValueError('prior input changed')
    with gzip.open(CATALOG,'rt') as f:stars=list(csv.DictReader(f))
    probes=next(f['sources'] for f in read(prior('13-geometry'))['frames'] if f['id']==RID)
    extraction=ROOT/frame['extraction']
    data=fits.getdata(extraction);header=fits.getheader(extraction)
    if (header['IMAGEW'],header['IMAGEH'])!=(rec['width'],rec['height']):raise ValueError('extraction dimensions differ')
    xy=np.column_stack((data['X'],data['Y'])).astype(float)-1
    camera=next(c['camera'] for c in read(prior('30-results'))['cases'] if c['name']=='original22_external')
    targets=[]
    for sid in IDS:
        p=next(s for s in probes if s['source_id']==sid)
        prev=next(s for s in frame['points'] if s['id']==sid)
        chosen=next(c for c in prev['choices'] if c['row']==prev['selected_row'])
        if [chosen['x'],chosen['y']]!=p['xy'] or not np.array_equal(xy[chosen['row']],p['xy']):raise ValueError('centroid changed')
        near=neighbours(stars,p['hyg_id'])
        pattern=near['pattern']
        positions=relative_positions(camera,p['radec'],p['xy'],[n['radec'] for n in pattern])
        for n,pos in zip(pattern,positions):
            order=np.argsort(np.linalg.norm(xy-pos,axis=1))[:3]
            n.update(relative_prediction_xy=pos.tolist(),choices=[dict(row=int(i),xy=xy[i].tolist(),distance_px=float(np.linalg.norm(xy[i]-pos))) for i in order if np.linalg.norm(xy[i]-pos)<=20])
        targets.append(dict(**p,axy_row=prev['selected_row'],previous_review=prev,**near))
    out.mkdir(parents=True)
    write_json(out/'input.json',dict(id=RID,split='development',image=frame['image'],targets=targets,camera=camera))
    paths=[Path(__file__).resolve(),image,CATALOG,extraction,out/'input.json',*(prior(n) for n in ['5-review','5-protocol','5-geometry','13-geometry','30-results','32-protocol','32-results']),
           *(ROOT/'prototype/eval'/n for n in ['robustness_geometry.py','robustness_single22_residuals.py','compare_local_wcs.py','collection_run.py']),
           *(ROOT/'data/samples/sky-samples'/n for n in ['manifest.json','robustness-results.json','collection-wcs-review.json'])]
    write_json(out/'protocol.json',dict(iteration=33,id=RID,split='development',holdout=False,source_ids=list(IDS),
        selection='both left-edge probes singled out by iteration32; targeted diagnostic, not unbiased sample',
        neighbours='six closest HYG entries without magnitude cut; separate six closest <=8mag within3deg for pattern review',
        pattern='saved original22_external projection translated to target centroid; no refit; three nearest existing extractions within20px are candidates only',
        morphology='unchanged native_centroid r8/r12 and annulus16-22 on Rec709 luma and each RGB channel; peak and clipping within radius12',
        criteria=['retain both source IDs and old labels regardless of findings','compare measured centroid variation to frozen camera residual vectors; no fitted correction',
                  'visual morphology and local pattern evidence remain conditional; no absolute identity/GT promotion',
                  'no LM, rematch, holdout, detection or acceptance change'],hashes={str(p.relative_to(ROOT)):digest(p) for p in paths}))
    write_json(out/'review.json',dict(id=RID,split='development',targets=[dict(source_id=s,label='pending',note='',pattern_note='') for s in IDS]))
    render(out)


def validate(out):
    p=read(out/'protocol.json')
    if (p['id'],p['split'],p['holdout'],p['source_ids'])!=(RID,'development',False,list(IDS)):raise ValueError('frozen development required')
    selected([RID])
    for name,sha in p['hashes'].items():
        if digest(ROOT/name)!=sha:raise ValueError('frozen input changed: '+name)
    return p


def render(out):
    validate(out)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    inp=read(out/'input.json');rgb=np.asarray(Image.open(ROOT/inp['image']).convert('RGB'))
    fig,axes=plt.subplots(2,3,figsize=(15,10))
    for row,p in enumerate(inp['targets']):
        x,y=p['xy']
        for col,radius in enumerate((32,100,230)):
            ax=axes[row,col];x0,x1=max(0,int(x)-radius),min(rgb.shape[1],int(x)+radius+1);y0,y1=max(0,int(y)-radius),min(rgb.shape[0],int(y)+radius+1)
            ax.imshow(np.clip(rgb[y0:y1,x0:x1].astype(float)*2/255,0,1),extent=(x0-.5,x1-.5,y1-.5,y0-.5),interpolation='nearest')
            ax.scatter([x],[y],s=170,facecolors='none',edgecolors='cyan')
            if col==0:
                for rr in (8,12):ax.add_patch(plt.Circle((x,y),rr,fill=False,color='cyan',alpha=.5))
            if col==2:
                for j,n in enumerate(p['pattern']):
                    a,b=n['relative_prediction_xy'];ax.scatter([a],[b],marker='+',color='yellow');ax.annotate(str(j),(a,b),color='yellow')
            ax.set_xlim(x0-.5,x1-.5);ax.set_ylim(y1-.5,y0-.5);ax.set_title(f"{p['source_id']} {p['name']} / +/-{radius}px")
    fig.suptitle('Fixed source audit: x2 display only; cyan=old centroid; yellow=translated catalog pattern\nPhotograph: oliwok CC BY4.0; HYG: Astronexus CC BY-SA4.0; conditional identities')
    fig.tight_layout();fig.savefig(out/'source-audit.png',dpi=140);plt.close(fig)


def image_metrics(rgb,center):
    gray=rgb.astype(np.float32)@np.array([.2126,.7152,.0722],dtype=np.float32)
    positions={f'{band}_r{r}':native_centroid(array,center,r) for band,array in [('luma',gray),*[(n,rgb[:,:,i]) for i,n in enumerate('RGB')]] for r in (8,12)}
    x,y=center;x0,x1=int(x)-13,int(x)+14;y0,y1=int(y)-13,int(y)+14
    patch=rgb[y0:y1,x0:x1];yy,xx=np.mgrid[y0:y1,x0:x1];mask=np.hypot(xx-x,yy-y)<=12
    lum=gray[y0:y1,x0:x1];peak=np.unravel_index(np.argmax(np.where(mask,lum,-np.inf)),lum.shape)
    return dict(positions=positions,shifts_px={k:float(np.linalg.norm(np.array(v)-center)) for k,v in positions.items()},
        peak_xy=[int(xx[peak]),int(yy[peak])],aperture_pixels=int(mask.sum()),clipped_pixels_any_channel=int(np.any(patch[mask]==255,axis=1).sum()),
        maximum_rgb=patch[mask].max(axis=0).tolist())


def measure(out):
    validate(out);inp=read(out/'input.json');review=read(out/'review.json')
    if review['id']!=RID or review['split']!='development' or [p['source_id'] for p in review['targets']]!=list(IDS):raise ValueError('review scope changed')
    if any(p['label'] not in ('single_core','ambiguous','blend','not_visible') or not p['note'] or not p['pattern_note'] for p in review['targets']):raise ValueError('complete visual review required')
    rgb=np.asarray(Image.open(ROOT/inp['image']).convert('RGB'))
    alt=next(f['sources'] for f in read(prior('5-geometry'))['frames'] if f['id']==RID)
    saved=read(prior('32-results'));rows=[]
    for p,v in zip(inp['targets'],review['targets']):
        m=image_metrics(rgb,p['xy']);old=next(s for s in alt if s['source_id']==p['source_id'])
        for r in (8,12):
            if not np.allclose(m['positions'][f'luma_r{r}'],old[f'native_r{r}_xy'],rtol=0,atol=1e-8):raise ValueError('old luma centroid not reproduced')
        residuals={c['name']:next(s['residual_xy_px'] for s in c['sources'] if s['source_id']==p['source_id']) for c in saved['cases']}
        rows.append(dict(source_id=p['source_id'],hyg_id=p['hyg_id'],name=p['name'],xy=p['xy'],review=v,measurements=m,
            saved_fit19_residuals_px=residuals,nearest_all_magnitudes=p['nearest_all_magnitudes'],pattern=p['pattern']))
    write_json(out/'results.json',dict(iteration=33,id=RID,split='development',holdout=False,protocol_sha256=digest(out/'protocol.json'),
        review_sha256=digest(out/'review.json'),targets=rows,old_luma_centroids_reproduced=True,new_refinements=0,new_searches=0,
        production_changed=False,ground_truth_promoted=False,automatic_improvement_confirmed=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=('prepare','render','measure','validate'));parser.add_argument('--out-dir',type=Path,required=True)
    args=parser.parse_args();globals()[args.stage](args.out_dir.resolve())
