#!/usr/bin/env python3
"""Camera-independent local pattern review of the frozen Tabit/Gomeisa sources."""
import argparse
import csv
import gzip
from pathlib import Path
import time

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from PIL import Image

from collection_run import ROOT, digest, write_json
from robustness_orion_fit import RID, WIDTH, HEIGHT, previous, read
from robustness_reference_audit import selected, CATALOG, MANIFEST
from robustness_left_source_audit import neighbours, image_metrics
from review_external_stars import registration_check

IDS=(15,13)


def tangent(radec,points):
    w=WCS(naxis=2);w.wcs.crpix=[1,1];w.wcs.crval=radec
    w.wcs.cdelt=[-1,1];w.wcs.ctype=['RA---TAN','DEC--TAN']
    return w.all_world2pix(points,0).tolist()


def registration(target,review):
    """Use the first four visible catalog-selected neighbours, never the target."""
    accepted=[];used=set()
    for p,v in zip(target['pattern'],review['neighbours']):
        if v['hyg_id']!=p['hyg_id']:raise ValueError('review identity order changed')
        if v['label']=='visible_source':
            row=v['axy_row']
            if row in used or row==target['axy_row']:raise ValueError('duplicate or target anchor')
            point=next((a for a in target['extractions'] if a['row']==row),None)
            if point is None:raise ValueError('review outside frozen crop')
            used.add(row);accepted.append((p,point))
        elif v['label'] not in ('ambiguous','not_visible','blend'):
            raise ValueError('finish every neighbour review')
    if len(accepted)<5:return dict(status='insufficient_visible_neighbours',count=len(accepted))
    rows=[dict(source_id=p['hyg_id'],primary_xy=p['tangent_xy'],role='anchor' if i<4 else 'held_out') for i,(p,a) in enumerate(accepted)]
    points=[dict(id=p['hyg_id'],x=a['xy'][0],y=a['xy'][1],label='visible_source') for p,a in accepted]
    rows.append(dict(source_id=target['hyg_id'],primary_xy=[0.,0.],role='held_out'))
    points.append(dict(id=target['hyg_id'],x=target['xy'][0],y=target['xy'][1],label='visible_source'))
    try:
        result=registration_check(dict(points=rows),dict(points=points))
    except ValueError as e:
        return dict(status='degenerate_registration',reason=str(e),count=len(accepted))
    return dict(status='conditional_local_registration',count=len(accepted),**result)


def prepare(out):
    rec=selected()
    if out.exists():raise ValueError('fresh output required')
    paths=[]
    for n in ['40-results','40-protocol','51-results']:
        p=previous(n);check=previous(n[:2]+'-checks')
        if digest(p)!=read(check)['artifact_hashes'][str(p.relative_to(ROOT))]:raise ValueError('published evidence changed: '+n)
        paths.extend([p,check])
    hashes=read(previous('40-protocol'))['hashes']
    extraction=ROOT/next(p for p in hashes if p.endswith('.axy'))
    for p in [CATALOG,extraction]:
        if digest(p)!=hashes[str(p.relative_to(ROOT))]:raise ValueError('catalog/extraction changed')
        paths.append(p)
    image=MANIFEST.parent/rec['file'];protected=read(previous('51-checks'))['protected_hashes']
    for n,sha in protected.items():
        if digest(ROOT/n)!=sha:raise ValueError('protected data changed')
        paths.append(ROOT/n)
    with gzip.open(CATALOG,'rt') as f:stars=list(csv.DictReader(f))
    data=fits.getdata(extraction);xy=np.column_stack((data['X'],data['Y'])).astype(float)-1
    sources=read(previous('40-results'))['sources'];targets=[]
    for sid in IDS:
        s=next(s for s in sources if s['id']==sid);center=s['coordinates']['external_centroid']
        if not np.array_equal(xy[s['axy_row']],center):raise ValueError('saved centroid changed')
        near=neighbours(stars,s['hyg_id']);pattern=near['pattern']
        for p,t in zip(pattern,tangent(s['radec'],[p['radec'] for p in pattern])):p['tangent_xy']=t
        inside=np.all(np.abs(xy-center)<=180,axis=1)
        targets.append(dict(source_id=sid,hyg_id=s['hyg_id'],name=s['name'],radec=s['radec'],xy=center,axy_row=s['axy_row'],
            old_coordinates=s['coordinates'],**near,
            extractions=[dict(row=int(i),xy=xy[i].tolist(),flux=float(data['FLUX'][i])) for i in np.flatnonzero(inside)]))
    out.mkdir(parents=True)
    write_json(out/'input.json',dict(id=RID,split='development',holdout=False,image=str(image.relative_to(ROOT)),targets=targets))
    paths += [out/'input.json',Path(__file__).resolve(),*(Path(__file__).with_name(n) for n in ['collection_run.py','robustness_orion_fit.py',
        'robustness_reference_audit.py','robustness_left_source_audit.py','robustness_geometry.py','review_external_stars.py','compare_local_wcs.py'])]
    write_json(out/'protocol.json',dict(iteration=52,id=RID,split='development',holdout=False,source_ids=list(IDS),
        selection='six nearest HYG<=8mag within3deg plus six nearest without magnitude limit; reuse33; no camera or residual selection',
        review='raw fixed +/-180px crop and independent north-up/east-left TAN chart; all saved extractions labelled; no current camera or WCS projection used',
        registration='only after review frozen: first4 visible neighbours in catalogue order as homography anchors; target and remaining >=1 neighbour held out; reuse registration_check',
        morphology='reuse33 image_metrics: fixed-center luma/RGB r8/r12; compare to saved positions, no refit',
        criteria=['retain both targets and all12 neighbours including ambiguous/blended/not-visible entries',
            'require >=5 visible distinct non-target neighbours for registration; otherwise uncertainty, no substitute or radius/magnitude tuning',
            'local fit is conditional pattern evidence, never full-field GT; no model-accuracy claim for target outside anchor convex hull',
            'reproduce old luma centroids to1e-8px; freeze all review choices before numerical evaluation',
            'no star exclusion or global camera change; no solver/WCS run, no holdout'],
        protected_hashes=protected,hashes={str(p.relative_to(ROOT)):digest(p) for p in sorted(set(paths))}))
    write_json(out/'review.json',dict(id=RID,split='development',targets=[dict(source_id=t['source_id'],label='pending',note='',
        neighbours=[dict(hyg_id=p['hyg_id'],label='pending',axy_row=None,note='') for p in t['pattern']]) for t in targets]))


def validate(out):
    p=read(out/'protocol.json')
    if (p['id'],p['split'],p['holdout'],p['source_ids'])!=(RID,'development',False,list(IDS)):raise ValueError('development only')
    selected()
    for name,sha in p['hashes'].items():
        if digest(ROOT/name)!=sha:raise ValueError('frozen input changed: '+name)
    return p


def render(out):
    validate(out)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    inp=read(out/'input.json');rgb=np.array(Image.open(ROOT/inp['image']).convert('RGB'))
    for t in inp['targets']:
        fig,axes=plt.subplots(1,3,figsize=(16,6),layout='constrained')
        x,y=t['xy']
        for ax,radius in zip(axes[:2],[28,180]):
            x0,x1=int(x)-radius,int(x)+radius;y0,y1=int(y)-radius,int(y)+radius
            patch=np.minimum(rgb[y0:y1,x0:x1].astype(float)*2,255).astype('uint8')
            ax.imshow(patch,extent=[x0-.5,x1-.5,y1-.5,y0-.5],interpolation='nearest')
            ax.scatter([x],[y],marker='+',s=70,c='cyan')
            if radius==28:
                for r in [8,12]:ax.add_patch(plt.Circle((x,y),r,fill=False,color='cyan',lw=.6))
            else:
                for p in t['extractions']:
                    a,b=p['xy'];ax.text(a+2,b+2,str(p['row']),color='yellow',fontsize=7)
            ax.set(xlabel='Original x, px',ylabel='Original y, px (down)',title=f"{t['name']} +/-{radius}px; extraction row {t['axy_row']}")
        ax=axes[2]
        for i,p in enumerate(t['pattern']):
            a,b=p['tangent_xy'];ax.scatter(a,b,s=12+10*(8-p['mag']),color='#245a90')
            ax.annotate(f"{i+1}: HYG {p['hyg_id']} / {p['mag']:.2f}",(a,b),xytext=(4,5),textcoords='offset points',fontsize=8)
        ax.scatter([0],[0],marker='+',s=70,color='#a22');ax.text(.02,-.07,t['name'],color='#a22')
        ax.set(xlim=(-1.8,1.8),ylim=(-1.8,1.8),aspect='equal',xlabel='TAN degrees, east left',ylabel='TAN degrees, north up',title='Catalogue only; no camera alignment')
        ax.grid(alpha=.2)
        fig.suptitle(f"{RID}: fixed local identity review; photo x2 display only\nDavydushta CC BY4.0; HYG/Astronexus CC BY-SA4.0; conditional identities")
        fig.savefig(out/f"source-{t['source_id']}.png",dpi=160);plt.close(fig)


def freeze(out):
    validate(out);review=read(out/'review.json');inp=read(out/'input.json')
    if (out/'review-protocol.json').exists():raise ValueError('review already frozen')
    if (review['id'],review['split'],[t['source_id'] for t in review['targets']])!=(RID,'development',list(IDS)):raise ValueError('review scope changed')
    for t,v in zip(inp['targets'],review['targets']):
        if v['label'] not in ('single_core','ambiguous','blend','not_visible') or not v['note']:raise ValueError('finish target review')
        if len(v['neighbours'])!=len(t['pattern']):raise ValueError('all neighbours required')
        for p,n in zip(t['pattern'],v['neighbours']):
            if n['hyg_id']!=p['hyg_id'] or n['label'] not in ('visible_source','ambiguous','blend','not_visible') or not n['note']:raise ValueError('finish ordered neighbour review')
            if n['label']=='visible_source' and not any(a['row']==n['axy_row'] for a in t['extractions']):raise ValueError('unknown extraction row')
    write_json(out/'review-protocol.json',dict(protocol_sha256=digest(out/'protocol.json'),review_sha256=digest(out/'review.json')))


def measure(out):
    validate(out);started=time.monotonic()
    if (out/'results.json').exists():raise ValueError('fresh analysis required')
    if read(out/'review-protocol.json')!=dict(protocol_sha256=digest(out/'protocol.json'),review_sha256=digest(out/'review.json')):raise ValueError('review changed')
    inp=read(out/'input.json');review=read(out/'review.json');rgb=np.array(Image.open(ROOT/inp['image']).convert('RGB'));rows=[]
    saved=read(previous('51-results'))['cases']
    for t,v in zip(inp['targets'],review['targets']):
        metrics=image_metrics(rgb,t['xy'])
        for r in [8,12]:
            if not np.allclose(metrics['positions'][f'luma_r{r}'],t['old_coordinates'][f'native_r{r}'],rtol=0,atol=1e-8):raise ValueError('old centroid mismatch')
        case=next(c for c in saved if c['method']=='external_centroid' and any(s['id']==t['source_id'] for s in c['sources']))
        index=next(i for i,s in enumerate(case['sources']) if s['id']==t['source_id'])
        residual=np.array(case['forward_residual_original_px'][index]);predicted=np.array(t['xy'])+residual
        rows.append(dict(source_id=t['source_id'],hyg_id=t['hyg_id'],name=t['name'],review=v,measurements=metrics,
            local_registration=registration(t,v),saved_residual_xy_px=residual.tolist(),
            alternative_residual_norms_px={k:float(np.linalg.norm(predicted-p)) for k,p in metrics['positions'].items()}))
    write_json(out/'results.json',dict(iteration=52,id=RID,split='development',holdout=False,protocol_sha256=digest(out/'protocol.json'),
        review_sha256=digest(out/'review.json'),targets=rows,elapsed_s=time.monotonic()-started,
        new_solver_calls=0,new_wcs_calls=0,new_camera_fits=0,production_changed=False,ground_truth_promoted=False,automatic_improvement_confirmed=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['prepare','render','freeze','measure','validate']);p.add_argument('--out-dir',type=Path,required=True)
    a=p.parse_args();globals()[a.stage](a.out_dir.resolve())
