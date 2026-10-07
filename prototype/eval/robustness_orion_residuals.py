#!/usr/bin/env python3
"""Describe saved Orion residual vectors; no fitting or source selection."""
import argparse
from pathlib import Path
import time

import numpy as np

from collection_run import ROOT, digest, write_json
from compare_local_wcs import project
from robustness_orion_fit import METHODS, WIDTH, HEIGHT, RID, read, previous
from robustness_reference_audit import selected
from robustness_refinement_report import lift
from robustness_residual_fields import decompose, groups, agreement
from robustness_single22_residuals import describe


def masks_for(sources, selection):
    ids = [s['id'] for s in sources]
    fit, check = selection['fit_ids'], selection['check_ids']
    if len(ids) != 47 or len(set(ids)) != 47 or len(fit) != 16 or len(check) != 31 or set(fit)&set(check) or set(fit+check) != set(ids):
        raise ValueError('unchanged disjoint16/31 selection required')
    xy = np.array([s['coordinates']['external_centroid'] for s in sources])
    rho, *_ = decompose(xy, np.zeros_like(xy), WIDTH, HEIGHT)
    masks = groups(xy, np.zeros(47, bool), WIDTH, HEIGHT, rho)
    masks.pop('outside_both_inputs')
    for name in {g for s in sources for g in s['groups']}:
        m = np.array([name in s['groups'] for s in sources])
        if name in masks and not np.array_equal(m, masks[name]):
            raise ValueError('legacy group membership changed')
        masks[name] = m
    return {part+'/'+name: m & np.array([s['id'] in members for s in sources])
        for part, members in [('all47', ids), ('fit16', fit), ('check31', check)] for name, m in sorted(masks.items())}


def prepare(out):
    selected()
    paths = [previous(n) for n in ['40-results', '40-checks', '42-results', '42-checks', '43-results', '43-checks']]
    for n in ['40', '42', '43']:
        p = previous(n+'-results')
        if digest(p) != read(previous(n+'-checks'))['artifact_hashes'][str(p.relative_to(ROOT))]:
            raise ValueError('published evidence changed')
    protected = read(previous('43-checks'))['protected_hashes']
    for name, sha in protected.items():
        if digest(ROOT/name) != sha:
            raise ValueError('protected data changed')
        paths.append(ROOT/name)
    saved = read(previous('43-results'))
    sources = read(previous('40-results'))['sources']
    masks = masks_for(sources, saved['selection'])
    if out.exists():
        raise ValueError('fresh output required')
    out.mkdir(parents=True)
    write_json(out/'input.json', dict(id=RID, split='development', source_ids=[s['id'] for s in sources],
        selection=saved['selection'], groups={k:[s['id'] for s, keep in zip(sources, m) if keep] for k,m in masks.items()}))
    paths += [Path(__file__).resolve(), out/'input.json', *(Path(__file__).with_name(n) for n in [
        'robustness_residual_fields.py', 'robustness_single22_residuals.py', 'robustness_refinement_report.py',
        'robustness_orion_fit.py', 'robustness_reference_audit.py', 'compare_local_wcs.py', 'collection_run.py'])]
    write_json(out/'protocol.json', dict(iteration=44, id=RID, split='development', holdout=False,
        analyses=['saved native control, three distributed16 replay cameras, saved pending external WCS projections',
            'reuse radial/tangential decomposition and vector metrics; nominal image center; x right,y down; predicted minus measured',
            'fixed original groups plus3x3 grid and radius bins0/.35/.70/1; all47,fit16,check31; preserve empty groups',
            'all three centroid methods, and agreement of matched-method distributed fits'],
        criteria=['reproduce all saved projections and residual norms within1e-8px',
            'fit/check partitions identical to42/43; no source removal or regrouping by residual',
            'radial plus tangential squared energy equals total; group memberships frozen before analysis',
            'descriptive only: no causal lens claim, added parameters, camera correction, acceptance change or ground truth'],
        plot_arrow_magnification=2., hashes={str(p.relative_to(ROOT)):digest(p) for p in paths},
        new_fits=0,new_solver_calls=0,new_wcs_calls=0))


def validate(out):
    p = read(out/'protocol.json')
    if (p['id'],p['split'],p['holdout']) != (RID,'development',False):
        raise ValueError('development only')
    selected()
    for name,sha in p['hashes'].items():
        if digest(ROOT/name) != sha:
            raise ValueError('frozen input changed: '+name)
    return p


def measure(out):
    p=validate(out); started=time.monotonic()
    old=read(previous('40-results')); saved=read(previous('43-results'))
    sources=old['sources']; masks=masks_for(sources,saved['selection'])
    inp=read(out/'input.json')
    if inp['source_ids'] != [s['id'] for s in sources] or inp['groups'] != {k:[s['id'] for s,v in zip(sources,m) if v] for k,m in masks.items()}:
        raise ValueError('frozen membership changed')
    positions={m:np.array([s['coordinates'][m] for s in sources]) for m in METHODS}
    projections={arm:np.array([s['predicted_xy'][key] for s in sources]) for arm,key in [('native_control','control'),('pending_external','external')]}
    for method in METHODS:
        case=next(c for c in saved['cases'] if c['name']=='replay_'+method)
        pred=lift(project(case['camera'],[s['radec'] for s in sources]),case['camera'],WIDTH,HEIGHT)
        if not np.allclose(pred,case['predicted_xy'],rtol=0,atol=1e-8) or not np.allclose(np.linalg.norm(pred-positions[method],axis=1),case['errors_px'],rtol=0,atol=1e-8):
            raise ValueError('saved distributed projection/residual mismatch')
        projections['distributed_'+method]=pred
    arms={name:describe(positions,pred,masks,WIDTH,HEIGHT,'external_centroid') for name,pred in projections.items()}
    for arm in arms.values():
        for method,data in arm['methods'].items():
            v=np.array(data['residual_xy_px']);rho,rr,tt,valid=decompose(positions[method],v,WIDTH,HEIGHT)
            if not np.allclose((rr**2+tt**2)[valid],np.sum(v**2,axis=1)[valid],rtol=1e-12,atol=1e-8):
                raise ValueError('vector energy decomposition failed')
            data.update(radial_px=rr.tolist(),tangential_px=tt.tolist(),radius_normalized=rho.tolist())
    vectors={m:projections['distributed_'+m]-positions[m] for m in METHODS}
    comparisons={m:{k:agreement(vectors['external_centroid'],v,mask) for k,mask in masks.items()}
        for m,v in vectors.items() if m!='external_centroid'}
    write_json(out/'results.json',dict(iteration=44,id=RID,split='development',holdout=False,
        protocol_sha256=digest(out/'protocol.json'),sources=[dict(id=s['id'],hyg_id=s['hyg_id'],name=s['name'],xy=s['coordinates']['external_centroid']) for s in sources],
        selection=saved['selection'],group_ids=inp['groups'],arms=arms,matched_fit_agreement=comparisons,
        elapsed_s=time.monotonic()-started,projections_reproduced=True,new_fits=0,new_solver_calls=0,new_wcs_calls=0,
        production_changed=False,independent_ground_truth=False,automatic_improvement_confirmed=False))


def plot(out):
    p=validate(out);r=read(out/'results.json')
    if r['protocol_sha256'] != digest(out/'protocol.json'):
        raise ValueError('result protocol mismatch')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({'font.size':10,'svg.hashsalt':'starglyph-iteration-44'})
    xy=np.array([s['xy'] for s in r['sources']]);fit=np.array([s['id'] in r['selection']['fit_ids'] for s in r['sources']])
    colors=np.where(fit,'#155b9b','#c04c00');gain=p['plot_arrow_magnification']
    fig,axes=plt.subplots(1,3,figsize=(15,5),layout='constrained')
    for ax,name,title in zip(axes,['native_control','distributed_external_centroid','pending_external'],['Native control','Distributed16 fit','Pending external WCS']):
        data=r['arms'][name]['methods']['external_centroid'];v=np.array(data['residual_xy_px'])
        ax.add_patch(plt.Rectangle((0,0),WIDTH,HEIGHT,facecolor='#fafafa',edgecolor='#777',lw=.8,zorder=0))
        for f in (1/3,2/3):
            ax.axvline(WIDTH*f,color='#ddd',lw=.7,zorder=0);ax.axhline(HEIGHT*f,color='#ddd',lw=.7,zorder=0)
        q=ax.quiver(xy[:,0],xy[:,1],v[:,0],v[:,1],color=colors,angles='xy',scale_units='xy',scale=1/gain,width=.004)
        ax.scatter(xy[:,0],xy[:,1],c=colors,s=9)
        ax.quiverkey(q,.78,.94,100,'100 original px',coordinates='axes',labelpos='S')
        ax.set(xlim=(-.2*WIDTH,1.2*WIDTH),ylim=(1.2*HEIGHT,-.2*HEIGHT),aspect='equal',xlabel='x, original px',ylabel='y, original px (down)',title=title+'\ncheck31 RMS='+f"{data['groups']['check31/all_reviewed']['rms_px']:.2f} px")
    fig.suptitle(f'{RID}: predicted minus measured; arrows enlarged {gain:g}x; no new fit')
    fig.legend(handles=[Line2D([0],[0],color='#155b9b',marker='o',label='Fit16 membership (distributed camera)'),Line2D([0],[0],color='#c04c00',marker='o',label='Check31 membership')],loc='outside lower center',ncol=2)
    fig.savefig(out/'vectors.png',dpi=160);fig.savefig(out/'vectors.pdf',metadata={'CreationDate':None,'ModDate':None});plt.close(fig)
    write_json(out/'plot.json',dict(matplotlib=matplotlib.__version__,results_sha256=digest(out/'results.json'),
        files={n:digest(out/n) for n in ['vectors.png','vectors.pdf']}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['prepare','measure','validate','plot']);p.add_argument('--out-dir',type=Path,required=True)
    a=p.parse_args();globals()[a.stage](a.out_dir.resolve())
