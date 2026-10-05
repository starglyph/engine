#!/usr/bin/env python3
"""Fixed-count spatial coverage diagnostic using the saved production LM binary."""
import argparse
import copy
import json
import os
from pathlib import Path
import shutil

import numpy as np

from collection_run import ROOT, digest, write_json
from compare_local_wcs import project, stats
from robustness_candidate_trace import command
from robustness_centroid_origin_report import geometry_groups
from robustness_geometry import selected, preview
from robustness_single22_residuals import prior, read, RID
from robustness_single31_fit import validate as validate_fit28

OLD = ROOT/'prototype/artifacts/robustness-iteration-28'
METHODS = ('external','native_r8','native_r12')
ARMS = ('original22','coverage22')


def selection(singles, probes, width, height):
    """Coordinate-only intervention; residuals/brightness never enter ranking."""
    central = [p for p in probes if width/3 <= p['xy'][0] < width*2/3 and height/3 <= p['xy'][1] < height*2/3]
    central.sort(key=lambda p:(p['xy'][1],p['source_id']))
    if len(central) != 6 or len(singles) != 22:
        raise ValueError('frozen six central probes and22 singles required')
    added = [central[i]['source_id'] for i in (0,(len(central)-1)//2,len(central)-1)]
    cells = {}
    for p in singles:
        x,y = p['review_center']
        cell = (int(y/(height/3)),int(x/(width/3)))
        cells.setdefault(cell,[]).append(p)
    cell = min(cells,key=lambda k:(-len(cells[k]),k))
    center = [(cell[1]+.5)*width/3,(cell[0]+.5)*height/3]
    ranked = sorted(cells[cell],key=lambda p:(sum((a-b)**2 for a,b in zip(p['review_center'],center)),p['id']))
    if len(ranked) < 3:
        raise ValueError('three removable sources required')
    return dict(added_probe_ids=added,removed_pair_ids=[p['id'] for p in ranked[:3]],
                donor_cell=list(cell),donor_count=len(ranked),central_probe_ids=[p['source_id'] for p in central])


def inputs():
    selected([RID])
    review = read(prior('27-results'))
    singles = [p for p in review['points'] if p['label']=='visible_source']
    probes = next(f['sources'] for f in read(prior('13-geometry'))['frames'] if f['id']==RID)
    alternate = {s['source_id']:s for s in next(f['sources'] for f in read(prior('5-geometry'))['frames'] if f['id']==RID)}
    for p in probes:
        a = alternate[p['source_id']]
        if a['hyg_id'] != p['hyg_id'] or a['xy'] != p['xy']:
            raise ValueError('probe centroid identity changed')
    return singles,probes,alternate


def review(out):
    singles,probes,_ = inputs()
    chosen = selection(singles,probes,3648,5472)
    if out.exists():
        raise ValueError('fresh output required')
    out.mkdir(parents=True)
    frame = copy.deepcopy(next(f for f in read(prior('5-review'))['frames'] if f['id']==RID))
    frame['points'] = [p for p in frame['points'] if p['id'] in chosen['added_probe_ids']]
    if digest(ROOT/frame['image']) != frame['source_sha256']:
        raise ValueError('image changed')
    write_json(out/'selection.json',chosen)
    write_json(out/'review-candidates.json',frame)
    preview(frame,out)


def cases(singles, probes, alternate, chosen):
    old = read(prior('28-input'))
    baseline = copy.deepcopy(old['cases'][1])
    baseline.update(name='reproduce_single22',expected=read(prior('28-results'))['cases'][1]['camera'])
    if [(p['id'],p['world'],p['review_center']) for p in singles] != [(m['pair_id'],m['world'],m['xy']) for m in baseline['matches']]:
        raise ValueError('single22 identity changed')
    lookup = {p['source_id']:p for p in probes}
    result = [baseline]
    for method in METHODS:
        orig = [dict(pair_id=p['id'],hyg_id=p['hyg_id'],world=p['world'],xy=p['positions'][method]) for p in singles]
        kept = [m for m in orig if m['pair_id'] not in chosen['removed_pair_ids']]
        added = [dict(probe_id=i,hyg_id=lookup[i]['hyg_id'],world=lookup[i]['world'],
            xy=lookup[i]['xy'] if method=='external' else alternate[i][method+'_xy']) for i in chosen['added_probe_ids']]
        for arm,matches in zip(ARMS,(orig,kept+added)):
            if len(matches)!=22 or len({m['hyg_id'] for m in matches})!=22:
                raise ValueError('22 distinct catalogue identities required')
            result.append(dict(name=arm+'_'+method,initial=baseline['initial'],matches=matches,prior_weight=None,expected=None))
    return dict(id=RID,split='development',cases=result,probe_worlds=[p['world'] for p in probes])


def masks_for(probes, fit_cases, previous):
    xy = np.array([p['xy'] for p in probes])
    # Union includes all centroid variants and the exact detector replay control.
    masks = geometry_groups(xy,[np.array([m['xy'] for m in c['matches']]) for c in fit_cases],3648,5472,12)
    unused = masks.pop('outside_current_inputs')
    used_ids = {m['hyg_id'] for c in fit_cases for m in c['matches']}
    unused &= np.array([p['hyg_id'] not in used_ids for p in probes])
    result = {k: m & unused for k,m in masks.items() if k!='all_reviewed'}
    result['common_unused'] = unused
    result['strict_unused'] = unused & np.array([p['outside_all40_inputs'] for p in previous])
    result['central_unused'] = unused & (xy[:,0]>=1216)&(xy[:,0]<2432)&(xy[:,1]>=1824)&(xy[:,1]<3648)
    return result


def prepare(out):
    if (out/'protocol.json').exists():
        raise ValueError('protocol already frozen')
    validate_fit28(OLD)
    singles,probes,alternate = inputs()
    chosen = selection(singles,probes,3648,5472)
    if chosen != read(out/'selection.json'):
        raise ValueError('selection changed after review')
    visual = read(out/'visual-review.json')
    if visual['id']!=RID or visual['split']!='development' or sorted(p['source_id'] for p in visual['points'])!=sorted(chosen['added_probe_ids']):
        raise ValueError('selected sources need visual review')
    if any(p['label']!='visible_source' or not p['note'] for p in visual['points']):
        raise ValueError('ambiguous source: stop without substitution')
    inp = cases(singles,probes,alternate,chosen)
    previous = read(prior('28-results'))['cases'][0]['sources']
    if [(p['source_id'],p['hyg_id']) for p in probes] != [(p['source_id'],p['hyg_id']) for p in previous]:
        raise ValueError('old scoring order changed')
    masks = masks_for(probes,inp['cases'],previous)
    shutil.copy2(OLD/'pair-test',out/'pair-test')
    write_json(out/'input.json',inp)
    files = [Path(__file__).resolve(),out/'input.json',out/'pair-test',out/'selection.json',out/'visual-review.json',out/'review-candidates.json',
        OLD/'protocol.json',*(prior(n) for n in ['27-results','28-input','28-results','28-checks','13-geometry','5-geometry','5-review']),
        *(Path(__file__).with_name(n) for n in ['pair_refit.rs','robustness_single31_fit.py','robustness_single22_residuals.py',
            'robustness_geometry.py','robustness_centroid_origin_report.py','compare_local_wcs.py','robustness_candidate_trace.py','collection_run.py'])]
    files += [ROOT/'data/samples/sky-samples'/n for n in ['manifest.json','robustness-results.json','collection-wcs-review.json']]
    files.append(ROOT/read(out/'review-candidates.json')['image'])
    write_json(out/'protocol.json',dict(iteration=30,id=RID,split='development',holdout=False,manual_diagnostic=True,
        selection=chosen,case_names=[c['name'] for c in inp['cases']],
        group_probe_ids={k:[p['source_id'] for p,m in zip(probes,v) if m] for k,v in masks.items()},
        criteria=['reproduce saved detector-single22 camera with existing harness tolerances',
            'coverage favourable only if strict common-unused RMS falls>=10percent and no common-unused group RMS grows>0.5px',
            'apply criterion separately to each of3 matched centroid-method fits; all3 must pass',
            'always report every probe; fitted sources excluded from shared assessment, including edge groups',
            'no automatic improvement, solved status, ground truth or causal isolation of coverage from source identity'],
        budget='one existing production5parameter LM per case; same initial, natural k1 prior, no restarts/rematches; process limit60s',
        hashes={str(p.relative_to(ROOT)):digest(p) for p in files}))


def validate(out):
    p = read(out/'protocol.json')
    if (p['id'],p['split'],p['holdout']) != (RID,'development',False):
        raise ValueError('frozen development required')
    selected([RID])
    validate_fit28(OLD)
    for name,sha in p['hashes'].items():
        if digest(ROOT/name)!=sha:
            raise ValueError('frozen input changed: '+name)
    return p


def run(out):
    validate(out)
    if (out/'run.json').exists():
        raise ValueError('fresh run required')
    command(out,'run',[str(out/'pair-test'),'--exact','solve::pair_refit::fixed_pairs','--ignored'],
        {**os.environ,'STARGLYPH_PAIR_INPUT':str(out/'input.json'),'STARGLYPH_PAIR_OUTPUT':str(out/'fits.json')},60)
    validate(out)


def summarize(out):
    protocol = validate(out)
    _,probes,alternate = inputs()
    fit = read(out/'fits.json')
    if fit['id']!=RID or fit['split']!='development' or [c['name'] for c in fit['cases']]!=protocol['case_names']:
        raise ValueError('fit scope changed')
    if not fit['cases'][0]['reproduced_previous_first_fit']:
        raise ValueError('control not reproduced')
    masks = {k:np.array([s['source_id'] in ids for s in probes]) for k,ids in protocol['group_probe_ids'].items()}
    positions = dict(external=np.array([p['xy'] for p in probes]),**{m:np.array([alternate[p['source_id']][m+'_xy'] for p in probes]) for m in METHODS[1:]})
    rows = []
    for case in fit['cases']:
        if case['matches']!=22 or case['prior_scale']!=1.:
            raise ValueError('objective changed')
        cam = case['camera']
        rotated = np.array([p['world'] for p in probes]) @ np.array(cam['world_to_camera']).T
        r2 = np.sum((rotated[:,:2]/rotated[:,2,None])**2,axis=1)
        if np.any(rotated[:,2]<=0) or (cam['k1']<0 and np.any(r2>=-1/(3*cam['k1']))):
            raise ValueError('probe outside monotone camera domain')
        pred = project(cam,[p['radec'] for p in probes])
        errors = {m:np.linalg.norm(pred-pos,axis=1) for m,pos in positions.items()}
        rows.append(dict(name=case['name'],matches=case['matches'],camera=cam,elapsed_ms=case['elapsed_ms'],prior_weight=case['prior_weight'],
            reproduced_control=case['reproduced_previous_first_fit'],fit_rms_px=float(np.sqrt(np.mean(np.sum(np.array(case['residuals'])[:-1].reshape(-1,2)**2,axis=1)))),
            geometry={m:{k:stats(e[v]) if v.any() else dict(count=0) for k,v in masks.items()} for m,e in errors.items()},
            sources=[dict(source_id=p['source_id'],hyg_id=p['hyg_id'],predicted_xy=xy.tolist(),errors_px={m:float(e[i]) for m,e in errors.items()}) for i,(p,xy) in enumerate(zip(probes,pred))]))
    comparisons = {}
    by_name = {r['name']:r for r in rows}
    for method in METHODS:
        b,a = [by_name[arm+'_'+method]['geometry'][method] for arm in ARMS]
        regressions = {k:a[k]['rms_px']-b[k]['rms_px'] for k in masks if b[k]['count'] and a[k]['rms_px']-b[k]['rms_px']>.5}
        ratio = a['strict_unused']['rms_px']/b['strict_unused']['rms_px']
        comparisons[method] = dict(strict_rms_ratio=ratio,regressions=regressions,criterion_passed=ratio<=.9 and not regressions)
    write_json(out/'results.json',dict(iteration=30,id=RID,split='development',holdout=False,manual_diagnostic=True,
        protocol_sha256=digest(out/'protocol.json'),cases=rows,comparisons=comparisons,
        criterion_passed=all(v['criterion_passed'] for v in comparisons.values()),automatic_improvement_confirmed=False,
        independent_ground_truth=False,new_searches=0,new_refinements=len(rows),rematches=0))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['review','prepare','run','summarize','validate'])
    parser.add_argument('--out-dir',type=Path,required=True)
    args=parser.parse_args();globals()[args.stage](args.out_dir.resolve())
