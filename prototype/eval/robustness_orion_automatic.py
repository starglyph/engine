#!/usr/bin/env python3
"""Apply saved p1/p2 harness to frozen automatic initial16 Orion pairs."""
import argparse
import copy
import os
from pathlib import Path
import shutil

import numpy as np

from collection_run import ROOT, digest, write_json
from compare_local_wcs import stats
from robustness_candidate_trace import command
from robustness_centroid_origin_report import geometry_groups
from robustness_orion_fit import METHODS, WIDTH, HEIGHT, RID, previous, read
from robustness_reference_audit import selected
from robustness_refinement_report import lift
from robustness_tangential_fit import projection

OLD=ROOT/'prototype/artifacts/robustness-iteration-45/nonlinear'


def assessment_groups(sources,pairs,legacy):
    """Exclude used identities OR nearby physical centers, across all centroids."""
    ids=[s['id'] for s in sources]
    if len(ids)!=47 or len(set(ids))!=47 or len(pairs)!=16:
        raise ValueError('frozen47 review and16 automatic pairs required')
    used={p['hyg_id'] for p in pairs};det=np.array([p['xy'] for p in pairs])
    outside=np.array([s['hyg_id'] not in used for s in sources])
    for method in METHODS:
        xy=np.array([s['coordinates'][method] for s in sources])
        outside &= geometry_groups(xy,[det],WIDTH,HEIGHT,12)['outside_current_inputs']
    if int(sum(outside))<8:raise ValueError('at least8 unused probes required')
    groups={k:v for k,v in legacy.items() if k.startswith('all47/')}
    if groups['all47/all_reviewed']!=ids:raise ValueError('review order changed')
    for prefix,mask in [('outside_auto',outside),('overlap_auto',~outside)]:
        for k,members in list(groups.items()):
            if k.startswith('all47/'):
                groups[prefix+'/'+k.split('/',1)[1]]=[s['id'] for s,keep in zip(sources,mask) if keep and s['id'] in members]
    return groups


def build_input(trace,old_input,old_results):
    if (old_input['id'],old_input['split'],old_results['id'],old_results['split'])!=(RID,'development',RID,'development'):
        raise ValueError('frozen development required')
    step=trace['steps'][0]
    if step['stage']!='first_lm' or len(step['fit_matches'])!=16 or step['fit_matches']!=trace['input']['verification']['matches']:
        raise ValueError('automatic initial16 changed')
    cases=[]
    # Saved binary requires9 cases:6 exact calibration controls, then3 automatic cases.
    for method in METHODS:
        old=next(c for c in old_results['cases'] if c['method']==method)
        for arm in ['five','seven']:
            c=copy.deepcopy(next(c for c in old_input['cases'] if c['name']==arm+'_'+method))
            c.update(name='calibration_'+c['name'],expected=old['cases'][arm]['camera'])
            cases.append(c)
    weight=cases[0]['prior_weight']
    if any(c['prior_weight']!=weight for c in cases):raise ValueError('shared saved prior required')
    for arm in ['replay','five','seven']:
        cases.append(dict(name='automatic_'+arm,initial=trace['input']['camera'] if arm=='replay' else step['camera'],
            matches=step['fit_matches'],prior_weight=weight,extra=arm=='seven',expected=step['camera'] if arm=='replay' else None))
    return dict(id=RID,split='development',cases=cases,probe_worlds=old_input['probe_worlds'])


def prepare(out):
    selected()
    paths=[previous(n) for n in ['40-results','40-checks','41-trace','41-results','41-checks','44-input','44-checks',
        '45-nonlinear-input','45-nonlinear-results','45-nonlinear-binary','45-checks']]
    for name in ['40-results','41-trace','41-results','44-input','45-nonlinear-input','45-nonlinear-results','45-nonlinear-binary']:
        path=previous(name)
        if digest(path)!=read(previous(name[:2]+'-checks'))['artifact_hashes'][str(path.relative_to(ROOT))]:
            raise ValueError('published input changed: '+name)
    for name,sha in read(previous('45-checks'))['protected_hashes'].items():
        if digest(ROOT/name)!=sha:raise ValueError('protected data changed')
        paths.append(ROOT/name)
    binary=read(previous('45-nonlinear-binary'))
    if digest(OLD/'tangential-test')!=binary['sha256']:raise ValueError('saved binary changed')
    trace=read(previous('41-trace'));bound=read(previous('41-results'))['stages'][0]['pairs']
    for m,p in zip(trace['steps'][0]['fit_matches'],bound):
        if not np.allclose(lift(m['xy'],trace['input']['camera'],WIDTH,HEIGHT),p['xy'],rtol=0,atol=1e-8):
            raise ValueError('catalog-bound pair coordinates changed')
    inp=build_input(trace,read(previous('45-nonlinear-input')),read(previous('45-nonlinear-results')))
    groups=assessment_groups(read(previous('40-results'))['sources'],bound,read(previous('44-input'))['groups'])
    if out.exists():raise ValueError('fresh output required')
    out.mkdir(parents=True);shutil.copy2(OLD/'tangential-test',out/'tangential-test')
    write_json(out/'input.json',inp);write_json(out/'assessment.json',dict(groups=groups,automatic_pairs=bound))
    paths += [Path(__file__).resolve(),out/'input.json',out/'assessment.json',out/'tangential-test',
        *(Path(__file__).with_name(n) for n in ['robustness_centroid_origin_report.py','robustness_orion_fit.py',
          'robustness_reference_audit.py','robustness_refinement_report.py','robustness_tangential_fit.py',
          'robustness_tangential_basis.py','robustness_candidate_trace.py','compare_local_wcs.py','collection_run.py'])]
    write_json(out/'protocol.json',dict(iteration=46,id=RID,split='development',holdout=False,
        case_names=[c['name'] for c in inp['cases']],
        model='unchanged existing guarded p1/p2 model; saved binary45; fixed prior3.5993484637547715; startp1=p2=0',
        selection='initial16 automatic verification pairs from41, unchanged identities/coordinates/order; no manual substitution or rematch',
        assessment='all47 plus outside_auto: exclude HYG identity OR center within12 original px of any input pair using union of3 centroid methods; fixed legacy spatial groups',
        budget=dict(total_lm_calls=9,calibration_calls=6,automatic_calls=3,max_iterations=30,damping_trials=10,wall_s=60),
        criteria=['six manual calibration cameras includingp1/p2 reproduce45 exactly; automatic replay camera reproduces41 exactly',
            'automatic five/seven share initial16, start and explicit prior; all rays projectable; no k1 bound hit',
            'outside_auto RMS falls>=10percent vs co-start five and no nonempty all47 or outside_auto group RMS grows>0.5 original px, for each centroid assessment',
            'record all47 and sparse/empty groups, actual fit budget, failures and timeouts',
            'fixed automatic-pair fit only: no full-pipeline solve, matching/acceptance change, independent GT or automatic improvement claim'],
        hashes={str(p.relative_to(ROOT)):digest(p) for p in paths}))


def validate(out):
    p=read(out/'protocol.json')
    if (p['id'],p['split'],p['holdout'])!=(RID,'development',False):raise ValueError('development only')
    selected()
    for name,sha in p['hashes'].items():
        if digest(ROOT/name)!=sha:raise ValueError('frozen input changed: '+name)
    return p


def run(out):
    validate(out)
    if (out/'run.json').exists():raise ValueError('fresh run required')
    command(out,'run',[str(out/'tangential-test'),'--exact','solve::tangential_refit::fixed_pairs','--ignored'],
        {**os.environ,'STARGLYPH_TANGENTIAL_INPUT':str(out/'input.json'),'STARGLYPH_TANGENTIAL_OUTPUT':str(out/'fits.json')},60)
    validate(out)


def transfer(before,after):
    if set(before)!=set(after) or any(v['count']!=after[k]['count'] for k,v in before.items()):raise ValueError('group scope changed')
    ratio=after['outside_auto/all_reviewed']['rms_px']/before['outside_auto/all_reviewed']['rms_px']
    regressions={k:after[k]['rms_px']-v['rms_px'] for k,v in before.items()
        if k.startswith(('all47/','outside_auto/')) and v['count'] and after[k]['rms_px']-v['rms_px']>.5}
    return dict(outside_rms_ratio=ratio,regressions=regressions,passed=ratio<=.9 and not regressions)


def summarize(out):
    p=validate(out);raw=read(out/'fits.json');inp=read(out/'input.json');assessment=read(out/'assessment.json')
    if (raw['id'],raw['split'],[c['name'] for c in raw['cases']])!=(RID,'development',p['case_names']):raise ValueError('fit scope changed')
    old=read(previous('45-nonlinear-results'));checks={}
    for method in METHODS:
        saved=next(c for c in old['cases'] if c['method']==method)
        for arm in ['five','seven']:
            name='calibration_'+arm+'_'+method;c=next(c for c in raw['cases'] if c['name']==name)
            checks[name]=c['control_exact'] and c['camera']==saved['cases'][arm]['camera']
    if not all(checks.values()):raise ValueError('calibration not exact')
    sources=read(previous('40-results'))['sources'];ids=[s['id'] for s in sources];rows=[]
    for c,original in zip(raw['cases'][6:],inp['cases'][6:]):
        if c['matches']!=16 or c['prior_weight']!=original['prior_weight']:raise ValueError('objective changed')
        if c['name']=='automatic_replay' and not c['control_exact']:raise ValueError('automatic replay failed')
        for key,worlds in [('fit_projected_xy',[m['world'] for m in original['matches']]),('probe_projected_xy',inp['probe_worlds'])]:
            if any(v is None for v in c[key]):raise ValueError('nonprojectable ray')
            if not np.allclose(projection(c['camera'],worlds),c[key],rtol=0,atol=1e-8):raise ValueError('Rust/Python projection mismatch')
        pred=lift(c['probe_projected_xy'],c['camera'],WIDTH,HEIGHT)
        metrics={}
        for method in METHODS:
            residual=pred-[s['coordinates'][method] for s in sources];errors=np.linalg.norm(residual,axis=1)
            metrics[method]=dict(residual_xy_px=residual.tolist(),groups={g:stats(errors[[ids.index(i) for i in members]]) if members else dict(count=0) for g,members in assessment['groups'].items()})
        rows.append(dict(**c,geometry=metrics,predicted_xy_original=pred.tolist(),
            fit_rms_working_px=float(np.sqrt(np.mean(np.sum(np.array(c['residuals'])[:-1].reshape(-1,2)**2,axis=1))))))
    verdicts={m:transfer(rows[1]['geometry'][m]['groups'],rows[2]['geometry'][m]['groups']) for m in METHODS}
    no_bound=abs(rows[2]['camera']['k1'])<.5-1e-12
    write_json(out/'results.json',dict(iteration=46,id=RID,split='development',holdout=False,
        protocol_sha256=digest(out/'protocol.json'),calibration_exact=checks,cases=rows,transfer=verdicts,no_k1_bound_hit=no_bound,
        descriptive_transfer_passed=no_bound and all(v['passed'] for v in verdicts.values()),
        sources=[dict(id=s['id'],hyg_id=s['hyg_id'],name=s['name']) for s in sources],group_ids=assessment['groups'],
        run=read(out/'run.json'),nonlinear_fits=9,new_solver_calls=0,new_wcs_calls=0,
        production_changed=False,independent_ground_truth=False,automatic_improvement_confirmed=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['prepare','run','summarize','validate']);p.add_argument('--out-dir',type=Path,required=True)
    a=p.parse_args();globals()[a.stage](a.out_dir.resolve())
