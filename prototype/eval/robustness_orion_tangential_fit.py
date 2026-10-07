#!/usr/bin/env python3
"""Nonlinear confirmation of Orion p1/p2 transfer using the existing Rust harness."""
import argparse
import copy
import json
import os
from pathlib import Path
import shutil

import numpy as np

from collection_run import ROOT, digest, write_json
from compare_local_wcs import stats
from robustness_candidate_trace import command
from robustness_centroid_origin import prepare as prepare_workspace
from robustness_orion_tangential import METHODS, WIDTH, HEIGHT, RID, previous, read, selected, geometry, transfer, validate as validate_linear
from robustness_tangential_fit import generated_optimizer, projection

LINEAR=ROOT/'prototype/artifacts/robustness-iteration-45'


def adapted_harness(source):
    for before,after in [('assert_eq!(input.id, "wm_r_132162731");',f'assert_eq!(input.id, "{RID}");'),
        ('assert_eq!(input.cases.len(), 18);','assert_eq!(input.cases.len(), 9);'),
        ('assert_eq!(matches.len(), 22);','assert_eq!(matches.len(), 16);')]:
        if source.count(before)!=1:raise ValueError('harness guard changed')
        source=source.replace(before,after)
    return source


def build_cases(frozen,old):
    if (frozen['id'],frozen['split'],[c['method'] for c in frozen['cases']])!=(RID,'development',list(METHODS)):
        raise ValueError('three frozen development cases required')
    cases=[]
    for c in frozen['cases']:
        original=next(r for r in old['cases'] if r['name']=='replay_'+c['method'])
        if c['fit_pairs']!=original['matches'] or len(c['fit_pairs'])!=16:
            raise ValueError('fixed16 pairs changed')
        for arm in ['replay','five','seven']:
            cases.append(dict(name=arm+'_'+c['method'],initial=copy.deepcopy(original['initial'] if arm=='replay' else c['camera']),
                matches=copy.deepcopy(c['fit_pairs']),prior_weight=c['prior_weight'],extra=arm=='seven',
                expected=copy.deepcopy(c['camera']) if arm=='replay' else None))
    return dict(id=RID,split='development',cases=cases,probe_worlds=old['probe_worlds'])


def prepare(out):
    validate_linear(LINEAR);selected()
    if not read(LINEAR/'results.json')['descriptive_transfer_passed']:
        raise ValueError('predeclared linear transfer prerequisite failed')
    inp=build_cases(read(LINEAR/'input.json'),read(previous('43-input')))
    prepare_workspace(out)
    pipeline=out/'pipeline';target=pipeline/'crates/starglyph-core/src/solve.rs'
    source=ROOT/'prototype/crates/starglyph-core/src/solve.rs'
    target.write_text(source.read_text()+'\n#[cfg(test)]\nmod tangential_refit;\n')
    (target.parent/'solve/centroid_origin.rs').unlink()
    module=target.parent/'solve/tangential_refit.rs'
    module.write_text(adapted_harness(Path(__file__).with_name('tangential_refit.rs').read_text()))
    generated=module.with_name('tangential_optimizer.rs');generated.write_text(generated_optimizer(source.read_text()))
    command(out,'format-generated',['rustfmt','--edition','2021',str(generated)],timeout=30)
    write_json(out/'input.json',inp)
    paths=[Path(__file__).resolve(),out/'input.json',previous('43-input'),
        *(LINEAR/(n+'.json') for n in ['protocol','input','results']),
        *(Path(__file__).with_name(n) for n in ['tangential_refit.rs','robustness_tangential_fit.py','robustness_radial_fit.py',
          'robustness_orion_tangential.py','robustness_tangential_basis.py','robustness_centroid_origin.py','robustness_candidate_trace.py','collection_run.py','compare_local_wcs.py'])]
    for base in (ROOT/'prototype',pipeline):
        paths.extend([base/'Cargo.toml',base/'Cargo.lock'])
        for name in ['starglyph-core','starglyph-cli','simulator-core']:
            paths.append(base/'crates'/name/'Cargo.toml');paths.extend((base/'crates'/name/'src').rglob('*.rs'))
    write_json(out/'protocol.json',dict(iteration=45,phase='nonlinear',id=RID,split='development',holdout=False,
        case_names=[c['name'] for c in inp['cases']],
        model='existing guarded p1/p2 model35; unchanged generated LM; p1/p2 start0; only saved k1 prior; no added bounds or prior',
        budget=dict(lm_calls=9,max_iterations=30,damping_trials=10,wall_s=60,build_s=600),
        criteria=['three exact original replay cameras; five/seven share start,16 pairs and exact prior weight',
            'all47 probes and16 fit rays projectable under existing ray guard; no k1 bound hit',
            'check31 RMS falls>=10percent and no nonempty frozen check group regresses>0.5 original px vs co-start five, in all3 methods',
            'test forecast separately: nonlinear minus linear residual-vector RMS<=0.1px and max<=0.5px on fit16 and check31',
            'report all results and actual LM work; transfer and linear accuracy are separate verdicts; no automatic improvement or GT'],
        hashes={str(p.relative_to(ROOT)):digest(p) for p in paths}))


def validate(out):
    p=read(out/'protocol.json');validate_linear(LINEAR)
    if (p['id'],p['split'],p['holdout'],p['phase'])!=(RID,'development',False,'nonlinear'):
        raise ValueError('frozen development nonlinear phase required')
    for name,sha in p['hashes'].items():
        if digest(ROOT/name)!=sha:raise ValueError('frozen input changed: '+name)
    return p


def build(out):
    validate(out);workspace=out/'pipeline'
    command(out,'build',['cargo','test','--offline','--locked','--release','--manifest-path',str(workspace/'Cargo.toml'),
        '-p','starglyph-core','--features','serde_json/float_roundtrip','--lib','--no-run','--message-format','json'],
        {**os.environ,'CARGO_TARGET_DIR':str(workspace/'target')},600)
    artifacts=[json.loads(s) for s in (out/'build.log').read_text().splitlines() if s.startswith('{')]
    binaries=[a['executable'] for a in artifacts if a.get('reason')=='compiler-artifact' and a.get('executable') and a['target']['name']=='starglyph_core']
    if len(binaries)!=1:raise ValueError('unique core executable required')
    shutil.copy2(binaries[0],out/'tangential-test')
    command(out,'rust-tests',[str(out/'tangential-test')],timeout=120)
    write_json(out/'binary.json',dict(sha256=digest(out/'tangential-test'),protocol_sha256=digest(out/'protocol.json')))
    validate(out)


def run(out):
    validate(out)
    if (out/'run.json').exists():raise ValueError('fresh run required')
    b=read(out/'binary.json')
    if b['sha256']!=digest(out/'tangential-test') or b['protocol_sha256']!=digest(out/'protocol.json'):
        raise ValueError('binary freeze changed')
    command(out,'run',[str(out/'tangential-test'),'--exact','solve::tangential_refit::fixed_pairs','--ignored'],
        {**os.environ,'STARGLYPH_TANGENTIAL_INPUT':str(out/'input.json'),'STARGLYPH_TANGENTIAL_OUTPUT':str(out/'fits.json')},60)
    validate(out)


def summarize(out):
    p=validate(out);raw=read(out/'fits.json');frozen=read(LINEAR/'input.json');forecast=read(LINEAR/'results.json');inp=read(out/'input.json')
    if (raw['id'],raw['split'],[c['name'] for c in raw['cases']])!=(RID,'development',p['case_names']):
        raise ValueError('fit scope changed')
    rows=[]
    for c,prediction in zip(frozen['cases'],forecast['cases']):
        method=c['method'];cases={arm:next(r for r in raw['cases'] if r['name']==arm+'_'+method) for arm in ['replay','five','seven']}
        if not cases['replay']['control_exact']:raise ValueError('replay failed')
        for fit in cases.values():
            if fit['matches']!=16 or fit['prior_weight']!=c['prior_weight']:raise ValueError('objective changed')
            for key,worlds in [('fit_projected_xy',[m['world'] for m in c['fit_pairs']]),('probe_projected_xy',inp['probe_worlds'])]:
                if any(v is None for v in fit[key]):raise ValueError('nonprojectable ray')
                if not np.allclose(projection(fit['camera'],worlds),fit[key],rtol=0,atol=1e-8):
                    raise ValueError('Rust/Python projection mismatch')
        scale=np.array([WIDTH/c['camera']['width'],HEIGHT/c['camera']['height']])
        original=projection(c['camera']|dict(p1=0.,p2=0.),inp['probe_worlds'])
        observed=original-np.array(c['probe_residuals'])
        residuals={arm:(np.array(fit['probe_projected_xy'])-observed)*scale for arm,fit in cases.items()}
        metrics={arm:geometry(r,frozen['source_ids'],frozen['groups']) for arm,r in residuals.items()}
        diff=residuals['seven']-np.array(prediction['residual_xy_original_px']['seven'])
        agreement={group:stats(np.linalg.norm(diff[[frozen['source_ids'].index(i) for i in frozen['selection'][key]]],axis=1)) for group,key in [('fit16','fit_ids'),('check31','check_ids')]}
        verdict=transfer(metrics['five'],metrics['seven'])
        no_bound=abs(cases['seven']['camera']['k1'])<.5-1e-12
        rows.append(dict(method=method,cases=cases,geometry=metrics,residual_xy_original_px={k:r.tolist() for k,r in residuals.items()},
            transfer=verdict,no_k1_bound_hit=no_bound,descriptive_passed=verdict['passed'] and no_bound,
            linear_agreement=agreement,linear_agreement_passed=all(v['rms_px']<=.1 and v['max_px']<=.5 for v in agreement.values())))
    write_json(out/'results.json',dict(iteration=45,phase='nonlinear',id=RID,split='development',holdout=False,
        protocol_sha256=digest(out/'protocol.json'),selection=frozen['selection'],sources=frozen['sources'],group_ids=frozen['groups'],cases=rows,
        descriptive_transfer_passed=all(r['descriptive_passed'] for r in rows),linear_agreement_confirmed=all(r['linear_agreement_passed'] for r in rows),
        nonlinear_fits=9,new_solver_calls=0,new_wcs_calls=0,run=read(out/'run.json'),
        production_changed=False,independent_ground_truth=False,automatic_improvement_confirmed=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['prepare','build','run','summarize','validate']);p.add_argument('--out-dir',type=Path,required=True)
    a=p.parse_args();globals()[a.stage](a.out_dir.resolve())
