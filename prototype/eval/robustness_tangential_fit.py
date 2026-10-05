#!/usr/bin/env python3
"""Nonlinear validation of frozen iteration34 on both fit22 sets, no rematching."""
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
from robustness_centroid_origin import prepare as prepare_workspace
from robustness_radial_fit import optimizer_source
from robustness_geometry import selected
from robustness_single22_residuals import RID, prior, read
from robustness_tangential_basis import NAMES, geometry, transfer_verdict, tangential_columns


def generated_optimizer(source):
    code=optimizer_source(source)
    replacements=[('    extra: bool,','    extra: bool,\n    fixed_weight: f64,'),
        ('let dim = if extra { 6 }','let dim = if extra { 7 }'),
        ('let k1_weight = k1_reg_weight(initial.fov_x_deg());','let k1_weight = fixed_weight;'),
        ('if extra { params.push(0.); }','if extra { params.extend([0., 0.]); }'),
        ('            if extra { trial[5] = trial[5].clamp(-0.5, 0.5); }\n','')]
    for before,after in replacements:
        if code.count(before)!=1:raise ValueError('optimizer template changed: '+before)
        code=code.replace(before,after)
    return code


def build_cases(frozen, previous, saved):
    if frozen['id']!=RID or frozen['split']!='development' or [c['name'] for c in frozen['cases']]!=NAMES:
        raise ValueError('six development starts required')
    old={c['name']:c for c in previous['cases']};weights={c['name']:c['prior_weight'] for c in saved['cases']};cases=[]
    for c in frozen['cases']:
        name=c['name'];base=old[name]
        if c['fit_pairs']!=base['matches'] or len(c['fit_pairs'])!=22:raise ValueError('fixed22 mismatch')
        weight=weights[name]
        if any(v!=0 for v in c['jacobian'][-1][:4]) or weight<=0 or not np.isclose(weight,c['jacobian'][-1][4],rtol=0,atol=1e-10):raise ValueError('fixed prior required')
        for prefix in ('replay','five','seven'):
            cases.append(dict(name=prefix+'_'+name,initial=copy.deepcopy(base['initial'] if prefix=='replay' else c['camera']),
                matches=copy.deepcopy(c['fit_pairs']),prior_weight=weight,extra=prefix=='seven',
                expected=copy.deepcopy(c['camera']) if prefix=='replay' else None))
    return dict(id=RID,split='development',cases=cases,probe_worlds=[p['world'] for p in frozen['probes']])


def prepare(out):
    selected([RID])
    checks=read(prior('34-checks'))
    for n in ('protocol','input','results'):
        p=prior('34-'+n)
        if digest(p)!=checks['artifact_hashes'][str(p.relative_to(ROOT))]:raise ValueError('iteration34 changed')
    frozen=read(prior('34-input'));inp=build_cases(frozen,read(prior('30-input')),read(prior('30-results')))
    prepare_workspace(out)
    pipeline=out/'pipeline';solve=pipeline/'crates/starglyph-core/src/solve.rs'
    original=ROOT/'prototype/crates/starglyph-core/src/solve.rs'
    solve.write_text(original.read_text()+'\n#[cfg(test)]\nmod tangential_refit;\n')
    (solve.parent/'solve/centroid_origin.rs').unlink()
    rust=Path(__file__).with_name('tangential_refit.rs');shutil.copy2(rust,solve.parent/'solve/tangential_refit.rs')
    generated=solve.parent/'solve/tangential_optimizer.rs';generated.write_text(generated_optimizer(original.read_text()))
    command(out,'rustfmt',['rustfmt','--edition','2021',str(generated)],timeout=30)
    write_json(out/'input.json',inp)
    paths=[Path(__file__).resolve(),rust,out/'input.json',*(prior(n) for n in ('34-protocol','34-input','34-results','34-checks','30-input','30-results')),
        *(Path(__file__).with_name(n) for n in ('robustness_tangential_basis.py','robustness_radial_fit.py','robustness_centroid_origin.py','robustness_candidate_trace.py','collection_run.py','compare_local_wcs.py','robustness_geometry.py','robustness_single22_residuals.py')),
        *(ROOT/p for p in checks['protected_hashes'])]
    for base in [ROOT/'prototype',pipeline]:
        paths.extend([base/'Cargo.toml',base/'Cargo.lock'])
        for name in ['starglyph-core','starglyph-cli','simulator-core']:
            paths.extend((base/'crates'/name/'src').rglob('*.rs'));paths.append(base/'crates'/name/'Cargo.toml')
    write_json(out/'protocol.json',dict(iteration=35,id=RID,split='development',holdout=False,names=NAMES,
        case_names=[c['name'] for c in inp['cases']],
        model='exact iteration34 p1/p2 on undistorted image-oriented x,y; starts p1=p2=0; existing k1 prior only',
        optimizer='generated from production LM through existing radial generator; explicit fixed prior;7 dimensions; no p1/p2 clamp or new prior; existing k1 clamp retained',
        budget=dict(lm_calls=18,replay_controls=6,paired_five_seven=6,max_iterations=30,damping_trials=10,finite_difference='production:1e-4 except focal max(abs(f)*1e-4,1e-3)',wall_limit_s=60,build_limit_s=600),
        guard='geom::project first, including negative-k1 guard; extra conservative positive Jacobian bound along axis-to-source ray; no full-sensor certification',
        criteria=['six replay controls exactly equal saved cameras; five/seven share initial,22 pairs and prior',
            'all saved35 probe and22 fit rays projectable; no k1 bound hit',
            'linear versus nonlinear residual vector discrepancy RMS<=0.1px and max<=0.5px on22 and common19 in all6 cases',
            'same iteration34 descriptive transfer:strict15 RMS decreases>=10percent versus co-start five, no nonempty group RMS regression>0.5px in all6 cases',
            'report every case, source, budget and failure; no automatic improvement or independent GT claim'],
        hashes={str(p.relative_to(ROOT)):digest(p) for p in sorted(set(paths))}))


def validate(out):
    p=read(out/'protocol.json')
    if (p['id'],p['split'],p['holdout'],p['names'])!=(RID,'development',False,NAMES):raise ValueError('frozen development required')
    selected([RID])
    for name,sha in p['hashes'].items():
        if digest(ROOT/name)!=sha:raise ValueError('frozen input changed: '+name)
    return p


def build(out):
    validate(out);workspace=out/'pipeline'
    cargo=['cargo','test','--offline','--locked','--release','--manifest-path',str(workspace/'Cargo.toml'),'-p','starglyph-core',
        '--features','serde_json/float_roundtrip','--lib','--no-run','--message-format','json']
    command(out,'build',cargo,timeout=600)
    artifacts=[json.loads(s) for s in (out/'build.log').read_text().splitlines() if s.startswith('{')]
    binaries=[a['executable'] for a in artifacts if a.get('reason')=='compiler-artifact' and a.get('executable') and a['target']['name']=='starglyph_core']
    if len(binaries)!=1:raise ValueError('unique core test executable required')
    shutil.copy2(binaries[0],out/'tangential-test')
    command(out,'rust-tests',[str(out/'tangential-test')],timeout=60)
    write_json(out/'binary.json',dict(sha256=digest(out/'tangential-test'),protocol_sha256=digest(out/'protocol.json')))
    validate(out)


def run(out):
    validate(out)
    if (out/'run.json').exists():raise ValueError('fresh run required')
    binary=read(out/'binary.json')
    if binary['sha256']!=digest(out/'tangential-test') or binary['protocol_sha256']!=digest(out/'protocol.json'):raise ValueError('binary freeze changed')
    command(out,'run',[str(out/'tangential-test'),'--exact','solve::tangential_refit::fixed_pairs','--ignored'],
        {**os.environ,'STARGLYPH_TANGENTIAL_INPUT':str(out/'input.json'),'STARGLYPH_TANGENTIAL_OUTPUT':str(out/'fits.json')},60)


def projection(camera, worlds):
    world=np.asarray(worlds)
    angles=np.column_stack((np.rad2deg(np.arctan2(world[:,1],world[:,0])),np.rad2deg(np.arcsin(world[:,2]))))
    return project(camera,angles)+(tangential_columns(camera,world)@[camera['p1'],camera['p2']]).reshape(-1,2)


def summarize(out):
    protocol=validate(out);raw=read(out/'fits.json');frozen=read(prior('34-input'));linear=read(prior('34-results'))
    if raw['id']!=RID or raw['split']!='development' or [c['name'] for c in raw['cases']]!=protocol['case_names']:raise ValueError('fit scope changed')
    fitted={c['name']:c for c in raw['cases']};rows=[]
    for old,forecast in zip(frozen['cases'],linear['cases']):
        name=old['name'];replay=fitted['replay_'+name];five=fitted['five_'+name];seven=fitted['seven_'+name]
        if not replay['control_exact']:raise ValueError('replay did not reproduce')
        for c in [replay,five,seven]:
            if c['matches']!=22 or not np.isclose(c['prior_weight'],old['jacobian'][-1][4],rtol=0,atol=1e-10):raise ValueError('objective changed')
            for key,worlds in [('fit_projected_xy',[p['world'] for p in old['fit_pairs']]),('probe_projected_xy',[p['world'] for p in frozen['probes']])]:
                if any(x is None for x in c[key]):raise ValueError('nonprojectable saved ray')
                if not np.allclose(projection(c['camera'],worlds),c[key],rtol=0,atol=1e-8):raise ValueError('Rust/Python projection mismatch')
        original_projection=projection(old['camera']|dict(p1=0.,p2=0.),[p['world'] for p in frozen['probes']])
        positions=original_projection-np.array(old['probe_residuals'])
        residuals={stage:np.array(c['probe_projected_xy'])-positions for stage,c in [('five',five),('seven',seven)]}
        metrics={stage:geometry(v,frozen['source_ids'],frozen['groups']) for stage,v in residuals.items()}
        indices=[frozen['source_ids'].index(i) for i in frozen['groups']['common_unused']]
        linear_probe=np.array([s['seven_xy_px'] for s in forecast['sources']])
        fit_residual=np.array(seven['residuals'])[:-1].reshape(-1,2)
        delta=dict(fit22=fit_residual-np.array(forecast['seven']['fit22_residual_xy_px']),common19=(residuals['seven']-linear_probe)[indices])
        agreement={k:stats(np.linalg.norm(v,axis=1)) for k,v in delta.items()}
        verdict=transfer_verdict(metrics['five'],metrics['seven'])
        no_bound=abs(seven['camera']['k1'])<.5-1e-12
        rows.append(dict(name=name,replay_exact=True,five=five,seven=seven,geometry=metrics,
            control_max_projection_shift_px=float(np.linalg.norm(np.array(five['probe_projected_xy'])-original_projection,axis=1).max()),
            linear_agreement=agreement,linear_agreement_passed=all(v['rms_px']<=.1 and v['max_px']<=.5 for v in agreement.values()),
            transfer=verdict,no_k1_bound_hit=no_bound,descriptive_passed=verdict['passed'] and no_bound,
            sources=[dict(source_id=p['source_id'],hyg_id=p['hyg_id'],in_common19=p['source_id'] in frozen['groups']['common_unused'],
                five_xy_px=residuals['five'][i].tolist(),seven_xy_px=residuals['seven'][i].tolist(),linear_xy_px=linear_probe[i].tolist()) for i,p in enumerate(frozen['probes'])]))
    write_json(out/'results.json',dict(iteration=35,id=RID,split='development',holdout=False,protocol_sha256=digest(out/'protocol.json'),
        cases=rows,linear_agreement_confirmed=all(c['linear_agreement_passed'] for c in rows),descriptive_transfer_passed=all(c['descriptive_passed'] for c in rows),
        nonlinear_fits=18,new_searches=0,production_changed=False,automatic_improvement_confirmed=False,independent_ground_truth=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=('prepare','build','run','summarize','validate'));p.add_argument('--out-dir',type=Path,required=True)
    args=p.parse_args();globals()[args.stage](args.out_dir.resolve())
