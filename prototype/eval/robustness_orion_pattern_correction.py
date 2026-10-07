#!/usr/bin/env python3
"""Frozen centroid/model diagnostic; no fitting or solving, development only."""
import argparse
import math
from pathlib import Path
import shutil

import numpy as np

from collection_run import ROOT, digest, write_json
from robustness_candidate_trace import command
from robustness_orion_fit import METHODS, WIDTH, HEIGHT, RID, previous, read, working_xy
from robustness_reference_audit import selected
from robustness_tangential_fit import projection
from tetra3_internal import prepare as prepare_harness


def probes(sources, candidates, selection):
    """Three disjoint quartets: next brightest check31 star in each quadrant."""
    fit, check = set(selection['fit_ids']), set(selection['check_ids'])
    if len(fit)!=16 or len(check)!=31 or fit&check or fit|check!={s['id'] for s in sources}:
        raise ValueError('unchanged disjoint16/31 required')
    mags={p['id']:p['mag'] for p in candidates}
    bins={q:[] for q in range(4)}
    for s in sorted(sources,key=lambda s:(mags[s['id']],s['hyg_id'])):
        if s['id'] in check:
            x,y=s['coordinates']['external_centroid']
            bins[2*int(y>=HEIGHT/2)+int(x>=WIDTH/2)].append(s)
    if min(map(len,bins.values()))<3:
        raise ValueError('three preselected check stars per quadrant required')
    return [dict(name=f'check_quartet_{i+1}',sources=[bins[q][i] for q in range(4)]) for i in range(3)]


def make_input():
    sources=read(previous('40-results'))['sources']
    old=read(previous('49-input'));trace=read(previous('49-results'))
    saved=read(previous('45-nonlinear-results'))
    if any((x['id'],x['split'],x['holdout'])!=(RID,'development',False) for x in (old,trace,saved)):
        raise ValueError('development only')
    lookup={s['hyg_id']:s for s in sources}
    quartets=[dict(name=t['name'],sources=[lookup[i] for i in t['catalog_ids']],detection_indices=t['detection_indices']) for t in old['targets']]
    quartets+=probes(sources,read(previous('40-candidates'))['frames'][0]['points'],saved['selection'])
    base=next(c for c in saved['cases'] if c['method']=='external_centroid')['cases']
    cameras={k:base[k]['camera'] for k in ('five','seven')}
    none=cameras['five']|dict(k1=0.,p1=0.,p2=0.)
    ra,dec=np.deg2rad([s['radec'] for s in sources]).T
    worlds=np.column_stack((np.cos(dec)*np.cos(ra),np.cos(dec)*np.sin(ra),np.sin(dec)))
    arms=[]
    for name,camera,offset in [('raw49',none,0.),('same_origin',none,.5),('five',cameras['five'],.5),('seven',cameras['seven'],.5)]:
        observed=projection(camera,worlds)-[800,521]
        ideal=projection(camera|dict(k1=0.,p1=0.,p2=0.),worlds)-[800,521]
        arms.append(dict(name=name,camera=camera,origin_offset=offset,
            projection_checks=[dict(ideal=a.tolist(),observed=b.tolist()) for a,b in zip(ideal,observed)]))
    fovs=sorted({w['image']['fov'] for w in trace['windows']})
    radians=[float(np.float32(np.deg2rad(np.float32(f)))) for f in fovs]
    fixed={}
    for name,c in cameras.items():
        fixed[name]=len(radians);radians.append(float(np.float32(2*math.atan(800/c['focal_px']))))
    cases=[]
    for quartet in quartets:
        s=quartet['sources']
        methods=list(METHODS)+(['detector'] if 'detection_indices' in quartet else [])
        for method in methods:
            xy=([[old['cases'][0]['search'][i][v] for v in ('x','y')] for i in quartet['detection_indices']]
                if method=='detector' else working_xy([a['coordinates'][method] for a in s],none))
            cases.append(dict(name=quartet['name']+'/'+method,quartet=quartet['name'],method=method,
                source_ids=[a['id'] for a in s],catalog_ids=[a['hyg_id'] for a in s],xy=xy,
                fit_members=[a['id'] for a in s if a['id'] in saved['selection']['fit_ids']]))
    return dict(id=RID,split='development',holdout=False,database=old['databases'][3],arms=arms,
        fovs_rad=radians,visited_fovs_deg=fovs,fixed_indices=fixed,cases=cases,selection=saved['selection'])


def prepare(out,registry):
    selected()
    if out.exists():raise ValueError('fresh output required')
    names=['40-results','40-candidates','45-nonlinear-results','49-input','49-results','49-protocol']
    paths=[]
    for n in names:
        path=previous(n);checks=previous(n[:2]+'-checks')
        if digest(path)!=read(checks)['artifact_hashes'][str(path.relative_to(ROOT))]:
            raise ValueError('published evidence changed: '+n)
        paths.extend([path,checks])
    protected=read(previous('49-checks'))['protected_hashes']
    for name,sha in protected.items():
        if digest(ROOT/name)!=sha:raise ValueError('protected input changed')
        paths.append(ROOT/name)
    inp=make_input();db=Path(inp['database'])
    if digest(db)!=read(previous('49-protocol'))['hashes'][str(db.relative_to(ROOT))]:
        raise ValueError('database changed')
    prepare_harness(out,registry)
    project=out/'harness';vendor=project/'tetra3'
    rust=Path(__file__).with_name('tetra3_pattern_correction.rs')
    shutil.copy2(rust,vendor/'src/research_pattern_correction.rs')
    with (vendor/'src/lib.rs').open('a') as f:f.write('\npub mod research_pattern_correction;\n')
    (project/'src/main.rs').write_text('''fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    let input = serde_json::from_slice(&std::fs::read(args.get(1).ok_or("input")?)?)?;
    let result = tetra3::research_pattern_correction::measure(&input)?;
    std::fs::write(args.get(2).ok_or("output")?, serde_json::to_vec(&result)?)?;
    Ok(())
}
''')
    manifest=project/'Cargo.toml'
    manifest.write_text(manifest.read_text().replace('serde_json = "=1.0.149"','serde_json = { version = "=1.0.149", features = ["float_roundtrip"] }'))
    command(out,'resolve',['cargo','update','--offline','--manifest-path',str(manifest)],timeout=60)
    write_json(out/'input.json',inp)
    paths += [db,Path(__file__).resolve(),rust,out/'input.json',ROOT/'prototype/Cargo.lock',
        *(project.rglob('*.rs')),*(project.rglob('Cargo.toml')),project/'Cargo.lock',
        *(Path(__file__).with_name(n) for n in ['collection_run.py','robustness_candidate_trace.py','robustness_orion_fit.py',
         'robustness_reference_audit.py','robustness_tangential_fit.py','robustness_tangential_basis.py',
         'compare_local_wcs.py','tetra3_internal.py','tetra3_internal.rs','tetra3_internal_trace.rs'])]
    write_json(out/'protocol.json',dict(iteration=50,id=RID,split='development',holdout=False,
        hypothesis='saved Brown-Conrady correction changes failed distributed ratios substantially more than measured centroid choice',
        budget=dict(new_fits=0,new_solver_calls=0,new_wcs_calls=0,wall_s=60),
        selection='original49 two quartets plus3 disjoint check31 quartets, next brightest star per fixed quadrant; no residual selection',
        arms='raw49 center; same core center without correction; saved45 external five/seven cameras frozen across all centroid methods',
        criteria=['f32 replay49 ratio error<=1e-6 across all saved windows; catalog ratios exact',
            'existing Brown-Conrady forward/inverse matches saved projection on47 rays within1e-8 working px; f32 roundtrip<=0.001px',
            'centroid explanation unsupported if original distributed remains rejected at every visited FOV with all3 measured methods',
            'correction transfer only if distributed plus all3 unused quartets pass0.006 at fixed saved-seven focal for all3 methods; report every regression',
            'same253 FOV evaluations per case/arm; two appended camera FOVs fixed before measurement, never selected from residuals',
            'necessary edge-ratio condition only, not actual pattern lookup, verification, solve or independent GT'],
        protected_hashes=protected,hashes={str(p.relative_to(ROOT)):digest(p) for p in sorted(set(paths))}))


def validate(out):
    p=read(out/'protocol.json')
    if (p['id'],p['split'],p['holdout'])!=(RID,'development',False):raise ValueError('development only')
    selected()
    for name,sha in p['hashes'].items():
        if digest(ROOT/name)!=sha:raise ValueError('frozen input changed: '+name)
    return p


def build(out):
    validate(out)
    command(out,'build',['cargo','build','--release','--offline','--locked','--manifest-path',str(out/'harness/Cargo.toml')],timeout=600)
    shutil.copy2(out/'harness/target/release/starglyph-tetra3-internal-trace',out/'measure')
    write_json(out/'binary.json',dict(sha256=digest(out/'measure'),protocol_sha256=digest(out/'protocol.json')))


def summarize(inp,raw,prior):
    if [c['name'] for c in raw['cases']]!=[c['name'] for c in inp['cases']]:raise ValueError('case membership changed')
    if any('error' in c for c in raw['cases']):raise ValueError('selected quartet unavailable in catalog; no replacement')
    if any(c['forward_error_px']>1e-8 or c['inverse_error_px']>1e-8 for c in raw['projection_checks']):raise ValueError('model conversion mismatch')
    max_error=0.
    for w in prior['windows']:
        c=next(c for c in raw['cases'] if c['name']==w['target']+'/detector')
        expected=next(i for i in prior['inventory'][3]['targets'] if set(i['catalog_ids'])==set(next(t for t in prior['targets'] if t['name']==w['target'])['catalog_ids']))['metrics']['ratios']
        if c['catalog_ratios']!=expected:raise ValueError('catalog geometry mismatch')
        idx=inp['visited_fovs_deg'].index(w['image']['fov'])
        actual=c['arms'][0]['curves'][idx]['ratios']
        max_error=max(max_error,max(abs(a-b) for a,b in zip(actual,w['image']['ratios'])))
    if max_error>1e-6:raise ValueError('f32 replay failed')
    n=len(inp['visited_fovs_deg']);fixed=inp['fixed_indices'];rows=[]
    for spec,c in zip(inp['cases'],raw['cases']):
        arms={}
        for arm in c['arms']:
            if arm['roundtrip_after_f32_px']>.001:raise ValueError('inverse/rounding mismatch')
            curves=arm['curves'];best=min(range(n),key=lambda i:curves[i]['max_abs_delta'])
            arms[arm['name']]=dict(visited_pass_count=sum(r['ratios_pass'] for r in curves[:n]),
                visited_min_delta=curves[best]['max_abs_delta'],visited_min_fov=inp['visited_fovs_deg'][best],
                fixed={k:curves[i] for k,i in fixed.items()},roundtrip_after_f32_px=arm['roundtrip_after_f32_px'])
        rows.append({k:spec[k] for k in ['name','quartet','method','source_ids','catalog_ids','fit_members']}|dict(arms=arms))
    transfer=all(r['arms']['seven']['fixed']['seven']['ratios_pass'] for r in rows if r['method'] in METHODS and r['quartet']!='local_control')
    centroid_failure=all(r['arms']['raw49']['visited_pass_count']==0 for r in rows if r['quartet']=='distributed' and r['method'] in METHODS)
    regressions=[r['name'] for r in rows if r['arms']['same_origin']['fixed']['seven']['ratios_pass'] and not r['arms']['seven']['fixed']['seven']['ratios_pass']]
    return dict(cases=rows,projection_checks=raw['projection_checks'],replay49_max_ratio_error=max_error,
        centroid_only_still_rejected=centroid_failure,correction_transfer_passed=transfer,fixed_fov_pass_regressions=regressions)


def run(out):
    p=validate(out)
    if (out/'run.json').exists():raise ValueError('fresh measurement required')
    if read(out/'binary.json')!=dict(sha256=digest(out/'measure'),protocol_sha256=digest(out/'protocol.json')):raise ValueError('binary changed')
    command(out,'run',[str(out/'measure'),str(out/'input.json'),str(out/'raw.json')],timeout=p['budget']['wall_s'])
    report=summarize(read(out/'input.json'),read(out/'raw.json'),read(previous('49-results')))
    write_json(out/'results.json',dict(iteration=50,id=RID,split='development',holdout=False,
        protocol_sha256=digest(out/'protocol.json'),**report,run=read(out/'run.json'),
        new_fits=0,new_solver_calls=0,new_wcs_calls=0,production_changed=False,
        independent_ground_truth=False,automatic_improvement_confirmed=False))
    validate(out)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['prepare','build','run','validate']);p.add_argument('--out-dir',type=Path,required=True);p.add_argument('--registry',type=Path)
    a=p.parse_args()
    if a.stage=='prepare':
        if a.registry is None:p.error('--registry required')
        prepare(a.out_dir.resolve(),a.registry)
    else:globals()[a.stage](a.out_dir.resolve())
