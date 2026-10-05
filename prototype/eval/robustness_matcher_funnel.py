#!/usr/bin/env python3
"""Stage counters on frozen development matcher inputs; no acceptance changes."""
import argparse
from collections import Counter
import json
from pathlib import Path
import shutil

from collection_run import ROOT, digest, select_records, write_json
from robustness_candidate_trace import command
from tetra3_internal import prepare as prepare_harness, insert

RID='wm_r_134291495'
POSITIVE='wm_r_112929230'
MANIFEST=ROOT/'data/samples/sky-samples/manifest.json'
PRIOR=ROOT/'docs/experiments'
COUNTERS=('patterns','hash_candidates','fov_pass','ratios_pass','svd_pass','verifications','probability_pass','finalized','nonfinite_probability')


def read(p): return json.loads(p.read_text())


def prior(n): return PRIOR/f'robustness-iteration-{n}.json'


def instrument(text):
    additions=[
        ('        // True pinhole pixel scale (rad/px): ps = 1/f where f = (W/2) / tan(fov/2).',
         '        let mut funnel = crate::research_funnel::Funnel::new(fov_estimate.to_degrees());'),
        ('        let num_pattern_centroids = pattern_centroid_inds.len();','        funnel.retained = num_pattern_centroids;'),
        ('            // Get image pattern vectors','            funnel.patterns += 1;'),
        ('                    let cat_largest = entry.largest_edge;','                    funnel.hash_candidates += 1;'),
        ('                    // Full edge-ratio comparison','                    funnel.fov_pass += 1;'),
        ('                    // ── Estimate rotation via SVD ──','                    funnel.ratios_pass += 1;'),
        ('                    // Determine parity from the rotation determinant.','                    funnel.svd_pass += 1;'),
        ('''                        star_vectors,
                    );''','''                    if funnel.observe(prob_mismatch, match_threshold, current_matches.len()) {
                        funnel.best = Some(serde_json::json!({
                            "probability":prob_mismatch,"threshold":match_threshold,
                            "refined_fov_deg":fov.to_degrees(),"parity_flip":parity_flip,
                            "image_indices":image_pattern_local.map(|i| sorted_indices[i]),
                            "catalog_ids":entry.star_indices.map(|i| self.star_catalog_ids[i as usize]),
                            "pairs":current_matches.iter().map(|&(i,c)| (sorted_indices[i], self.star_catalog_ids[c])).collect::<Vec<_>>()
                        }));
                    }'''),
        ('                    // ── WCS TAN-projection refinement ──','                    funnel.probability_pass += 1;'),
        ('                        return Ok(result);','                        funnel.finalized += 1;')]
    for anchor,addition in additions:
        # The success counter must precede the return; every other hook follows
        # its anchor. Keep branch expressions and floating-point operations intact.
        if anchor=='                        return Ok(result);':
            if text.count(anchor)!=1: raise ValueError('unique success anchor required')
            text=text.replace(anchor,addition+'\n'+anchor)
        else: text=insert(text,anchor,addition)
    return text


def selected():
    rows=select_records(read(MANIFEST),'development',[RID,POSITIVE])
    if {r['id']for r in rows}!={RID,POSITIVE}:raise ValueError('both development records required')


def prepare(out,registry):
    selected()
    if out.exists():raise ValueError('fresh output directory required')
    checks=read(prior('37-checks'))
    for n in ('37-replay-input','37-replay-results'):
        path=prior(n)
        if digest(path)!=checks['artifact_hashes'][str(path.relative_to(ROOT))]:raise ValueError('prior input changed')
    checks36=read(prior('36-checks'))
    for n in ('36-input','36-results'):
        path=prior(n)
        if digest(path)!=checks36['artifact_hashes'][str(path.relative_to(ROOT))]:raise ValueError('positive fixture changed')
    inp=read(prior('37-replay-input'));inp['ids']=[RID,POSITIVE]
    inp['cases'].append(read(prior('36-input'))['cases'][-1])
    prepare_harness(out,registry)
    vendor=out/'harness/tetra3';source=vendor/'src/solver/solve.rs'
    source.write_text(instrument(source.read_text()))
    module=Path(__file__).with_name('tetra3_funnel.rs')
    shutil.copy2(module,vendor/'src/research_funnel.rs')
    with (vendor/'src/lib.rs').open('a')as f:f.write('\npub mod research_funnel;\n')
    caller=out/'harness/src/main.rs';code=caller.read_text()
    anchor='        if !matches!(name, "4080-default-control" | "4080-deep-control" | "positive-control") { continue; }'
    if code.count(anchor)!=1:raise ValueError('caller selection changed')
    caller.write_text(code.replace(anchor,'        // All frozen development cases.')+
        '\n#[cfg(test)]\n#[path = "../tetra3/src/research_funnel.rs"]\nmod funnel_tests;\n')
    write_json(out/'input.json',inp)
    paths=[Path(__file__).resolve(),module,MANIFEST,out/'input.json',
        *(prior(n)for n in ['37-replay-input','37-replay-results','37-checks','36-input','36-results','36-checks']),
        *(Path(__file__).with_name(n)for n in ['tetra3_internal.py','tetra3_internal.rs','tetra3_internal_trace.rs','collection_run.py','robustness_candidate_trace.py']),
        *(Path(p)for p in inp['databases']),
        *(ROOT/'data/samples/sky-samples'/n for n in ['robustness-results.json','collection-wcs-review.json'])]
    paths+=list((out/'harness').rglob('*.rs'))+list((out/'harness').rglob('Cargo.toml'))
    write_json(out/'protocol.json',dict(iteration=38,ids=[RID,POSITIVE],split='development',holdout=False,
        scope='observation-only funnel counters in isolated tetra3; existing 37 original/manual lists plus36 positive control',
        case_names=[c['name']for c in inp['cases']],budget=dict(attempts=432,per_attempt_ms=2500,wall_s=180),
        criteria=['384 target outcomes exactly match iteration37',
            'positive control reproduces previously successful attempts and matched identities; all other outcomes reported including timeout sensitivity',
            'every FOV has one thinning and one funnel event; stage counts and verification histogram balance',
            'no change in thresholds, source lists or budgets; instrumentation timing is not a production benchmark'],
        hashes={str(p.relative_to(ROOT)):digest(p)for p in sorted(set(paths))}))


def validate(out):
    p=read(out/'protocol.json')
    if (p['ids'],p['split'],p['holdout'])!=([RID,POSITIVE],'development',False):raise ValueError('development only')
    selected()
    for name,sha in p['hashes'].items():
        if digest(ROOT/name)!=sha:raise ValueError('frozen input changed: '+name)
    return p


def build(out):
    validate(out)
    cargo=['cargo','--offline','--manifest-path',str(out/'harness/Cargo.toml')]
    command(out,'build',[cargo[0],'build','--release',*cargo[1:]],timeout=600)
    command(out,'rust-tests',[cargo[0],'test','--release',*cargo[1:]],timeout=120)
    shutil.copy2(out/'harness/target/release/starglyph-tetra3-internal-trace',out/'trace')
    write_json(out/'binary.json',dict(sha256=digest(out/'trace'),protocol_sha256=digest(out/'protocol.json')))


def run(out):
    validate(out)
    if read(out/'binary.json')!=dict(sha256=digest(out/'trace'),protocol_sha256=digest(out/'protocol.json')):raise ValueError('binary changed')
    if (out/'run.json').exists():raise ValueError('fresh run required')
    command(out,'run',[str(out/'trace'),str(out/'input.json'),str(out/'raw.json')],timeout=180)


def validate_counts(f):
    if any(not isinstance(f[n],int)or f[n]<0 for n in COUNTERS):raise ValueError('invalid counter')
    if not (f['hash_candidates']>=f['fov_pass']>=f['ratios_pass']>=f['svd_pass']==f['verifications']>=f['probability_pass']>=f['finalized']):
        raise ValueError('stage count mismatch')
    if sum(f['matches_histogram'].values())!=f['verifications']:raise ValueError('histogram count mismatch')
    if f['retained']<4 and any(f[n]for n in COUNTERS):raise ValueError('search after TooFew')


def parse_trace(text):
    attempts=[];pending=None
    for line in text.splitlines():
        if not line.startswith('SGTRACE '):continue
        event=json.loads(line[8:]);stage=event['stage'];data=event['data']
        if stage=='attempt':
            if pending is not None:raise ValueError('unfinished FOV')
            attempts.append(dict(context=data,fovs=[]))
        elif stage=='thinning':
            if not attempts or pending is not None:raise ValueError('unattributed thinning')
            pending=data
        elif stage=='funnel':
            if pending is None or data['fov_deg']!=pending['fov'] or data['retained']!=len(pending['kept_input_indices']):raise ValueError('FOV attribution mismatch')
            validate_counts(data);attempts[-1]['fovs'].append(data);pending=None
    if pending is not None or not attempts or any(not a['fovs']for a in attempts):raise ValueError('incomplete trace')
    return attempts


def aggregate(fovs):
    result={n:sum(f[n]for f in fovs)for n in COUNTERS};hist=Counter()
    for f in fovs:hist.update(f['matches_histogram'])
    best=min((f for f in fovs if f['best_ratio']is not None),key=lambda f:f['best_ratio'],default=None)
    return result|dict(matches_histogram=dict(hist),best=best,sweeps=len(fovs))


def summarize(out):
    p=validate(out);cases=read(out/'raw.json')['cases'];trace=parse_trace((out/'run.log').read_text())
    if [c['name']for c in cases]!=p['case_names']or len(trace)!=432 or any(len(c['attempts'])!=48 for c in cases):raise ValueError('scope changed')
    previous=read(prior('37-replay-results'))['cases'];positive=read(prior('36-results'))['cases'][-1]
    rows=[];i=0
    for c in cases:
        attempts=[]
        for a in c['attempts']:
            t=trace[i];i+=1
            if t['context']!=dict(case=c['name'],db=a['db'],k=a['k'],fov=a['fov']):raise ValueError('attempt attribution changed')
            fovs=t['fovs'];s=aggregate(fovs)
            if s['finalized']!=int('error'not in a['result']):raise ValueError('finalization/result mismatch')
            attempts.append(a|dict(funnel=s))
        old=positive if c['name']=='positive-control'else next(r for r in previous if r['name']==c['name'])
        differences=[dict(db=a['db'],k=a['k'],fov=a['fov'],before=b['result'],after=a['result'])for a,b in zip(attempts,old['attempts'])if a['result']!=b['result']]
        if c['name']!='positive-control'and differences:raise ValueError('target outcomes changed')
        if c['name']=='positive-control'and any('error'not in d['before']for d in differences):raise ValueError('successful positive attempt changed')
        fovs=[f for t in trace[i-48:i]for f in t['fovs']]
        rows.append(dict(name=c['name'],id=POSITIVE if c['name']=='positive-control'else RID,
            statuses=dict(Counter(a['result'].get('error','candidate')for a in attempts)),elapsed_ms=sum(a['elapsed_ms']for a in attempts),
            funnel=aggregate(fovs),prior_differences=differences,attempts=attempts))
    write_json(out/'results.json',dict(iteration=38,ids=[RID,POSITIVE],split='development',holdout=False,
        protocol_sha256=digest(out/'protocol.json'),cases=rows,target_outcomes_reproduced=384,
        elapsed_s=read(out/'run.json')['elapsed_s'],production_changed=False,automatic_improvement_confirmed=False,independent_ground_truth=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['prepare','build','run','validate','summarize']);p.add_argument('--out-dir',type=Path,required=True);p.add_argument('--registry',type=Path)
    a=p.parse_args()
    if a.stage=='prepare':
        if a.registry is None:p.error('--registry required')
        prepare(a.out_dir.resolve(),a.registry)
    else:globals()[a.stage](a.out_dir.resolve())
