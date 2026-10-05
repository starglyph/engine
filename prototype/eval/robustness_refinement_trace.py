#!/usr/bin/env python3
"""One frozen development regression: observe then cross initial pose/matches."""
import argparse
import json
import os
from pathlib import Path
import shutil

from collection_run import ROOT, digest, write_json
from robustness_candidate_trace import command, HOOK_BEFORE, HOOK_AFTER, traces
from robustness_centroid_origin import prepare as prepare_workspace, records, MANIFEST
from robustness_huber_development import patch_source

FRAME = 'wm_r_132162731'
ARMS = ('control', 'huber')
START = '    let mut refined = refine_pose(&chosen.pose, &chosen.verify.matches);'
END = '    if (width, height) != (original.width, original.height) {'


def extract(source):
    if source.count(START) != 1 or source.count(END) != 1:
        raise ValueError('refinement boundary changed')
    start, end = source.index(START), source.index(END)
    return source[start:end]


def module(source):
    block = extract(source).replace('&chosen.pose', 'initial').replace('&chosen.verify.matches', 'initial_matches')
    block = block.replace('&verify, &detections', 'verify, detections')
    block = block.replace('let mut final_match =', 'let final_match =')
    first = '    let mut refined = refine_pose(initial, initial_matches);'
    block = block.replace(first, first+'\n    steps.push(json!({"stage":"first_lm","camera":camera(&refined),"fit_matches":matches(initial_matches)}));')
    after = '                refined = refine_pose(&refined, &rematch.matches);\n            }'
    if block.count(after) != 1:
        raise ValueError('rematch body changed')
    block = block.replace(after, after+'\n            steps.push(json!({"stage":"rematch","radius_px":radius,"refined":rematch.matches.len() >= REMATCH_MIN_MATCHES,"input":verification(&rematch),"camera":camera(&refined)}));')
    body = '''pub(super) fn run(initial: &CameraSolution, initial_matches: &[Match], verify: &VerifyStars, detections: &[Detection]) -> (CameraSolution, VerifyResult, Vec<Value>) {
    let mut steps = Vec::new();
'''+block+'''    steps.push(json!({"stage":"final","camera":camera(&refined),"verification":verification(&final_match)}));
    (refined, final_match, steps)
}
'''
    template = Path(__file__).with_name('refinement_trace.rs').read_text()
    return template.replace('// GENERATED_REFINEMENT_PATH', body)


def instrument(source):
    block = extract(source)
    if source.count(HOOK_BEFORE) != 1:
        raise ValueError('candidate hook changed')
    replacement = '''    refinement_trace::input(&chosen, &verify, &detections, (original.width, original.height), epoch);
    let (mut refined, mut final_match, steps) = refinement_trace::run(&chosen.pose, &chosen.verify.matches, &verify, &detections);
    eprintln!("REFINEMENT_PATH {}", serde_json::to_string(&steps).expect("diagnostic JSON"));
'''
    return source.replace(HOOK_BEFORE, HOOK_AFTER).replace(block, replacement)+'\nmod candidate_trace;\nmod refinement_trace;\n'


def selected():
    rows = [r for r in records() if r['id'] == FRAME]
    if len(rows) != 1 or rows[0]['track'] != 'solver':
        raise ValueError('frozen development solver required')
    return rows[0]


def prepare(out):
    row = selected()
    prepare_workspace(out)
    pipeline = out/'pipeline'
    source = ROOT/'prototype/crates/starglyph-core/src/solve.rs'
    target = pipeline/'crates/starglyph-core/src/solve.rs'
    (target.parent/'solve/centroid_origin.rs').unlink()
    for name in ['candidate_trace.rs', 'huber_roll.rs', 'robust_roll.rs']:
        shutil.copyfile(Path(__file__).with_name(name), target.parent/'solve'/name)
    (target.parent/'solve/refinement_trace.rs').write_text(module(source.read_text()))
    (out/'binaries').mkdir()
    for arm in ARMS:
        target.write_text(instrument(source.read_text() if arm == 'control' else patch_source(source.read_text())))
        shutil.copyfile(target, out/f'solve-{arm}.rs')
        command(out, 'build-'+arm, ['cargo','build','--offline','--locked','--release','--manifest-path',str(pipeline/'Cargo.toml'),'-p','starglyph-cli'],timeout=600)
        shutil.copy2(pipeline/'target/release/starglyph', out/'binaries'/arm)
    files = [Path(__file__).resolve(), Path(__file__).with_name('refinement_trace.rs'),
        *[out/f'solve-{a}.rs' for a in ARMS], *[out/'binaries'/a for a in ARMS],
        MANIFEST, MANIFEST.parent/'robustness-results.json',MANIFEST.parent/'collection-wcs-review.json',
        MANIFEST.parent/row['file'],ROOT/'data/catalogs/hyg_v42.csv.gz']
    for name in ['candidate_trace.rs','huber_roll.rs','robust_roll.rs','robustness_candidate_trace.py',
                 'robustness_centroid_origin.py','robustness_huber_development.py','robustness_huber_roll.py',
                 'collection_run.py','robustness_centroid_origin_report.py','compare_local_wcs.py']:
        files.append(Path(__file__).with_name(name))
    for name in ['robustness-iteration-24-case-132162731.json','robustness-iteration-24-development-results.json',
                 'robustness-iteration-13-geometry.json','robustness-iteration-5-geometry.json']:
        files.append(ROOT/'docs/experiments'/name)
    for base in [ROOT/'prototype', pipeline]:
        files.extend([base/'Cargo.toml',base/'Cargo.lock'])
        for name in ['starglyph-core','starglyph-cli','simulator-core']:
            files.extend((base/'crates'/name/'src').rglob('*.rs'))
            files.append(base/'crates'/name/'Cargo.toml')
    files.extend((ROOT/'prototype/artifacts/cache').glob('*.bin'))
    write_json(out/'protocol.json',dict(iteration=25,id=FRAME,split='development',holdout=False,
        wall_limit_s=120,algorithm_change=False,variant='unchanged iteration24 Huber, observational only',
        experiment='2x2 initial pose x initial verification match set; full production refinement/rematches',
        criteria=['instrumented final cameras, detection identities and quality reproduce iteration24',
                  'full chosen tetra3 geometry and pair lists compared, excluding timing only',
                  'offline diagonal paths reproduce captured steps within 1e-8 for scalar numeric values and exact discrete identities',
                  'crossovers diagnostic only, no selection or default promotion',
                  'same 35 conditional reviewed probes and native r8/r12 centroids; no accepted ground truth'],
        timing='instrumented single runs, no performance inference',
        hashes={str(p.relative_to(ROOT)):digest(p) for p in sorted(set(files))}))


def validate(out):
    selected()
    p = json.loads((out/'protocol.json').read_text())
    if (p['id'],p['split'],p['holdout']) != (FRAME,'development',False):
        raise ValueError('wrong selection')
    for name, sha in p['hashes'].items():
        if digest(ROOT/name) != sha:
            raise ValueError('frozen file changed: '+name)
    return p


def events(path, prefix):
    return [json.loads(line[len(prefix):]) for line in path.read_text().splitlines() if line.startswith(prefix)]


def run(out):
    p = validate(out)
    if any((out/f'run-{a}.json').exists() for a in ARMS):
        raise ValueError('fresh runs required')
    for arm in ARMS:
        command(out,'run-'+arm,[str(out/'binaries'/arm),'eval','--manifest',str(MANIFEST),'--ids',FRAME,
            '--catalog',str(ROOT/'data/catalogs/hyg_v42.csv.gz'),'--out-dir',str(out/arm)],
            {**os.environ,'STARGLYPH_SOLVE_DEBUG':'1'},p['wall_limit_s'])
    validate(out)


def replay(out):
    validate(out)
    if (out/'replay-plan.json').exists():
        raise ValueError('replay already frozen')
    inputs, paths, candidates = {}, {}, {}
    for arm in ARMS:
        log = out/f'run-{arm}.log'
        inp, path = events(log,'REFINEMENT_INPUT '), events(log,'REFINEMENT_PATH ')
        if len(inp) != 1 or len(path) != 1:
            raise ValueError('expected one chosen candidate per arm')
        inputs[arm], paths[arm] = inp[0], path[0]
        found = [t for t in traces(log) if all(t['camera'][k] == inp[0]['camera'][k] for k in t['camera'])]
        if len(found) != 1:
            raise ValueError('chosen candidate not uniquely identified')
        candidates[arm] = found[0]
    write_json(out/'chosen-candidates.json',candidates)
    write_json(out/'replay-plan.json',dict(id=FRAME,holdout=False,inputs=inputs))
    write_json(out/'observed-paths.json',paths)
    write_json(out/'replay-protocol.json',dict(iteration=25,id=FRAME,holdout=False,
        cases=[dict(pose_arm=a,match_arm=b) for a in ARMS for b in ARMS],
        hashes={str((out/name).relative_to(ROOT)):digest(out/name) for name in
            ['protocol.json','run-control.log','run-huber.log','chosen-candidates.json','replay-plan.json','observed-paths.json']}))
    command(out,'replay',['cargo','test','--offline','--locked','--release','--manifest-path',str(out/'pipeline/Cargo.toml'),
        '-p','starglyph-core','--features','serde_json/float_roundtrip','--lib','solve::refinement_trace::replay::export','--','--ignored'],
        {**os.environ,'STARGLYPH_REFINEMENT_PLAN':str(out/'replay-plan.json'),
         'STARGLYPH_REFINEMENT_OUTPUT':str(out/'replay-results.json')},600)
    validate(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['prepare','run','replay','validate'])
    parser.add_argument('--out-dir',type=Path,required=True)
    args = parser.parse_args()
    globals()[args.stage](args.out_dir.resolve())


if __name__ == '__main__':
    main()
