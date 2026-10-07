#!/usr/bin/env python3
"""One fixed bright-catalog rematch replay; automatic seed, development only."""
import argparse
import json
import os
from pathlib import Path
import shutil

from collection_run import ROOT, digest, write_json
from robustness_candidate_trace import command
from robustness_centroid_origin import prepare as prepare_workspace
from robustness_orion_fit import RID, previous, read
from robustness_reference_audit import selected
from robustness_refinement_trace import module

OLD = ROOT/'prototype/artifacts/robustness-iteration-41/trace.json'


def replay_module(source):
    """Reuse the production-generated refinement and existing replay parser."""
    code = module(source)
    replacements = [
        ('assert_eq!(plan["id"], "wm_r_132162731");', f'assert_eq!(plan["id"], "{RID}");\n        assert_eq!(plan["split"], "development");'),
        ('"huber"', '"bright55"'),
        ('for match_arm in ["control", "bright55"] {', '{\n                let match_arm = "control";'),
        ('json!([initial.width, initial.height]),\n                "lift not supported in this diagnostic"',
         'json!([4485, 2920]),\n                "fixed Orion original dimensions"'),
        ('let (p, v, steps) = run(&initial, &rows, &stars, &dets);',
         '''let rematch_stars = VerifyStars {
                    list: stars.list.iter().copied()
                        .filter(|(_, mag)| pose_arm == "control" || *mag <= 5.5).collect(),
                    by_id: HashMap::new(),
                };
                let (p, _, steps) = run(&initial, &rows, &rematch_stars, &dets);
                // Final verification always uses the full catalogue; no acceptance relaxation.
                let v = match_predictions(&p, &stars, &dets, FINAL_RADIUS_PX);'''),
        ('"verification":verification(&v),"steps":steps,"elapsed_ms":',
         '"verification":verification(&v),"rematch_catalog_count":rematch_stars.list.len(),"steps":steps,"elapsed_ms":'),
    ]
    for before, after in replacements:
        if before != '"huber"' and code.count(before) != 1:
            raise ValueError('replay template changed: '+before)
        code = code.replace(before, after)
    # Observer entry point is not used in an isolated replay.
    start = code.index('pub(super) fn input('); end = code.index('pub(super) fn run(')
    return code[:start] + code[end:]


def prepare(out):
    selected()
    paths = [previous(n) for n in ['40-results', '40-checks', '41-trace', '41-results', '41-checks', '53-checks']]
    for name in ['40-results', '41-trace', '41-results']:
        p = previous(name)
        if digest(p) != read(previous(name[:2]+'-checks'))['artifact_hashes'][str(p.relative_to(ROOT))]:
            raise ValueError('published input changed: '+name)
    if digest(OLD) != read(previous('41-checks'))['artifact_hashes'][str(OLD.relative_to(ROOT))]:
        raise ValueError('saved full trace changed')
    protected = read(previous('53-checks'))['protected_hashes']
    for name, sha in protected.items():
        if digest(ROOT/name) != sha:
            raise ValueError('protected data changed')
        paths.append(ROOT/name)
    trace = read(OLD); inp = trace['input']
    if len(inp['detections']) != 40 or len(inp['verification']['matches']) != 16:
        raise ValueError('fixed automatic seed required')
    prepare_workspace(out)
    target = out/'pipeline/crates/starglyph-core/src/solve.rs'
    source = ROOT/'prototype/crates/starglyph-core/src/solve.rs'
    target.write_text(source.read_text()+'\n#[cfg(test)]\nmod refinement_trace;\n')
    (target.parent/'solve/centroid_origin.rs').unlink()
    generated = target.parent/'solve/refinement_trace.rs'
    generated.write_text(replay_module(source.read_text()))
    command(out, 'rustfmt', ['rustfmt', '--edition', '2021', str(generated)], timeout=30)
    write_json(out/'input.json', dict(id=RID, split='development', holdout=False,
               inputs=dict(control=inp, bright55=inp)))
    paths += [OLD, out/'input.json', ROOT/'data/catalogs/hyg_v42.csv.gz', Path(__file__).resolve()]
    paths += [Path(__file__).with_name(n) for n in ['robustness_orion_rematch_report.py',
        'test_robustness_orion_rematch.py', 'robustness_refinement_trace.py', 'refinement_trace.rs',
        'robustness_centroid_origin.py', 'robustness_candidate_trace.py', 'collection_run.py',
        'robustness_orion_fit.py', 'robustness_reference_audit.py', 'robustness_refinement_report.py',
        'robustness_refinement_crossover_report.py', 'robustness_reference_coverage.py', 'compare_local_wcs.py']]
    for base in [ROOT/'prototype', out/'pipeline']:
        paths += [base/'Cargo.toml', base/'Cargo.lock']
        for crate in ['starglyph-core', 'starglyph-cli', 'simulator-core']:
            paths += [base/f'crates/{crate}/Cargo.toml', *(base/f'crates/{crate}/src').rglob('*.rs')]
    write_json(out/'protocol.json', dict(iteration=54, id=RID, split='development', holdout=False,
        arms=['control', 'bright55'], change='only rematch/refit catalogue mag<=5.5; unchanged automatic initial16 and camera; final working verification uses full catalogue',
        cutoff_origin='pre-existing40 reviewed catalogue magnitude limit; one diagnostic cutoff, no sweep',
        budget=dict(cases=2, max_lm_calls_per_case=4, max_lm_iterations=30, damping_trials=10, wall_s=60, build_s=600),
        assessment='all47 and common outside-fit union across BOTH arms: exclude used HYG identity OR center within12 originalpx of any fit detection across any of3 centroid methods; retain original spatial groups',
        criteria=['control steps reproduce41 within1e-8 and discrete identities exactly; both first LM steps identical',
            'outside-common count>=8; final outside-common RMS falls>=10percent for all3 methods; no nonempty all47/outside-common spatial RMS regression>0.5originalpx',
            'final full-catalog reviewed-agree pair count not smaller and width coverage strictly larger; reviewed-disagree count not larger',
            'unknown pairs remain unknown; ID disagreements are conditional on40, close components not automatically false',
            'record every stage, fit count, geometry, pair identities, process failures and timeouts; no parameter tuning',
            'working refinement only, no native fit or final acceptance; no solver/WCS/holdout or automatic improvement claim'],
        protected_hashes=protected, hashes={str(p.relative_to(ROOT)): digest(p) for p in sorted(set(paths))}))


def validate(out):
    p = read(out/'protocol.json')
    if (p['id'], p['split'], p['holdout']) != (RID, 'development', False):
        raise ValueError('development only')
    selected()
    for name, sha in p['hashes'].items():
        if digest(ROOT/name) != sha:
            raise ValueError('frozen input changed: '+name)
    return p


def build(out):
    validate(out); workspace = out/'pipeline'
    env = {**os.environ, 'CARGO_TARGET_DIR': str(ROOT/'prototype/target')}
    command(out, 'build', ['cargo', 'test', '--offline', '--locked', '--release', '--manifest-path', str(workspace/'Cargo.toml'),
        '-p', 'starglyph-core', '--features', 'serde_json/float_roundtrip', '--lib', '--no-run', '--message-format', 'json'], env, 600)
    artifacts = [json.loads(s) for s in (out/'build.log').read_text().splitlines() if s.startswith('{')]
    binaries = [a['executable'] for a in artifacts if a.get('reason') == 'compiler-artifact' and a.get('executable') and a['target']['name'] == 'starglyph_core']
    if len(binaries) != 1:
        raise ValueError('unique core executable required')
    shutil.copy2(binaries[0], out/'replay-test')
    command(out, 'rust-tests', [str(out/'replay-test')], timeout=120)
    command(out, 'clippy', ['cargo', 'clippy', '--offline', '--locked', '--release', '--manifest-path', str(workspace/'Cargo.toml'),
        '-p', 'starglyph-core', '--features', 'serde_json/float_roundtrip', '--lib', '--tests', '--no-deps', '--', '-D', 'warnings'], env, 600)
    write_json(out/'binary.json', dict(sha256=digest(out/'replay-test'), protocol_sha256=digest(out/'protocol.json')))
    validate(out)


def run(out):
    validate(out)
    if (out/'run.json').exists():
        raise ValueError('fresh run required')
    if read(out/'binary.json') != dict(sha256=digest(out/'replay-test'), protocol_sha256=digest(out/'protocol.json')):
        raise ValueError('binary freeze changed')
    command(out, 'run', [str(out/'replay-test'), '--exact', 'solve::refinement_trace::replay::export', '--ignored'],
        {**os.environ, 'STARGLYPH_REFINEMENT_PLAN': str(out/'input.json'), 'STARGLYPH_REFINEMENT_OUTPUT': str(out/'replay.json')}, 60)
    validate(out)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['prepare', 'build', 'run', 'validate'])
    p.add_argument('--out-dir', type=Path, required=True)
    a = p.parse_args(); globals()[a.stage](a.out_dir.resolve())
