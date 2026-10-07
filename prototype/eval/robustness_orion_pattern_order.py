#!/usr/bin/env python3
"""Frozen full-pipeline Orion pilot: reorder retained tetra3 pattern stars only."""
import argparse
import os
from pathlib import Path
import shutil

from collection_run import ROOT, digest, write_json
from robustness_candidate_trace import command
from robustness_centroid_origin import prepare as prepare_workspace
from robustness_matcher_funnel import instrument as funnel_instrument
from robustness_orion_fit import RID, previous, read
from robustness_reference_audit import MANIFEST, CATALOG, selected
from tetra3_internal import prepare as prepare_vendor, insert

OLD = ROOT/'prototype/artifacts/robustness-iteration-25/fixed'
ARMS = ('control', 'farthest')


def instrument(source):
    anchor = '        let num_pattern_centroids = pattern_centroid_inds.len();'
    addition = '''
        let pattern_order = if std::env::var("STARGLYPH_PATTERN_ORDER").as_deref() == Ok("farthest") {
            let positions: Vec<[f32; 2]> = sorted_indices.iter()
                .map(|&i| [centroids[i].x, centroids[i].y]).collect();
            crate::research_pattern_order::farthest_first(&pattern_centroid_inds, &positions)
        } else {
            pattern_centroid_inds.clone()
        };
        eprintln!("SGORDER {}", serde_json::json!({
            "fov_deg": fov_estimate.to_degrees(),
            "retained": pattern_centroid_inds.iter().map(|&i| sorted_indices[i]).collect::<Vec<_>>(),
            "priority": pattern_order.iter().map(|&i| sorted_indices[i]).collect::<Vec<_>>()
        }));'''
    loop = 'BreadthFirstCombinations::<PATTERN_SIZE>::new(&pattern_centroid_inds)'
    if source.count(loop) != 1:
        raise ValueError('unique existing pattern iterator required')
    return insert(source, anchor, addition).replace(loop, 'BreadthFirstCombinations::<PATTERN_SIZE>::new(&pattern_order)')


def prepare(out, registry):
    selected()
    for name in ('40-results', '41-trace', '41-native-report', '47-results'):
        p = previous(name)
        if digest(p) != read(previous(name[:2]+'-checks'))['artifact_hashes'][str(p.relative_to(ROOT))]:
            raise ValueError('published evidence changed: '+name)
    old_protocol = read(previous('41-protocol'))
    saved = OLD/'solve-control.rs'
    trace_module = OLD/'pipeline/crates/starglyph-core/src/solve/refinement_trace.rs'
    for p in (saved, trace_module):
        if digest(p) != old_protocol['hashes'][str(p.relative_to(ROOT))]:
            raise ValueError('observer changed')
    protected = read(previous('47-checks'))['protected_hashes']
    for name, sha in protected.items():
        if digest(ROOT/name) != sha:
            raise ValueError('protected data changed')
    prepare_workspace(out)
    pipeline = out/'pipeline'; solve = pipeline/'crates/starglyph-core/src/solve.rs'
    solve.write_text(saved.read_text())
    (solve.parent/'solve/centroid_origin.rs').unlink()
    shutil.copy2(trace_module, solve.parent/'solve/refinement_trace.rs')
    shutil.copy2(Path(__file__).with_name('candidate_trace.rs'), solve.parent/'solve/candidate_trace.rs')
    # Reuse the existing instrumented dependency, without the footprint patch.
    prepare_vendor(out/'vendor', registry)
    vendor = out/'vendor/harness/tetra3'; source = vendor/'src/solver/solve.rs'
    source.write_text(instrument(funnel_instrument(source.read_text())))
    for name in ('tetra3_pattern_order', 'tetra3_funnel'):
        module = 'research_pattern_order' if name == 'tetra3_pattern_order' else 'research_funnel'
        shutil.copy2(Path(__file__).with_name(name+'.rs'), vendor/'src'/f'{module}.rs')
        with (vendor/'src/lib.rs').open('a') as f:
            f.write(f'\npub mod {module};\n')
    with (pipeline/'Cargo.toml').open('a') as f:
        f.write('\n[patch.crates-io]\ntetra3 = { path = "../vendor/harness/tetra3" }\n')
    command(out, 'resolve', ['cargo', 'update', '--offline', '--manifest-path', str(pipeline/'Cargo.toml'), '-p', 'tetra3'], timeout=60)
    paths = [Path(__file__).resolve(), saved, trace_module, MANIFEST, CATALOG, previous('41-protocol'),
        *(previous(n) for n in ('40-results', '40-checks', '41-trace', '41-native-report', '41-checks', '47-results', '47-checks')),
        *(ROOT/n for n in protected), *(ROOT/'prototype/artifacts/cache').glob('*.bin'),
        *(Path(__file__).with_name(n) for n in ('tetra3_pattern_order.rs', 'tetra3_funnel.rs', 'tetra3_internal.py',
            'tetra3_internal.rs', 'tetra3_internal_trace.rs', 'robustness_matcher_funnel.py',
            'robustness_candidate_trace.py', 'candidate_trace.rs', 'robustness_centroid_origin.py', 'collection_run.py'))]
    for base in (pipeline, vendor):
        paths.extend(base.rglob('*.rs')); paths.extend(base.rglob('Cargo.toml'))
    paths += [pipeline/'Cargo.lock', ROOT/'prototype/Cargo.toml', ROOT/'prototype/Cargo.lock']
    write_json(out/'protocol.json', dict(iteration=48, id=RID, split='development', holdout=False, arms=ARMS,
        scope='single development pilot; unchanged full pipeline except tetra3 pattern priority after existing thinning',
        variant='brightest survivor first, then maximize minimum squared pixel distance to selected survivors; brightness tie-break; no new threshold',
        preserved=['photo', 'detection coordinates and flux', 'brightness sort', 'cluster-buster membership',
            'verification order', 'database', 'FOV sweep', 'pattern and acceptance tolerances', 'LM and rematch'],
        budget=dict(processes=2, per_attempt_ms=2500, wall_s_per_process=120, search_ladder='unchanged'),
        criteria=['control exactly reproduces41 camera/status/quality/ordered detector DTO and candidate geometry/identities',
            'same first matching detector list; preserve every thinning survivor exactly once; count actual work and timeouts',
            'retain solved status; improve RMS of frozen outside_both_starglyph_inputs18 by>=10percent',
            'no nonempty all47 spatial group RMS regression>0.5 original px under any of three centroid methods',
            'inspect correspondence identities before any automatic improvement claim; no default promotion from this pilot',
            'if geometry criterion fails, stop without broad development or holdout run'],
        hashes={str(p.relative_to(ROOT)): digest(p) for p in sorted(set(paths))}))


def validate(out):
    p = read(out/'protocol.json')
    if (p['id'], p['split'], p['holdout'], p['arms']) != (RID, 'development', False, list(ARMS)):
        raise ValueError('one frozen development pilot required')
    selected()
    for name, sha in p['hashes'].items():
        if digest(ROOT/name) != sha:
            raise ValueError('frozen input changed: '+name)
    return p


def build(out):
    validate(out)
    env = dict(os.environ, CARGO_TARGET_DIR=str(out/'pipeline/target'))
    command(out, 'build', ['cargo', 'build', '--release', '--offline', '--locked', '--manifest-path',
        str(out/'pipeline/Cargo.toml'), '-p', 'starglyph-cli', '--features', 'serde_json/float_roundtrip'], env, 600)
    shutil.copy2(out/'pipeline/target/release/starglyph', out/'observer')
    write_json(out/'binary.json', dict(sha256=digest(out/'observer'), protocol_sha256=digest(out/'protocol.json')))


def run(out):
    p = validate(out)
    if read(out/'binary.json') != dict(sha256=digest(out/'observer'), protocol_sha256=digest(out/'protocol.json')):
        raise ValueError('binary freeze changed')
    if any((out/(arm+'.json')).exists() for arm in ARMS):
        raise ValueError('fresh paired run required')
    for arm in ARMS:
        command(out, arm, [str(out/'observer'), 'eval', '--manifest', str(MANIFEST), '--ids', RID,
            '--catalog', str(CATALOG), '--out-dir', str(out/arm)],
            dict(os.environ, STARGLYPH_SOLVE_DEBUG='1', STARGLYPH_PATTERN_ORDER=arm), p['budget']['wall_s_per_process'])
    validate(out)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=('prepare', 'build', 'run', 'validate'))
    p.add_argument('--out-dir', type=Path, required=True); p.add_argument('--registry', type=Path)
    a = p.parse_args()
    if a.stage == 'prepare':
        if a.registry is None: p.error('--registry required')
        prepare(a.out_dir.resolve(), a.registry)
    else:
        globals()[a.stage](a.out_dir.resolve())
