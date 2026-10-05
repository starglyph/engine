#!/usr/bin/env python3
"""Frozen synthetic audit of actual production pose conversion and refinement."""
import argparse
import itertools
import json
import os
from pathlib import Path
import re
import shutil

from collection_run import ROOT, bounded, digest, write_json

RUST = ['centre_transfer.rs', 'centre_transfer_fixture.rs']
PLAN = dict(sizes=[[1600, 1200], [3000, 4000], [1989, 1498]],
            fovs_deg=[22., 55., 85.], sky_tan_deg=[[30., 20., 15.], [210., -35., -27.]],
            centre_offsets_px=[0., .5], representation_shifts_px=[0., .5], supports=['full', 'upper'])


def prepare(out):
    if out.exists():
        raise ValueError('fresh output directory required')
    workspace = out / 'pipeline'
    workspace.mkdir(parents=True)
    for name in ['starglyph-core', 'simulator-core']:
        shutil.copytree(ROOT / 'prototype/crates' / name, workspace / 'crates' / name)
    manifest = (ROOT / 'prototype/Cargo.toml').read_text()
    manifest, count = re.subn(r'members = \[.*?\]',
        'members = ["crates/starglyph-core", "crates/simulator-core"]', manifest, count=1, flags=re.S)
    if count != 1:
        raise ValueError('workspace members changed')
    (workspace / 'Cargo.toml').write_text(manifest)
    shutil.copyfile(ROOT / 'prototype/Cargo.lock', workspace / 'Cargo.lock')
    solve = workspace / 'crates/starglyph-core/src/solve.rs'
    solve.write_text(solve.read_text() + '\n#[cfg(test)]\nmod centre_transfer;\n')
    for name in RUST:
        shutil.copyfile(Path(__file__).with_name(name), solve.parent / 'solve' / name)
    result = bounded(['cargo', 'update', '--offline', '--workspace', '--manifest-path', str(workspace / 'Cargo.toml')],
                     out / 'lock-resolution.log', 120)
    if result['status'] != 'completed':
        raise RuntimeError('isolated lock resolution failed')
    write_json(out / 'plan.json', PLAN)
    sources = [Path(__file__).resolve(), Path(__file__).with_name('collection_run.py'),
               *[Path(__file__).with_name(name) for name in RUST],
               ROOT / 'prototype/Cargo.toml', ROOT / 'prototype/Cargo.lock',
               out / 'plan.json', workspace / 'Cargo.toml', workspace / 'Cargo.lock']
    for tree in [ROOT / 'prototype/crates', workspace / 'crates']:
        for name in ['starglyph-core', 'simulator-core']:
            sources.extend((tree / name / 'src').rglob('*.rs'))
            sources.append(tree / name / 'Cargo.toml')
    sources.extend(ROOT / 'data/samples/sky-samples' / name for name in
                   ['manifest.json', 'robustness-results.json', 'collection-wcs-review.json'])
    sources.extend((ROOT / 'docs/experiments').glob('robustness-iteration-16-*.json'))
    write_json(out / 'protocol.json', dict(iteration=17, split='synthetic', real_images=False, holdout=False,
        hashes={str(p.relative_to(ROOT)): digest(p) for p in sorted(set(sources))}, plan=PLAN,
        case_count=144, fixed_pairs=20, independent_probes=63,
        control_max_probe_error_px=.01, large_error_threshold_px=2.5,
        fit_budget=dict(lm_iterations=30, damping_attempts_per_iteration=10, prior='unchanged production'),
        wall_timeout_s=240,
        criteria=['all 72 matched-centre controls must recover probes within 0.01 px',
                  'report every case and edge; no fitting to the 63 independent probes',
                  'measure whether mismatched-centre errors after fit exceed 2.5 px',
                  'use actual pose_from_solution and refine_pose, with identical pairs/budget',
                  'do not equate synthetic pose conversion with tetra3 search, acceptance or rematch'],
        limitations=['noise-free TAN fields; k1 truth zero; non-mirrored fields',
                     'ideal tetra3 input record; no pattern search or centroid detector',
                     'no resize, foreground, uncertain matches or unknown image processing',
                     'matched-centre arm is a known-truth control, not an automatic real-image correction']))


def validate(out):
    protocol = json.loads((out / 'protocol.json').read_text())
    if protocol['split'] != 'synthetic' or protocol['holdout'] or protocol['real_images']:
        raise ValueError('synthetic only')
    if protocol['plan'] != PLAN:
        raise ValueError('fixed synthetic plan changed')
    for path, sha in protocol['hashes'].items():
        if digest(ROOT / path) != sha:
            raise ValueError('frozen input changed: ' + path)
    return protocol


def run(out):
    protocol = validate(out)
    if (out / 'raw.json').exists() or (out / 'run.json').exists():
        raise ValueError('run already attempted; keep its artifacts')
    command = ['env', 'CARGO_TARGET_DIR=' + str(out / 'pipeline/target'),
               'STARGLYPH_CENTRE_PLAN=' + str(out / 'plan.json'),
               'STARGLYPH_CENTRE_OUTPUT=' + str(out / 'raw.json'),
               'cargo', 'test', '--offline', '--locked', '--release', '--manifest-path',
               str(out / 'pipeline/Cargo.toml'), '-p', 'starglyph-core', 'solve::centre_transfer',
               '--', '--include-ignored', '--nocapture']
    result = bounded(command, out / 'run.log', protocol['wall_timeout_s'])
    write_json(out / 'run.json', result)
    if result['status'] != 'completed':
        raise RuntimeError('synthetic experiment failed: see run.log')


def summarize(out):
    protocol = validate(out)
    rows = json.loads((out / 'raw.json').read_text())['cases']
    expected = set(itertools.product(map(tuple, PLAN['sizes']), PLAN['fovs_deg'],
        map(tuple, PLAN['sky_tan_deg']), PLAN['centre_offsets_px'], PLAN['supports'], PLAN['representation_shifts_px']))
    actual = [(tuple(r['size']), r['truth_fov_deg'], tuple(r['truth_sky_tan_deg']),
               r['truth_centre_offset_from_geometric_px'], r['support'], r['representation_shift_px']) for r in rows]
    if len(rows) != protocol['case_count'] or len(set(actual)) != len(actual) or set(actual) != expected:
        raise ValueError('incomplete or duplicated case matrix')
    if json.loads((out / 'run.json').read_text())['status'] != 'completed':
        raise ValueError('run did not complete')
    for row in rows:
        if row['fixed_pairs'] != 20:
            raise ValueError('training membership changed')
        for stage in ['before', 'after']:
            if row[stage]['training']['count'] != 20 or row[stage]['probes']['count'] != 63:
                raise ValueError('probe membership changed')
            del row[stage]['probe_residuals_px']
    controls = [r for r in rows if r['centres_aligned']]
    mismatches = [r for r in rows if not r['centres_aligned']]
    grouped = []
    for support, fov in itertools.product(PLAN['supports'], PLAN['fovs_deg']):
        group = [r for r in mismatches if r['support'] == support and r['truth_fov_deg'] == fov]
        grouped.append(dict(support=support, fov_deg=fov, count=len(group),
            max_probe_rms_px=max(r['after']['probes']['rms_px'] for r in group),
            max_probe_error_px=max(r['after']['probes']['max_px'] for r in group)))
    write_json(out / 'results.json', dict(iteration=17, split='synthetic',
        protocol_sha256=digest(out / 'protocol.json'), raw_sha256=digest(out / 'raw.json'),
        run=json.loads((out / 'run.json').read_text()), control_count=len(controls),
        control_max_probe_error_px=max(r['after']['probes']['max_px'] for r in controls),
        controls_pass=all(r['after']['probes']['max_px'] <= protocol['control_max_probe_error_px'] for r in controls),
        mismatched_count=len(mismatches),
        mismatched_large_error_count=sum(r['after']['probes']['max_px'] > protocol['large_error_threshold_px'] for r in mismatches),
        groups=grouped, cases=rows, limitations=protocol['limitations']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'run', 'summarize'])
    parser.add_argument('--out-dir', type=Path, required=True)
    args = parser.parse_args()
    out = args.out_dir.resolve()
    os.chdir(ROOT / 'prototype')
    globals()[args.stage](out)


if __name__ == '__main__':
    main()
