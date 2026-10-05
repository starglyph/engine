#!/usr/bin/env python3
"""Development-only full-pipeline A/B of consistent centroid-origin subtraction."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys

from collection_run import ROOT, digest, select_records, write_json

MANIFEST = ROOT / 'data/samples/sky-samples/manifest.json'
ARMS = ('control', 'centered')
BEFORE = '    let cx = (width - 1) as f32 / 2.0;\n    let cy = (height - 1) as f32 / 2.0;'
AFTER = '    let cx = width as f32 / 2.0;\n    let cy = height as f32 / 2.0;'
TEST_BEFORE = '        let cx = (w - 1) as f32 / 2.0;\n        let cy = (h - 1) as f32 / 2.0;'
TEST_AFTER = '        let (cx, cy) = geom::principal_point(w, h);\n        let (cx, cy) = (cx as f32, cy as f32);'
MODULE = '\n#[cfg(test)]\nmod centroid_origin;\n'


def patch_source(text):
    if text.count(BEFORE) != 1 or text.count(TEST_BEFORE) != 1:
        raise ValueError('unexpected centroid conversion source')
    return (text.replace(BEFORE, AFTER).replace(TEST_BEFORE, TEST_AFTER)
            .replace('centroid_roundtrip_matches_spike_convention', 'centroid_roundtrip_matches_geom_convention') + MODULE)


def records():
    return select_records(json.loads(MANIFEST.read_text()), 'development')


def prepare(out):
    if out.exists():
        raise ValueError('fresh output directory required')
    selected = records()
    if len(selected) != 45:
        raise ValueError('development membership changed')
    workspace = out / 'pipeline'
    workspace.mkdir(parents=True)
    for name in ['starglyph-core', 'starglyph-cli', 'simulator-core']:
        shutil.copytree(ROOT / 'prototype/crates' / name, workspace / 'crates' / name)
    source = workspace / 'crates/starglyph-core/src/solve.rs'
    source.write_text(patch_source(source.read_text()))
    shutil.copyfile(Path(__file__).with_name('centroid_origin.rs'), source.parent / 'solve/centroid_origin.rs')
    manifest = (ROOT / 'prototype/Cargo.toml').read_text()
    manifest, count = re.subn(r'members = \[.*?\]',
        'members = ["crates/starglyph-core", "crates/starglyph-cli", "crates/simulator-core"]', manifest, count=1, flags=re.S)
    if count != 1:
        raise ValueError('workspace members changed')
    (workspace / 'Cargo.toml').write_text(manifest)
    shutil.copyfile(ROOT / 'prototype/Cargo.lock', workspace / 'Cargo.lock')
    (out / 'data').symlink_to(ROOT / 'data', target_is_directory=True)
    (out / 'prototype').symlink_to(workspace, target_is_directory=True)
    (workspace / 'artifacts').mkdir()
    (workspace / 'artifacts/cache').symlink_to(ROOT / 'prototype/artifacts/cache', target_is_directory=True)
    with (out / 'lock-resolution.log').open('w') as log:
        subprocess.run(['cargo', 'update', '--offline', '--workspace', '--manifest-path', str(workspace / 'Cargo.toml')],
                       cwd=ROOT / 'prototype', stdout=log, stderr=subprocess.STDOUT, check=True)


def freeze(out):
    if (out / 'protocol.json').exists():
        raise ValueError('protocol already frozen')
    selected = records()
    workspace = out / 'pipeline'
    source = ROOT / 'prototype/crates/starglyph-core/src/solve.rs'
    if (workspace / 'crates/starglyph-core/src/solve.rs').read_text() != patch_source(source.read_text()):
        raise ValueError('candidate differs from the single prescribed change')
    built = dict(control=ROOT / 'prototype/target/release/starglyph', centered=workspace / 'target/release/starglyph')
    (out / 'binaries').mkdir()
    binaries = {arm:out / 'binaries' / arm for arm in ARMS}
    for arm in ARMS:
        shutil.copy2(built[arm], binaries[arm])
    files = [Path(__file__).resolve(), Path(__file__).with_name('centroid_origin.rs'),
             Path(__file__).with_name('robustness_centroid_origin_report.py'),
             Path(__file__).with_name('collection_run.py'), Path(__file__).with_name('compare_local_wcs.py'),
             MANIFEST, MANIFEST.parent / 'robustness-results.json', MANIFEST.parent / 'collection-wcs-review.json',
             ROOT / 'data/catalogs/hyg_v42.csv.gz', *binaries.values(), out / 'gates.json', out / 'final-gates.json',
             out / 'regression-recheck.json']
    for base in [ROOT / 'prototype', workspace]:
        files.extend([base / 'Cargo.toml', base / 'Cargo.lock'])
        for name in ['starglyph-core', 'starglyph-cli', 'simulator-core']:
            files.extend((base / 'crates' / name / 'src').rglob('*.rs'))
            files.append(base / 'crates' / name / 'Cargo.toml')
    files.extend(MANIFEST.parent / r['file'] for r in selected)
    for record in selected:
        reference = ROOT / f"prototype/artifacts/robustness-baseline/wcs-run/{record['id']}/reference.json"
        if reference.exists():
            files.append(reference)
            data = json.loads(reference.read_text())
            if data['status'] == 'solved_candidate':
                files.extend(ROOT / 'prototype' / data[key] for key in ('wcs_file', 'correspondences_file'))
    files.extend((ROOT / 'prototype/artifacts/cache').glob('*.bin'))
    files.extend(ROOT / 'docs/experiments' / name for name in [
        'robustness-iteration-13-geometry.json', 'robustness-iteration-5-geometry.json',
        'robustness-iteration-17-protocol.json', 'robustness-iteration-17-results.json'])
    write_json(out / 'protocol.json', dict(iteration=18, split='development', holdout=False,
        ids=[r['id'] for r in selected],
        binaries={a:str(p.relative_to(ROOT)) for a,p in binaries.items()},
        hashes={str(p.relative_to(ROOT)):digest(p) for p in sorted(set(files))},
        variant='only remove -1 from width/height in detection_to_centroid; preserve f32 arithmetic order',
        other_experiments=False, wall_limit_s=120, batch_size=6,
        environment=dict(python=platform.python_version(), platform=platform.platform(),
            available_cpus=len(os.sched_getaffinity(0)), rayon_num_threads=os.environ.get('RAYON_NUM_THREADS')),
        order='alternate first arm each batch; one CLI at a time; shared prewarmed cache',
        geometry=dict(source='iteration-13 frozen sources, all 117; iteration-5 alternate centroids',
                      excluded_detection_radius_px=12, max_group_rms_regression_px=.5,
                      improvement_fraction=.1, minimum_outside_probes_for_improvement=8),
        criteria=['retain every control solver success and all eight development negative rejections',
                  'no additional process failure or wall timeout',
                  'report baseline/control status differences instead of silently replacing baseline',
                  'on three reviewed fields, no all/outside-current-input/edge RMS regression greater than 0.5 px',
                  'improvement requires a new independently reviewed solve or >=10% outside-input RMS improvement with >=8 probes',
                  'new solves without reviewed field geometry remain unresolved; no default promotion from solve rate alone'],
        time_interpretation='descriptive single paired run, same limits; not a latency benchmark'))


def validate(out):
    p = json.loads((out / 'protocol.json').read_text())
    if p['split'] != 'development' or p['holdout']:
        raise ValueError('development only')
    selected = records()
    if [r['id'] for r in selected] != p['ids']:
        raise ValueError('frozen development selection changed')
    for path, sha in p['hashes'].items():
        if digest(ROOT / path) != sha:
            raise ValueError('frozen input changed: ' + path)
    return p


def run(out):
    p = validate(out)
    with (out / '.iteration.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        batch = 0
        while True:
            complete = True
            order = ARMS if batch % 2 == 0 else tuple(reversed(ARMS))
            for arm in order:
                folder = out / 'development' / arm
                prior = json.loads((folder / 'summary.json').read_text()) if (folder / 'summary.json').exists() else None
                if prior and prior['completed'] == len(p['ids']):
                    continue
                complete = False
                command = [sys.executable, str(ROOT / 'prototype/eval/collection_run.py'), 'starglyph',
                           '--split', 'development', '--binary', str(ROOT / p['binaries'][arm]),
                           '--out-dir', str(folder), '--batch-size', str(p['batch_size'])]
                if prior:
                    command.append('--resume')
                print('batch', batch, arm, flush=True)
                subprocess.run(command, cwd=ROOT / 'prototype', check=True,
                               env={**os.environ, 'STARGLYPH_SOLVE_DEBUG':'1'})
            if complete:
                break
            batch += 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'freeze', 'run'])
    parser.add_argument('--out-dir', type=Path, required=True)
    args = parser.parse_args()
    globals()[args.stage](args.out_dir.resolve())


if __name__ == '__main__':
    main()
