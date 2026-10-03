#!/usr/bin/env python3
"""Isolated automatic footprint experiment. Development only; no default change."""
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

from collection_run import ROOT, digest, select_records, write_json
from tetra3_internal import SOLVE_SHA

PATCH = Path(__file__).with_name('tetra3_verify_footprint.patch').resolve()


def prepare(out, registry):
    if digest(registry / 'src/solver/solve.rs') != SOLVE_SHA:
        raise ValueError('unexpected tetra3 version')
    project = out / 'pipeline'
    project.mkdir(parents=True, exist_ok=False)
    shutil.copytree(ROOT / 'prototype/crates', project / 'crates')
    shutil.copyfile(ROOT / 'prototype/Cargo.lock', project / 'Cargo.lock')
    manifest = (ROOT / 'prototype/Cargo.toml').read_text().replace('  "apps/desktop",\n', '')
    manifest += '\n[patch.crates-io]\ntetra3 = { path = "tetra3" }\n'
    (project / 'Cargo.toml').write_text(manifest)
    vendor = project / 'tetra3'
    shutil.copytree(registry, vendor)
    cm = vendor / 'Cargo.toml'
    cm.write_text(re.sub(r'(?ms)^\[dev-dependencies\.[^\]]+\]\n.*?(?=^\[|\Z)', '', cm.read_text()))
    subprocess.run(['patch', '--batch', '-p1', '-i', str(PATCH)], cwd=vendor, check=True)
    # CLI's compiled-in data root points one level above the copied workspace.
    (out / 'data').symlink_to(ROOT / 'data', target_is_directory=True)
    (project / 'artifacts').mkdir()
    (project / 'artifacts/cache').symlink_to(ROOT / 'prototype/artifacts/cache', target_is_directory=True)


def freeze(out):
    manifest = ROOT / 'data/samples/sky-samples/manifest.json'
    records = select_records(json.loads(manifest.read_text()), 'development')
    binaries = {'control': ROOT / 'prototype/target/release/starglyph',
                'footprint': out / 'pipeline/target/release/starglyph'}
    sources = [Path(__file__).resolve(), PATCH, ROOT / 'prototype/eval/collection_run.py',
               manifest, manifest.parent / 'robustness-results.json', manifest.parent / 'collection-wcs-review.json',
               ROOT / 'data/catalogs/hyg_v42.csv.gz', out / 'pipeline/Cargo.lock']
    sources += sorted((out / 'pipeline/crates').glob('*/src/**/*.rs'))
    sources += sorted((out / 'pipeline/tetra3/src').rglob('*.rs'))
    sources += sorted((ROOT / 'prototype/artifacts/cache').glob('*.bin'))
    sources += [manifest.parent / r['file'] for r in records]
    sources += list(binaries.values())
    write_json(out / 'protocol.json', dict(split='development',
        binaries={k: str(v.resolve()) for k, v in binaries.items()},
        hashes={str(p.resolve()): digest(p) for p in sources},
        ids=[r['id'] for r in records],
        variant='clip projected verification catalogue to rectangular sensor plus existing match radius before 2N truncation',
        unchanged=['detections', 'DBs', 'search budget', 'probability threshold', 'Starglyph acceptance', 'refinement'],
        criteria=['retain each previous solver success', 'reject all development negatives',
                  'no new process errors or wall timeouts', 'new solutions require independent geometry review'],
        wall_timeout_per_frame_s=120, quantile_threshold=False, blob_concentration=False,
        solve_debug=True, cache='same prewarmed cache for both binaries',
        holdout='not permitted in this executor'))


def run(out):
    protocol_path = out / 'protocol.json'
    if not protocol_path.exists():
        freeze(out)
    protocol = json.loads(protocol_path.read_text())
    if protocol['split'] != 'development':
        raise ValueError('development only')
    for path, sha in protocol['hashes'].items():
        if digest(Path(path)) != sha:
            raise ValueError(f'frozen input changed: {path}')
    remaining = set(protocol['binaries'])
    while remaining:
        for arm, binary in protocol['binaries'].items():
            if arm not in remaining:
                continue
            folder = out / 'development' / arm
            prior = json.loads((folder / 'summary.json').read_text()) if (folder / 'summary.json').exists() else None
            if prior and prior['completed'] == len(protocol['ids']):
                remaining.remove(arm)
                continue
            command = [sys.executable, str(ROOT / 'prototype/eval/collection_run.py'), 'starglyph',
                       '--split', 'development', '--binary', binary, '--out-dir', str(folder), '--batch-size', '6']
            if prior:
                command.append('--resume')
            subprocess.run(command, cwd=ROOT / 'prototype', check=True,
                           env={**os.environ, 'STARGLYPH_SOLVE_DEBUG': '1'})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('prepare', 'run'))
    p.add_argument('out', type=Path)
    p.add_argument('--registry', type=Path)
    args = p.parse_args()
    out = args.out.resolve()
    if args.command == 'prepare':
        if args.registry is None:
            p.error('--registry required to prepare')
        prepare(out, args.registry)
    else:
        run(out)


if __name__ == '__main__':
    main()
