#!/usr/bin/env python3
"""Isolated development candidate/refinement replay; no production mutations."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from collection_run import ROOT, digest, select_records, write_json
from tetra3_footprint import prepare as prepare_pipeline

RID = 'wm_r_143159342'
RUST = Path(__file__).with_name('geometry_replay.rs')


def prepare(out, registry):
    manifest = ROOT / 'data/samples/sky-samples/manifest.json'
    records = select_records(json.loads(manifest.read_text()), 'development', [RID])
    if out.exists():
        raise ValueError('fresh output directory required')
    image = manifest.parent / records[0]['file']
    if digest(image) != records[0]['clean_sha256']:
        raise ValueError('image changed')
    out.mkdir(parents=True)
    for arm in ('control', 'footprint'):
        base = out / arm
        # Reuse the isolated-workspace builder. Reverse only the footprint patch
        # for control, preserving all production source and dependency versions.
        prepare_pipeline(base, registry)
        project = base / 'pipeline'
        if arm == 'control':
            for name in ('solve.rs', 'track.rs'):
                shutil.copyfile(registry / 'src/solver' / name, project / 'tetra3/src/solver' / name)
        solve = project / 'crates/starglyph-core/src/solve.rs'
        solve.write_text(solve.read_text() + '\n#[cfg(test)]\nmod geometry_replay;\n')
        shutil.copyfile(RUST, solve.parent / 'solve/geometry_replay.rs')
    write_json(out / 'input.json', dict(id=RID, split='development', image=str(image),
        catalog=str(ROOT / 'data/catalogs/hyg_v42.csv.gz'), cache=str(ROOT / 'prototype/artifacts/cache')))
    files = [manifest, image, RUST, Path(__file__), ROOT / 'prototype/Cargo.toml', ROOT / 'prototype/Cargo.lock',
             ROOT / 'docs/experiments/robustness-iteration-5-review.json',
             ROOT / 'docs/experiments/robustness-iteration-5-geometry.json',
             ROOT / 'data/catalogs/hyg_v42.csv.gz', out / 'input.json']
    files += list((ROOT / 'prototype/artifacts/cache').glob('*.bin'))
    for arm in ('control', 'footprint'):
        project = out / arm / 'pipeline'
        files += list((project / 'crates').glob('*/src/**/*.rs'))
        files += list((project / 'tetra3/src').rglob('*.rs'))
        files += [project / 'Cargo.toml', project / 'tetra3/Cargo.toml']
    write_json(out / 'protocol.json', dict(iteration=6, id=RID, split='development',
        hashes={str(p.resolve()):digest(p) for p in files},
        diagnostic='unchanged default and deep matching; refine every returned candidate separately',
        search_budget='same production prefixes, bands and per-attempt timeouts for both variants and tiers',
        criteria=['reproduce saved selected cameras and detector coordinates',
                  'report all stages on unchanged iteration-5 observations',
                  'retain all pairs and excluded/unchosen candidates; do not tune by errors'],
        holdout=False, wall_timeout_s=240))


def run(out):
    protocol = json.loads((out / 'protocol.json').read_text())
    if protocol['split'] != 'development' or protocol['id'] != RID:
        raise ValueError('frozen development ID required')
    manifest = json.loads((ROOT / 'data/samples/sky-samples/manifest.json').read_text())
    select_records(manifest, 'development', [protocol['id']])
    for p, expected in protocol['hashes'].items():
        if digest(Path(p)) != expected:
            raise ValueError('frozen input changed: ' + p)
    records = []
    for arm in ('control', 'footprint'):
        project = out / arm / 'pipeline'
        # Copied path crates can share Cargo fingerprints despite different
        # contents and preserved mtimes. Each variant needs its own target tree.
        env = {**os.environ, 'CARGO_TARGET_DIR': str(project / 'target'),
               'STARGLYPH_GEOMETRY_INPUT': str(out / 'input.json'),
               'STARGLYPH_GEOMETRY_OUTPUT': str(out / arm / 'trace.json')}
        command = ['cargo', 'test', '--offline', '--release', '--manifest-path', str(project / 'Cargo.toml'),
                   '-p', 'starglyph-core', 'solve::geometry_replay::candidate_geometry', '--', '--ignored', '--nocapture']
        start = time.monotonic()
        with (out / arm / 'run.log').open('w') as log:
            try:
                result = subprocess.run(command, cwd=ROOT / 'prototype', env=env, stdout=log, stderr=subprocess.STDOUT,
                                        timeout=protocol['wall_timeout_s'], check=False)
                row = dict(arm=arm, exit_code=result.returncode, status='completed' if result.returncode == 0 else 'failed')
            except subprocess.TimeoutExpired:
                row = dict(arm=arm, exit_code=None, status='timeout')
        row.update(elapsed_s=time.monotonic()-start, command=command)
        records.append(row)
        write_json(out / 'runs.json', records)
        if row['status'] != 'completed':
            raise RuntimeError(f'{arm}: {row["status"]}; see run.log')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['prepare', 'run'])
    p.add_argument('out', type=Path)
    p.add_argument('--registry', type=Path)
    a = p.parse_args()
    if a.command == 'prepare':
        if a.registry is None:
            p.error('--registry required')
        prepare(a.out.resolve(), a.registry.resolve())
    else:
        run(a.out.resolve())


if __name__ == '__main__':
    main()
