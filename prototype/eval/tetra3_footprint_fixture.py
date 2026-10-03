#!/usr/bin/env python3
"""Run deterministic verification fixtures against original and patched tetra3."""
import argparse
from pathlib import Path
import re
import shutil
import subprocess

from collection_run import ROOT, digest
from tetra3_internal import SOLVE_SHA


def prepare(out, registry, footprint=False):
    if digest(registry / 'src/solver/solve.rs') != SOLVE_SHA:
        raise ValueError('unexpected tetra3 source')
    out.mkdir(parents=True, exist_ok=False)
    vendor = out / 'tetra3'
    shutil.copytree(registry, vendor)
    cm = vendor / 'Cargo.toml'
    cm.write_text(re.sub(r'(?ms)^\[dev-dependencies\.[^\]]+\]\n.*?(?=^\[|\Z)', '', cm.read_text()))
    if footprint:
        subprocess.run(['patch', '--batch', '--fuzz=0', '-p1', '-i',
                        str(Path(__file__).with_name('tetra3_verify_footprint.patch').resolve())], cwd=vendor, check=True)
    solve = vendor / 'src/solver/solve.rs'
    fixture = Path(__file__).with_suffix('.rs').read_text().replace(
        '/*PIXEL_SCALE_ARG*/', 'pixel_scale,' if footprint else '')
    solve.write_text(solve.read_text() + '\n' + fixture)
    with (vendor / 'src/solver/mod.rs').open('a') as f:
        f.write('\npub use self::solve::research_footprint_fixture;\n')
    (out / 'Cargo.toml').write_text('''[package]
name="starglyph-footprint-fixture"
version="0.0.0"
edition="2021"
[workspace]
[dependencies]
tetra3={path="tetra3",features=["parallel"]}
''')
    shutil.copyfile(ROOT / 'prototype/Cargo.lock', out / 'Cargo.lock')
    (out / 'src').mkdir()
    (out / 'src/main.rs').write_text('''fn main() {
    let db = std::env::args().nth(1).expect("cache database");
    let patched = std::env::args().nth(2).as_deref() == Some("patched");
    for portrait in [false,true] {
        for edge in [false,true] {
            for distractors in [false,true] {
                for refined_ratio in [1.0, 0.85] {
                    let n = tetra3::solver::research_footprint_fixture(&db,portrait,distractors,edge,refined_ratio).unwrap();
                    let expected = if distractors && !patched {0} else {12};
                    assert_eq!(n,expected,"portrait={portrait} edge={edge} distractors={distractors} ratio={refined_ratio}");
                    println!("portrait={portrait} edge={edge} distractors={distractors} ratio={refined_ratio}: {n}");
                }
            }
        }
    }
}
''')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('out', type=Path)
    p.add_argument('--registry', type=Path, required=True)
    p.add_argument('--footprint', action='store_true')
    a = p.parse_args()
    prepare(a.out, a.registry, a.footprint)


if __name__ == '__main__':
    main()
