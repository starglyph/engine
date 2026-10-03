#!/usr/bin/env python3
"""Build an isolated, hash-checked tetra3 0.8.0 observation-only trace harness.

The Cargo registry and production workspace/lockfile are never edited.
"""
import argparse
import json
from pathlib import Path
import re
import shutil
import subprocess

from collection_run import ROOT, digest, write_json

SOLVE_SHA = '62847229edf3ed141c09158fa7f0011e3538bb0a37d1cfc91b0fcfa9ae6f5835'


def insert(text, anchor, addition):
    if text.count(anchor) != 1:
        raise ValueError(f'upstream trace anchor must occur once: {anchor[:80]}')
    return text.replace(anchor, anchor + '\n' + addition)


def instrument(text):
    text = insert(text, '        let num_pattern_centroids = pattern_centroid_inds.len();', '''
        crate::research_trace::emit("thinning", serde_json::json!({
            "fov": fov_estimate.to_degrees(), "kept_input_indices": pattern_centroid_inds.iter().map(|&i| sorted_indices[i]).collect::<Vec<_>>()
        }));''')
    text = insert(text, '            let image_largest_edge = edge_angles[NUM_EDGES - 1];', '''
            let trace_image = crate::research_trace::image_is_target(image_pattern_local.map(|i| sorted_indices[i]));
            if trace_image { crate::research_trace::emit("image_pattern", serde_json::json!({
                "fov": fov_estimate.to_degrees(), "ratios": image_ratios,
                "input_indices": image_pattern_local.map(|i| sorted_indices[i])
            })); }''')
    text = insert(text, '                    let cat_largest = entry.largest_edge;', '''
                    let trace_target = trace_image && crate::research_trace::catalog_is_target(entry.star_indices.map(|i| self.star_catalog_ids[i as usize]));
                    if trace_target { crate::research_trace::emit("lookup", serde_json::json!({
                        "fov": fov_estimate.to_degrees(), "ids": entry.star_indices.map(|i| self.star_catalog_ids[i as usize]),
                        "implied_fov": (cat_largest / image_largest_edge * fov_estimate).to_degrees(),
                        "fov_error": config.fov_max_error_rad.map(|f| f.to_degrees())
                    })); }''')
    text = insert(text, '''                    let ratios_ok = (0..NUM_EDGE_RATIOS)
                        .all(|i| cat_ratios[i] > ratio_min[i] && cat_ratios[i] < ratio_max[i]);''', '''
                    if trace_target { crate::research_trace::emit("ratios", serde_json::json!({
                        "fov": fov_estimate.to_degrees(), "pass": ratios_ok, "image": image_ratios, "catalog": cat_ratios, "tolerance": p_max_err
                    })); }''')
    text = insert(text, '                    let matched_cat: [[f32; 3]; 4] = std::array::from_fn(|i| cat_vecs[i]);', '''
                    if trace_target { crate::research_trace::emit("pairing", serde_json::json!({
                        "image_input_indices": img_order.map(|i| sorted_indices[image_pattern_local[i]]),
                        "catalog_ids": cat_pat.map(|i| self.star_catalog_ids[i as usize]),
                        "fov_seed": fov_estimate.to_degrees(), "fov_refined": fov.to_degrees()
                    })); }''')
    text = insert(text, '                    // Determine parity from the rotation determinant.', '''
                    if trace_target { crate::research_trace::emit("rotation", serde_json::json!({
                        "det": rotation_matrix.det(),
                        "boresight": [rotation_matrix[(2,0)], rotation_matrix[(2,1)], rotation_matrix[(2,2)]]
                    })); }
                    crate::research_trace::set_active(trace_target);''')
    text = insert(text, '''                        star_vectors,
                    );

                    if prob_mismatch >= match_threshold'''.removesuffix('\n\n                    if prob_mismatch >= match_threshold'), '''
                    crate::research_trace::set_active(false);
                    if trace_target { crate::research_trace::emit("verification", serde_json::json!({
                        "fov": fov_estimate.to_degrees(), "matches": current_matches.len(),
                        "pairs": current_matches.iter().map(|&(i,c)| (sorted_indices[i], self.star_catalog_ids[c])).collect::<Vec<_>>(),
                        "prob_mismatch": prob_mismatch, "threshold": match_threshold,
                        "pass": prob_mismatch < match_threshold
                    })); }''')
    text = insert(text, '        // Limit to 2x the number of image centroids (like tetra3)', '''
        if crate::research_trace::active() { crate::research_trace::emit("verification_pool", serde_json::json!({
            "before": nearby_cam_positions.len(), "limit": 2 * match_centroid_count,
            "ids_before": nearby_cam_positions.iter().map(|&(i,_,_)| self.star_catalog_ids[i]).collect::<Vec<_>>(),
            "projected_before": nearby_cam_positions.iter().map(|&(i,x,y)| (self.star_catalog_ids[i],x,y)).collect::<Vec<_>>()
        })); }''')
    text = insert(text, '        if wcs_result.matches.len() < min_matches {', '''
            crate::research_trace::emit("refinement_reject", serde_json::json!({"matches": wcs_result.matches.len(), "minimum": min_matches}));''')
    return text


def prepare(out, registry, footprint=False):
    source = registry / 'src/solver/solve.rs'
    if digest(source) != SOLVE_SHA:
        raise ValueError('unexpected tetra3 solve.rs; review instrumentation against this version')
    project = out / 'harness'
    vendor = project / 'tetra3'
    if project.exists():
        raise ValueError('harness already exists; choose a fresh output directory')
    project.mkdir(parents=True)
    shutil.copytree(registry, vendor)
    # Upstream dev-only fixtures require uncached packages; they are not part of
    # the library under observation and are not built by this standalone caller.
    manifest = vendor / 'Cargo.toml'
    manifest.write_text(re.sub(r'(?ms)^\[dev-dependencies\.[^\]]+\]\n.*?(?=^\[|\Z)', '', manifest.read_text()))
    if footprint:
        subprocess.run(['patch', '--batch', '--fuzz=0', '-p1', '-i',
                        str(Path(__file__).with_name('tetra3_verify_footprint.patch').resolve())], cwd=vendor, check=True)
    (vendor / 'src/solver/solve.rs').write_text(instrument((vendor / 'src/solver/solve.rs').read_text()))
    with (vendor / 'src/lib.rs').open('a') as f:
        f.write('\npub mod research_trace;\n')
    with (vendor / 'Cargo.toml').open('a') as f:
        f.write('\n[dependencies.serde_json]\nversion = "=1.0.149"\n')
    shutil.copyfile(Path(__file__).with_name('tetra3_internal_trace.rs'), vendor / 'src/research_trace.rs')
    (project / 'src').mkdir()
    shutil.copyfile(Path(__file__).with_suffix('.rs'), project / 'src/main.rs')
    (project / 'Cargo.toml').write_text('''[package]
name = "starglyph-tetra3-internal-trace"
version = "0.0.0"
edition = "2021"
[workspace]
[dependencies]
tetra3 = { path = "tetra3", features = ["parallel"] }
serde_json = "=1.0.149"
''')
    shutil.copyfile(ROOT / 'prototype/Cargo.lock', project / 'Cargo.lock')
    write_json(out / 'instrumentation.json', dict(upstream_solve_sha256=digest(source), footprint=footprint,
        patched_solve_sha256=digest(vendor / 'src/solver/solve.rs'),
        source_hashes={str(p.relative_to(ROOT)): digest(p) for p in [Path(__file__).resolve(),
            Path(__file__).with_suffix('.rs').resolve(), Path(__file__).with_name('tetra3_internal_trace.rs').resolve()]},
        scope=('footprint patch plus observation' if footprint else 'observation only')
              + '; isolated copy, no production dependency/lockfile mutation'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    parser.add_argument('--registry', type=Path, required=True)
    parser.add_argument('--footprint', action='store_true')
    args = parser.parse_args()
    prepare(args.out, args.registry, args.footprint)


if __name__ == '__main__':
    main()
