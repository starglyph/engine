#!/usr/bin/env python3
"""Observe one distributed quartet and a local control with unchanged tetra3."""
import argparse
from pathlib import Path
import shutil

from collection_run import ROOT, digest, write_json
from robustness_candidate_trace import command
from robustness_orion_fit import RID, previous, read
from robustness_reference_audit import selected
from tetra3_internal import prepare as prepare_harness, insert


def choose(trace, reviewed, prior):
    if prior['id'] != RID or prior['split'] != 'development' or prior['holdout']:
        raise ValueError('frozen development required')
    last = prior['attempts'][-2]
    if 'k=20 ' not in last['outcome']:
        raise ValueError('saved first dense candidate attempt required')
    retained = last['sweeps'][-1]['order']['retained']
    near = {}
    for s in reviewed['sources']:
        if s['nearest_detection_distance_px'] < 12:
            index = s['nearest_detection_index']
            if index in near:
                raise ValueError('ambiguous reviewed detection')
            near[index] = s
    distributed = sorted(i for i in retained if i in near)[:4]
    local = sorted(set(retained) & {p['detection_index'] for p in trace['candidates'][0]['pairs']})
    if len(distributed) != 4 or len(local) != 4 or any(i not in near for i in local):
        raise ValueError('two reviewed quartets required')
    return [dict(name=name, detection_indices=indices,
                 catalog_ids=[near[i]['hyg_id'] for i in indices],
                 reviewed_sources=[near[i] for i in indices])
            for name, indices in [('distributed', distributed), ('local_control', local)]]


def target_module(source, targets):
    anchor = '    ids == [73486, 74164, 76644, 78029]'
    if source.count(anchor) != 1:
        raise ValueError('existing target hook changed')
    arrays = ', '.join(str(sorted(t['catalog_ids'])) for t in targets)
    return source.replace(anchor, f'    [{arrays}].contains(&ids)')


def prepare(out, registry):
    selected()
    if out.exists():
        raise ValueError('fresh output required')
    paths = []
    for name in ('40-results', '41-trace', '41-results', '48-control-trace'):
        path = previous(name); check = previous(name[:2]+'-checks')
        if digest(path) != read(check)['artifact_hashes'][str(path.relative_to(ROOT))]:
            raise ValueError('published evidence changed: '+name)
        paths += [path, check]
    trace = read(previous('41-trace')); reviewed = read(previous('41-results'))
    prior = read(previous('48-control-trace')) | dict(id=RID, split='development', holdout=False)
    targets = choose(trace, reviewed, prior)
    detections = trace['candidates'][0]['detections']
    ids = [None]*len(detections)
    for t in targets:
        for index, ident in zip(t['detection_indices'], t['catalog_ids']):
            if ids[index] not in (None, ident):
                raise ValueError('conflicting quartet identities')
            ids[index] = ident
    dbs = read(previous('36-input'))['databases']
    known = read(previous('48-protocol'))['hashes']
    for path in [Path(p) for p in dbs]:
        if digest(path) != known[str(path.relative_to(ROOT))]:
            raise ValueError('database changed')
        paths.append(path)
    protected = read(previous('48-checks'))['protected_hashes']
    for name, sha in protected.items():
        if digest(ROOT/name) != sha:
            raise ValueError('protected input changed')
        paths.append(ROOT/name)
    prepare_harness(out, registry)
    project = out/'harness'; vendor = project/'tetra3'
    module = vendor/'src/research_trace.rs'
    module.write_text(target_module(module.read_text(), targets))
    inventory = Path(__file__).with_name('tetra3_pattern_inventory.rs')
    shutil.copy2(inventory, vendor/'src/research_pattern_inventory.rs')
    with (vendor/'src/lib.rs').open('a') as f:
        f.write('\npub mod research_pattern_inventory;\n')
    source = vendor/'src/solver/solve.rs'
    source.write_text(insert(source.read_text(), '            // Build list of candidate pattern keys, sorted by distance from image_key', '''
            if trace_image { crate::research_trace::emit("target_window", serde_json::json!({
                "fov":fov_estimate.to_degrees(), "input_indices":image_pattern_local.map(|i| sorted_indices[i]),
                "largest_edge":image_largest_edge,"ratio_min":ratio_min,"ratio_max":ratio_max,
                "key_min":key_min,"key_max":key_max,"tolerance":p_max_err
            })); }'''))
    caller = project/'src/main.rs'; code = caller.read_text()
    replacements = {
        '        if !matches!(name, "4080-default-control" | "4080-deep-control" | "positive-control") { continue; }':
            '        if name != "orion-control" { return Err("unexpected case".into()); }',
        '    let mut results = Vec::new();': '''    let targets: Vec<[i64;4]> = serde_json::from_value(input["target_ids"].clone())?;
    let inventory: Vec<_> = dbs.iter().map(|db| tetra3::research_pattern_inventory::inventory(db,&targets)).collect();
    let mut results = Vec::new();''',
        '''Ok(s) => json!({"matches":s.num_matches,"prob":s.prob,"catalog_ids":s.matched_catalog_ids,
                        "centroid_indices":s.matched_centroid_indices,"fov_rad":s.fov_rad,"crval_rad":s.crval_rad}),''':
            'Ok(s) => json!({"solution":s}),',
        'json!({"cases":results})': 'json!({"cases":results,"inventory":inventory})'
    }
    for before, after in replacements.items():
        if code.count(before) != 1:
            raise ValueError('caller anchor changed')
        code = code.replace(before, after)
    caller.write_text(code)
    manifest = project/'Cargo.toml'
    manifest.write_text(manifest.read_text().replace('serde_json = "=1.0.149"',
        'serde_json = { version = "=1.0.149", features = ["float_roundtrip"] }'))
    command(out, 'resolve', ['cargo', 'update', '--offline', '--manifest-path', str(manifest)], timeout=60)
    write_json(out/'input.json', dict(id=RID, split='development', holdout=False, databases=dbs,
        targets=targets, target_ids=[sorted(t['catalog_ids']) for t in targets], cases=[dict(name='orion-control',
            width=1600, height=1042, search=[dict(x=x, y=y, flux=f) for x,y,f in detections], search_ids=ids)]))
    paths += [Path(__file__).resolve(), inventory, out/'input.json', previous('48-protocol'), previous('48-checks'),
        previous('36-input'), ROOT/'prototype/Cargo.lock', *(project.rglob('*.rs')), *(project.rglob('Cargo.toml')), project/'Cargo.lock',
        *(Path(__file__).with_name(n) for n in ('tetra3_internal.py','tetra3_internal.rs','tetra3_internal_trace.rs',
            'collection_run.py','robustness_candidate_trace.py','robustness_orion_fit.py','robustness_reference_audit.py'))]
    write_json(out/'protocol.json', dict(iteration=49,id=RID,split='development',holdout=False,
        scope='observation only; inventory two frozen quartets and replay original48 tetra3 attempts',
        selection='four brightest reviewed survivors at saved k20 successful FOV; local control is intersection of same survivors and saved first tetra3 pairs',
        targets=targets,budget=dict(attempts=48,per_attempt_ms=2500,wall_s=120),
        criteria=['same48 attempt outcomes and full two solutions except solve_time_ms',
            'database inventory separate from lookup; absence of lookup alone never proves absence from database',
            'record every target pattern/window, FOV/ratio/verification rejection and unvisited stage',
            'conditional reviewed identities only; no new solve-rate, accepted WCS or threshold adjustment'],
        hashes={str(p.relative_to(ROOT)):digest(p) for p in sorted(set(paths))}))


def validate(out):
    p = read(out/'protocol.json')
    if (p['id'],p['split'],p['holdout']) != (RID,'development',False):
        raise ValueError('development only')
    selected()
    for name,sha in p['hashes'].items():
        if digest(ROOT/name) != sha:
            raise ValueError('frozen input changed: '+name)
    return p


def build(out):
    validate(out)
    command(out,'build',['cargo','build','--release','--offline','--locked','--manifest-path',str(out/'harness/Cargo.toml')],timeout=600)
    shutil.copy2(out/'harness/target/release/starglyph-tetra3-internal-trace',out/'observer')
    write_json(out/'binary.json',dict(sha256=digest(out/'observer'),protocol_sha256=digest(out/'protocol.json')))


def run(out):
    p = validate(out)
    if read(out/'binary.json') != dict(sha256=digest(out/'observer'),protocol_sha256=digest(out/'protocol.json')):
        raise ValueError('binary changed')
    if (out/'run.json').exists():
        raise ValueError('fresh observation required')
    command(out,'run',[str(out/'observer'),str(out/'input.json'),str(out/'raw.json')],timeout=p['budget']['wall_s'])
    validate(out)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=('prepare','build','run','validate'))
    p.add_argument('--out-dir',type=Path,required=True);p.add_argument('--registry',type=Path)
    a=p.parse_args()
    if a.stage=='prepare':
        if a.registry is None:p.error('--registry required')
        prepare(a.out_dir.resolve(),a.registry)
    else:globals()[a.stage](a.out_dir.resolve())
