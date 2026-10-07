//! Read-only inventory using the dependency's own f32 pattern mathematics.
use crate::solver::pattern::{
    compute_edge_ratios, compute_pattern_key, compute_pattern_key_hash, compute_sorted_edge_angles,
};
use crate::SolverDatabase;
use serde_json::{json, Value};

pub fn same_ids(mut actual: [i64; 4], mut expected: [i64; 4]) -> bool {
    actual.sort_unstable();
    expected.sort_unstable();
    actual == expected
}

pub fn inventory(db: &SolverDatabase, targets: &[[i64; 4]]) -> Value {
    let mut rows = Vec::new();
    for &target in targets {
        let indices: Vec<_> = target
            .iter()
            .map(|id| db.star_catalog_ids.iter().position(|i| i == id))
            .collect();
        let mut entries = Vec::new();
        for (slot, entry) in db.pattern_catalog.entries.iter().enumerate() {
            if !entry.is_empty()
                && same_ids(
                    entry.star_indices.map(|i| db.star_catalog_ids[i as usize]),
                    target,
                )
            {
                entries.push(json!({"slot":slot,"star_indices":entry.star_indices,
                    "catalog_ids":entry.star_indices.map(|i| db.star_catalog_ids[i as usize]),
                    "largest_edge":entry.largest_edge,"key_hash16":entry.key_hash}));
            }
        }
        let metrics = if let [Some(a), Some(b), Some(c), Some(d)] = indices.as_slice() {
            let vectors = [*a, *b, *c, *d].map(|i| db.star_vectors[i]);
            let edges = compute_sorted_edge_angles(&vectors);
            let ratios = compute_edge_ratios(&edges);
            let key = compute_pattern_key(&ratios, db.props.pattern_bins);
            json!({"vectors":vectors,"edges_rad":edges,"ratios":ratios,"key":key,
                "key_hash":compute_pattern_key_hash(&key,db.props.pattern_bins)})
        } else {
            Value::Null
        };
        rows.push(json!({"catalog_ids":target,"star_indices":indices,"entries":entries,"metrics":metrics}));
    }
    json!({"properties":db.props,"table_slots":db.pattern_catalog.entries.len(),"targets":rows})
}

#[cfg(test)]
mod tests {
    use super::same_ids;

    #[test]
    fn accepts_only_permutations_of_complete_identity() {
        assert!(same_ids([4, 2, 1, 3], [1, 2, 3, 4]));
        assert!(!same_ids([1, 2, 3, 5], [1, 2, 3, 4]));
        assert!(!same_ids([1, 2, 3, 3], [1, 2, 3, 4]));
    }
}
