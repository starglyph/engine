"""Protect diagnostic source selection, coordinate frames and holdout boundary."""
import copy
from pathlib import Path
import tempfile
import unittest

from collection_run import write_json
from robustness_orion_pattern_correction import make_input, probes, validate, previous, read, RID


class PatternCorrectionTests(unittest.TestCase):
    def test_unused_probes_disjoint_and_not_residual_selected(self):
        sources=read(previous('40-results'))['sources']
        candidates=read(previous('40-candidates'))['frames'][0]['points']
        selection=read(previous('45-nonlinear-results'))['selection']
        chosen=probes(sources,candidates,selection)
        ids=[s['id'] for q in chosen for s in q['sources']]
        self.assertEqual(len(ids),12);self.assertEqual(len(set(ids)),12)
        self.assertFalse(set(ids)&set(selection['fit_ids']))
        changed=copy.deepcopy(sources)
        for s in changed:s['predicted_xy']={'arbitrary':[1e20,-1e20]}
        again=probes(list(reversed(changed)),candidates,selection)
        self.assertEqual([[s['id'] for s in q['sources']] for q in chosen],[[s['id'] for s in q['sources']] for q in again])
        with self.assertRaises(ValueError):probes(sources,candidates,selection|dict(check_ids=selection['fit_ids']))

    def test_fixed_models_shared_across_methods_and_origin_control(self):
        inp=make_input()
        self.assertEqual(len(inp['cases']),17)
        self.assertEqual([a['origin_offset'] for a in inp['arms']],[0.,.5,.5,.5])
        self.assertEqual(inp['arms'][0]['camera'],inp['arms'][1]['camera'])
        for q in ('check_quartet_1','check_quartet_2','check_quartet_3'):
            rows=[c for c in inp['cases'] if c['quartet']==q]
            self.assertEqual(len(rows),3)
            self.assertTrue(all(not c['fit_members'] for c in rows))
            self.assertEqual(rows[0]['catalog_ids'],rows[2]['catalog_ids'])
        self.assertEqual(len(inp['fovs_rad']),len(inp['visited_fovs_deg'])+2)
        self.assertEqual(inp['fixed_indices']['seven'],len(inp['fovs_rad'])-1)

    def test_holdout_refused_before_hash_access(self):
        with tempfile.TemporaryDirectory() as folder:
            out=Path(folder);write_json(out/'protocol.json',dict(id=RID,split='holdout',holdout=True))
            with self.assertRaisesRegex(ValueError,'development'):validate(out)


if __name__=='__main__':unittest.main()
