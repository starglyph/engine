"""Protect the frozen visual subset and reuse of the production fit harness."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import robustness_single31_fit as fit


class Single31FitTests(unittest.TestCase):
    def fixture(self):
        pairs=[dict(world=[float(i),0.,1.],xy=[float(i),2.]) for i in range(31)]
        points=[dict(id=i,detection_index=i+1,world=m['world'],review_center=m['xy'],hyg_id=100+i,
            label='visible_source' if i<22 else 'blend') for i,m in enumerate(pairs)]
        path=dict(steps=[dict(camera=dict(roll_deg=1.)),dict(camera=dict(roll_deg=2.),
            input=dict(matches=pairs,original_detection_indices=list(range(1,32))))])
        return path,points

    def test_only_frozen_labels_change_fit_membership(self):
        path,points=self.fixture();saved=copy.deepcopy((path,points))
        a,b=fit.build_cases(path,points)
        self.assertEqual((path,points),saved)
        self.assertEqual(a['matches'][:22],b['matches'])
        self.assertEqual(a['initial'],b['initial'])
        self.assertIsNone(a['prior_weight']);self.assertIsNone(b['prior_weight'])
        self.assertEqual(a['expected'],path['steps'][1]['camera']);self.assertIsNone(b['expected'])

    def test_changed_identity_or_subset_size_rejected(self):
        for change in ['world','index','label']:
            path,points=self.fixture()
            if change=='world':points[0]['world']=[99.,0.,1.]
            elif change=='index':points[0]['detection_index']=999
            else:points[0]['label']='ambiguous'
            with self.assertRaises(ValueError):fit.build_cases(path,points)

    def test_rust_harness_changes_only_exact_frame_guard(self):
        source=Path(fit.__file__).with_name('pair_refit.rs').read_text()
        modified=fit.adapted_harness(source)
        self.assertEqual(modified.replace(fit.RID,'wm_r_143159342'),source)
        with self.assertRaises(ValueError):fit.adapted_harness('unexpected source')

    def test_holdout_guard_precedes_hash_reads(self):
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory)
            (out/'protocol.json').write_text(json.dumps(dict(id=fit.RID,split='holdout',holdout=True,cases=fit.NAMES,hashes={'x':'sha'})))
            with patch.object(fit,'digest',side_effect=AssertionError('must not read')):
                with self.assertRaisesRegex(ValueError,'development'):fit.validate(out)


if __name__=='__main__':
    unittest.main()
