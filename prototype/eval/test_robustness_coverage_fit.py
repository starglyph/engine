"""Protect equal counts, coordinate-only selection and shared excluded probes."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import robustness_coverage_fit as fit


class CoverageFitTests(unittest.TestCase):
    def test_selection_ignores_residuals_and_input_order(self):
        singles,probes,_=fit.inputs()
        chosen=fit.selection(singles,probes,3648,5472)
        singles=copy.deepcopy(singles[::-1]);probes=copy.deepcopy(probes[::-1])
        for p in singles+probes:p['residual']=1e9;p['mag']=-999
        self.assertEqual(chosen,fit.selection(singles,probes,3648,5472))
        self.assertEqual(chosen['added_probe_ids'],[20,1,6])
        self.assertEqual(chosen['donor_cell'],[0,1])

    def test_same_count_initial_prior_and_retained_coordinates(self):
        singles,probes,alternate=fit.inputs()
        chosen=fit.selection(singles,probes,3648,5472)
        before=copy.deepcopy((singles,probes,alternate))
        data=fit.cases(singles,probes,alternate,chosen)
        self.assertEqual(before,(singles,probes,alternate))
        self.assertEqual(len(data['cases']),7)
        for c in data['cases']:
            self.assertEqual(len(c['matches']),22)
            self.assertEqual(c['initial'],data['cases'][0]['initial'])
            self.assertIsNone(c['prior_weight'])
        for i in (1,3,5):
            a,b=data['cases'][i:i+2]
            self.assertEqual([m for m in a['matches'] if m['pair_id'] not in chosen['removed_pair_ids']],b['matches'][:19])
            self.assertEqual([m['probe_id'] for m in b['matches'][19:]],chosen['added_probe_ids'])

    def test_both_arms_all_centroids_and_identities_excluded(self):
        probes=[dict(source_id=i,hyg_id=i,xy=xy) for i,xy in enumerate([[10.,10.],[100.,100.],[200.,200.],[300.,300.],[500.,500.]])]
        cases=[dict(matches=[dict(hyg_id=0,xy=[10.,10.])]),dict(matches=[dict(hyg_id=1,xy=[100.,100.])]),
               dict(matches=[dict(hyg_id=99,xy=[209.,200.]),dict(hyg_id=3,xy=[999.,999.])])]
        old=[dict(outside_all40_inputs=True) for _ in probes]
        masks=fit.masks_for(probes,cases,old)
        np.testing.assert_array_equal(masks['common_unused'],[False,False,False,False,True])
        for name,mask in masks.items():
            self.assertFalse(mask[:4].any(),name)

    def test_holdout_rejected_before_binary_validation(self):
        with tempfile.TemporaryDirectory() as folder:
            out=Path(folder)
            (out/'protocol.json').write_text(json.dumps(dict(id=fit.RID,split='holdout',holdout=True)))
            with patch.object(fit,'validate_fit28',side_effect=AssertionError('must not read')):
                with self.assertRaisesRegex(ValueError,'development'):fit.validate(out)


if __name__=='__main__':unittest.main()
