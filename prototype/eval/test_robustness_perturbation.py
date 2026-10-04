"""Freeze cell membership and distinguish coherent shifts from noise variance."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from robustness_perturbation import DELTAS, agreement_gate, cell_selection, perturbation_vector, trial_metrics, validate


class PerturbationTests(unittest.TestCase):
    def test_cells_partition_only_training_and_keep_empty_cells(self):
        f=dict(initial=dict(width=300,height=300),sources=[
            dict(xy=[10,10],outside_both_inputs=False),dict(xy=[20,20],outside_both_inputs=True),
            dict(xy=[210,210],outside_both_inputs=False)])
        old=copy.deepcopy(f);cells=cell_selection(f)
        self.assertEqual(f,old);self.assertEqual(len(cells),9)
        self.assertEqual(cells[0]['training_indices'],[0]);self.assertEqual(cells[-1]['training_indices'],[1])
        self.assertEqual(sum(bool(c['training_indices']) for c in cells),2)

    def test_signed_shifts_are_symmetric_and_coordinates_are_scoped(self):
        self.assertEqual(len(DELTAS),8)
        for d in DELTAS:
            self.assertIn([-v for v in d],DELTAS)
            v=perturbation_vector(3,[1],d).reshape(3,2)
            np.testing.assert_array_equal(v[[0,2]],0.)
            np.testing.assert_array_equal(v[1],d)
        with self.assertRaisesRegex(ValueError,'membership'):perturbation_vector(2,[0,0],[1,0])
        with self.assertRaisesRegex(ValueError,'membership'):perturbation_vector(2,[2],[1,0])

    def test_coherent_response_uses_signed_sum_not_independent_noise_norm(self):
        propagation=np.array([[.5,0.,.5,0.],[0.,.5,0.,.5]])
        coherent=propagation@perturbation_vector(2,[0,1],[1.,0.])
        np.testing.assert_array_equal(coherent,[1.,0.])
        self.assertNotEqual(coherent[0],np.linalg.norm(propagation[0]))

    def test_motion_mismatch_gate_and_invalid_projections(self):
        actual=np.array([[.2,0.],[np.nan,np.nan]])
        m=trial_metrics(actual,np.array([[.19,0.],[0.,0.]]),np.array([2.,np.nan]),np.array([True,False]))
        self.assertAlmostEqual(m['mismatch_rms_px'],.01);self.assertTrue(agreement_gate(m))
        m['mismatch_rms_px']=.03;self.assertFalse(agreement_gate(m))
        bad=trial_metrics(actual,np.zeros((2,2)),np.array([2.,np.nan]),np.array([True,True]))
        self.assertEqual(bad['invalid_count'],1);self.assertFalse(agreement_gate(bad))
        self.assertFalse(agreement_gate(dict(count=0,invalid_count=0)))

    def test_holdout_rejected_before_hash_access(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp);(p/'protocol.json').write_text(json.dumps(dict(split='holdout',holdout=True)))
            with patch('robustness_perturbation.digest',side_effect=AssertionError('must not read')):
                with self.assertRaisesRegex(ValueError,'development'):validate(p)


if __name__=='__main__':unittest.main()
