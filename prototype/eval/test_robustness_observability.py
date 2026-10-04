"""Linear diagnostics must exclude prior noise and retain spatial partitions."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from robustness_observability import linear_system, spatial_influence, propagation_summary, validate
from robustness_pair_report import sensitivity


class ObservabilityTests(unittest.TestCase):
    def test_two_priors_are_fixed_not_noisy_observations(self):
        jac=np.vstack((np.eye(6),np.diag([0.,0.,0.,0.,2.,3.])[-2:]))
        info,response=linear_system(jac,6)
        self.assertEqual(response.shape,(6,6))
        np.testing.assert_allclose(response,np.diag([1.,1.,1.,1.,1/5,1/10]),atol=1e-14)
        self.assertEqual(info['data_rank'],6)

    def test_column_units_do_not_change_predicted_response(self):
        rng=np.random.default_rng(42);jac=rng.normal(size=(14,6));probes=rng.normal(size=(8,6))
        info,a=linear_system(jac,12)
        scale=np.array([1e-3,3.,5.,1e3,2.,.5]);other,b=linear_system(jac*scale,12)
        np.testing.assert_allclose(probes@a,(probes*scale)@b,atol=1e-12)
        self.assertAlmostEqual(info['data_condition'],other['data_condition'])

    def test_one_prior_agrees_with_existing_helper(self):
        rng=np.random.default_rng(17);jac=rng.normal(size=(21,5));probes=rng.normal(size=(8,5))
        _,response=linear_system(jac,20)
        expected=sensitivity(jac,np.vstack((probes,np.zeros((1,5)))))
        factors=np.sqrt(np.sum((probes@response).reshape(-1,2,20)**2,axis=(1,2)))
        np.testing.assert_allclose(factors,expected['probe_amplification'],atol=1e-12)

    def test_spatial_noise_partition_and_empty_probe_groups(self):
        prop=np.eye(4);cells={'a':np.array([True,False]),'b':np.array([False,True]),'empty':np.array([False,False])}
        result=spatial_influence(prop,np.array([True,True]),cells)
        self.assertAlmostEqual(sum(v['noise_energy_share'] for v in result.values()),1.)
        self.assertEqual(result['empty']['noise_energy_share'],0.)
        self.assertEqual(result['a']['noise_energy_share'],.5)
        self.assertEqual(propagation_summary(prop,np.array([False,False])),{'count':0})
        empty=spatial_influence(prop,np.array([False,False]),cells)
        self.assertTrue(all(v['noise_energy_share'] is None for v in empty.values()))

    def test_singular_system_does_not_report_false_certainty(self):
        jac=np.ones((6,2));info,response=linear_system(jac,4)
        self.assertEqual(info['augmented_rank'],1);self.assertIsNone(response);self.assertIsNone(info['data_condition'])
        with self.assertRaisesRegex(ValueError,'unobserved'):linear_system(np.zeros((4,2)),4)
        with self.assertRaisesRegex(ValueError,'invalid'):linear_system(np.full((4,2),np.nan),4)

    def test_holdout_rejected_before_hash_access(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp);(p/'protocol.json').write_text(json.dumps(dict(split='holdout',holdout=True)))
            with patch('robustness_observability.digest',side_effect=AssertionError('must not read')):
                with self.assertRaisesRegex(ValueError,'development'):validate(p)


if __name__=='__main__':unittest.main()
