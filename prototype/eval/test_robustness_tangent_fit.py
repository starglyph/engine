"""Check correction sign, nonzero prior and pair exclusion for tangent diagnostics."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import robustness_tangent_fit as fit


class TangentFitTests(unittest.TestCase):
    def test_known_parameter_error_is_cancelled_with_scaled_columns(self):
        jac=np.vstack((np.diag([.001,1,10,100,1000]),np.diag([.002,3,4,5,6])))
        error=np.array([2.,-3.,4.,-.5,.01])
        delta,info=fit.correction(jac,jac@error,np.zeros(5),0.)
        np.testing.assert_allclose(delta,-error,rtol=0,atol=1e-10)
        self.assertEqual(info['augmented_rank'],5)

    def test_nonzero_prior_residual_affects_step(self):
        delta,_=fit.correction(np.eye(5),np.zeros(5),[0,0,0,0,2],.5)
        np.testing.assert_allclose(delta,[0,0,0,0,-.2],atol=1e-12)

    def test_excluded_xy_cannot_change_its_predicted_correction(self):
        rng=np.random.default_rng(31);jac=rng.normal(size=(18,5));r=rng.normal(size=(9,2))
        original=fit.leave_one_probe_out(jac,r,np.zeros(5),0.)
        changed=r.copy();changed[4]+=[1000,-2000]
        after=fit.leave_one_probe_out(jac,changed,np.zeros(5),0.)
        np.testing.assert_array_equal(original[4]['delta_parameters'],after[4]['delta_parameters'])
        np.testing.assert_array_equal(original[4]['predicted_change_xy_px'],after[4]['predicted_change_xy_px'])
        self.assertFalse(np.allclose(original[0]['delta_parameters'],after[0]['delta_parameters']))

    def test_degenerate_system_and_nonfinite_prior_fail(self):
        with self.assertRaisesRegex(ValueError,'rank deficient'):
            fit.correction(np.ones((10,5)),np.ones(10),np.zeros(5),0.)
        with self.assertRaisesRegex(ValueError,'non-finite'):
            fit.correction(np.eye(5),np.ones(5),np.zeros(5),float('nan'))

    def test_holdout_rejected_before_input_access(self):
        with tempfile.TemporaryDirectory() as folder:
            out=Path(folder)
            (out/'protocol.json').write_text(json.dumps(dict(id=fit.RID,split='holdout',holdout=True)))
            with patch.object(fit,'selected',side_effect=AssertionError('must not read')):
                with self.assertRaisesRegex(ValueError,'development'):fit.validate(out)


if __name__=='__main__':unittest.main()
