"""Coordinate-sign, prior, rank, split and spatial-transfer contracts."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import robustness_tangential_basis as basis


class TangentialBasisTests(unittest.TestCase):
    def test_known_axes_and_cross_term_use_image_y_down(self):
        cam=dict(world_to_camera=np.eye(3),focal_px=100,k1=.9)
        actual=basis.tangential_columns(cam,[[0,0,1],[1,0,1],[0,1,1],[.2,-.3,1]])
        # Row order X/Y. y_up=+1 becomes image y=-1; cross terms keep this sign.
        expected=[[0,0],[0,0],[0,300],[100,0],[0,100],[300,0],[12,21],[31,12]]
        np.testing.assert_allclose(actual,expected,atol=1e-12)
        np.testing.assert_allclose(basis.tangential_columns(cam,[[.2,.3,1]]),[[-12,21],[31,-12]],atol=1e-12)
        cam['k1']=-.1
        np.testing.assert_array_equal(actual,basis.tangential_columns(cam,[[0,0,1],[1,0,1],[0,1,1],[.2,-.3,1]]))

    def test_seven_parameter_error_recovered_with_unchanged_nonzero_prior(self):
        rng=np.random.default_rng(34)
        j=np.vstack((rng.normal(size=(44,5))*[.1,1,10,100,1000],[0,0,0,0,3]))
        extra=rng.normal(size=(44,2))*100
        augmented=np.column_stack((j,np.vstack((extra,np.zeros((1,2))))))
        error=np.array([.3,-.5,.2,.01,.004,.001,-.003]);res=augmented@error
        result=basis.fit_step(j,res,extra)
        np.testing.assert_allclose(result['delta_parameters'],-error,atol=1e-10)
        self.assertLess(result['fit22_rms_px'],1e-10)
        self.assertLess(abs(result['prior_residual']),1e-10)
        self.assertEqual(result['data_information']['data_rank'],7)

    def test_dependent_new_columns_fail_instead_of_claiming_identification(self):
        rng=np.random.default_rng(35);j=rng.normal(size=(45,5));j[-1]=[0,0,0,0,2]
        with self.assertRaisesRegex(ValueError,'rank deficient'):
            basis.fit_step(j,np.ones(45),j[:-1,:2])

    def test_edge_regression_rejects_large_strict_improvement(self):
        before=dict(strict_unused=dict(count=15,rms_px=10),left_10pct=dict(count=2,rms_px=2),bottom_10pct=dict(count=0))
        after=dict(strict_unused=dict(count=15,rms_px=5),left_10pct=dict(count=2,rms_px=2.6),bottom_10pct=dict(count=0))
        verdict=basis.transfer_verdict(before,after)
        self.assertFalse(verdict['passed']);self.assertIn('left_10pct',verdict['regressions'])
        after['left_10pct']['rms_px']=2.4
        self.assertTrue(basis.transfer_verdict(before,after)['passed'])

    def test_holdout_and_back_facing_directions_fail(self):
        with self.assertRaisesRegex(ValueError,'outside camera'):
            basis.tangential_columns(dict(world_to_camera=np.eye(3),focal_px=100),[[0,0,-1]])
        with tempfile.TemporaryDirectory() as folder:
            out=Path(folder)
            (out/'protocol.json').write_text(json.dumps(dict(id=basis.RID,split='holdout',holdout=True,names=basis.NAMES)))
            with patch.object(basis,'selected',side_effect=AssertionError('must not read')):
                with self.assertRaisesRegex(ValueError,'development'):basis.validate(out)


if __name__=='__main__':unittest.main()
