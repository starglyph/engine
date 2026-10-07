"""Verify16-pair adaptation and transfer gates without changing the old harness."""
import unittest

import numpy as np

from robustness_orion_tangential import fit16, transfer
from robustness_tangential_basis import fit_step


class OrionTangentialTests(unittest.TestCase):
    @staticmethod
    def system():
        rng=np.random.default_rng(45)
        j=np.vstack((rng.normal(size=(32,5))*[.1,1,10,100,1000],[0,0,0,0,3.6]))
        extra=rng.normal(size=(32,2))*100
        full=np.column_stack((j,np.vstack((extra,np.zeros((1,2))))))
        delta=np.array([.3,-.5,.2,.01,.004,.001,-.003])
        return j,extra,full@delta,delta

    def test_seven_parameters_and_nonzero_prior_recovered(self):
        j,extra,r,delta=self.system();result=fit16(j,r,extra)
        np.testing.assert_allclose(result['delta_parameters'],-delta,atol=1e-10)
        np.testing.assert_allclose(result['residuals_working_px'],0,atol=1e-10)
        self.assertEqual(result['data_information']['data_rank'],7)

    def test_old22_harness_agrees_when_extra_rows_are_zero(self):
        j,extra,r,_=self.system()
        old_j=np.vstack((j[:-1],np.zeros((12,5)),j[-1]))
        old_r=np.r_[r[:-1],np.zeros(12),r[-1]]
        old_extra=np.vstack((extra,np.zeros((12,2))))
        for x,y in [(None,None),(extra,old_extra)]:
            a,b=fit16(j,r,x),fit_step(old_j,old_r,y)
            np.testing.assert_allclose(a['delta_parameters'],b['delta_parameters'],atol=1e-10)
            self.assertAlmostEqual(a['objective'],b['objective'],places=8)

    def test_dependent_extra_columns_fail(self):
        j,_,r,_=self.system()
        with self.assertRaisesRegex(ValueError,'rank deficient'):fit16(j,r,j[:-1,:2])

    def test_wrong_pair_shape_rejected(self):
        with self.assertRaises(ValueError):fit16(np.zeros((45,5)),np.zeros(45))

    def test_grid_regression_rejects_aggregate_gain(self):
        before={'check31/all_reviewed':dict(count=31,rms_px=50),'check31/cell_0_0':dict(count=4,rms_px=5),
            'check31/empty':dict(count=0),'fit16/all_reviewed':dict(count=16,rms_px=1)}
        after={'check31/all_reviewed':dict(count=31,rms_px=20),'check31/cell_0_0':dict(count=4,rms_px=5.6),
            'check31/empty':dict(count=0),'fit16/all_reviewed':dict(count=16,rms_px=2)}
        result=transfer(before,after)
        self.assertFalse(result['passed']);self.assertIn('check31/cell_0_0',result['regressions'])
        after['check31/cell_0_0']['rms_px']=5.4
        self.assertTrue(transfer(before,after)['passed'])

    def test_changed_check_count_rejected(self):
        with self.assertRaises(ValueError):
            transfer({'check31/all_reviewed':dict(count=31,rms_px=50)},
                {'check31/all_reviewed':dict(count=30,rms_px=20)})


if __name__=='__main__':unittest.main()
