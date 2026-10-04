"""Residual signs, orthogonal decomposition and fixed spatial groups."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from robustness_residual_fields import agreement, decompose, groups, validate, vector_metrics


class ResidualFieldTests(unittest.TestCase):
    def test_radial_and_clockwise_tangential_signs(self):
        xy=np.array([[75.,50.],[50.,75.],[25.,50.],[50.,25.]])
        radial=np.array([[2.,0.],[0.,2.],[-2.,0.],[0.,-2.]])
        tangent=np.array([[0.,3.],[-3.,0.],[0.,-3.],[3.,0.]])
        _,rr,tt,valid=decompose(xy,radial+tangent,100,100)
        np.testing.assert_allclose(rr,2.)
        np.testing.assert_allclose(tt,3.)
        self.assertTrue(valid.all())
        np.testing.assert_allclose(rr**2+tt**2,np.sum((radial+tangent)**2,axis=1))

    def test_center_has_no_arbitrary_radial_direction(self):
        rho,rr,tt,valid=decompose([[50.,50.]],[[3.,4.]],100,100)
        result=vector_metrics(np.array([[3.,4.]]),rr,tt,valid,np.hypot(50,50))
        self.assertEqual(result['rms_px'],5.)
        self.assertEqual(result['radial_valid_count'],0)
        self.assertNotIn('radial_energy_fraction',result)
        self.assertEqual(rho[0],0.)

    def test_empty_groups_and_partition_boundaries(self):
        xy=np.array([[0.,0.],[50.,50.],[99.,99.]])
        rho,*_=decompose(xy,np.zeros_like(xy),100,100)
        masks=groups(xy,[True,False,True],100,100,rho)
        self.assertTrue(masks['outside_both_inputs'].tolist()==[True,False,True])
        np.testing.assert_equal(sum(m for k,m in masks.items() if k.startswith('cell_')),1)
        np.testing.assert_equal(sum(m for k,m in masks.items() if k.startswith('radius_')),1)
        self.assertEqual(vector_metrics(np.empty((0,2)),np.array([]),np.array([]),np.array([],dtype=bool),1),{'count':0})

    def test_field_comparison_uses_declared_mask_and_ignores_tiny_angles(self):
        a=np.array([[2.,0.],[.1,0.],[9.,0.]])
        b=np.array([[-2.,0.],[0.,.1],[9.,0.]])
        r=agreement(a,b,np.array([True,True,False]))
        self.assertEqual(r['count'],2)
        self.assertEqual(r['angular_sample_count'],1)
        self.assertEqual(r['median_cosine'],-1.)

    def test_holdout_protocol_rejected_before_input_reads(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)
            (out/'protocol.json').write_text(json.dumps(dict(split='holdout',holdout=True)))
            with self.assertRaisesRegex(ValueError,'development'):validate(out)


if __name__=='__main__':unittest.main()
