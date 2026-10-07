"""Check edge identities, ratio decomposition and signed endpoint attribution."""
from pathlib import Path
import tempfile
import unittest

import numpy as np

from collection_run import write_json
from robustness_pattern_edge_geometry import angles, signature, ratio_terms, attribute, identified_ratios
from robustness_orion_pattern_edges import build_input, validate, RID


class EdgeGeometryTests(unittest.TestCase):
    def test_known_spherical_edges(self):
        edges=angles([[1,0,0],[0,1,0],[0,0,1],[-1,0,0]])
        np.testing.assert_allclose(edges,[np.pi/2,np.pi/2,np.pi,np.pi/2,np.pi/2,np.pi/2],atol=1e-15)

    def test_sorted_signature_does_not_preserve_edge_identity(self):
        xy=np.array([[10.,20.],[100.,-40.],[-250.,20.],[40.,400.]])
        _,order,r=signature(xy,1000)
        _,swapped,s=signature(xy[[1,0,2,3]],1000)
        np.testing.assert_allclose(r,s,atol=1e-15)
        self.assertFalse(np.array_equal(order,swapped))
        self.assertGreater(np.max(np.abs(identified_ratios(xy[[1,0,2,3]],1000,order)-r)),.01)

    def test_common_scale_cancels_between_numerator_and_denominator(self):
        catalog=np.array([1.,2.,3.,4.,5.,6.]);observed=catalog*1.1
        numerator,denominator=ratio_terms(observed,catalog,np.arange(6))
        self.assertGreater(np.max(np.abs(numerator)),.01)
        np.testing.assert_allclose(numerator+denominator,0,atol=1e-15)

    def test_signed_contributions_sum_and_split(self):
        ideal=np.array([[10.,20.],[100.,-40.],[-250.,20.],[40.,400.]])
        displacement=np.array([[.1,-.05],[0.,0.],[-.03,.07],[.02,.04]])
        _,order,_=signature(ideal,1000)
        a=attribute(ideal,ideal+displacement,1000,order)
        np.testing.assert_allclose(np.array(a['radial_per_source'])+a['transverse_per_source'],a['per_source'],atol=1e-15)
        np.testing.assert_allclose(np.sum(a['per_source'],axis=1),a['actual_change'],atol=1e-7)
        self.assertLess(a['max_step_change'],1e-9)
        np.testing.assert_allclose(np.array(a['per_source'])[:,1],0,atol=0)

    def test_all_prior_cases_and_origin_retained(self):
        inp=build_input()
        self.assertEqual(len(inp['cases']),17)
        self.assertEqual(inp['focus'],['check_quartet_1','local_control'])
        self.assertTrue(all(len(c['arms'])==4 and len(c['sources'])==4 for c in inp['cases']))
        check=next(c for c in inp['cases'] if c['name']=='check_quartet_1/external_centroid')
        self.assertFalse(any(s['fit'] for s in check['sources']))

    def test_holdout_and_invalid_geometry_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            out=Path(folder);write_json(out/'protocol.json',dict(id=RID,split='holdout',holdout=True))
            with self.assertRaisesRegex(ValueError,'development'):validate(out)
        with self.assertRaises(ValueError):signature(np.zeros((4,2)),1000)
        with self.assertRaises(ValueError):signature(np.ones((4,2)),0)


if __name__=='__main__':unittest.main()
