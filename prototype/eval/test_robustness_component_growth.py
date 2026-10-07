import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from robustness_component_growth import exact_source, unique_anchor, veto_reason
import robustness_component_growth as growth


class ComponentGrowthTests(unittest.TestCase):
    def trace(self):
        return dict(point=[3.1,4.2],distance_to_component=(.1**2+.2**2)**.5,
            component=dict(bounds=[1,1,8,8],centroid=[3.1,4.2],rank_before_top_k=7,outcome='selected'))

    def test_unique_pixel_and_ambiguous_tie(self):
        t=self.trace();self.assertEqual(unique_anchor(t),[3,4])
        t['point']=[3.5,4];t['distance_to_component']=.5
        self.assertIsNone(unique_anchor(t))

    def test_box_alone_does_not_establish_membership(self):
        t=self.trace();t['distance_to_component']=.12345
        self.assertIsNone(unique_anchor(t))
        t['component']=None;self.assertIsNone(unique_anchor(t))

    def test_source_rank_and_centroid_both_required(self):
        t=self.trace();source=dict(x=3.1,y=4.2)
        self.assertTrue(exact_source(t,source,7))
        self.assertFalse(exact_source(t,source,8))
        self.assertFalse(exact_source(t,dict(x=3.11,y=4.2),7))

    def test_veto_requires_shared_pixel_and_lower_threshold(self):
        high=self.trace();high['distance_to_component']=0
        low=copy.deepcopy(high);low['component']['outcome']='elongation'
        hs=dict(sigma=1,sigma_convolved=1,background_median=1,statistics_pixels=100,threshold=2.5)
        ls={**hs,'threshold':2.0};source=dict(x=3.1,y=4.2)
        args=[high,low,[3,4],source,7,hs,ls]
        self.assertEqual(veto_reason(*args),'veto_elongation')
        low['distance_to_component']=.1
        self.assertEqual(veto_reason(*args),'retain_unproven_shared_pixel')
        low['distance_to_component']=0;ls['threshold']=4
        self.assertEqual(veto_reason(*args),'retain_threshold_not_lower')
        ls['threshold']=2;ls['sigma']=2
        with self.assertRaises(ValueError):veto_reason(*args)

    def test_no_veto_from_other_rejection_or_unknown_anchor(self):
        high=self.trace();high['distance_to_component']=0;low=copy.deepcopy(high)
        hs=dict(sigma=1,sigma_convolved=1,background_median=1,statistics_pixels=100,threshold=2.5)
        ls={**hs,'threshold':2.0};source=dict(x=3.1,y=4.2)
        low['component']['outcome']='concentration'
        self.assertEqual(veto_reason(high,low,[3,4],source,7,hs,ls),'retain_lower_component')
        self.assertEqual(veto_reason(high,low,None,source,7,hs,ls),'retain_uncertain_default_association')

    def test_process_metadata_does_not_overwrite_anchor_mapping(self):
        def fake_trace(folder,stage,inp):
            # Reproduce command()'s real side effect, independently of trace data.
            (folder/f'{stage}.json').write_text(json.dumps(dict(exit_code=0,elapsed_s=0)))
            rows=[]
            for quantile,blob in [(False,False),(True,False),(False,True)]:
                for width,height in [(1600,1117),(2772,1935)]:
                    for tier in ['default','deep']:
                        rows.append(dict(quantile=quantile,blob_concentration=blob,width=width,height=height,
                            tier=tier,result=dict(detections=[dict(x=3.1,y=4.2)]),
                            probes=[{**self.trace(),'component':{**self.trace()['component'],'rank_before_top_k':0}}
                                    for _ in inp['probes']]))
            return dict(tiers=rows)
        with tempfile.TemporaryDirectory() as td:
            out=Path(td)
            protocol=dict(inputs=[dict(id=growth.IDS[0],image='unused.png',width=2772,height=1935)])
            with patch.object(growth,'validate',return_value=protocol), patch.object(growth,'trace_run',side_effect=fake_trace):
                growth.run(out)
            folder=out/growth.IDS[0]
            self.assertEqual(json.loads((folder/'anchors.json').read_text())['exit_code'],0)
            self.assertEqual(len(json.loads((folder/'anchor-mapping.json').read_text())['anchors']),2)


if __name__=='__main__':unittest.main()
