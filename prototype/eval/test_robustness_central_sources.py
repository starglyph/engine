"""Protect the diagnostic intervention, pixel convention and frozen split."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from collection_run import digest, write_json
from robustness_central_replay import promote
from robustness_central_sources import RID, probe_row, validate


class CentralSourcesTests(unittest.TestCase):
    def test_promotion_changes_only_membership_order_and_rank(self):
        ds=[dict(x=i+.25,y=i+.75,flux=100-i,rank=i)for i in range(20)]
        before=copy.deepcopy(ds);ranks=[10,12,14,16,18]
        result=promote(ds,ranks,8)
        self.assertEqual(len(result),8)
        self.assertEqual([r['rank']for r in result],list(range(8)))
        self.assertEqual([r['flux']for r in result],[ds[i]['flux']for i in ranks+[0,1,2]])
        self.assertEqual([r['x']for r in result],[ds[i]['x']for i in ranks+[0,1,2]])
        self.assertEqual(ds,before)

    def test_promotion_rejects_duplicates_selected_or_missing_sources(self):
        ds=[dict(rank=i)for i in range(20)]
        for ranks in [[10,12,14,16,16],[0,12,14,16,18],[10,12,14,16,20],[10,12,14,16]]:
            with self.assertRaisesRegex(ValueError,'distinct sources below top-K'):
                promote(ds,ranks,8)

    def test_probe_offset_uses_pixel_centers_and_anisotropic_scale(self):
        meta=dict(id='P01',visual_class='bright_compact',xy=[100.,200.])
        point=[(100+.5)*.25-.5,(200+.5)*.5-.5]
        trace=dict(point=point,distance_to_component=1.,component=dict(outcome='top_k',centroid=point))
        row=probe_row(meta,trace,.25,.5)
        self.assertEqual(row['centroid_original_xy'],[100.,200.])
        self.assertEqual(row['centroid_offset_original_px'],0.)
        trace['component']=None
        self.assertEqual(probe_row(meta,trace,.25,.5)['outcome'],'no_component_in_radius')
        self.assertNotIn('centroid_offset_original_px',probe_row(meta,trace,.25,.5))

    def test_holdout_rejected_before_image_access(self):
        with tempfile.TemporaryDirectory() as folder:
            out=Path(folder);write_json(out/'protocol.json',dict(id=RID,split='holdout',holdout=True))
            with patch('robustness_central_sources.selected') as select:
                with self.assertRaisesRegex(ValueError,'development only'):validate(out)
                select.assert_not_called()

    def test_changed_input_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            out=Path(folder);source=out/'input';source.write_text('before')
            write_json(out/'protocol.json',dict(id=RID,split='development',holdout=False,hashes={str(source):digest(source)}))
            source.write_text('after')
            with patch('robustness_central_sources.selected'):
                with self.assertRaisesRegex(ValueError,'frozen input changed'):validate(out)
