import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import robustness_left_source_audit as audit


class LeftSourceAuditTests(unittest.TestCase):
    def test_nearest_catalogue_search_keeps_faint_component(self):
        def star(i,ra,mag):return dict(id=str(i),ra=str(ra),dec='0',mag=str(mag),proper='',bf='')
        rows=[star(1,0,4),star(2,.001,12),star(3,.01,7)]
        found=audit.neighbours(rows,1)
        self.assertEqual([p['hyg_id'] for p in found['nearest_all_magnitudes']],[2,3])
        self.assertEqual([p['hyg_id'] for p in found['pattern']],[3])

    def test_target_translation_preserves_pattern_vectors(self):
        cam=dict(width=100,height=80,focal_px=100,k1=0,world_to_camera=[[0,1,0],[0,0,1],[1,0,0]])
        target=[0,0];others=[[.1,.2],[.2,-.1]]
        before=audit.project(cam,others);actual=audit.relative_positions(cam,target,[32,27],others)
        np.testing.assert_allclose(actual[1]-actual[0],before[1]-before[0])
        np.testing.assert_allclose(audit.relative_positions(cam,target,[32,27],[target]),[[32,27]])

    def test_channel_offsets_and_clipping_are_measured_without_recentering(self):
        yy,xx=np.mgrid[:80,:80];rgb=np.empty((80,80,3),dtype=np.uint8)
        for i,offset in enumerate((-1,0,1)):
            rgb[:,:,i]=np.rint(10+200*np.exp(-((xx-40-offset)**2+(yy-40)**2)/2))
        original=rgb.copy();m=audit.image_metrics(rgb,[40,40])
        self.assertEqual(m['clipped_pixels_any_channel'],0)
        for band,expected in zip('RGB',(39,40,41)):
            self.assertAlmostEqual(m['positions'][band+'_r8'][0],expected,places=4)
        np.testing.assert_array_equal(rgb,original)
        rgb[40,40,0]=255
        self.assertEqual(audit.image_metrics(rgb,[40,40])['clipped_pixels_any_channel'],1)

    def test_holdout_guard_precedes_any_image_or_catalogue_read(self):
        with tempfile.TemporaryDirectory() as folder:
            out=Path(folder)
            (out/'protocol.json').write_text(json.dumps(dict(id=audit.RID,split='holdout',holdout=True,source_ids=[24,29])))
            with patch.object(audit,'selected',side_effect=AssertionError('must not read')):
                with self.assertRaisesRegex(ValueError,'development'):audit.validate(out)


if __name__=='__main__':unittest.main()
