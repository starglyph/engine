import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import robustness_tangential_fit as fit


class NonlinearTangentialTests(unittest.TestCase):
    def test_generated_lm_keeps_budget_and_unregularized_extra_terms(self):
        source=(fit.ROOT/'prototype/crates/starglyph-core/src/solve.rs').read_text()
        generated=fit.generated_optimizer(source)
        for text in ('for _ in 0..30','for _ in 0..10','params.extend([0., 0.])','let k1_weight = fixed_weight','trial[4] = trial[4].clamp'):
            self.assertIn(text,generated)
        self.assertNotIn('trial[5] =',generated)
        with self.assertRaisesRegex(ValueError,'optimizer'):
            fit.generated_optimizer(source.replace('for _ in 0..30','for _ in 0..31'))

    def test_paired_models_share_starts_matches_and_exact_saved_weight(self):
        args=[fit.read(fit.prior(n)) for n in ('34-input','30-input','30-results')];before=copy.deepcopy(args)
        inp=fit.build_cases(*args)
        self.assertEqual(args,before);self.assertEqual(len(inp['cases']),18)
        for i in range(0,18,3):
            replay,five,seven=inp['cases'][i:i+3]
            self.assertEqual(five['initial'],seven['initial']);self.assertEqual(five['matches'],seven['matches'])
            self.assertEqual(five['prior_weight'],2.5);self.assertEqual(seven['prior_weight'],2.5)
            self.assertEqual(replay['expected'],five['initial']);self.assertFalse(five['extra']);self.assertTrue(seven['extra'])
        args[0]['cases'][0]['fit_pairs'][0]['xy'][0]+=1
        with self.assertRaisesRegex(ValueError,'fixed22'):fit.build_cases(*args)

    def test_projection_adds_known_image_oriented_terms(self):
        c=dict(width=100,height=100,focal_px=100,k1=0,world_to_camera=np.eye(3),p1=.002,p2=-.003)
        w=np.array([[.2,-.3,1.] ]);w/=np.linalg.norm(w,axis=1)[:,None]
        np.testing.assert_allclose(fit.projection(c,w),[[69.961,80.026]],atol=1e-10)

    def test_holdout_rejected_before_reading_other_files(self):
        with tempfile.TemporaryDirectory() as folder:
            out=Path(folder);(out/'protocol.json').write_text(json.dumps(dict(id=fit.RID,split='holdout',holdout=True,names=fit.NAMES)))
            with patch.object(fit,'selected',side_effect=AssertionError('must not read')):
                with self.assertRaisesRegex(ValueError,'development'):fit.validate(out)


if __name__=='__main__':unittest.main()
