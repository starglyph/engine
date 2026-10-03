import unittest

import numpy as np

from robustness_diagnose import recover


class RecoveryTests(unittest.TestCase):
    def test_one_detection_cannot_recover_two_probes(self):
        matches = recover(np.array([[10.,10.],[11.,10.]]), [{'x':10.,'y':10.}],100,100,100,100)
        self.assertEqual(len(matches),1)
        self.assertEqual(matches[0]['probe'],0)

    def test_half_pixel_resize_and_radius(self):
        matches = recover(np.array([[20.5,20.5],[90.,90.]]), [{'x':10.,'y':10.}],100,100,200,200)
        self.assertEqual(matches,[{'probe':0,'rank':0,'distance_px':0.}])
        self.assertEqual(recover(np.empty((0,2)),[],100,100,100,100),[])


if __name__ == '__main__':
    unittest.main()
