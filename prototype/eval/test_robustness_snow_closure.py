import unittest
from robustness_snow_closure import sky_coverage


class SnowClosureTests(unittest.TestCase):
    def test_coverage_does_not_create_verified_stars(self):
        result=sky_coverage([dict(x=10,y=10),dict(x=90,y=40)],100,100)
        self.assertEqual(result['occupied_sky_cells'],2)
        self.assertAlmostEqual(result['conservative_sky']['width_fraction'],.8)
        self.assertIsNone(result['catalogue_verified_star_coverage'])

    def test_sky_boundary_is_strict_and_foreground_is_excluded(self):
        result=sky_coverage([dict(x=50,y=45),dict(x=20,y=80),dict(x=10,y=44.9)],100,100)
        self.assertEqual(result['all_detections'],3)
        self.assertEqual(result['conservative_sky']['count'],1)
        self.assertEqual(sum(sum(r) for r in result['sky_grid_4x2']),1)

    def test_empty_and_invalid_coordinates(self):
        result=sky_coverage([],100,100)
        self.assertIsNone(result['conservative_sky']['bounds_xyxy'])
        self.assertEqual(result['occupied_sky_cells'],0)
        with self.assertRaises(ValueError):sky_coverage([dict(x=-1,y=10)],100,100)


if __name__=='__main__':unittest.main()
