import unittest

import numpy as np
from PIL import Image, ImageDraw

from automatic_sky_compare import rasterize
from automatic_sky_mask import algorithm_settings, estimate
from gradient_sky_mask import estimate_gradient


def gradient_scene(width=160, height=160, cloud=False):
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    values = np.stack([8+.12*x+.55*y, 18+.16*x+.50*y, 30+.20*x+.45*y], axis=-1)
    if cloud:
        values += (35*np.exp(-((y-75)/22)**2))[..., None]
    return Image.fromarray(np.clip(values, 0, 255).astype('uint8'))


class GradientSkyTests(unittest.TestCase):
    def test_linear_illumination_gradient_keeps_sky_where_constant_rgb_truncates(self):
        image = gradient_scene()
        polygon, _ = estimate_gradient(image)
        self.assertTrue(rasterize(polygon, 160, 160).all())
        old, _ = estimate(image)
        self.assertFalse(rasterize(old, 160, 160)[100:].any())

    def test_broad_cloud_and_nonlinear_gradient_are_followed(self):
        polygon, _ = estimate_gradient(gradient_scene(cloud=True))
        self.assertTrue(rasterize(polygon, 160, 160).all())

    def test_sharp_roof_and_tree_stop_tracking_after_gradient(self):
        image = gradient_scene()
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 125, 159, 159), fill=(210, 160, 90))
        draw.rectangle((60, 80, 90, 159), fill=(1, 2, 1))
        polygon, _ = estimate_gradient(image)
        mask = rasterize(polygon, 160, 160)
        self.assertTrue(mask[60, 75])
        self.assertFalse(mask[85, 75])
        self.assertTrue(mask[110, 20])
        self.assertFalse(mask[130:].any())

    def test_rejected_foreground_must_not_train_the_model(self):
        image = gradient_scene()
        ImageDraw.Draw(image).rectangle((0, 65, 159, 159), fill=(230, 220, 210))
        polygon, _ = estimate_gradient(image)
        self.assertFalse(rasterize(polygon, 160, 160)[70:].any())

    def test_sparse_stars_do_not_terminate_columns(self):
        image = gradient_scene(cloud=True)
        draw = ImageDraw.Draw(image)
        for x, y in ((15, 30), (70, 90), (140, 50)):
            draw.point((x, y), fill='white')
        polygon, _ = estimate_gradient(image)
        self.assertTrue(rasterize(polygon, 160, 160).all())

    def test_portrait_dimensions_and_determinism(self):
        image = gradient_scene(width=90, height=240)
        a = estimate_gradient(image)
        self.assertEqual(a, estimate_gradient(image))
        self.assertTrue(rasterize(a[0], 90, 240).all())

    def test_algorithm_selection_explicit_and_unknown_rejected(self):
        self.assertIs(algorithm_settings('rgb_skyline_v1')[1], estimate)
        self.assertIs(algorithm_settings('rgb_gradient_v2')[1], estimate_gradient)
        with self.assertRaisesRegex(ValueError, 'unknown mask algorithm'):
            algorithm_settings('typo')


if __name__ == '__main__':
    unittest.main()
