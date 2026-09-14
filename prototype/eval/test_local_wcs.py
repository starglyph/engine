import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from PIL import Image

from compare_local_wcs import compare, project
from solve_local_wcs import sha256, solve_one


class LocalWcsTests(unittest.TestCase):
    def test_tan_reference_uses_top_down_zero_based_pixels_and_ra_wrap(self):
        width, height, focal = 1000, 600, 900
        camera = {"width": width, "height": height, "focal_px": focal, "k1": 0,
                  "world_to_camera": [[0, -1, 0], [0, 0, 1], [1, 0, 0]]}
        wcs = WCS(naxis=2)
        wcs.wcs.crpix = [width / 2 + 1, height / 2 + 1]
        wcs.wcs.crval = [0, 0]
        wcs.wcs.cd = np.diag([-180 / np.pi / focal] * 2)
        wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        pixels = np.array([[0, 0], [width - 1, height - 1], [100, 500], [500, 300]])
        world = wcs.all_pix2world(pixels, 0)
        self.assertGreater(world[1, 0], 300)  # RA wraps through zero.
        np.testing.assert_allclose(project(camera, world), pixels, atol=1e-9)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "reference.wcs"
            fits.PrimaryHDU(header=wcs.to_header()).writeto(path)
            reference = {"source_sha256": "source", "width": width, "height": height,
                         "pixel_convention": "top_left_zero_based", "status": "solved_candidate",
                         "review_status": "pending", "wcs_file": str(path), "wcs_sha256": sha256(path),
                         "reference_points": [{"x": x, "y": y} for x, y in pixels]}
            artifact = {"source_sha256": "source", "width": width, "height": height,
                        "pixel_convention": "top_left_zero_based", "report": {"status": "solved"},
                        "camera": camera}
            result = compare(reference, artifact)
            self.assertEqual(result["review_status"], "pending")
            self.assertLess(result["full_field_wcs_difference"]["rms_px"], 1e-9)
            artifact["source_sha256"] = "another image"
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                compare(reference, artifact)
            artifact["source_sha256"] = "source"
            artifact["width"] += 1
            with self.assertRaisesRegex(ValueError, "width mismatch"):
                compare(reference, artifact)

    def test_rerun_cannot_accept_stale_wcs_and_preserves_sensor_range(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "sensor.tiff"
            pixels = np.full((32, 48), 1000, dtype=np.uint16)
            pixels[10, 20] = 4000
            Image.fromarray(pixels).save(source)
            executable = root / "no-solution"
            executable.write_text("#!/bin/sh\nexit 0\n")
            executable.chmod(0o700)
            attempt = root / "output/sensor/downsample-4"
            attempt.mkdir(parents=True)
            (attempt / "field.solved").write_bytes(b"\x01")
            (attempt / "field.wcs").write_text("old result")
            args = SimpleNamespace(solve_field=str(executable), config=None, cpu_limit=1)
            result = solve_one(source, root / "output", args)
            self.assertEqual(result["status"], "unresolved")
            self.assertFalse((attempt / "field.wcs").exists())
            np.testing.assert_array_equal(fits.getdata(root / "output/sensor/input.fits"), pixels)
            json.dumps(result, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
