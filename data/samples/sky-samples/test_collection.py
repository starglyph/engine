"""Offline integrity tests: no network or real image fixtures required."""
import contextlib
import copy
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

import fetch_sample as fetcher
from validate_collection import validate


def encoded(image, format="PNG", **kwargs):
    stream = io.BytesIO()
    image.save(stream, format=format, **kwargs)
    return stream.getvalue()


class FetchTests(unittest.TestCase):
    def record(self, raw):
        clean, pixels, size = fetcher.normalize(raw)
        return {"id": "test", "file": "images/test.png", "license": "CC0",
                "orig_sha256": fetcher.sha256(raw), "clean_sha256": fetcher.sha256(clean),
                "width": size[0], "height": size[1],
                "processing": {"mode": "oriented_png_v1", "pixel_sha256": pixels}}

    def test_orientation_and_metadata_stripping(self):
        im = Image.new("RGB", (3, 2))
        im.putdata([(n * 30, 0, 0) for n in range(6)])
        exif = Image.Exif()
        exif[274], exif[315] = 6, "private author"
        raw = encoded(im, exif=exif)
        clean, pixels, size = fetcher.normalize(raw)
        result = Image.open(io.BytesIO(clean))
        self.assertEqual(size, (2, 3))
        self.assertFalse(result.getexif())
        self.assertEqual(result.tobytes(), im.transpose(Image.Transpose.ROTATE_270).tobytes())
        self.assertEqual(fetcher.normalize(raw), (clean, pixels, size))

    def test_tiff_retains_sensor_range(self):
        im = Image.frombytes("I;16", (2, 1), b"\x00\x01\xff\xff")
        clean, _, _ = fetcher.normalize(encoded(im, "TIFF"))
        decoded = Image.open(io.BytesIO(clean))
        self.assertEqual(list(decoded.getdata()), [256, 65535])

    def test_png_alpha_is_preserved(self):
        im = Image.new("RGBA", (2, 1), (20, 40, 60, 128))
        clean, _, _ = fetcher.normalize(encoded(im))
        self.assertEqual(Image.open(io.BytesIO(clean)).tobytes(), im.tobytes())

    def test_wrong_source_never_overwrites_image(self):
        raw = encoded(Image.new("RGB", (2, 2), "red"))
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            target = root / "images/test.png"
            target.parent.mkdir()
            target.write_bytes(b"previous accepted file")
            with patch.object(fetcher, "source_bytes", return_value=b"changed source"):
                with self.assertRaisesRegex(ValueError, "source SHA"):
                    fetcher.fetch(self.record(raw), root)
            self.assertEqual(target.read_bytes(), b"previous accepted file")

    def test_wrong_output_never_overwrites_image(self):
        raw = encoded(Image.new("RGB", (2, 2), "red"))
        rec = self.record(raw)
        rec["clean_sha256"] = "0" * 64
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            target = root / rec["file"]
            target.parent.mkdir()
            target.write_bytes(b"previous")
            with patch.object(fetcher, "source_bytes", return_value=raw):
                with self.assertRaisesRegex(ValueError, "output SHA"):
                    fetcher.fetch(rec, root)
            self.assertEqual(target.read_bytes(), b"previous")

    def test_success_is_reproducible_from_original_cache(self):
        raw = encoded(Image.new("RGB", (2, 3), "blue"))
        rec = self.record(raw)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with patch.object(fetcher, "source_bytes", return_value=raw), contextlib.redirect_stdout(io.StringIO()):
                fetcher.fetch(rec, root)
            expected = (root / rec["file"]).read_bytes()
            (root / rec["file"]).unlink()
            with patch.object(fetcher, "curl", side_effect=AssertionError("network")), contextlib.redirect_stdout(io.StringIO()):
                fetcher.fetch(rec, root)
            self.assertEqual((root / rec["file"]).read_bytes(), expected)

    def test_unknown_id_exits_nonzero(self):
        result = subprocess.run([sys.executable, str(fetcher.HERE / "fetch_sample.py"), "not-an-id"],
                                capture_output=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(b"unknown IDs", result.stderr)

    def test_legacy_recipe_unchanged(self):
        im = Image.new("RGB", (5, 3), "green")
        raw = encoded(im)
        expected = io.BytesIO()
        im.resize((3, 2), Image.Resampling.LANCZOS).save(expected, format="JPEG", quality=92)
        self.assertEqual(fetcher.process(raw, 3, 2), expected.getvalue())

    def test_failed_cli_download_is_nonzero(self):
        with tempfile.TemporaryDirectory() as folder:
            manifest = Path(folder) / "manifest.json"
            rec = self.record(encoded(Image.new("RGB", (1, 1))))
            manifest.write_text(json.dumps([rec]))
            with patch.object(sys, "argv", ["fetch", "--manifest", str(manifest)]), \
                    patch.object(fetcher, "fetch", side_effect=OSError("offline")), \
                    contextlib.redirect_stdout(io.StringIO()):
                self.assertTrue(fetcher.main())


class ManifestTests(unittest.TestCase):
    def reviewed_record(self):
        return copy.deepcopy(next(r for r in json.loads((fetcher.HERE / "manifest.json").read_text())
                                  if r.get("research", {}).get("collection") == "robustness-2026-10"))

    def test_declared_license_cannot_hide_restrictive_license_url(self):
        row = self.reviewed_record()
        row["license_url"] = "https://creativecommons.org/licenses/by-nc/4.0/"
        self.assertTrue(any("license URL" in e for e in validate([row], Path("."))))

    def test_author_cannot_leak_by_changing_group_id(self):
        first = self.reviewed_record()
        second = copy.deepcopy(first)
        second.update(id="other", file="images/other.png", orig_sha256="0" * 64)
        second["processing"]["pixel_sha256"] = "1" * 64
        second["research"].update(group_id="different", session_group="different",
                                  split="holdout" if first["research"]["split"] == "development" else "development")
        self.assertTrue(any("author" in e and "leaks" in e for e in validate([first, second], Path("."))))

    def test_new_image_cannot_duplicate_historical_original(self):
        row = self.reviewed_record()
        old = {"id": "old", "file": "images/old.jpg", "orig_sha256": row["orig_sha256"]}
        self.assertTrue(any("duplicate source" in e for e in validate([row, old], Path("."))))

    def test_legacy_records_remain_readable(self):
        self.assertEqual(validate([{"id": "old", "file": "images/old.jpg"}], Path(".")), [])

    def test_duplicate_id_is_rejected(self):
        rows = [{"id": "same", "file": name} for name in ("a.jpg", "b.jpg")]
        self.assertTrue(any("duplicate ID" in e for e in validate(rows, Path("."))))

    def test_incomplete_new_record_cannot_pass_as_reviewed(self):
        row = {"id": "new", "file": "images/new.png", "research": {"collection": "robustness-2026-10"}}
        errors = validate([row], Path("."))
        self.assertTrue(any("license not verified" in e for e in errors))
        self.assertTrue(any("missing visual review" in e for e in errors))

    def test_acceptance_does_not_treat_small_sample_as_complete(self):
        self.assertTrue(any("60–80" in e for e in validate([], Path("."), acceptance=True)))

    def test_results_cannot_change_inputs_or_lose_candidates(self):
        from validate_collection import validate_results
        rec = self.reviewed_record()
        row = dict(id=rec["id"], source_sha256=rec["clean_sha256"], track=rec["track"],
                   split=rec["research"]["split"], negative=rec["research"]["negative"],
                   external_wcs=dict(status="solved_candidate", review_status="pending", wcs_sha256="a" * 64))
        report = dict(provenance=dict(manifest_sha256="frozen"), frames=[row], accepted_ground_truth_count=0)
        row["starglyph"] = dict(status="completed", solve_status="failed")
        report["summaries"] = [dict(split=split, track=track,
            count=int((split, track) == (row["split"], row["track"])),
            starglyph_status={"failed": 1} if (split, track) == (row["split"], row["track"]) else {},
            external_wcs_status={"solved_candidate": 1} if (split, track) == (row["split"], row["track"]) else {})
            for split in ("development", "holdout") for track in ("solver", "stress", "scene")]
        review = dict(frames=[dict(id=rec["id"], source_sha256=rec["clean_sha256"],
                                  wcs_sha256="a" * 64, review_status="pending", diagnostic_note="Unverified")])
        self.assertEqual(validate_results([rec], "frozen", report, review), [])
        row["source_sha256"] = "changed"
        self.assertTrue(any("input metadata" in e for e in validate_results([rec], "frozen", report, review)))
        row["source_sha256"] = rec["clean_sha256"]
        review["frames"] = []
        self.assertTrue(any("each external candidate" in e for e in validate_results([rec], "frozen", report, review)))
        report["frames"] = []
        self.assertTrue(any("missing" in e for e in validate_results([rec], "frozen", report, review)))
        self.assertTrue(any("manifest hash" in e for e in validate_results([rec], "changed", report, review)))


if __name__ == "__main__":
    unittest.main()
