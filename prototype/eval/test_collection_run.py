"""Exercise timeout/report semantics without catalogs, images or external solvers."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from collection_run import bounded, eligible, digest, write_json, resume_rows
from export_collection import read_run


class CollectionRunTests(unittest.TestCase):
    def test_resume_preserves_prefix_and_records_executor_change(self):
        prior = {"runner_sha256": "old", "input_count": 2, "inputs": [{"id": "a"}, {"id": "b"}], "cpu_limit_s": 30}
        current = {**prior, "runner_sha256": "new"}
        summary = {"plan_sha256": "plan", "input_count": 2, "completed": 1, "frames": [{"id": "a"}]}
        self.assertEqual(resume_rows(prior, current, summary, "plan"), [{"id": "a"}])
        with self.assertRaisesRegex(ValueError, "settings"):
            resume_rows(prior, {**current, "cpu_limit_s": 60}, summary, "plan")

    def test_resume_rejects_missing_prefix_or_changed_plan(self):
        plan = {"runner_sha256": "old", "input_count": 2, "inputs": [{"id": "a"}, {"id": "b"}]}
        summary = {"plan_sha256": "plan", "input_count": 2, "completed": 1, "frames": [{"id": "b"}]}
        with self.assertRaisesRegex(ValueError, "prefix"):
            resume_rows(plan, plan, summary, "plan")
        summary["frames"] = [{"id": "a"}]
        with self.assertRaisesRegex(ValueError, "prefix"):
            resume_rows(plan, plan, summary, "modified")

    def test_partial_run_cannot_be_exported_as_complete(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            inputs = [{"id": "a"}, {"id": "b"}]
            write_json(root / "plan.json", {"manifest_sha256": "frozen", "inputs": inputs})
            write_json(root / "summary.json", {"plan_sha256": digest(root / "plan.json"),
                       "completed": 1, "input_count": 2, "frames": [{"id": "a"}]})
            with self.assertRaisesRegex(ValueError, "incomplete"):
                read_run(root, "frozen", inputs)

    def test_duplicate_rows_cannot_replace_missing_id(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            inputs = [{"id": "a"}, {"id": "b"}]
            write_json(root / "plan.json", {"manifest_sha256": "frozen", "inputs": inputs})
            write_json(root / "summary.json", {"plan_sha256": digest(root / "plan.json"),
                       "completed": 2, "input_count": 2, "frames": [{"id": "a"}, {"id": "a"}]})
            with self.assertRaisesRegex(ValueError, "duplicated"):
                read_run(root, "frozen", inputs)

    def test_timeout_is_distinct_from_solver_failure(self):
        with tempfile.TemporaryDirectory() as folder:
            result = bounded([sys.executable, "-c", "import time; time.sleep(10)"],
                             Path(folder) / "process.log", .1)
            self.assertEqual(result["status"], "wall_timeout")
            self.assertNotIn("solve_status", result)

    def test_nonzero_process_is_not_an_unsolved_image(self):
        with tempfile.TemporaryDirectory() as folder:
            result = bounded([sys.executable, "-c", "raise SystemExit(7)"],
                             Path(folder) / "process.log", 3)
            self.assertEqual(result["status"], "process_error")
            self.assertEqual(result["exit_code"], 7)

    def test_negative_is_run_by_starglyph_but_not_wcs(self):
        rec = {"track": "stress", "research": {"negative": True}}
        self.assertTrue(eligible(rec, "starglyph")[0])
        self.assertFalse(eligible(rec, "wcs")[0])

    def test_scene_is_never_silently_counted_as_solver_failure(self):
        rec = {"track": "scene", "research": {}}
        self.assertFalse(eligible(rec, "starglyph")[0])
        self.assertFalse(eligible(rec, "wcs")[0])


if __name__ == "__main__":
    unittest.main()
