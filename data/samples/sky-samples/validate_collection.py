#!/usr/bin/env python3
"""Offline validation of the additive robustness collection contract."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
from urllib.parse import urlparse

from fetch_sample import pixel_sha256

COLLECTION = "robustness-2026-10"
ALLOWED = {"CC0", "CC BY 2.0", "CC BY 3.0", "CC BY 4.0"}


def validate(records, root, check_files=False, acceptance=False):
    errors = []
    ids, files, pixels = set(), set(), set()
    sources = {r["orig_sha256"] for r in records if r.get("orig_sha256")
               and r.get("research", {}).get("collection") != COLLECTION}
    groups, authors, series = defaultdict(set), Counter(), Counter()
    author_splits, session_splits = defaultdict(set), defaultdict(set)
    selected = []
    for rec in records:
        rid = rec.get("id", "<missing>")
        def require(condition, message):
            if not condition:
                errors.append(f"{rid}: {message}")
        require(rid not in ids, "duplicate ID")
        require(rec.get("file") not in files, "duplicate file")
        ids.add(rid)
        files.add(rec.get("file"))
        if rec.get("research", {}).get("collection") != COLLECTION:
            continue
        selected.append(rec)
        research, review = rec["research"], rec.get("license_review", {})
        relative = Path(rec.get("file", ""))
        require(not relative.is_absolute() and ".." not in relative.parts, "unsafe file path")
        require(relative.stem == rid, "filename must match ID")
        require(rec.get("license") in ALLOWED, "license is not allowed")
        license_path = {"CC0": "/publicdomain/zero/1.0", "CC BY 2.0": "/licenses/by/2.0",
                        "CC BY 3.0": "/licenses/by/3.0", "CC BY 4.0": "/licenses/by/4.0"}.get(rec.get("license"))
        url = urlparse(rec.get("license_url", ""))
        require(url.hostname == "creativecommons.org" and license_path is not None
                and (url.path == license_path or url.path.startswith(license_path + "/")),
                "license URL does not match declared license")
        require(rec.get("track") in ("solver", "scene", "stress"), "invalid track")
        require(review.get("status") == "verified", "license not verified")
        for key in ("checked_at", "evidence_url", "evidence_sha256", "basis", "reviewer"):
            require(bool(review.get(key)), f"missing license_review.{key}")
        require(bool(re.fullmatch("[0-9a-f]{64}", review.get("evidence_sha256", ""))),
                "invalid license evidence checksum")
        for key in ("author", "credit", "attribution_text", "page_url", "license_url", "download_url"):
            require(bool(rec.get(key)), f"missing {key}")
        require("copy exact" not in rec.get("attribution_text", "").lower(), "placeholder attribution")
        for key in ("orig_sha256", "clean_sha256"):
            require(bool(re.fullmatch("[0-9a-f]{64}", rec.get(key, ""))), f"invalid {key}")
        original = rec.get("orig_sha256")
        require(original not in sources, "duplicate source bytes")
        sources.add(original)
        process = rec.get("processing", {})
        require(process.get("mode") == "oriented_png_v1", "missing lossless normalization recipe")
        digest = process.get("pixel_sha256", "")
        require(bool(re.fullmatch("[0-9a-f]{64}", digest)), "missing decoded-pixel checksum")
        require(digest not in pixels, "duplicate normalized pixels")
        pixels.add(digest)
        require(research.get("split") in ("development", "holdout"), "invalid split")
        require(bool(research.get("group_id")), "missing provenance group")
        groups[research.get("group_id")].add(research.get("split"))
        authors[rec.get("author")] += 1
        series[research.get("session_group")] += 1
        author_splits[rec.get("author")].add(research.get("split"))
        session_splits[research.get("session_group")].add(research.get("split"))
        require(bool(research.get("session_group")), "missing conservative session group")
        require(bool(research.get("visual_review")), "missing visual review")
        require(bool(research.get("projection_basis")), "missing projection evidence")
        require(bool(research.get("risks")), "missing risk annotation")
        for risk in research.get("risks", []):
            require(risk.get("id") in {f"R{i:02d}" for i in range(1, 25)}, "unknown research risk")
            require(risk.get("basis") in ("visual", "source", "metadata", "hypothesis", "test_purpose"),
                    "invalid risk evidence type")
            require(bool(risk.get("note")), "risk without evidence")
        if check_files:
            path = root / relative
            require(path.is_file(), "image is missing")
            if path.is_file():
                from PIL import Image
                require(hashlib.sha256(path.read_bytes()).hexdigest() == rec["clean_sha256"], "image hash mismatch")
                with Image.open(path) as image:
                    require(image.size == (rec["width"], rec["height"]), "dimensions mismatch")
                    require(pixel_sha256(image) == digest, "pixel checksum mismatch")
                    require(not image.getexif(), "normalized image retains EXIF")
    for group, splits in groups.items():
        if len(splits) > 1:
            errors.append(f"{group}: provenance group leaks between splits")
    for kind, mapping in (("author", author_splits), ("session", session_splits)):
        for group, splits in mapping.items():
            if len(splits) > 1:
                errors.append(f"{kind} {group}: leaks between splits")
    for author, count in authors.items():
        if count > 10:
            errors.append(f"{author}: exceeds 10-image author cap")
    for group, count in series.items():
        if count > 4:
            errors.append(f"{group}: exceeds four-image session cap")
    if acceptance:
        count = len(selected)
        if not 60 <= count <= 80:
            errors.append(f"collection has {count} new images; expected 60–80")
        if sum(r["track"] == "solver" for r in selected) * 3 < count * 2:
            errors.append("fewer than two thirds are single-frame solver inputs")
        if sum(r["research"].get("negative", False) for r in selected) < 6:
            errors.append("fewer than six negative examples")
        holdout = sum(r["research"]["split"] == "holdout" for r in selected)
        if count and not .25 <= holdout / count <= .4:
            errors.append("holdout must contain 25–40% after grouping")
    return errors


def validate_results(records, manifest_sha, report, review):
    """Check portable measurements against the frozen input and diagnostic review."""
    errors = []
    selected = {r["id"]: r for r in records
                if r.get("research", {}).get("collection") == COLLECTION}
    if report.get("provenance", {}).get("manifest_sha256") != manifest_sha:
        errors.append("results: frozen manifest hash mismatch")
    rows = report.get("frames", [])
    ids = [r["id"] for r in rows]
    if len(ids) != len(set(ids)) or set(ids) != set(selected):
        errors.append("results: missing, extra or duplicate IDs")
    candidates = {}
    for row in rows:
        rec = selected.get(row["id"])
        if rec is None:
            continue
        expected = {"source_sha256": rec["clean_sha256"], "track": rec["track"],
                    "split": rec["research"]["split"], "negative": rec["research"]["negative"]}
        if any(row.get(k) != v for k, v in expected.items()):
            errors.append(f"{row['id']}: results input metadata mismatch")
        if row.get("external_wcs", {}).get("status") == "solved_candidate":
            candidates[row["id"]] = row
            if row["external_wcs"].get("review_status") != "pending":
                errors.append(f"{row['id']}: candidate must remain pending")
    if report.get("accepted_ground_truth_count") != 0:
        errors.append("results: this collection has no accepted ground truth")
    summaries = []
    for split in ("development", "holdout"):
        for track in ("solver", "stress", "scene"):
            group = [r for r in rows if r.get("split") == split and r.get("track") == track]
            summaries.append(dict(split=split, track=track, count=len(group),
                starglyph_status=dict(Counter(r.get("starglyph", {}).get("solve_status",
                    r.get("starglyph", {}).get("status", "missing")) for r in group)),
                external_wcs_status=dict(Counter(r.get("external_wcs", {}).get("status", "missing") for r in group))))
    if report.get("summaries") != summaries:
        errors.append("results: aggregate counts differ from per-image outcomes")
    reviews = review.get("frames", [])
    review_ids = [r["id"] for r in reviews]
    if len(review_ids) != len(set(review_ids)) or set(review_ids) != set(candidates):
        errors.append("review: must cover each external candidate exactly once")
    for row in reviews:
        candidate = candidates.get(row["id"])
        if candidate and (row.get("source_sha256") != candidate["source_sha256"]
                          or row.get("wcs_sha256") != candidate["external_wcs"].get("wcs_sha256")
                          or row.get("review_status") != "pending"
                          or not row.get("diagnostic_note")):
            errors.append(f"{row['id']}: review identity or status mismatch")
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path(__file__).with_name("manifest.json"))
    parser.add_argument("--check-files", action="store_true")
    parser.add_argument("--acceptance", action="store_true")
    parser.add_argument("--results", type=Path, help="also validate a portable measurement report")
    parser.add_argument("--wcs-review", type=Path, default=Path(__file__).with_name("collection-wcs-review.json"))
    args = parser.parse_args()
    records = json.loads(args.manifest.read_text())
    errors = validate(records, args.manifest.parent, args.check_files, args.acceptance)
    if args.results:
        errors.extend(validate_results(records, hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
                                       json.loads(args.results.read_text()), json.loads(args.wcs_review.read_text())))
    for error in errors:
        print(error)
    print(f"{len(records)} entries; {len(errors)} errors")
    return bool(errors)


if __name__ == "__main__":
    raise SystemExit(main())
