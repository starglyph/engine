#!/usr/bin/env python3
"""Discover Commons candidates; this does NOT approve licenses or select a dataset.

Responses are retained locally for per-file review. Run with an ignored --out-dir.
"""
import argparse
import hashlib
import html
import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

API = "https://commons.wikimedia.org/w/api.php"
UA = "Starglyph/1.0 (open sky-image research; https://github.com/starglyph/engine)"
LICENSES = {"CC0", "CC BY 2.0", "CC BY 3.0", "CC BY 4.0"}
QUERIES = [
    '"night sky"', '"stars" "clouds"', '"milky way" "trees"',
    '"milky way" "light pollution"', '"night sky" "clouds"',
    '"star trails"', '"satellite trail"', '"night" "fog"',
    '"night" "cloudy"', '"orion" "constellation"',
    '"night sky" "moon"', '"milky way" "smartphone"',
    '"milky way" "noise"', '"aurora" "stars"',
    '"stars" "Jupiter"', '"stars" "airplane"',
    '"night" "street lights"', '"milky way" "panorama"',
]


def plain(value):
    return html.unescape(re.sub(r"<[^>]*>", "", value)).strip()


def request(params):
    url = API + "?" + urllib.parse.urlencode(params)
    for attempt in range(4):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=60) as response:
                body = response.read()
            result = json.loads(body)
            if "error" in result:
                raise ValueError(result["error"])
            return url, body, result
        except (OSError, ValueError):
            if attempt == 3:
                raise
            time.sleep(2 * (attempt + 1))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--query", action="append")
    parser.add_argument("--limit", type=int, default=50)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    candidates, rejected = {}, []
    for number, query in enumerate(args.query or QUERIES):
        params = dict(action="query", format="json", generator="search",
                      gsrsearch=query + " filetype:bitmap", gsrnamespace=6,
                      gsrlimit=args.limit, prop="imageinfo|revisions",
                      iiprop="url|size|sha1|timestamp|extmetadata",
                      rvprop="ids|timestamp|content", rvslots="main")
        url, body, result = request(params)
        evidence = f"search-{number:02d}.json"
        (args.out_dir / evidence).write_bytes(body)
        for page in result.get("query", {}).get("pages", {}).values():
            if "imageinfo" not in page:
                continue
            info = page["imageinfo"][0]
            meta = info.get("extmetadata", {})
            get = lambda key: plain(str(meta.get(key, {}).get("value", "")))
            license_name = get("LicenseShortName")
            record = dict(page_id=page["pageid"], title=page["title"],
                          info=info, revision=page.get("revisions", [{}])[0],
                          evidence=evidence, evidence_sha256=hashlib.sha256(body).hexdigest(),
                          evidence_url=url, query=query)
            if license_name not in LICENSES:
                rejected.append(dict(title=page["title"], reason="license_not_allowed",
                                     license=license_name, evidence=evidence))
                continue
            if info["width"] * info["height"] > 80_000_000 or info["size"] > 35_000_000:
                rejected.append(dict(title=page["title"], reason="collection_size_limit",
                                     evidence=evidence))
                continue
            candidates.setdefault(page["pageid"], record)
        (args.out_dir / "candidates.json").write_text(json.dumps(list(candidates.values()), indent=2))
        (args.out_dir / "rejected.json").write_text(json.dumps(rejected, indent=2))
        print(query, "cumulative permissive candidates:", len(candidates), flush=True)
        time.sleep(1)


if __name__ == "__main__":
    main()
