# Third-Party Licenses

This file tracks third-party datasets and data files used by `starglyph`.

Original photographs by the project author, including the smartphone series, are
first-party material. Their CC BY 4.0 license and attribution are documented in
[`ATTRIBUTION.md`](ATTRIBUTION.md) and
[`data/input/smartphone/README.md`](data/input/smartphone/README.md).

## 1) HYG catalog

- **Purpose:** baseline star catalog for synthetic generation and matching.
- **Upstream:** [https://codeberg.org/astronexus/hyg](https://codeberg.org/astronexus/hyg)
- **License:** Creative Commons Attribution-ShareAlike 4.0 International (`CC BY-SA 4.0`)
- **Summary of obligations:**
  - keep attribution to the original source;
  - include a copy/link to the CC BY-SA 4.0 license;
  - if distributing adapted/derived dataset artifacts, preserve share-alike terms as required by license.

## 2) d3-celestial: constellation lines

- **Purpose:** constellation stick figures for overlay rendering.
- **Upstream:** [https://github.com/ofrohn/d3-celestial](https://github.com/ofrohn/d3-celestial)
- **File:** `data/constellations.lines.json` (vendored: `data/celestial/constellations.lines.json`)
- **Pinned commit:** `7e720a3de062059d4c5400a379146a601d9010e0`
- **SHA256:** `294f66bef5d5cf50b1e17f16d2efa1d97a15131612c68dd935adef6e7373e13c`
- **License:** BSD 3-Clause (`BSD-3-Clause`), text vendored at `data/celestial/LICENSE.d3-celestial`
- **Summary of obligations:**
  - retain copyright notice;
  - retain license text in source distributions;
  - avoid using contributor names for endorsement without permission.

## 3) d3-celestial: constellation names

- **Purpose:** canonical constellation abbreviations and names.
- **Upstream:** [https://github.com/ofrohn/d3-celestial](https://github.com/ofrohn/d3-celestial)
- **File:** `data/constellations.json` (vendored: `data/celestial/constellations.json`)
- **Pinned commit:** `7e720a3de062059d4c5400a379146a601d9010e0`
- **SHA256:** `ab4ae692027cbc042c0d6791a84456a65eb7c55656107fd00c58ff6e55d4d8b2`
- **License:** BSD 3-Clause (`BSD-3-Clause`), text vendored at `data/celestial/LICENSE.d3-celestial`
- **Summary of obligations:** same as above.

## Compliance notes

- `starglyph` uses stable IAU abbreviation keys (`Ori`, `UMa`, etc.) as internal identifiers.
- If datasets are vendored into this repository, include:
  - exact source URL;
  - upstream commit/tag or release identifier;
  - checksum of local copy.
- Before external release, ensure user-facing docs include:
  - attribution section;
  - third-party license list;
  - links to full license texts.

## 4) Tycho-2 positions for diagnostic correspondences

- **Purpose:** independent sparse star correspondences for smartphone solve review.
- **File:** `data/input/smartphone/external-star-correspondences.json` (`stars` positions).
- **Source:** Astrometry.net Tycho-2 index 17, Debian package
  [`astrometry-data-tycho2-10-19-littleendian`, version 2-4](https://deb.debian.org/debian/pool/main/a/astrometry-data-tycho2/astrometry-data-tycho2-10-19-littleendian_2-4_all.deb).
- **Provenance:** exact index name and SHA-256 in the correspondence file's `catalog_provenance`.
- **License:** BSD-2-Clause; original packaging copyright and permission record
  vendored in `data/input/smartphone/LICENSE.tycho2`.
- **Copyright:** 2000 Høg E., Fabricius C., Makarov V.V., Urban S., Corbin T.,
  Wycoff G., Bastian U., Schwekendiek P., Wicenec A.; Debian files © 2015 Ole Streicher.
- **Obligations:** retain copyright, license conditions and disclaimer.
  The full indices are local dependencies and are not bundled.

The same Tycho-2 terms cover eight positions from index 19 in
`data/input/smartphone/external-star-correspondences-215934.json`; its provenance
includes the exact index SHA-256. The mixed research snapshot
`docs/experiments/sky-edge-review-2026-10-02.json` additionally contains positions
derived from HYG v4.2 and is distributed under CC BY-SA 4.0 with HYG/Astronexus
attribution; original Tycho-2 notices are retained. No complete indices are vendored.

## 5) Public sky sample photographs

- **Registry:** [`data/samples/sky-samples/manifest.json`](data/samples/sky-samples/manifest.json), with per-file source, revision, author, license and checksums.
- **New collection:** `robustness-2026-10`, CC0 or CC BY 2.0/3.0/4.0 only.
- **Historical collection:** also includes ESA tetra3 Apache-2.0 examples and two separately stored CC BY-SA 4.0 trail photographs; those terms remain in force.
- **Required credits:** [`data/samples/sky-samples/ATTRIBUTION.md`](data/samples/sky-samples/ATTRIBUTION.md). Keep author/source/license links and indicate modifications when distributing normalized copies or previews. CC0 credits are retained as provenance.
- **Scope:** photographs and image-derived annotations retain the applicable media license, independently of the repository's code license. A manifest/download script does not remove redistribution obligations. Original files and derived previews are local ignored artifacts.
- **Measurements:** local Astrometry.net uses the Tycho-2 indices documented in §4; no full indices or star-coordinate tables are added with this collection. Starglyph uses HYG (§1).
