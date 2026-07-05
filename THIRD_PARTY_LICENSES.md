# Third-Party Licenses

This file tracks third-party datasets and data files used by `starglyph`.

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
