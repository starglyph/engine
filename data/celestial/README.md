# Constellation data (d3-celestial)

Vendored data files for constellation overlay rendering.

| File | Purpose | License |
|------|---------|---------|
| `constellations.lines.json` | Stick-figure lines (GeoJSON `MultiLineString`, positions `[lon, lat]` where `ra_deg = lon < 0 ? lon + 360 : lon`, `dec_deg = lat`) | BSD-3-Clause |
| `constellations.json` | Canonical IAU abbreviations + localized names | BSD-3-Clause |
| `LICENSE.d3-celestial` | Upstream license text | — |

## Provenance

- Upstream: <https://github.com/ofrohn/d3-celestial>
- Commit: `7e720a3de062059d4c5400a379146a601d9010e0` (fetched 2026-07-03)
- SHA256:
  - `constellations.lines.json` — `294f66bef5d5cf50b1e17f16d2efa1d97a15131612c68dd935adef6e7373e13c`
  - `constellations.json` — `ab4ae692027cbc042c0d6791a84456a65eb7c55656107fd00c58ff6e55d4d8b2`

Bright-star proper names come from the HYG catalog `proper` column (see `../catalogs/`), no extra source needed.
