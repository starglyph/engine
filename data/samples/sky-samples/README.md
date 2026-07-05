# sky-samples — real-frame acceptance/stress set

Representative sample for the blind plate solver: **23 frames**.

## Tracks (see [`GROUND-TRUTH.md`](GROUND-TRUTH.md) and the scene-localization research note)

Each frame is tagged with a **`track`** in the manifest — which problem it belongs to:

- **`solver`** (8) — geometry / quad-matching is the right tool (single-shot frames). Main solver-eval targets; 7 have WCS ground truth, 1 (`flickr_torchbearer_ladia`) is a hard single-shot.
- **`scene`** (13) — stitched panoramas / heavy landscape / processed composites. Blind quad-matching **cannot** solve these (a stitched panorama has no single pinhole pose). Material for a **separate research track**: Milky-Way-band / scene-pattern localization, or tile-and-solve. NOT counted as solver failures.
- **`stress`** (2) — Tier-B adversarial (star-trails); ShareAlike, kept in `images-stress-tier-b/`.

> Why `scene` is separate: quad-matching + WCS assume one pinhole projection. Wide FOV alone is fine
> (`flickr_orion_rahn` solved at 71°); stitching/foreground/processing is what breaks it. See the research note.

## No binaries in git — reconstruct on demand

This directory stores **only provenance** (`manifest.json` + docs), never the image files:

```bash
python3 fetch_sample.py            # download, verify sha256, strip EXIF, resize
python3 fetch_sample.py --list     # ids/licenses without fetching
```

See [`../../../docs/data-catalog.md`](../../../docs/data-catalog.md) for the full source catalog and licensing tiers.

## Frames

| # | id | Track | Solve | License | Size | Attribution |
|---|----|-------|-------|---------|------|-------------|
| 1 | `tetra3_alt40` | solver | ✅ solved | Apache-2.0 | 1024×768 | Test image from ESA tetra3 (https://github.com/esa/tetra3), (c) ESA, Apache-2.0. |
| 2 | `tetra3_alt60` | solver | ✅ solved | Apache-2.0 | 1024×768 | Test image from ESA tetra3 (https://github.com/esa/tetra3), (c) ESA, Apache-2.0. |
| 3 | `flickr_cygnus_fermion` | solver | ✅ solved | CC0 1.0 | 1024×633 | Photo by 'Fermion', CC0 1.0. |
| 4 | `flickr_orion_rahn` | solver | ✅ solved | CC0 1.0 | 984×1024 | Photo by Stephen Rahn, CC0 1.0. |
| 5 | `flickr_torchbearer_ladia` | solver | ❌ unsolved | CC0 1.0 | 683×1024 | Photo by Neeraj Ladia, CC0 1.0. |
| 6 | `wm_constellation_orion` | solver | ✅ solved | CC0 | 1689×1171 | Madonka, CC0, via Wikimedia Commons. |
| 7 | `flickr_m41_donatiello` | solver | ✅ solved | CC0 1.0 | 1024×1024 | Photo by Giuseppe Donatiello, CC0 1.0. |
| 8 | `eso_mw_panorama_0932a` | scene | — n/a | CC BY 4.0 | 6000×3000 | ESO/S. Brunier, CC BY 4.0. |
| 9 | `eso_vista_mw_1242b` | scene | ❌ unsolved | CC BY 4.0 | 3042×2025 | ESO/Serge Brunier, CC BY 4.0. |
| 10 | `noirlab_iotw2334a` | scene | ❌ unsolved | CC BY 4.0 | 1280×1025 | NOIRLab/NSF/AURA (copy exact credit from page), CC BY 4.0. |
| 11 | `noirlab_iotw2452a` | scene | ❌ unsolved | CC BY 4.0 | 1280×1131 | CTIO/NOIRLab/NSF/AURA (copy exact credit from page), CC BY 4.0. |
| 12 | `wm_milkyway_arch` | scene | ❌ unsolved | CC BY 4.0 | 4000×1212 | Bruno Gilli/ESO, CC BY 4.0, via Wikimedia Commons. |
| 13 | `comet_neowise` | scene | ❌ unsolved | CC BY 2.0 | 3000×2000 | RuggyBearLA, CC BY 2.0, via Wikimedia Commons. |
| 14 | `eso_cerro_armazones` | solver | ✅ solved | CC BY 4.0 | 4000×886 | ESO/H. Carrasco, CC BY 4.0, via Wikimedia Commons. |
| 15 | `mw_arc_of_creation` | scene | ❌ unsolved | CC BY 3.0 | 2048×1248 | Burak Demir, CC BY 3.0, via Wikimedia Commons. |
| 16 | `mw_first_attempt` | scene | ❌ unsolved | CC BY 2.0 | 4000×3000 | Josef Laimer, CC BY 2.0, via Wikimedia Commons. |
| 17 | `mw_heart_valentine` | scene | ❌ unsolved | CC BY 3.0 | 4000×2667 | ESO/J. Girard, CC BY 3.0, via Wikimedia Commons. |
| 18 | `mw_himalayas_tents` | scene | ❌ unsolved | CC BY 2.0 | 4000×2473 | Rajarshi MITRA from Mumbai, India, CC BY 2.0, via Wikimedia Commons. |
| 19 | `mw_searching_portrait` | scene | ❌ unsolved | CC BY 2.0 | 2857×4000 | herdiephoto, CC BY 2.0, via Wikimedia Commons. |
| 20 | `mw_sochi_ru` | scene | ❌ unsolved | CC BY 4.0 | 4000×2667 | Илья Бунин, CC BY 4.0, via Wikimedia Commons. |
| 21 | `mw_time_panorama` | scene | — n/a | CC BY 2.0 | 4000×1386 | Jason Jacobs from Honolulu, USA, CC BY 2.0, via Wikimedia Commons. |
| 22 | `mw_satellite_trail` | stress | — n/a | CC BY-SA 4.0 | 2667×4000 | Martin Bernardi, CC BY-SA 4.0, via Wikimedia Commons. |
| 23 | `startrails_la_hague` | stress | — n/a | CC BY-SA 4.0 | 4000×2250 | Antoine Lamielle, CC BY-SA 4.0, via Wikimedia Commons. |

## Guarantees

- **Licenses:** Tier-A permissive (Apache-2.0 / CC0 / CC BY 2.0/3.0/4.0); Tier-B star-trails are CC BY-SA 4.0 (segregated).
- **EXIF/GPS:** `fetch_sample.py` strips all metadata on rebuild; no GPS in any source frame.
- **Provenance:** per asset in `manifest.json` — page_url, download_url, license_url, orig+clean sha256, size, `track`, `solve_status`, `wcs`.
- **NOIRLab:** copy the exact per-image credit from the source page before public use.

## Ground truth & attribution

WCS sidecars for solved frames: `ground-truth/<id>.wcs.json` (facts only). Method + solve-status: [`GROUND-TRUTH.md`](GROUND-TRUTH.md). Required credit lines: [`ATTRIBUTION.md`](ATTRIBUTION.md).

> Note: the ✅/❌ **Solve** column above records **astrometry.net** bootstrap results (upper
> bound / GT source). The **live `starglyph-core` solver** is measured separately by the
> eval harness — `cd prototype && make eval` (see [`docs/evaluation.md`](../../../docs/evaluation.md) §6);
> its solve-rate counts only `track:solver`, `scene` is expected-unsolvable, `stress` is opt-in.
