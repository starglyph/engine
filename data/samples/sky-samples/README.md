# sky-samples — real-frame acceptance/stress set

**89 frames:** 23 historical entries and 66 new photographs in `robustness-2026-10`.

The new collection has **54 solver / 1 scene / 11 stress** inputs (nine negatives,
two extended-trail images), split by author into **45 development / 21 holdout**.
See [the collection protocol and measurements](../../../docs/robustness-collection.md).
Per-image licenses were checked on 2026-10-02: new records allow **CC0 or CC BY
2.0/3.0/4.0**. License evidence and visual R01–R24 annotations live in `manifest.json`.
The historical records below retain their IDs, pixels, track and WCS/baseline fields.

## Historical tracks (see [`GROUND-TRUTH.md`](GROUND-TRUTH.md) and the scene-localization research note)

Each frame is tagged with a **`track`** in the manifest — which problem it belongs to:

- **`solver`** (8) — geometry / quad-matching is the right tool (single-shot frames). Main solver-eval targets; 7 have WCS ground truth, 1 (`flickr_torchbearer_ladia`) is a hard single-shot.
- **`scene`** (13 historical entries) — images historically assigned to a separate research track. Confirmed stitched panoramas and non-TAN projections are outside the current single-pose evaluation. Foreground, processing, or an unsuccessful solve alone do not establish incompatible geometry; the historical assignments are not new geometric certifications.
- **`stress`** (2) — Tier-B adversarial (star-trails); ShareAlike, kept in `images-stress-tier-b/`.

> Why `scene` is separate: quad-matching + WCS assume one pinhole projection. Wide FOV alone is fine
> (`flickr_orion_rahn` solved at 71°). Stitching may break the geometry; foreground
> and clouds may instead impair detection. New assignments use projection evidence,
> never the success/failure of a solver.

## No binaries in git — reconstruct on demand

This directory stores **only provenance** (`manifest.json` + docs), never the image files:

```bash
python3 -m pip install -r requirements.txt
python3 fetch_sample.py --collection robustness-2026-10
python3 validate_collection.py --check-files --acceptance
# Optional: reconstruct every entry, including the historical BY-SA stress files:
python3 fetch_sample.py
python3 fetch_sample.py --list     # ids/licenses without fetching
```

See [`../../../docs/data-catalog.md`](../../../docs/data-catalog.md) for the full source catalog and licensing tiers.

## Historical frames

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
| 10 | `noirlab_iotw2334a` | scene | ❌ unsolved | CC BY 4.0 | 1280×1025 | CTIO/NOIRLab/NSF/AURA/T. Slovinský, CC BY 4.0. |
| 11 | `noirlab_iotw2452a` | scene | ❌ unsolved | CC BY 4.0 | 1280×1131 | CTIO/NOIRLab/NSF/AURA/P. Horálek (Institute of Physics in Opava), M. Kosari, CC BY 4.0. |
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
- **EXIF/GPS:** normalized files contain no source EXIF/GPS. Originals remain in the ignored `.originals/` cache; no claim is made that upstream originals lack metadata.
- **Provenance:** per asset in `manifest.json` — page_url, download_url, license_url, orig+clean sha256, size, `track`, `solve_status`, `wcs`.
- **NOIRLab:** exact per-image credits are recorded; retain them visibly when distributing images.

## Ground truth & attribution

WCS sidecars for solved frames: `ground-truth/<id>.wcs.json` (facts only). Method + solve-status: [`GROUND-TRUTH.md`](GROUND-TRUTH.md). Required credit lines: [`ATTRIBUTION.md`](ATTRIBUTION.md).

> Note: the ✅/❌ **Solve** column above records **astrometry.net** bootstrap results (upper
> bound / GT source). The **live `starglyph-core` solver** is measured separately by the
> eval harness — `cd prototype && make eval` (see [`docs/evaluation.md`](../../../docs/evaluation.md) §6);
> its solve-rate counts only `track:solver`; `scene` is excluded from single-pose evaluation, and `stress` is opt-in. Historical solve results are not collection measurements.

## New collection artifacts

[Browse all 66 photographs with source links, risk labels and split](COLLECTION.md).

- `manifest.json`: immutable image identity, lossless normalization recipe, licenses,
  visual review and grouped split; `research.external_reference` points to the separate measurement report.
- `robustness-results.json`: all per-ID outcomes, including unresolved and skipped cases;
  external WCS remains a candidate, never automatically promoted to accepted ground truth.
- `collection-rejections.json`: discovery exclusions and reviewed candidates not selected.
- `collection-wcs-review.json`: image/WCS hashes and diagnostic visual observations;
  reviewed candidates still remain pending, without full-field GT certification.
- `preview_collection.py --out-dir ../../../prototype/artifacts/robustness-previews`:
  regenerate ignored overview/native-crop contact sheets and their credit list.

New normalized PNG/TIFF files keep full oriented resolution and decoded pixel values;
no resizing or extra JPEG loss is introduced. Historical entries retain their original
resize/JPEG recipe. Checksums fail closed: source changes or codec drift require explicit
review and never silently replace an accepted image. `requirements.txt` pins Pillow;
codec versions and decoded-pixel hashes in each new record help diagnose drift across platforms.
