# sky-samples — real-frame acceptance/stress set

Representative sample for the blind plate solver: **23 frames** (21 Tier-A + 2 Tier-B stress).

## No binaries in git — reconstruct on demand

This directory stores **only provenance** (`manifest.json` + docs), never the
image files. Rebuild the frames locally:

```bash
python3 fetch_sample.py            # download, verify sha256, strip EXIF, resize
python3 fetch_sample.py --list     # show ids/licenses without fetching
```

Why: keeps the public repo light, avoids re-hosting third-party images, and
`orig_sha256` in the manifest catches source drift. See [`../../../docs/data-catalog.md`](../../../docs/data-catalog.md) for the full source catalog and licensing tiers.

## Frames

| # | id | Tier | License | Author/Credit | Size | Attribution |
|---|----|------|---------|---------------|------|-------------|
| 1 | `tetra3_alt40` | A | Apache-2.0 | ESA / tetra3 | 1024×768 | Test image from ESA tetra3 (https://github.com/esa/tetra3), (c) ESA, Apache-2.0. |
| 2 | `tetra3_alt60` | A | Apache-2.0 | ESA / tetra3 | 1024×768 | Test image from ESA tetra3 (https://github.com/esa/tetra3), (c) ESA, Apache-2.0. |
| 3 | `flickr_cygnus_fermion` | A | CC0 1.0 | Fermion (CC0) | 1024×633 | Photo by 'Fermion', CC0 1.0. |
| 4 | `flickr_orion_rahn` | A | CC0 1.0 | Stephen Rahn (CC0) | 984×1024 | Photo by Stephen Rahn, CC0 1.0. |
| 5 | `flickr_torchbearer_ladia` | A | CC0 1.0 | Neeraj Ladia (CC0) | 683×1024 | Photo by Neeraj Ladia, CC0 1.0. |
| 6 | `wm_constellation_orion` | A | CC0 | Own work | 1689×1171 | Madonka, CC0, via Wikimedia Commons. |
| 7 | `flickr_m41_donatiello` | A | CC0 1.0 | Giuseppe Donatiello (CC0) | 1024×1024 | Photo by Giuseppe Donatiello, CC0 1.0. |
| 8 | `eso_mw_panorama_0932a` | A | CC BY 4.0 | ESO/S. Brunier | 6000×3000 | ESO/S. Brunier, CC BY 4.0. |
| 9 | `eso_vista_mw_1242b` | A | CC BY 4.0 | ESO/Serge Brunier | 3042×2025 | ESO/Serge Brunier, CC BY 4.0. |
| 10 | `noirlab_iotw2334a` | A | CC BY 4.0 | see page: .../NOIRLab/NSF/AURA/<photographer> | 1280×1025 | NOIRLab/NSF/AURA (copy exact credit from page), CC BY 4.0. |
| 11 | `noirlab_iotw2452a` | A | CC BY 4.0 | see page: .../NOIRLab/NSF/AURA/<photographer> | 1280×1131 | CTIO/NOIRLab/NSF/AURA (copy exact credit from page), CC BY 4.0. |
| 12 | `wm_milkyway_arch` | A | CC BY 4.0 | https://www.eso.org/public/images/milkyway/ | 4000×1212 | Bruno Gilli/ESO, CC BY 4.0, via Wikimedia Commons. |
| 13 | `comet_neowise` | A | CC BY 2.0 | https://www.flickr.com/photos/21874566@N07/50668678147/ | 3000×2000 | RuggyBearLA, CC BY 2.0, via Wikimedia Commons. |
| 14 | `eso_cerro_armazones` | A | CC BY 4.0 | This media was produced by the European Southern Observatory (ESO), under the identifier armazones-1
This tag does not indicate the copyright status of the attached work. A normal copyright tag is still required. See Commons:Licensing. | 4000×886 | ESO/H. Carrasco, CC BY 4.0, via Wikimedia Commons. |
| 15 | `mw_arc_of_creation` | A | CC BY 3.0 | Imported from 500px (archived version) by the Archive Team. (detail page) | 2048×1248 | Burak Demir, CC BY 3.0, via Wikimedia Commons. |
| 16 | `mw_first_attempt` | A | CC BY 2.0 | Milky Way - First attempt | 4000×3000 | Josef Laimer, CC BY 2.0, via Wikimedia Commons. |
| 17 | `mw_heart_valentine` | A | CC BY 3.0 | http://www.eso.org/public/images/potw1207a/ | 4000×2667 | ESO/J. Girard, CC BY 3.0, via Wikimedia Commons. |
| 18 | `mw_himalayas_tents` | A | CC BY 2.0 | Milkyway with green tents from Himalayas | 4000×2473 | Rajarshi MITRA from Mumbai, India, CC BY 2.0, via Wikimedia Commons. |
| 19 | `mw_searching_portrait` | A | CC BY 2.0 | Searching | 2857×4000 | herdiephoto, CC BY 2.0, via Wikimedia Commons. |
| 20 | `mw_sochi_ru` | A | CC BY 4.0 | Own work | 4000×2667 | Илья Бунин, CC BY 4.0, via Wikimedia Commons. |
| 21 | `mw_time_panorama` | A | CC BY 2.0 | Milky Way Time | 4000×1386 | Jason Jacobs from Honolulu, USA, CC BY 2.0, via Wikimedia Commons. |
| 22 | `mw_satellite_trail` | B | CC BY-SA 4.0 | Own work | 2667×4000 | Martin Bernardi, CC BY-SA 4.0, via Wikimedia Commons. |
| 23 | `startrails_la_hague` | B | CC BY-SA 4.0 | Own work | 4000×2250 | Antoine Lamielle, CC BY-SA 4.0, via Wikimedia Commons. |

## Categories

- **A_sensor_narrowfov** — Tier A — (2): Real sensor frames, narrow FOV (~11°) — closest to solver input
- **B_amateur_widefield_cc0** — Tier A — (4): Amateur wide-field with constellations (CC0)
- **C_starcluster_cc0** — Tier A — (1): Dense field / open cluster (CC0)
- **D_institutional_ccby** — Tier A — (5): Institutional wide-field / all-sky (CC BY 4.0)
- **E_widefield_varied** — Tier A — (9): Wide-field, varied regions/conditions: comet, noise, landscape, RU region, panorama (CC BY 2.0/3.0/4.0)
- **F_stress_startrails** — Tier B — (2): ⚠ Tier B (ShareAlike): star-trails / satellite-trail — adversarial stress (CC BY-SA 4.0)

## Guarantees

- **Tier-A licenses:** Apache-2.0 / CC0 / CC BY 2.0/3.0/4.0 — no ShareAlike, no NC. **Tier-B** (`images-stress-tier-b/`): CC BY-SA 4.0 — kept separate; ShareAlike propagates to derivatives.
- **EXIF/GPS:** `fetch_sample.py` strips all metadata on rebuild; no GPS was present in any source frame (verified 2026-07-05).
- **Provenance:** per asset in `manifest.json` — page_url, download_url, license_url, orig+clean sha256, original/output size.
- **CC BY 2.0 (Flickr imports via Commons):** license confirmed via Commons API; for public use re-verify at the Flickr origin.
- **NOIRLab:** copy the exact per-image credit (photographer) from the source page before public use.

## Ground truth (WCS)

See [`GROUND-TRUTH.md`](GROUND-TRUTH.md). WCS sidecars (`ground-truth/<id>.wcs.json`) are
produced with [`solve_wcs.py`](solve_wcs.py) (astrometry.net API, free key) or the engine's
own blind solver. The `.wcs.json` files are committed (facts); `.wcs.fits` are gitignored.

## Attribution

Required credit lines: [`ATTRIBUTION.md`](ATTRIBUTION.md). CC0 frames need none (listed for provenance).
