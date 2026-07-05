# Catalog Data

- `hyg_v3.csv` — full HYG catalog used by `dataset-cli` (ignored by git, generated locally).
- `hyg_v42.csv.gz` — upstream `.gz` archive, **committed and redistributed** here under
  CC BY-SA 4.0 (see `../../ATTRIBUTION.md`); byte-for-byte pinned via `HYG_GZ_SHA256`.
- `hyg_v3.sample.csv` — tiny tracked sample for tests/docs.
- `hyg_v3.source.env` — repo URL, path inside the repo, and pinned SHA256 of the `.gz` file.

Upstream lives on Codeberg: [astronexus/hyg](https://codeberg.org/astronexus/hyg).

## Why not `curl` on the raw URL?

Large files in that repository are stored with **Git LFS**. An HTTP `raw` link returns a short **LFS pointer file** (text starting with `version https://git-lfs.github.com/spec/v1`), not the CSV. That is why you must use **git** and **git-lfs**.

## Prerequisites

- `git`
- `git-lfs` ([install](https://git-lfs.com/), e.g. `sudo apt install git-lfs` then `git lfs install`)

## Download full catalog

From `prototype/`:

```bash
make fetch-catalog
```

First run: leave `HYG_GZ_SHA256=""` in `hyg_v3.source.env`. The script will print a line to paste into the env file for reproducible checksums on later runs.

Then open `hyg_v3.csv` — it should start with a CSV header and many data rows (tens of thousands of stars), not an LFS pointer.