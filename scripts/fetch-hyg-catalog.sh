#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DATA_DIR="${REPO_ROOT}/data/catalogs"
ENV_FILE="${DATA_DIR}/hyg_v3.source.env"
OUTPUT_CSV="${DATA_DIR}/hyg_v3.csv"
OUTPUT_GZ="${DATA_DIR}/hyg_v42.csv.gz"
TMP_DIR=""

cleanup() {
  if [[ -n "${TMP_DIR}" && -d "${TMP_DIR}" ]]; then
    rm -rf "${TMP_DIR}"
  fi
}
trap cleanup EXIT

if [[ ! -f "${ENV_FILE}" ]]; then
  echo "Missing source config: ${ENV_FILE}" >&2
  exit 1
fi

# shellcheck disable=SC1090
source "${ENV_FILE}"

if [[ -z "${HYG_REPO_URL:-}" ]]; then
  echo "HYG_REPO_URL is empty in ${ENV_FILE}" >&2
  exit 1
fi

if [[ -z "${HYG_GZ_REL_PATH:-}" ]]; then
  echo "HYG_GZ_REL_PATH is empty in ${ENV_FILE}" >&2
  exit 1
fi

BOOTSTRAP_SHA=0
if [[ -z "${HYG_GZ_SHA256:-}" ]]; then
  echo "HYG_GZ_SHA256 is empty — verifying checksum is skipped (bootstrap)." >&2
  echo "After success, add this line to ${ENV_FILE}:" >&2
  BOOTSTRAP_SHA=1
fi

if ! command -v git >/dev/null 2>&1; then
  echo "git is required to fetch the HYG catalog (Git LFS)." >&2
  exit 1
fi

if ! command -v git-lfs >/dev/null 2>&1; then
  echo "git-lfs is required (large files are not available via plain HTTP raw URLs)." >&2
  echo "Install: https://git-lfs.com/  (e.g. sudo apt install git-lfs && git lfs install)" >&2
  exit 1
fi

if ! command -v gzip >/dev/null 2>&1; then
  echo "gzip is required to unpack the catalog archive." >&2
  exit 1
fi

mkdir -p "${DATA_DIR}"
TMP_DIR="$(mktemp -d)"

echo "Cloning ${HYG_REPO_URL} (shallow)..."
git clone --depth 1 "${HYG_REPO_URL}" "${TMP_DIR}/hyg"

echo "Fetching Git LFS objects..."
git -C "${TMP_DIR}/hyg" lfs pull

GZ_SRC="${TMP_DIR}/hyg/${HYG_GZ_REL_PATH}"
if [[ ! -f "${GZ_SRC}" ]]; then
  echo "Expected file missing after LFS pull: ${HYG_GZ_REL_PATH}" >&2
  exit 1
fi

# Reject LFS pointer accidentally checked in as file (should not happen after lfs pull)
if head -n1 "${GZ_SRC}" | grep -q '^version https://git-lfs.github.com/spec/v1'; then
  echo "File is still a Git LFS pointer; git lfs pull did not materialize the blob." >&2
  echo "Try: git -C ... lfs install && git -C ... lfs pull" >&2
  exit 1
fi

cp "${GZ_SRC}" "${OUTPUT_GZ}"
ACTUAL_SHA="$(sha256sum "${OUTPUT_GZ}" | awk '{print $1}')"
if [[ "${BOOTSTRAP_SHA}" -eq 1 ]]; then
  echo "HYG_GZ_SHA256=\"${ACTUAL_SHA}\"" >&2
else
  if [[ "${ACTUAL_SHA}" != "${HYG_GZ_SHA256}" ]]; then
    rm -f "${OUTPUT_GZ}"
    echo "Checksum mismatch for ${OUTPUT_GZ}" >&2
    echo "Expected: ${HYG_GZ_SHA256}" >&2
    echo "Actual:   ${ACTUAL_SHA}" >&2
    exit 1
  fi
fi

gzip -dc "${OUTPUT_GZ}" > "${OUTPUT_CSV}"
echo "Catalog saved to ${OUTPUT_CSV} (from ${OUTPUT_GZ})"
