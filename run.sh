#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if ! git lfs version >/dev/null 2>&1; then
  echo 'Git LFS is required. Install it, run "git lfs install", and retry.' >&2
  exit 1
fi
git lfs pull

UV_EXE="${UV_EXE:-}"
if [[ -z "$UV_EXE" ]]; then
  UV_EXE="$(command -v uv || true)"
fi
if [[ -z "$UV_EXE" ]]; then
  echo "uv was not found. Install uv or set UV_EXE to its executable path." >&2
  exit 1
fi

if [[ ! -x .venv/bin/weblfp ]]; then
  "$UV_EXE" sync --locked
fi

if [[ -z "${SKIP_FRONTEND_BUILD:-}" ]]; then
  if ! command -v npm >/dev/null 2>&1; then
    echo "npm is required to build the frontend." >&2
    exit 1
  fi
  (
    cd frontend
    npm ci
    npm run build
  )
fi

exec "$UV_EXE" run --no-sync weblfp
