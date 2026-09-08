#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UV_BIN="${UV_BIN:-uv}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
POLICY_ENV="${POLICY_ENV:-${REPO_ROOT}/.venv-policy}"
REWARD_ENV="${REWARD_ENV:-${REPO_ROOT}/.venv-reward}"

command -v "${UV_BIN}" >/dev/null 2>&1 || {
  echo "uv is required: https://docs.astral.sh/uv/getting-started/installation/" >&2
  exit 1
}

if [[ ! -x "${POLICY_ENV}/bin/python" ]]; then
  "${UV_BIN}" venv --python "${PYTHON_VERSION}" "${POLICY_ENV}"
fi
UV_PROJECT_ENVIRONMENT="${POLICY_ENV}" "${UV_BIN}" sync --extra dev

# OpenPI's PyTorch Pi0.5 backend relies on its patched Gemma, PaliGemma, and
# SigLIP implementations. Upstream installs these files directly into the
# Transformers package; make that non-standard step explicit and idempotent.
"${POLICY_ENV}/bin/python" - "${REPO_ROOT}" <<'PY'
import shutil
import site
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
source = root / "vendor/openpi/openpi/models_pytorch/transformers_replace/models"
target = Path(site.getsitepackages()[0]) / "transformers/models"
if not source.is_dir() or not target.is_dir():
    raise SystemExit(f"missing OpenPI replacement source or Transformers target: {source}, {target}")
for model_dir in source.iterdir():
    if model_dir.is_dir():
        shutil.copytree(model_dir, target / model_dir.name, dirs_exist_ok=True)
PY

"${POLICY_ENV}/bin/python" - <<'PY'
from transformers.models.siglip import check

if not check.check_whether_transformers_replace_is_installed_correctly():
    raise SystemExit("OpenPI Transformers replacement validation failed")
PY

if [[ ! -x "${REWARD_ENV}/bin/python" ]]; then
  "${UV_BIN}" venv --python "${PYTHON_VERSION}" "${REWARD_ENV}"
fi
"${UV_BIN}" pip install --python "${REWARD_ENV}/bin/python" \
  -r "${REPO_ROOT}/environments/reward/requirements.lock"

for env_dir in "${POLICY_ENV}" "${REWARD_ENV}"; do
  "${env_dir}/bin/python" - "${REPO_ROOT}" <<'PY'
import site
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
paths = [
    root,
    root / "vendor" / "openpi",
    root / "vendor" / "rlinf",
    root / "vendor" / "robometer",
    root / "vendor" / "libero",
]
site_dir = Path(site.getsitepackages()[0])
(site_dir / "robometer_pi05_sac_dsrl_vendor.pth").write_text(
    "\n".join(map(str, paths)) + "\n", encoding="utf-8"
)
PY
done

mkdir -p "${REPO_ROOT}/assets" "${REPO_ROOT}/outputs" "${REPO_ROOT}/.runtime/libero"
if [[ ! -f "${REPO_ROOT}/.env" ]]; then
  cp "${REPO_ROOT}/.env.example" "${REPO_ROOT}/.env"
fi

echo "Policy environment: ${POLICY_ENV}"
echo "Reward environment: ${REWARD_ENV}"
echo "Next: source scripts/load_env.sh && python scripts/download_assets.py --all"
