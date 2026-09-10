#!/usr/bin/env bash

_LRDS_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -f "${_LRDS_REPO_ROOT}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${_LRDS_REPO_ROOT}/.env"
  set +a
fi

export REPO_ROOT="${REPO_ROOT:-${_LRDS_REPO_ROOT}}"
export ASSET_ROOT="${ASSET_ROOT:-${REPO_ROOT}/assets}"
export RUNS_ROOT="${RUNS_ROOT:-${REPO_ROOT}/outputs}"
[[ "${ASSET_ROOT}" = /* ]] || ASSET_ROOT="${REPO_ROOT}/${ASSET_ROOT}"
[[ "${RUNS_ROOT}" = /* ]] || RUNS_ROOT="${REPO_ROOT}/${RUNS_ROOT}"
export PI05_LIBERO_CHECKPOINT="${PI05_LIBERO_CHECKPOINT:-${ASSET_ROOT}/RLinf-Pi05-LIBERO-SFT}"
export PALIGEMMA_TOKENIZER_PATH="${PALIGEMMA_TOKENIZER_PATH:-${ASSET_ROOT}/paligemma_tokenizer.model}"
export DINOV2_MODEL="${DINOV2_MODEL:-${ASSET_ROOT}/dinov2-base}"
export ROBOMETER_MODEL="${ROBOMETER_MODEL:-${ASSET_ROOT}/Robometer-4B}"
export ROBOMETER_BASE_MODEL="${ROBOMETER_BASE_MODEL:-${ASSET_ROOT}/Qwen3-VL-4B-Instruct}"
export LIBERO_ASSETS="${LIBERO_ASSETS:-${ASSET_ROOT}/LIBERO}"
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-${REPO_ROOT}/.runtime/libero}"
export PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv-policy/bin/python}"
export REWARD_PYTHON_BIN="${REWARD_PYTHON_BIN:-${REPO_ROOT}/.venv-reward/bin/python}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/vendor/openpi:${REPO_ROOT}/vendor/rlinf:${REPO_ROOT}/vendor/robometer:${REPO_ROOT}/vendor/libero${PYTHONPATH:+:${PYTHONPATH}}"

if [[ "${POSTTRAIN_OFFLINE:-1}" == "1" ]]; then
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
  export HF_DATASETS_OFFLINE=1
fi

mkdir -p "${LIBERO_CONFIG_PATH}" "${RUNS_ROOT}" "${LIBERO_ASSETS}/datasets"
_LRDS_LIBERO_PACKAGE_ASSETS="${REPO_ROOT}/vendor/libero/libero/libero/assets"
if [[ -L "${_LRDS_LIBERO_PACKAGE_ASSETS}" ]]; then
  rm -f "${_LRDS_LIBERO_PACKAGE_ASSETS}"
fi
if [[ ! -e "${_LRDS_LIBERO_PACKAGE_ASSETS}" ]]; then
  ln -s "${LIBERO_ASSETS}" "${_LRDS_LIBERO_PACKAGE_ASSETS}"
fi
cat >"${LIBERO_CONFIG_PATH}/config.yaml" <<EOF
benchmark_root: ${REPO_ROOT}/vendor/libero/libero/libero
bddl_files: ${REPO_ROOT}/vendor/libero/libero/libero/bddl_files
init_states: ${REPO_ROOT}/vendor/libero/libero/libero/init_files
datasets: ${ASSET_ROOT}/LIBERO/datasets
assets: ${LIBERO_ASSETS}
EOF

unset _LRDS_LIBERO_PACKAGE_ASSETS _LRDS_REPO_ROOT
