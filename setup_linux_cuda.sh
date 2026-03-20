#!/usr/bin/env bash
set -euo pipefail

# Bootstrap gsplat-lidar for Linux + NVIDIA GPUs.
# Tuned defaults target A100 (sm_80), but can be overridden via env vars.

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${VENV_DIR:-${REPO_DIR}/.venv}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"
TORCH_PACKAGES="${TORCH_PACKAGES:-torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2}"
GSPLAT_WHL_INDEX_URL="${GSPLAT_WHL_INDEX_URL:-https://docs.gsplat.studio/whl/pt21cu121}"
INSTALL_EXAMPLES_DEPS="${INSTALL_EXAMPLES_DEPS:-1}"
FORCE_TORCH_REINSTALL="${FORCE_TORCH_REINSTALL:-0}"
INSTALL_FROM_SOURCE="${INSTALL_FROM_SOURCE:-0}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "[ERR] Python not found: ${PYTHON_BIN}" >&2
  exit 1
fi

if ! command -v git >/dev/null 2>&1; then
  echo "[ERR] git is required but not installed." >&2
  exit 1
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "[WARN] nvidia-smi not found. CUDA runtime may be unavailable."
else
  echo "[INFO] Detected GPU(s):"
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
fi

echo "[INFO] Creating virtual environment: ${VENV_DIR}"
"${PYTHON_BIN}" -m venv "${VENV_DIR}"
# shellcheck disable=SC1090
source "${VENV_DIR}/bin/activate"

python -m pip install --upgrade pip wheel ninja
# torch cpp_extension still imports pkg_resources. Keep setuptools below 81.
python -m pip install --upgrade "setuptools<81"

if [[ "${FORCE_TORCH_REINSTALL}" == "1" ]] || ! python -c "import torch" >/dev/null 2>&1; then
  echo "[INFO] Installing PyTorch from ${TORCH_INDEX_URL}"
  python -m pip install --index-url "${TORCH_INDEX_URL}" ${TORCH_PACKAGES}
else
  echo "[INFO] PyTorch already installed in venv. Skipping install."
fi

# A100 architecture (sm_80) to reduce CUDA build time and binary size.
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}"
echo "[INFO] TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}"

if command -v nvcc >/dev/null 2>&1; then
  CUDA_HOME="${CUDA_HOME:-$(dirname "$(dirname "$(command -v nvcc)")")}"
  export CUDA_HOME
  echo "[INFO] nvcc detected at $(command -v nvcc)"
  echo "[INFO] CUDA_HOME=${CUDA_HOME}"
  HAS_NVCC=1
else
  echo "[WARN] nvcc not found. Falling back to prebuilt gsplat wheel."
  HAS_NVCC=0
fi

if [[ "${INSTALL_FROM_SOURCE}" == "1" ]]; then
  if [[ "${HAS_NVCC}" != "1" ]]; then
    echo "[ERR] INSTALL_FROM_SOURCE=1 but nvcc is unavailable. Install CUDA toolkit first." >&2
    exit 1
  fi
  echo "[INFO] Installing gsplat-lidar from local source (editable)"
  python -m pip install --no-build-isolation -e "${REPO_DIR}"
else
  echo "[INFO] Installing prebuilt gsplat wheel from ${GSPLAT_WHL_INDEX_URL}"
  python -m pip install --index-url "${GSPLAT_WHL_INDEX_URL}" gsplat
fi

if [[ "${INSTALL_EXAMPLES_DEPS}" == "1" ]]; then
  if [[ "${HAS_NVCC}" == "1" ]]; then
    echo "[INFO] Installing full examples dependencies"
    python -m pip install --no-build-isolation -r "${REPO_DIR}/examples/requirements.txt"
  else
    echo "[INFO] Installing examples dependencies without CUDA-extension extras"
    TMP_REQS="$(mktemp)"
    grep -Ev 'fused-ssim|fused-bilagrid|ppisp' "${REPO_DIR}/examples/requirements.txt" > "${TMP_REQS}"
    python -m pip install --no-build-isolation -r "${TMP_REQS}"
    rm -f "${TMP_REQS}"
  fi
else
  echo "[INFO] Skipping examples dependencies (INSTALL_EXAMPLES_DEPS=${INSTALL_EXAMPLES_DEPS})"
fi

echo "[INFO] Verifying installation"
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("cuda devices:", torch.cuda.device_count())
    print("gpu 0:", torch.cuda.get_device_name(0))
import gsplat
print("gsplat import: OK")
PY

echo "[OK] Setup complete. Activate with: source ${VENV_DIR}/bin/activate"
