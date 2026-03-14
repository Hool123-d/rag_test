#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"
PYTHON_BIN="${PYTHON_BIN:-python3}"
MODE="${1:-web}"

if [[ ! -d "${VENV_DIR}" ]]; then
  echo "[setup] create local venv at ${VENV_DIR}"
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

if [[ ! -f "${VENV_DIR}/.deps_ready" ]] || [[ "${ROOT_DIR}/requirements.txt" -nt "${VENV_DIR}/.deps_ready" ]]; then
  echo "[setup] install/update dependencies"
  python -m pip install --upgrade pip >/dev/null
  python -m pip install -r "${ROOT_DIR}/requirements.txt"
  date > "${VENV_DIR}/.deps_ready"
fi

if [[ ! -f "${ROOT_DIR}/.env" ]]; then
  cp "${ROOT_DIR}/.env.example" "${ROOT_DIR}/.env"
  echo "[setup] .env created from .env.example, please fill in secrets before querying"
fi

export HF_HOME="${ROOT_DIR}/.cache/huggingface"
export TRANSFORMERS_CACHE="${ROOT_DIR}/.cache/huggingface"

cd "${ROOT_DIR}"

case "${MODE}" in
  ingest)
    python src/ingest.py
    ;;
  chat)
    python src/chat.py
    ;;
  web)
    python -m streamlit run src/web_ui.py
    ;;
  *)
    echo "Usage: ./scripts/dev.sh [ingest|chat|web]"
    exit 1
    ;;
esac
