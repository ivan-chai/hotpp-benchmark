#!/usr/bin/env bash
set -uo pipefail

DS="${1:?usage: $0 <dataset> [--force]}"
FORCE="${2:-}"

ROOT="/root/.nv/hotpp-benchmark"
EXP_DIR="${ROOT}/experiments/${DS}"
CFG_DIR="${EXP_DIR}/configs"
RES_DIR="${EXP_DIR}/results/mamba_sizes"
CKPT_DIR="${EXP_DIR}/checkpoints/mamba_sizes"

if [[ ! -d "${CFG_DIR}" ]]; then
    echo "no configs for dataset=${DS} at ${CFG_DIR}" >&2
    exit 1
fi
mkdir -p "${RES_DIR}" "${CKPT_DIR}"

PAIRS=(
    "32 1"
    "32 2"
    "64 2"
    "64 4"
    "128 2"
    "128 4"
    "256 2"
    "256 4"
)

cd "${EXP_DIR}"

for pair in "${PAIRS[@]}"; do
    D="${pair% *}"
    L="${pair#* }"
    TAG="d${D}_L${L}"
    OUT="${RES_DIR}/${TAG}.yaml"
    if [[ -f "${OUT}" && "${FORCE}" != "--force" ]]; then
        echo "[skip] ${DS}:${TAG} already has ${OUT}"
        continue
    fi
    echo "===== [${DS}] training ${TAG} (D=${D}, L=${L}) ====="
    "${ROOT}/.venv/bin/python" -m hotpp.train \
        --config-dir configs --config-name next_item_mamba \
        name=mamba_sizes/${TAG} \
        transformer_hidden_size=${D} \
        transformer_layers=${L} \
        hydra.run.dir="outputs/size_sweep/${TAG}" \
        || echo "[fail] ${DS}:${TAG} (continuing)"
done
echo "done: ${DS}"
