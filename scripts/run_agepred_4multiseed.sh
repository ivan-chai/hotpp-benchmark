#!/usr/bin/env bash
set -uo pipefail

ROOT="/root/.nv/hotpp-benchmark"
EXP_DIR="${ROOT}/experiments/transactions-agepred"
PY="${ROOT}/.venv/bin/python"
LOG_DIR="${ROOT}/logs"
TS="$(date +%Y%m%d_%H%M%S)"
LOG="${LOG_DIR}/agepred_4multiseed_${TS}.log"

mkdir -p "${LOG_DIR}"
exec > >(tee -a "${LOG}") 2>&1

export WANDB_MODE=offline
cd "${EXP_DIR}"

CONFIGS=(
    "next_item_mamba"
    "next_item_mamba_structural_channel"
    "next_item_mamba_jump"
    "s2p2"
)

mkdir -p outputs/multiseed
mkdir -p results
for name in "${CONFIGS[@]}"; do
    mkdir -p "checkpoints/${name}"
    mkdir -p "outputs/multiseed/${name}"
done

echo "=========================================================="
echo "AGEPRED multiseed sweep started at $(date)"
echo "log:    ${LOG}"
echo "python: ${PY}"
echo "exp:    ${EXP_DIR}"
echo "configs: ${CONFIGS[*]}"
echo "=========================================================="

run_one() {
    local idx="$1"
    local total="$2"
    local name="$3"
    local started ended status
    started="$(date)"
    echo
    echo "===== [agepred ${idx}/${total}] ${name}  start: ${started} ====="
    "${PY}" -m hotpp.train_multiseed \
        --config-dir configs \
        --config-name "${name}" \
        hydra.run.dir="outputs/multiseed/${name}"
    status=$?
    ended="$(date)"
    if [[ ${status} -eq 0 ]]; then
        echo "===== [agepred ${idx}/${total}] ${name}  OK    end: ${ended} ====="
    else
        echo "===== [agepred ${idx}/${total}] ${name}  FAIL (exit ${status})  end: ${ended} ====="
    fi
    rm -rf lightning_logs 2>/dev/null || true
}

total="${#CONFIGS[@]}"
for i in "${!CONFIGS[@]}"; do
    run_one "$((i+1))" "${total}" "${CONFIGS[$i]}"
done

echo
echo "=========================================================="
echo "AGEPRED multiseed sweep finished at $(date)"
echo "Result files (if all runs succeeded):"
for name in "${CONFIGS[@]}"; do
    rep="results/multiseed_${name}.yaml"
    if [[ -f "${rep}" ]]; then
        echo "  [ok]   ${rep}"
    else
        echo "  [miss] ${rep}"
    fi
done
echo "=========================================================="
