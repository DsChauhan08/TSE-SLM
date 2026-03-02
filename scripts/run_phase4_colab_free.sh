#!/usr/bin/env bash
set -euo pipefail

# Google Colab Free runner for Phase 4 sweeps.
# Usage examples:
#   bash scripts/run_phase4_colab_free.sh --suite main
#   bash scripts/run_phase4_colab_free.sh --suite controls
#   bash scripts/run_phase4_colab_free.sh --suite repeats
#   bash scripts/run_phase4_colab_free.sh --suite all --limit 10
#
# Notes:
# - This script uses GGUF models only for stability and consistent run_tag behavior.
# - Resume is automatic because evaluate_gguf_local.py reuses existing result files.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

SUITE="all"                 # main | controls | repeats | all
LIMIT="10"
STARTUP_TIMEOUT="300"
N_THREADS="4"
N_GPU_LAYERS="-1"
TEMPERATURES=""             # empty = 0.0..2.0 step 0.1
DO_SETUP="1"
LOG_FILE="logs/phase4_colab_suite.log"
SYNC_DIR=""                 # optional persistent path (e.g., /content/drive/MyDrive/slm-temp-sync)

DATASETS_MAIN=("arc_easy" "arc_challenge" "commonsense_qa" "piqa" "hellaswag")
MODELS=(
  "unsloth/Llama-3.1-8B-Instruct-GGUF"
  "TeichAI/Qwen3-4B-Thinking-2507-Claude-4.5-Opus-High-Reasoning-Distill-GGUF"
  "unsloth/Phi-4-mini-reasoning-GGUF"
  "unsloth/gemma-3-4b-it-GGUF"
)

usage() {
  cat <<'EOF'
run_phase4_colab_free.sh

Options:
  --suite <main|controls|repeats|all>  Which part to run (default: all)
  --limit <int>                         Samples per temperature (default: 10)
  --startup-timeout <sec>               llama.cpp server startup timeout (default: 300)
  --n-threads <int>                     llama.cpp CPU threads (default: 4)
  --n-gpu-layers <int>                  llama.cpp GPU layers (default: -1; all)
  --temperatures <csv>                  Optional subset, e.g. 0.0,0.7,1.4
  --dataset <name>                      Add main-suite dataset (repeatable)
  --model <repo_id>                     Add model (repeatable; overrides defaults)
  --log-file <path>                     Log file path (default: logs/phase4_colab_suite.log)
  --sync-dir <path>                     Optional directory to copy logs/results after each block
  --run-only                            Skip dependency setup
  --setup-only                          Setup dependencies only, do not run suite
  -h, --help                            Show this help
EOF
}

custom_models=0
custom_datasets=0
RUN_SUITE=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --suite) SUITE="${2:?}"; shift 2 ;;
    --limit) LIMIT="${2:?}"; shift 2 ;;
    --startup-timeout) STARTUP_TIMEOUT="${2:?}"; shift 2 ;;
    --n-threads) N_THREADS="${2:?}"; shift 2 ;;
    --n-gpu-layers) N_GPU_LAYERS="${2:?}"; shift 2 ;;
    --temperatures) TEMPERATURES="${2:?}"; shift 2 ;;
    --dataset)
      if [[ $custom_datasets -eq 0 ]]; then
        DATASETS_MAIN=()
        custom_datasets=1
      fi
      DATASETS_MAIN+=("${2:?}")
      shift 2
      ;;
    --model)
      if [[ $custom_models -eq 0 ]]; then
        MODELS=()
        custom_models=1
      fi
      MODELS+=("${2:?}")
      shift 2
      ;;
    --log-file) LOG_FILE="${2:?}"; shift 2 ;;
    --sync-dir) SYNC_DIR="${2:?}"; shift 2 ;;
    --run-only) DO_SETUP="0"; shift ;;
    --setup-only) RUN_SUITE=0; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 1 ;;
  esac
done

mkdir -p "$(dirname "$LOG_FILE")" data/results logs
exec > >(tee -a "$LOG_FILE") 2>&1

if [[ -d /content ]]; then
  : "${HF_HOME:=/content/hf_cache}"
else
  : "${HF_HOME:=$ROOT_DIR/.hf_cache}"
fi
export HF_HOME
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
mkdir -p "$HUGGINGFACE_HUB_CACHE" "$TRANSFORMERS_CACHE"

setup_colab() {
  echo "[setup] Installing build deps..."
  if command -v apt-get >/dev/null 2>&1; then
    if [[ "$(id -u)" -eq 0 ]]; then
      apt-get -qq update
      apt-get -qq install -y build-essential cmake ninja-build pkg-config libopenblas-dev >/dev/null
    elif command -v sudo >/dev/null 2>&1; then
      sudo apt-get -qq update
      sudo apt-get -qq install -y build-essential cmake ninja-build pkg-config libopenblas-dev >/dev/null
    fi
  fi

  echo "[setup] Installing Python deps..."
  python -m pip install --quiet --upgrade pip setuptools wheel
  python -m pip install --quiet lm-eval huggingface_hub requests datasets pandas matplotlib tqdm scipy scikit-learn transformers accelerate bitsandbytes

  echo "[setup] Reinstalling llama-cpp-python with best backend..."
  python -m pip uninstall -y llama-cpp-python >/dev/null 2>&1 || true
  if command -v nvidia-smi >/dev/null 2>&1; then
    CMAKE_ARGS="-DGGML_CUDA=on" FORCE_CMAKE=1 python -m pip install --quiet --no-cache-dir "llama-cpp-python>=0.2.90"
  else
    python -m pip install --quiet --no-cache-dir "llama-cpp-python>=0.2.90"
  fi

  echo "[setup] Runtime check:"
  python - <<'PY'
import torch
gpu = torch.cuda.is_available()
name = torch.cuda.get_device_name(0) if gpu else "none"
try:
    from llama_cpp import llama_cpp as low_level
    fn = getattr(low_level, "llama_supports_gpu_offload", None)
    offload = bool(fn()) if fn else False
except Exception:
    offload = False
print({"torch_cuda": gpu, "cuda_device": name, "llama_gpu_offload": offload})
PY
}

sync_outputs() {
  if [[ -z "$SYNC_DIR" ]]; then
    return
  fi
  mkdir -p "$SYNC_DIR/results" "$SYNC_DIR/logs"
  cp -f data/results/*.json "$SYNC_DIR/results/" 2>/dev/null || true
  cp -f data/results/*.csv "$SYNC_DIR/results/" 2>/dev/null || true
  cp -f logs/*.log "$SYNC_DIR/logs/" 2>/dev/null || true
  echo "[sync] Copied artifacts to $SYNC_DIR"
}

run_queue() {
  local dataset="$1"
  local top_p="$2"
  local top_k="$3"
  local run_tag="$4"
  local seed="$5"

  local cmd=(python -u scripts/run_phase1_queue.py)
  for model_id in "${MODELS[@]}"; do
    cmd+=(--model "$model_id")
  done
  cmd+=(
    --dataset "$dataset"
    --limit "$LIMIT"
    --output_dir "data/results"
    --log_dir "logs/phase1"
    --startup_timeout "$STARTUP_TIMEOUT"
    --n_threads "$N_THREADS"
    --n_gpu_layers "$N_GPU_LAYERS"
    --top_p "$top_p"
    --top_k "$top_k"
    --run_tag "$run_tag"
  )
  if [[ -n "$TEMPERATURES" ]]; then
    cmd+=(--temperatures "$TEMPERATURES")
  fi
  if [[ "$seed" != "-1" ]]; then
    cmd+=(--seed "$seed")
  fi

  echo
  echo "[run] dataset=$dataset run_tag=$run_tag top_p=$top_p top_k=$top_k seed=$seed"
  "${cmd[@]}"
}

if [[ "$DO_SETUP" == "1" ]]; then
  setup_colab
fi

if [[ "$RUN_SUITE" -eq 0 ]]; then
  echo "[done] Setup completed."
  exit 0
fi

case "$SUITE" in
  main|controls|repeats|all) ;;
  *) echo "Invalid --suite value: $SUITE" >&2; exit 1 ;;
esac

echo "[config] suite=$SUITE limit=$LIMIT n_threads=$N_THREADS n_gpu_layers=$N_GPU_LAYERS startup_timeout=$STARTUP_TIMEOUT"
echo "[config] models=${MODELS[*]}"
echo "[config] datasets(main)=${DATASETS_MAIN[*]}"
echo "[config] log_file=$LOG_FILE"
echo "[config] hf_cache=$HF_HOME"
if [[ -n "$TEMPERATURES" ]]; then
  echo "[config] temperatures=$TEMPERATURES"
fi
if [[ -n "$SYNC_DIR" ]]; then
  echo "[config] sync_dir=$SYNC_DIR"
fi

if [[ "$SUITE" == "main" || "$SUITE" == "all" ]]; then
  for ds in "${DATASETS_MAIN[@]}"; do
    run_queue "$ds" "1.0" "0" "phase4-main" "-1"
    sync_outputs
  done
fi

if [[ "$SUITE" == "controls" || "$SUITE" == "all" ]]; then
  run_queue "gsm8k" "0.9" "0" "phase4-control-top_p_0p9" "-1"
  sync_outputs
  run_queue "gsm8k" "1.0" "40" "phase4-control-top_k_40" "-1"
  sync_outputs
fi

if [[ "$SUITE" == "repeats" || "$SUITE" == "all" ]]; then
  run_queue "gsm8k" "1.0" "0" "phase4-repeat-seed42" "42"
  sync_outputs
  run_queue "gsm8k" "1.0" "0" "phase4-repeat-seed314" "314"
  sync_outputs
fi

echo "[done] Phase4 Colab suite segment finished."
echo "[monitor] tail -n 40 $LOG_FILE"
