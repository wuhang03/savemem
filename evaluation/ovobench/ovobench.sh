#!/usr/bin/env bash
trap "kill 0" SIGINT

export CUDA_VISIBLE_DEVICES=0,1
export FORCE_QWENVL_VIDEO_READER=decord

RUN_NAME="ovobench_qwen2.5_7b_real-time"
CKPT_PATH="Qwen/Qwen2.5-VL-7B-Instruct"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RESULT_DIR="eval_results/ovobench/${RUN_NAME}_${TIMESTAMP}"
TASKS="OCR ACR ATR STU FPD OJR" # real time
# TASKS="EPM ASI HLD" # backward
TASK_JSON="ovo_bench_new.json"
VIDEO_DIR="ovo/"
NUM_GPUS=2

# Video/sampling config
MAX_PIXELS=401408     # 512*28*28
MAX_FRAMES=256
FPS=1
TIME_WINDOW_SIZE=256

# Memory config
ENABLE_MEMORY=true
SHORT_FRAMES=4
MEDIUM_FRAMES=16

# SaveMem toggle (set to false for baseline runs without token compression).
# Independent of ENABLE_MEMORY: when ENABLE_MEMORY=true and USE_SAVEMEM=false,
# memory-related args still pass through but --use_savemem is omitted.
USE_SAVEMEM=true

# real-time 0.1
# backward 2.0
RECENCY_GATE_DROP_RATIO="0.1"
MAX_MEMORY_TOKENS="2048"
BBOX_ATTENTION_BIAS=""
SAVE_PATH="eval_results/ovobench/memory_stats.jsonl"

# Arg parsing (override defaults)
while [[ $# -gt 0 ]]; do
  case "$1" in
    --task-json) TASK_JSON="$2"; shift 2 ;;
    --video-dir) VIDEO_DIR="$2"; shift 2 ;;
    --ckpt-path) CKPT_PATH="$2"; shift 2 ;;
    --run-name) RUN_NAME="$2"; shift 2 ;;
    --num-gpus) NUM_GPUS="$2"; shift 2 ;;
    --time-window) TIME_WINDOW_SIZE="$2"; shift 2 ;;
    --video-chunk-t) VIDEO_CHUNK_T="$2"; shift 2 ;;
    --save-path) SAVE_PATH="$2"; shift 2 ;;
    --recency-gate-drop-ratio) RECENCY_GATE_DROP_RATIO="$2"; shift 2 ;;
    --max-memory-tokens) MAX_MEMORY_TOKENS="$2"; shift 2 ;;
    --bbox-attention-bias) BBOX_ATTENTION_BIAS="$2"; shift 2 ;;
    --enable-memory) ENABLE_MEMORY=true; shift ;;
    --disable-memory) ENABLE_MEMORY=false; shift ;;
    --use-savemem) USE_SAVEMEM=true; shift ;;
    --no-use-savemem) USE_SAVEMEM=false; shift ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

mkdir -p "${RESULT_DIR}"

echo "Run name: ${RUN_NAME}"
echo "Checkpoint: ${CKPT_PATH}"
echo "Tasks: ${TASKS}"
echo "GPUs: ${NUM_GPUS}"
echo "Task JSON: ${TASK_JSON}"
echo "Video dir: ${VIDEO_DIR}"

ARGS=(
  "evaluation/ovobench/ovobench.py"
  "--multi_gpu"
  "--run_name" "${RUN_NAME}"
  "--ckpt_path" "${CKPT_PATH}"
  "--result_dir" "${RESULT_DIR}"
  "--task_json" "${TASK_JSON}"
  "--video_dir" "${VIDEO_DIR}"
  "--max_pixels" "${MAX_PIXELS}"
  "--max_frames" "${MAX_FRAMES}"
  "--fps" "${FPS}"
  "--num_gpus" "${NUM_GPUS}"
  "--task" ${TASKS}
)

if [[ -n "${TIME_WINDOW_SIZE}" ]]; then
  ARGS+=("--time_window_size" "${TIME_WINDOW_SIZE}")
fi

if [[ -n "${VIDEO_CHUNK_T}" ]]; then
  ARGS+=("--video_chunk_t" "${VIDEO_CHUNK_T}")
fi

if [[ "${ENABLE_MEMORY}" == "true" ]]; then
  if [[ "${USE_SAVEMEM}" == "true" ]]; then
    ARGS+=("--use_savemem")
  fi
  ARGS+=("--short_frames" "${SHORT_FRAMES}" "--medium_frames" "${MEDIUM_FRAMES}")
  if [[ -n "${RECENCY_GATE_DROP_RATIO}" ]]; then
    ARGS+=("--recency_gate_drop_ratio" "${RECENCY_GATE_DROP_RATIO}")
  fi
  if [[ -n "${MAX_MEMORY_TOKENS}" ]]; then
    ARGS+=("--max_memory_tokens" "${MAX_MEMORY_TOKENS}")
  fi
  if [[ -n "${BBOX_ATTENTION_BIAS}" ]]; then
    ARGS+=("--bbox_attention_bias" "${BBOX_ATTENTION_BIAS}")
  fi
  if [[ -n "${SAVE_PATH}" ]]; then
    ARGS+=("--save_path" "${SAVE_PATH}")
  fi
fi

python "${ARGS[@]}"

echo "Done"
