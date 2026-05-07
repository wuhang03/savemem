#!/bin/bash
trap "kill 0" SIGINT

TASK_CSV="./StreamingBench/StreamingBench/Real_Time_Visual_Understanding.csv"
VIDEO_DIR="./StreamingBench/data/real"
CKPT_PATH="Qwen/Qwen2.5-VL-3B-Instruct"
MAX_PIXELS=401408          # 512*28*28
MAX_NUM_FRAMES=256
FPS=1
TIMEWINDOW=256

# Multi-GPU
MULTI_GPU=true
NUM_GPUS=2

# SaveMem toggles
USE_SAVEMEM=true
SHORT_FRAMES=8
MEDIUM_FRAMES=16
RECENCY_GATE_DROP_RATIO="3.0"
MAX_MEMORY_TOKENS="8192"
BBOX_ATTENTION_BIAS=""

MAX_SAMPLES=-1
START_SAMPLE=0
TASK_TYPE="Event Understanding"


# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --multi-gpu) MULTI_GPU=true; shift ;;
        --num-gpus) NUM_GPUS="$2"; shift 2 ;;
        --task-csv) TASK_CSV="$2"; shift 2 ;;
        --video-dir) VIDEO_DIR="$2"; shift 2 ;;
        --ckpt-path) CKPT_PATH="$2"; shift 2 ;;
        --max-pixels) MAX_PIXELS="$2"; shift 2 ;;
        --max-num-frames) MAX_NUM_FRAMES="$2"; shift 2 ;;
        --fps) FPS="$2"; shift 2 ;;
        --timewindow) TIMEWINDOW="$2"; shift 2 ;;
        --save-path) SAVE_PATH="$2"; shift 2 ;;
        # Memory flags
        --use-memory) USE_SAVEMEM=true; shift ;;
        --short-frames) SHORT_FRAMES="$2"; shift 2 ;;
        --medium-frames) MEDIUM_FRAMES="$2"; shift 2 ;;
        --recency-gate-drop-ratio) RECENCY_GATE_DROP_RATIO="$2"; shift 2 ;;
        --max-memory-tokens) MAX_MEMORY_TOKENS="$2"; shift 2 ;;
        --bbox-attention-bias) BBOX_ATTENTION_BIAS="$2"; shift 2 ;;
        --task-type) TASK_TYPE="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Create output directory
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RESULT_DIR="eval_results/streamingbench_${TIMESTAMP}"
mkdir -p "$RESULT_DIR"/{log,output}

SAVE_PATH="$RESULT_DIR/log/memory_stats.jsonl"

echo "StreamingBench Evaluation (Memory)"
echo "================================="
echo "Model: $CKPT_PATH"
echo "Mode: $([ "$MULTI_GPU" = true ] && echo "Multi-GPU ($NUM_GPUS GPUs)" || echo "Single-GPU")"
echo "Output: $RESULT_DIR"
echo "Memory: USE_SAVEMEM=$USE_SAVEMEM SHORT_FRAMES=$SHORT_FRAMES MEDIUM_FRAMES=$MEDIUM_FRAMES MAX_PIXELS=$MAX_PIXELS MAX_NUM_FRAMES=$MAX_NUM_FRAMES FPS=$FPS TIMEWINDOW=${TIMEWINDOW:-none}"
if [ -n "$TASK_TYPE" ]; then
  echo "Task type filter: $TASK_TYPE"
fi
if [ -n "$PAIR_SIM_THRESHOLD" ]; then
  echo "Pair similarity threshold provided (bypass Otsu): $PAIR_SIM_THRESHOLD"
fi
if [ -n "$SAVE_PATH" ]; then
    echo "Will save memory stats to: $SAVE_PATH"
fi

# Build common memory args
MEMORY_ARGS=( )
if [ "$USE_SAVEMEM" = true ]; then MEMORY_ARGS+=("--use_savemem"); fi
MEMORY_ARGS+=("--short_frames" "$SHORT_FRAMES" "--medium_frames" "$MEDIUM_FRAMES")
if [ -n "$RECENCY_GATE_DROP_RATIO" ]; then MEMORY_ARGS+=("--recency_gate_drop_ratio" "$RECENCY_GATE_DROP_RATIO"); fi
if [ -n "$MAX_MEMORY_TOKENS" ]; then MEMORY_ARGS+=("--max_memory_tokens" "$MAX_MEMORY_TOKENS"); fi
if [ -n "$BBOX_ATTENTION_BIAS" ]; then MEMORY_ARGS+=("--bbox_attention_bias" "$BBOX_ATTENTION_BIAS"); fi

# Video sampling/processing args to forward
VIDEO_ARGS=("--max-pixels" "$MAX_PIXELS" "--max-num-frames" "$MAX_NUM_FRAMES" "--fps" "$FPS")
# Pass time window (in seconds) to python as --time_window_size if provided
if [ -n "$TIMEWINDOW" ]; then
    VIDEO_ARGS+=("--time_window_size" "$TIMEWINDOW")
fi
# If fixed threshold is provided, forward to Python to bypass Otsu
if [ -n "$PAIR_SIM_THRESHOLD" ]; then
    VIDEO_ARGS+=("--pair_sim_threshold" "$PAIR_SIM_THRESHOLD")
fi
if [ -n "$SAVE_PATH" ]; then
    VIDEO_ARGS+=("--save_path" "$SAVE_PATH")
fi

# Task type filter
TASK_TYPE_ARGS=( )
if [ -n "$TASK_TYPE" ]; then
    TASK_TYPE_ARGS+=("--task_type" "$TASK_TYPE")
fi

if [ "$MULTI_GPU" = true ]; then
    python evaluation/streamingbench/streamingbench.py \
        --ckpt_path "$CKPT_PATH" \
        --task_csv "$TASK_CSV" \
        --video_dir "$VIDEO_DIR" \
        --result_dir "$RESULT_DIR" \
        --run_name "streamingbench" \
        --multi_gpu \
        --num_gpus "$NUM_GPUS" \
        --max_sample "$MAX_SAMPLES" \
        --start_sample "$START_SAMPLE" \
        --parallel_inference \
        "${MEMORY_ARGS[@]}" \
        "${VIDEO_ARGS[@]}" \
        "${TASK_TYPE_ARGS[@]}"
else
    OUTPUT_JSONL="$RESULT_DIR/output/results_${TIMESTAMP}.jsonl"
    LOG_PATH="$RESULT_DIR/log/eval_${TIMESTAMP}.log"

    python evaluation/streamingbench/streamingbench.py \
        --ckpt_path "$CKPT_PATH" \
        --task_csv "$TASK_CSV" \
        --video_dir "$VIDEO_DIR" \
        --output_jsonl "$OUTPUT_JSONL" \
        --log_path "$LOG_PATH" \
        --result_dir "$RESULT_DIR" \
        --run_name "streamingbench" \
        --max_sample "$MAX_SAMPLES" \
        --start_sample "$START_SAMPLE" \
        --parallel_inference \
        "${MEMORY_ARGS[@]}" \
        "${VIDEO_ARGS[@]}" \
        "${TASK_TYPE_ARGS[@]}"
fi

PYTHON_EXIT=$?

# Run scoring if we have any JSONL results; warn if the evaluation exited non‑zero.
if find "$RESULT_DIR" -maxdepth 2 -name '*.jsonl' | grep -q .; then
    if [ $PYTHON_EXIT -ne 0 ]; then
        echo "Warning: evaluation exited with code $PYTHON_EXIT; attempting to score existing results..."
    else
        echo "Running scoring..."
    fi
    python evaluation/streamingbench/score.py \
        --result_dir "$RESULT_DIR" \
        --model_name "$(basename "$CKPT_PATH")"
else
    echo "No JSONL results found in $RESULT_DIR; skipping scoring."
fi

echo "Evaluation complete. Results in: $RESULT_DIR"
