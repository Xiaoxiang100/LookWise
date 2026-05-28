#!/bin/bash

set -e

GPU_LIST="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
MODEL_PATH="${MODEL_PATH:-/your/model/path/Qwen2.5-VL-3B-Instruct}"
ANNO_PATH="${ANNO_PATH:-/your/data/adaptive_vr_data}"
BENCHMARK="hr-bench_4k"
CONF_THRESHOLD="${CONF_THRESHOLD:-1.00}"

MODE_FLAG=""
MERGE_FILE_SUFFIX="_adaptive_vr"
if [[ "$1" == "direct" ]]; then
    MODE_FLAG="--direct-answer"
    MERGE_FILE_SUFFIX="_direct"
    echo "Mode: direct answer baseline"
else
    echo "Mode: AdaptiveVR"
fi

IFS=',' read -ra GPULIST <<< "$GPU_LIST"
CHUNKS=${#GPULIST[@]}

MODEL_BASENAME=$(basename "$MODEL_PATH")
ANSWERS_DIR="adaptive_vr/eval/answers/${BENCHMARK}/${MODEL_BASENAME}"

mkdir -p "${ANSWERS_DIR}"
echo "========================================="
echo "  AdaptiveVR evaluation"
echo "========================================="
echo "Model path: ${MODEL_PATH}"
echo "Annotation path: ${ANNO_PATH}"
echo "Benchmark: ${BENCHMARK}"
echo "Output directory: ${ANSWERS_DIR}"
echo "Parallel GPUs: ${CHUNKS}"
echo "Confidence threshold: ${CONF_THRESHOLD}"
echo "========================================="

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python adaptive_vr/eval/run_inference.py \
        --model-path "${MODEL_PATH}" \
        --annotation_path "${ANNO_PATH}" \
        --benchmark "${BENCHMARK}" \
        --answers-file "${ANSWERS_DIR}/${CHUNKS}_${IDX}.jsonl" \
        --num-chunks "$CHUNKS" \
        --chunk-idx "$IDX" \
        --conf-threshold "${CONF_THRESHOLD}" \
        ${MODE_FLAG} &
done

echo "All inference jobs started. Waiting for completion..."
wait
echo "All inference jobs completed."

THRESHOLD_STR=$(printf "%.2f" "${CONF_THRESHOLD}" | tr -d '.')
MERGE_FILE="${ANSWERS_DIR}/hr-bench_4k_test_threshold${MERGE_FILE_SUFFIX}_th${THRESHOLD_STR}.jsonl"
echo "Merging results into ${MERGE_FILE}"

> "${MERGE_FILE}"
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat "${ANSWERS_DIR}/${CHUNKS}_${IDX}.jsonl" >> "${MERGE_FILE}"
done

echo "========================================="
echo "Evaluation finished."
echo "Result file: ${MERGE_FILE}"
echo "Score with:"
echo "python adaptive_vr/eval/eval_results_hr-bench.py --answers-file ${MERGE_FILE}"
echo "========================================="
