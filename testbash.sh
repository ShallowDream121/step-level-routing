#!/usr/bin/env bash

if [ -f .env ]; then
    echo "📂 加载 .env 环境变量..."
    export $(grep -v '^#' .env | xargs)
fi

export VLLM_WORKER_MULTIPROC_METHOD='spawn'

BIG_MODEL="/home/lcy/models/DeepSeek-R1-Distill-Llama-8B"
SMALL_MODEL="/home/lcy/models/DeepSeek-R1-Distill-Qwen-1.5B"

MODE="small_only"
BIG_GPU="0"
SMALL_GPU="1"

SMALL_PORT=8001
LARGE_PORT=8002

TASKS="./lm_tasks"
OUTPUT_PATH="./log_traces/1.5B-only-aime24"

CONFIDENCE_THRESHOLD=0.7
MAX_STEP_TOKENS=2048
MAX_GEN_TOKENS=16384
LIMIT=""  
NUM_FEWSHOT=0
BATCH_SIZE="auto"
DEBUG=""


echo "=============================================================================="
echo "   启动 Speculative Routing 评测"
echo "   大模型: $BIG_MODEL (GPU: $BIG_GPU)"
echo "   小模型: $SMALL_MODEL (GPU: $SMALL_GPU)"
echo "   模式：$MODE"
echo "   任务集: $TASKS"
echo "=============================================================================="

CMD=(
    python run.py
    --big_model "$BIG_MODEL"
    --mode "$MODE"
    --small_model "$SMALL_MODEL"
    --big_gpu "$BIG_GPU"
    --small_gpu "$SMALL_GPU"
    --large_port "$LARGE_PORT"
    --small_port "$SMALL_PORT"
    --tasks "$TASKS"
    --output_path "$OUTPUT_PATH"
    --batch_size "$BATCH_SIZE"
    --confidence_threshold "$CONFIDENCE_THRESHOLD"
    --max_step_tokens "$MAX_STEP_TOKENS"
    --max_gen_tokens "$MAX_GEN_TOKENS"
    --num_fewshot "$NUM_FEWSHOT"
)

if [ ! -z "$LIMIT" ]; then
    CMD+=($LIMIT)
fi

if [ ! -z "$DEBUG" ]; then
    CMD+=($DEBUG)
fi

CMD+=(--apply_chat_template)


"${CMD[@]}"

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "评测完成！结果已保存至: $OUTPUT_PATH"
else
    echo ""
    echo "评测失败 (Exit Code: $EXIT_CODE)"
fi

exit $EXIT_CODE