#!/usr/bin/env bash

execute_eval() {
    if [ -f .env ]; then
        export $(grep -v '^#' .env | xargs)
    fi

    export USE_TEMPLATE=1
    export VLLM_WORKER_MULTIPROC_METHOD='spawn'
    export BIG_MODEL="/home/lcy/models/DeepSeek-R1-Distill-Qwen-7B"
    export SMALL_MODEL="/home/lcy/models/DeepSeek-R1-Distill-Qwen-1.5B"
    export BIG_GPU="0"
    export SMALL_GPU="1"
    export SMALL_PORT=8001
    export LARGE_PORT=8002
    export MAX_GEN_TOKENS=16384
    export NUM_FEWSHOT=0
    export BATCH_SIZE="auto"

    
echo "=============================================================================="
echo "   启动 Speculative Routing 评测"
echo "   大模型: $BIG_MODEL (GPU: $BIG_GPU)"
echo "   小模型: $SMALL_MODEL (GPU: $SMALL_GPU)"
echo "   模式：$MODE"
echo "   任务集: $TASKS"
echo "   结果日志: $OUTPUT_PATH"
echo "=============================================================================="

    CMD=(
        python 
        run.py
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
        --plan ""
        --apply_chat_template
    )

    if [ ! -z "$LIMIT" ]; then
        CMD+=($LIMIT)
    fi
    #CMD+=(--debug)

    "${CMD[@]}"
    return $?
}

BASE_OUTPUT="./log_traces"

export MODE="block" 
export TASKS="gpqa_diamond_mine"
export OUTPUT_PATH="${BASE_OUTPUT}/1.5B-7B-block-512-gpqa/new/0.51"
export CONFIDENCE_THRESHOLD=0.51
export MAX_STEP_TOKENS=512
export LIMIT="--limit 80"  

for i in {1..5}; do
    echo ">>> 第 $i 次运行"
    execute_eval
    if [ $? -eq 0 ]; then
        echo "第 $i 次运行成功"
    else
        echo "第 $i 次运行失败"
    fi
done

echo ">>> 开始第二阶段"

export MODE="steer" 
export TASKS="gpqa_diamond_mine"
export OUTPUT_PATH="${BASE_OUTPUT}/1.5B-7B-steer-gpqa/0.6"
export CONFIDENCE_THRESHOLD=0.6
export MAX_STEP_TOKENS=512
export LIMIT="--limit 80"  

for i in {1..4}; do
    execute_eval
    if [ $? -eq 0 ]; then
        echo "第 $i 次运行成功"
    else
        echo "第 $i 次运行失败"
    fi
done

echo "全部任务完成！"