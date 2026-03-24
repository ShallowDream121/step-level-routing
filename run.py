import argparse
import subprocess
import sys
import time
import os
import signal
from pathlib import Path

def start_vllm_server(model_path, port, gpu_id):
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.api_server",
        "--model", model_path,
        "--port", str(port),
        "--tensor-parallel-size", "1",
        "--gpu-memory-utilization", "0.9",
        "--disable-log-stats",
        "--enforce-eager"
    ]
    
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    print(f"Starting vllm server: GPU {gpu_id}, Port {port}, Model {model_path}")
    proc = subprocess.Popen(cmd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    return proc

def wait_for_health(port, timeout=180):
    import requests
    url = f"http://localhost:{port}/health"
    start = time.time()
    while time.time() - start < timeout:
        try:
            if requests.get(url, timeout=2).status_code == 200:
                return True
        except:
            pass
        time.sleep(2)
    return False

def parse_arguments():
    """
    解析命令行参数。
    """
    parser = argparse.ArgumentParser(
        description="Run lm-eval with step-by-step llm Router"
    )
    
    # --- 模型配置 ---
    parser.add_argument(
        "--big_model", type=str, required=True, 
        help="Path or HF ID for the large model."
    )
    parser.add_argument(
        "--small_model", type=str, required=True, 
        help="Path or HF ID for the small model."
    )
    parser.add_argument(
        "--big_gpu", type=str, default="0", 
        help="GPU ID for the large model (e.g., '0' or '0,1')."
    )
    parser.add_argument(
        "--small_gpu", type=str, default="1", 
        help="GPU ID for the small model."
    )
    parser.add_argument(
        "--small_port", type=int, default=8001, 
        help="Port for the small model server."
    )
    parser.add_argument(
        "--large_port", type=int, default=8002, 
        help="Port for the large model server."
    )
    
    # --- 路由参数 ---
    parser.add_argument(
        "--confidence_threshold", type=float, default=0.6, 
        help="Threshold for the router to decide which model to use for next step."
    )
    parser.add_argument(
        "--max_step_tokens", type=int, default=2048, 
        help="Max tokens to generate in one speculation step."
    )
    parser.add_argument(
        "--max_gen_tokens", type=int, default=16384, 
        help="Maximum total tokens to generate per sample."
    )
    
    # --- lm-eval 参数 ---
    parser.add_argument(
        "--tasks", type=str, required=True, 
        help="Comma-separated list of tasks to evaluate (e.g., aime25_nofigures,mmlu)."
    )
    parser.add_argument(
        "--output_path", type=str, default="./logs", 
        help="Directory to save evaluation results."
    )
    parser.add_argument(
        "--batch_size", type=str, default="auto", 
        help="Batch size for evaluation ('auto' or integer)."
    )
    parser.add_argument(
        "--limit", type=int, default=None, 
        help="Limit the number of samples per task (debug)."
    )
    parser.add_argument(
        "--num_fewshot", type=int, default=0, 
        help="Number of few-shot examples."
    )
    parser.add_argument(
        "--apply_chat_template", action="store_true", 
        help="Apply chat template to inputs (recommended for chat models)."
    )
    
    return parser.parse_args()

def main():
    args = parse_arguments()
    
    procs = []
    try:
        p_small = start_vllm_server(args.small_model, args.small_port, args.small_gpu)
        print("Waiting for small model...")
        if not wait_for_health(args.small_port):
            print("Small model failed to start!!!")
            raise RuntimeError("Small model failed to start")

        p_large = start_vllm_server(args.big_model, args.large_port, args.big_gpu)
        print("Waiting for large model...")
        if not wait_for_health(args.large_port):
            print("Large model failed to start!!!")
            raise RuntimeError("Large model failed to start")

        procs = [p_small, p_large]
        print("✅ 所有model服务已就绪，开始运行 lm-eval...")

        
        pre_load_code = (
            "import route; "  
            "from lm_eval.__main__ import cli_main; " 
            "cli_main()"
        )

        model_args_str = (
            f"small_model_url=http://localhost:{args.small_port},"
            f"large_model_url=http://localhost:{args.large_port},"
            f"small_model_name={args.small_model},"
            f"large_model_name={args.big_model},"
            f"confidence_threshold={args.confidence_threshold},"
            f"max_step_tokens={args.max_step_tokens},"
            f"max_gen_tokens={args.max_gen_tokens}"
        )

        eval_cmd = [
            sys.executable,
            "-c", pre_load_code,

            "--model", "vllm_router",
            "--model_args", model_args_str,
            "--tasks", args.tasks,
            "--output_path", args.output_path,
            "--batch_size", str(args.batch_size),
            "--log_samples"
        ]
        
        if args.limit:
            eval_cmd.extend(["--limit", str(args.limit)])
        if args.num_fewshot:
            eval_cmd.extend(["--num_fewshot", str(args.num_fewshot)])
        if args.apply_chat_template:
            eval_cmd.append("--apply_chat_template")
            
        result = subprocess.run(eval_cmd)
        
        if result.returncode != 0:
            print("lm-eval 执行失败!!!")
            sys.exit(1)
            
        print("评估完成")
        
    except Exception as e:
        print(f"error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("! 正在关闭 vLLM 服务...")
        for p in procs:
            if p:
                p.terminate()
                try:
                    p.wait(timeout=5)
                except:
                    p.kill()
        print("✅ 服务已关闭。")

if __name__ == "__main__":
    main()