import argparse
import subprocess
import sys
import random
import time
import os
import json
import logging
import signal
import yaml
import glob
import requests
import vllm_plan_router
import vllm_router
import vllm_small
import vllm_big
import vllm_random
import traceback
from lm_eval import evaluator, tasks
from lm_eval.tasks import TaskManager 
from lm_eval.utils import make_table
from pathlib import Path


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def start_vllm_server(model_path, port, gpu_id):
    random_seed = random.randint(1, 1000000)
    cmd = [
        sys.executable, 
        "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_path,
        "--port", str(port),
        "--host", "0.0.0.0",       
        "--dtype", "float16",
        "--tensor-parallel-size", "1",
        "--gpu-memory-utilization", "0.9",
        "--max-model-len", "32768",  
        "--max-logprobs", "20",    
        "--seed", str(random_seed),
        "--disable-log-stats",
        "--enforce-eager"  
    ]
    
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["MLIR_ENABLE_REMARK"] = "1" 

    print(f"Starting vllm server: GPU {gpu_id}, Port {port}, Model {model_path}")
    proc = subprocess.Popen(cmd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

    return proc

def wait_for_health(port, timeout=180):
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

    # --- 模式选择 ---
    parser.add_argument(
        "--mode", type=str, default="mix",
        help="Mode to run test."
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

    # ---- 其他参数 ---
    parser.add_argument(
        "--debug", action="store_true", default=False,
        help="Enable debug mode."
    )

    return parser.parse_args()

def main():
    args = parse_arguments()
    
    procs = []
    try:
        if not args.mode == "big_only":
            logger.info(f"正在启动小模型服务: {args.small_model} (Port: {args.small_port})")
            p_small = start_vllm_server(args.small_model, args.small_port, args.small_gpu)
            print("Waiting for small model...")
            if not wait_for_health(args.small_port):
                raise RuntimeError("Small model failed to start")
            procs.append(p_small)
            
        if not args.mode == "small_only":
            logger.info(f"正在启动大模型服务: {args.big_model} (Port: {args.large_port})")
            p_large = start_vllm_server(args.big_model, args.large_port, args.big_gpu)
            print("Waiting for large model...")
            if not wait_for_health(args.large_port):
                raise RuntimeError("Large model failed to start")
            procs.append(p_large)

        print("所有 model 服务已就绪，开始准备评估环境...")

        current_dir = os.path.dirname(os.path.abspath(__file__))
        if current_dir not in sys.path:
            sys.path.insert(0, current_dir)

        logger.info("正在强制导入 router 模块...")
        try:
            logger.info("router 导入成功，模型已注册到 lm-eval。")
        except Exception as e:
            logger.error(f"导入 router 失败！请检查缩进或语法错误。\n错误详情: {e}")
            traceback.print_exc()
            sys.exit(1)

        model_args_dict = {}
        model_args_str = (
            f"small_url=http://localhost:{args.small_port},"
            f"large_url=http://localhost:{args.large_port},"
            f"small_model_name={args.small_model},"
            f"large_model_name={args.big_model},"
            f"confidence_threshold={args.confidence_threshold},"
            f"max_step_tokens={args.max_step_tokens},"
            f"max_gen_tokens={args.max_gen_tokens},"
            f"debug={str(args.debug)}"
        )
        
        for item in model_args_str.split(","):
            if "=" in item:
                k, v = item.split("=", 1)
                model_args_dict[k.strip()] = v.strip()
        
        model_args_dict["trust_remote_code"] = "True"
        
        final_task_list = args.tasks.split(",") if isinstance(args.tasks, str) else args.tasks
        
        logger.info(f"即将执行的任务列表: {final_task_list}")

        try:
            logger.info("开始运行 simple_evaluate (这可能需要几分钟)...")
            
            gen_kwargs_override = {
                "do_sample": True,
                "until": ["</think>"],                 
                "add_bos_token": False
            
            }

            results = None
            if args.mode == "mix":
                logger.info("使用混合路由进行评估...")
                results = evaluator.simple_evaluate(
                    model="vllm_plan_router",       
                    model_args=model_args_dict,  
                    tasks=final_task_list,       
                    batch_size=str(args.batch_size),
                    log_samples=True,
                    gen_kwargs=gen_kwargs_override,
                    apply_chat_template=args.apply_chat_template,
                    random_seed=0,
                    numpy_random_seed=1234,
                    torch_random_seed=1234,
                    fewshot_random_seed=1234,
                    limit=args.limit,
                    num_fewshot=args.num_fewshot if args.num_fewshot else None
                )
            elif args.mode == "random":
                logger.info("使用随机路由进行评估...")
                results = evaluator.simple_evaluate(
                    model="vllm_random",       
                    model_args=model_args_dict,  
                    tasks=final_task_list,       
                    batch_size=str(args.batch_size),
                    log_samples=True,
                    gen_kwargs=gen_kwargs_override,
                    apply_chat_template=args.apply_chat_template,
                    random_seed=0,
                    numpy_random_seed=1234,
                    torch_random_seed=1234,
                    fewshot_random_seed=1234,
                    limit=args.limit,
                    num_fewshot=args.num_fewshot if args.num_fewshot else None
                )
            elif args.mode == "big_only":
                logger.info("仅使用大模型进行评估...")
                results = evaluator.simple_evaluate(
                    model="vllm_big",       
                    model_args=model_args_dict,  
                    tasks=final_task_list,       
                    batch_size=str(args.batch_size),
                    log_samples=True,
                    gen_kwargs=gen_kwargs_override,
                    apply_chat_template=args.apply_chat_template,
                    random_seed=0,
                    numpy_random_seed=1234,
                    torch_random_seed=1234,
                    fewshot_random_seed=1234,
                    limit=args.limit,
                    num_fewshot=args.num_fewshot if args.num_fewshot else None
                )
            else:
                logger.info("仅使用小模型进行评估...")
                results = evaluator.simple_evaluate(
                    model="vllm_small",       
                    model_args=model_args_dict,  
                    tasks=final_task_list,       
                    batch_size=str(args.batch_size),
                    log_samples=True,
                    gen_kwargs=gen_kwargs_override,
                    apply_chat_template=args.apply_chat_template,
                    random_seed=0,
                    numpy_random_seed=1234,
                    torch_random_seed=1234,
                    fewshot_random_seed=1234,
                    limit=args.limit,
                    num_fewshot=args.num_fewshot if args.num_fewshot else None
                )
            
            if not results:
                raise ValueError("评估返回了无效的结果")

            if args.output_path:
                from datetime import datetime
                os.makedirs(args.output_path, exist_ok=True)

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_file = os.path.join(args.output_path, f"results_{timestamp}.json")
                
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=2, ensure_ascii=False, default=str)
                
                logger.info(f"评估结果已保存至: {output_file}")
            else:
                logger.info("未指定 output_path，结果仅打印在控制台。")

            print("\n" + "="*50)
            print("评估完成！结果摘要:")
            print("="*50)
            try:
                print(make_table(results))
            except Exception as e:
                print("表格生成失败，直接打印结果字典:")
                print(results.get("results", {}))
                
        except Exception as e:
            logger.error(f"评估过程中发生严重错误: {e}")
            traceback.print_exc()
            sys.exit(1)
            
        print("评估流程结束，正在关闭服务...")

    except Exception as e:
        print(f"主程序发生异常: {e}")
        traceback.print_exc()
    finally:
        print("! 正在关闭 vLLM 服务...")
        for p in procs:
            if p:
                try:
                    p.terminate()
                    p.wait(timeout=5)
                except Exception:
                    p.kill()
        print("服务已关闭。")


if __name__ == "__main__":
    main()