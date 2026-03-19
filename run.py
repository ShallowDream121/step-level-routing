import argparse
import subprocess
import sys
import time
import os
import signal
from pathlib import Path

def start_vllm_server(model_path, port, gpu_id):
    """启动 vLLM 服务"""
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.api_server",
        "--model", model_path,
        "--port", str(port),
        "--tensor-parallel-size", "1",
        "--gpu-memory-utilization", "0.9",
        "--disable-log-stats",
        "--enforce-eager" # 调试时可加，生产环境可去掉
    ]
    
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    print(f"🚀 启动服务: GPU {gpu_id}, Port {port}, Model {os.path.basename(model_path)}")
    proc = subprocess.Popen(cmd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    return proc

def wait_for_health(port, timeout=120):
    """等待服务健康"""
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

def main():
    parser = argparse.ArgumentParser(description="Run lm-eval with Speculative Router")
    
    # 模型配置
    parser.add_argument("--big_model", type=str, required=True)
    parser.add_argument("--small_model", type=str, required=True)
    parser.add_argument("--big_gpu", type=str, default="0")
    parser.add_argument("--small_gpu", type=str, default="1")
    parser.add_argument("--small_port", type=int, default=8000)
    parser.add_argument("--large_port", type=int, default=8001)
    
    # 路由参数
    parser.add_argument("--confidence_threshold", type=float, default=0.6)
    parser.add_argument("--max_step_tokens", type=int, default=256)
    parser.add_argument("--max_gen_tokens", type=int, default=2048)
    
    # lm-eval 参数 (透传)
    parser.add_argument("--tasks", type=str, required=True, help="e.g., aime25_nofigures,mmlu")
    parser.add_argument("--output_path", type=str, default="./results")
    parser.add_argument("--batch_size", type=str, default="auto")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--num_fewshot", type=int, default=0)
    parser.add_argument("--apply_chat_template", action="store_true")
    
    args = parser.parse_args()
    
    procs = []
    try:
        # 1. 启动双模型服务
        p_small = start_vllm_server(args.small_model, args.small_port, args.small_gpu)
        p_large = start_vllm_server(args.big_model, args.large_port, args.big_gpu)
        procs = [p_small, p_large]
        
        # 2. 等待就绪
        if not wait_for_health(args.small_port):
            raise RuntimeError("Small model failed to start")
        if not wait_for_health(args.large_port):
            raise RuntimeError("Large model failed to start")
        print("✅ 所有服务已就绪，开始运行 lm-eval...")
        
        # 3. 构建 lm-eval 命令
        # 关键点：--model vllm_spec_router (对应 register_model 的名字)
        # --model_args 传递 URL 和阈值
        model_args_str = (
            f"small_model_url=http://localhost:{args.small_port},"
            f"large_model_url=http://localhost:{args.large_port},"
            f"confidence_threshold={args.confidence_threshold},"
            f"max_step_tokens={args.max_step_tokens},"
            f"max_gen_tokens={args.max_gen_tokens}"
        )
        
        eval_cmd = [
            sys.executable, "-m", "lm_eval",
            "--model", "vllm_spec_router",
            "--model_args", model_args_str,
            "--tasks", args.tasks,
            "--output_path", args.output_path,
            "--batch_size", args.batch_size,
            "--log_samples"
        ]
        
        if args.limit:
            eval_cmd.extend(["--limit", str(args.limit)])
        if args.num_fewshot:
            eval_cmd.extend(["--num_fewshot", str(args.num_fewshot)])
        if args.apply_chat_template:
            eval_cmd.append("--apply_chat_template")
            
        # 4. 执行 lm-eval
        # 注意：这里直接调用 subprocess run，它会阻塞直到 lm-eval 结束
        result = subprocess.run(eval_cmd)
        
        if result.returncode != 0:
            print("❌ lm-eval 执行失败")
            sys.exit(1)
            
        print("🎉 评估完成！")
        
    except Exception as e:
        print(f"❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 5. 清理进程
        print("🛑 正在关闭 vLLM 服务...")
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