#!/usr/bin/env python3
"""
Evaluate local GGUF models directly by spinning up an ephemeral llama-cpp-python server.
"""

import argparse
import json
import os
import time
import subprocess
import socket
import signal
from pathlib import Path
import tempfile

def wait_for_port(port, host='localhost', timeout=60.0):
    start_time = time.time()
    while True:
        try:
            with socket.create_connection((host, port), timeout=1):
                return True
        except OSError:
            time.sleep(1)
            if time.time() - start_time > timeout:
                return False

def run_evaluation_direct(model_path: str, model_name: str, dataset: str, max_samples: int, output_dir: str):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    print("\n=======================================================")
    print(f"Evaluating: {model_name}")
    print(f"Path: {model_path}")
    print("=======================================================")
    
    if not os.path.exists(model_path):
        print(f"Error: Model path does not exist: {model_path}")
        return
        
    port = 8000
    server_cmd = [
        "python", "-m", "llama_cpp.server",
        "--model", model_path,
        "--n_ctx", "4096",
        "--port", str(port),
        "--host", "127.0.0.1"
    ]
    
    print(f"Starting ephemeral llama_cpp server...")
    server_process = subprocess.Popen(
        server_cmd, 
        stdout=subprocess.DEVNULL, 
        stderr=subprocess.DEVNULL,
        preexec_fn=os.setsid
    )
    
    if not wait_for_port(port):
        print("Error: Server failed to start within timeout.")
        try:
            os.killpg(os.getpgid(server_process.pid), signal.SIGTERM)
        except:
            pass
        return

    print("Server is up and running!")

    temperatures = [round(i * 0.1, 1) for i in range(21)]
    results_summary = []
    
    base_lm_eval_cmd = [
        "lm_eval",
        "--model", "local-chat-completions",
        "--tasks", dataset,
        "--apply_chat_template",
        "--num_fewshot", "0"
    ]
    
    if max_samples > 0:
        base_lm_eval_cmd.extend(["--limit", str(max_samples)])

    for temp in temperatures:
        print(f"\n--- Evaluating {model_name} at T={temp} ---")
        
        do_sample = temp > 0.0
        # If not sampling, pass T=0.0
        model_args = f"model={model_name},base_url=http://127.0.0.1:{port}/v1,temperature={temp if do_sample else 0.0}"
        if do_sample:
            model_args += ",top_p=1.0"

        with tempfile.TemporaryDirectory() as tmpdir:
            cmd = base_lm_eval_cmd + [
                "--model_args", model_args,
                "--output_path", tmpdir
            ]
            
            start_time = time.time()
            try:
                # We use subprocess here to interact with the CLI because the python API
                # for local-completions is occasionally finicky with dynamic argument injection
                result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                elapsed = time.time() - start_time
                
                # Read the results from the temporary directory
                # lm-eval creates a nested structure in the output_path
                result_file = None
                for root, _, files in os.walk(tmpdir):
                    for file in files:
                        if file.endswith(".json") and "results" in file:
                            result_file = os.path.join(root, file)
                            break
                    if result_file:
                        break
                
                if result_file and os.path.exists(result_file):
                    with open(result_file, 'r') as f:
                        eval_data = json.load(f)
                    
                    task_results = eval_data.get('results', {}).get(dataset, {})
                    acc = task_results.get('exact_match,strict-match', 
                          task_results.get('exact_match,flexible-extract', 
                          task_results.get('acc,none', 0.0)))
                    
                    print(f"Result for T={temp}: {acc} (Time: {elapsed:.2f}s)")
                    
                    record = {
                        "temperature": temp,
                        "accuracy": acc,
                        "time_seconds": elapsed,
                        "raw_results": task_results
                    }
                else:
                    raise FileNotFoundError("Results JSON not found in output directory")
                
            except subprocess.CalledProcessError as e:
                print(f"Evaluation failed at T={temp}")
                print(f"Stderr: {e.stderr}")
                elapsed = time.time() - start_time
                record = {
                    "temperature": temp,
                    "error": "Subprocess failed",
                    "time_seconds": elapsed
                }
            except Exception as e:
                print(f"Failed to parse results: {e}")
                elapsed = time.time() - start_time
                record = {
                    "temperature": temp,
                    "error": str(e),
                    "time_seconds": elapsed
                }

        results_summary.append(record)
        
        # Save incremental results
        safe_model_name = model_name.replace('/', '_').replace(' ', '_')
        out_file = Path(output_dir) / f"{safe_model_name}_{dataset}_results.json"
        with open(out_file, "w") as f:
            json.dump(results_summary, f, indent=2)
            
    print(f"\nEvaluation sweep complete for {model_name}!")
    
    # Clean up the server
    print("Shutting down server...")
    try:
        os.killpg(os.getpgid(server_process.pid), signal.SIGTERM)
        server_process.wait(timeout=10)
    except Exception as e:
        print(f"Error shutting down server: {e}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate multiple local GGUF models directly.")
    parser.add_argument("--dataset", type=str, default="gsm8k", help="lm-eval task name")
    parser.add_argument("--limit", type=int, default=10, help="Number of samples to evaluate (0 for full dataset)")
    parser.add_argument("--output_dir", type=str, default="data/results", help="Output directory")
    args = parser.parse_args()

    # Pre-defined map of the exact paths found on your system
    models_to_test = [
        {
            "name": "Llama-3.1-Minitron-4B-Width-Base",
            "path": "/home/regulus/.lmstudio/models/mradermacher/Llama-3.1-Minitron-4B-Width-Base-GGUF/Llama-3.1-Minitron-4B-Width-Base.Q4_K_S.gguf"
        },
        {
            "name": "Qwen3-4B-Instruct-2507",
            "path": "/home/regulus/.lmstudio/models/unsloth/Qwen3-4B-Instruct-2507-GGUF/Qwen3-4B-Instruct-2507-Q4_K_S.gguf"
        },
        {
            "name": "Qwen3-4B-Thinking-2507-Claude-4.5-Opus-High-Reasoning-Distill",
            "path": "/home/regulus/.lmstudio/models/TeichAI/Qwen3-4B-Thinking-2507-Claude-4.5-Opus-High-Reasoning-Distill-GGUF/Qwen3-4B-Thinking-2507-Claude-4.5-Opus-High-Reasoning-Distill.q4_k_m.gguf"
        },
        {
            "name": "Phi-4-mini-instruct",
            "path": "/home/regulus/.lmstudio/models/unsloth/Phi-4-mini-instruct-GGUF/Phi-4-mini-instruct-Q4_K_M.gguf"
        }
    ]

    print("Starting automated local GGUF evaluation pipeline...")
    for model_info in models_to_test:
        run_evaluation_direct(
            model_path=model_info["path"], 
            model_name=model_info["name"], 
            dataset=args.dataset, 
            max_samples=args.limit, 
            output_dir=args.output_dir
        )
        
if __name__ == "__main__":
    main()