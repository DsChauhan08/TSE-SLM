#!/usr/bin/env python3
import argparse
import json
import os
import time
import subprocess
import socket
import sys
from pathlib import Path
from huggingface_hub import list_repo_files, hf_hub_download

import lm_eval
from lm_eval.models.gguf import GGUFLM

def patched_completion(self, context, continuation=None, stop=None, retries=3, delay=5, **kwargs):
    import requests
    from requests.exceptions import RequestException
    import logging
    logger = logging.getLogger(__name__)
    
    for _ in range(retries):
        try:
            prompt = context
            request = {
                "prompt": prompt,
                "logprobs": self.logprobs,
                "temperature": self.temperature,
                "top_p": getattr(self, "top_p", 1.0)
            }
            top_k = getattr(self, "top_k", 0)
            if top_k and top_k > 0:
                request["top_k"] = int(top_k)
            if continuation:
                prompt += continuation
                request.update({"prompt": prompt, "max_tokens": 1, "echo": True})
            if stop is not None:
                request["stop"] = stop
            response = requests.post(f"{self.base_url}/v1/completions", json=request)
            response.raise_for_status()
            return response.json()
        except RequestException as e:
            logger.error(f"RequestException: {e}")
            time.sleep(delay)
    raise RuntimeError(f"Failed to get a valid response after {retries} retries.")

GGUFLM.gguf_completion = patched_completion

def supports_gpu_offload() -> bool:
    try:
        from llama_cpp import llama_cpp as low_level
        fn = getattr(low_level, "llama_supports_gpu_offload", None)
        if fn is None:
            return False
        return bool(fn())
    except Exception:
        return False

def pick_q4_gguf_file(files):
    q4_files = [f for f in files if f.lower().endswith(".gguf") and "Q4" in f.upper()]
    if not q4_files:
        return None

    preferred_markers = ["Q4_K_M", "Q4_K_S", "Q4_0", "Q4_1"]
    for marker in preferred_markers:
        for candidate in q4_files:
            if marker in candidate.upper():
                return candidate
    return q4_files[0]

def wait_for_port(port, host='localhost', timeout=60.0):
    start_time = time.monotonic()
    while True:
        try:
            with socket.create_connection((host, port), timeout=1):
                return True
        except OSError:
            time.sleep(1)
            if time.monotonic() - start_time > timeout:
                return False

def parse_temperatures(raw_temperatures: str):
    if not raw_temperatures:
        return [round(i * 0.1, 1) for i in range(21)]

    parsed = []
    seen = set()
    for chunk in raw_temperatures.split(","):
        value = round(float(chunk.strip()), 1)
        if value not in seen:
            parsed.append(value)
            seen.add(value)
    if not parsed:
        raise ValueError("No valid temperatures provided.")
    return parsed

def run_evaluation(
    repo_id: str,
    dataset: str,
    max_samples: int,
    output_dir: str,
    temperatures,
    startup_timeout: int,
    n_threads: int,
    n_gpu_layers: int,
    top_p: float,
    top_k: int,
    run_tag: str,
    seed: int,
):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    safe_repo_name = repo_id.replace('/', '_')
    safe_run_tag = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in run_tag).strip("_")
    tag_suffix = f"_{safe_run_tag}" if safe_run_tag else ""
    out_file = Path(output_dir) / f"{safe_repo_name}_{dataset}{tag_suffix}_results.json"
    
    print("\n======================================")
    print(f"Starting pipeline for {repo_id}")
    print("======================================")

    results_by_temp = {}
    if out_file.exists():
        try:
            loaded = json.loads(out_file.read_text())
            if isinstance(loaded, list):
                for row in loaded:
                    if not isinstance(row, dict):
                        continue
                    temp = row.get("temperature")
                    if temp is None:
                        continue
                    try:
                        key = round(float(temp), 1)
                    except (TypeError, ValueError):
                        continue
                    results_by_temp[key] = row
            print(f"Loaded existing results from {out_file} (records={len(results_by_temp)}).")
        except Exception as e:
            print(f"Warning: could not parse existing results file {out_file}: {e}")

    completed_temps = {t for t, row in results_by_temp.items() if "error" not in row}
    temperatures_to_run = [t for t in temperatures if round(float(t), 1) not in completed_temps]
    if not temperatures_to_run:
        print("All requested temperatures already completed; skipping evaluation run.")
        return True
    
    print(f"Looking for a 4-bit GGUF file in {repo_id}...")
    files = list_repo_files(repo_id)
    filename = pick_q4_gguf_file(files)

    if not filename:
        print(f"Error: No Q4 GGUF file found in {repo_id}")
        return False

    print(f"Found: {filename}")
    
    print(f"Downloading {filename} (this may take a while)...")
    file_path = hf_hub_download(repo_id=repo_id, filename=filename, local_dir=".")
    print(f"Downloaded to {file_path}")
    
    port = 8000
    server_cmd = [
        sys.executable, "-m", "llama_cpp.server",
        "--model", file_path,
        "--n_ctx", "4096",
        "--port", str(port),
        "--host", "127.0.0.1"
    ]
    if n_threads and n_threads > 0:
        server_cmd.extend(["--n_threads", str(n_threads)])
    if n_gpu_layers != 0:
        if supports_gpu_offload():
            server_cmd.extend(["--n_gpu_layers", str(n_gpu_layers)])
        else:
            print("Warning: GPU offload requested but current llama.cpp build has no GPU-offload support; continuing on CPU.")
    if seed >= 0:
        server_cmd.extend(["--seed", str(seed)])
    
    print(f"Starting llama_cpp server: {' '.join(server_cmd)}")
    server_process = subprocess.Popen(
        server_cmd, 
        stdout=subprocess.DEVNULL, 
        stderr=subprocess.DEVNULL,
        preexec_fn=os.setsid
    )
    
    if not wait_for_port(port, timeout=float(startup_timeout)):
        print(f"Error: Server failed to start within {startup_timeout}s timeout.")
        server_process.kill()
        return False

    print("Server is up and running!")
    
    lm = GGUFLM(base_url=f"http://127.0.0.1:{port}")
    lm.top_p = top_p
    if top_k > 0:
        lm.top_k = top_k

    for temp in temperatures_to_run:
        print(f"\n--- Evaluating {filename} at T={temp} ---")
        lm.temperature = temp
        
        start_time = time.time()
        try:
            results = lm_eval.simple_evaluate(
                model=lm,
                tasks=[dataset],
                num_fewshot=0,
                limit=max_samples if max_samples > 0 else None,
            )
            
            elapsed = time.time() - start_time
            task_results = results.get('results', {}).get(dataset, {})
            
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
        except Exception as e:
            print(f"Evaluation failed at T={temp} with error: {e}")
            elapsed = time.time() - start_time
            record = {
                "temperature": temp,
                "error": str(e),
                "time_seconds": elapsed
            }

        results_by_temp[round(float(temp), 1)] = record
        sorted_results = [results_by_temp[t] for t in sorted(results_by_temp.keys())]
        with open(out_file, "w") as f:
            json.dump(sorted_results, f, indent=2)
            
    print(f"\nEvaluation sweep complete for {repo_id}!")
    
    print("Shutting down server...")
    import signal
    try:
        os.killpg(os.getpgid(server_process.pid), signal.SIGTERM)
        server_process.wait(timeout=10)
    except Exception as e:
        print(f"Error shutting down server: {e}")
    
    print(f"Deleting model file {file_path} to free space...")
    try:
        os.remove(file_path)
        print("Deleted successfully.")
    except Exception as e:
        print(f"Failed to delete file: {e}")

    had_error = any("error" in row for row in results_by_temp.values())
    return not had_error

def main():
    parser = argparse.ArgumentParser(description="Evaluate multiple GGUF SLMs locally.")
    parser.add_argument("--dataset", type=str, default="gsm8k", help="lm-eval task name")
    parser.add_argument("--limit", type=int, default=10, help="Number of samples to evaluate (0 for full dataset)")
    parser.add_argument("--output_dir", type=str, default="data/results", help="Output directory")
    parser.add_argument("--model", action="append", help="Hugging Face GGUF repo ID (repeat for multiple)")
    parser.add_argument("--temperatures", type=str, default="", help="Comma-separated temperature list (e.g. 0.0,0.7,1.4)")
    parser.add_argument("--startup_timeout", type=int, default=300, help="llama.cpp server startup timeout in seconds")
    parser.add_argument("--n_threads", type=int, default=0, help="llama.cpp server CPU threads (0 = default)")
    parser.add_argument("--n_gpu_layers", type=int, default=0, help="llama.cpp GPU layers (-1 = all layers, 0 = CPU/default)")
    parser.add_argument("--top_p", type=float, default=1.0, help="Sampling top_p value")
    parser.add_argument("--top_k", type=int, default=0, help="Sampling top_k value (0 = disabled)")
    parser.add_argument("--run_tag", type=str, default="", help="Optional suffix tag for output filename")
    parser.add_argument("--seed", type=int, default=-1, help="Server seed (-1 = default/random)")
    args = parser.parse_args()

    default_models = [
        "mradermacher/Llama-3.1-Minitron-4B-Width-Base-GGUF",
        "unsloth/Qwen3-4B-Instruct-2507-GGUF",
        "TeichAI/Qwen3-4B-Thinking-2507-Claude-4.5-Opus-High-Reasoning-Distill-GGUF",
        "unsloth/Phi-4-mini-instruct-GGUF"
    ]
    models_to_test = args.model if args.model else default_models
    temperatures = parse_temperatures(args.temperatures)
    had_failures = False

    print("Starting LOCAL CPU evaluation pipeline...")
    for model_id in models_to_test:
        ok = run_evaluation(
            model_id,
            args.dataset,
            args.limit,
            args.output_dir,
            temperatures,
            args.startup_timeout,
            args.n_threads,
            args.n_gpu_layers,
            args.top_p,
            args.top_k,
            args.run_tag,
            args.seed,
        )
        if not ok:
            had_failures = True

    if had_failures:
        raise SystemExit(1)
        
if __name__ == "__main__":
    main()
