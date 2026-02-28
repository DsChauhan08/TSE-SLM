#!/usr/bin/env python3
"""
Evaluate a model loaded in LM Studio with lm-eval over a sweep of temperatures.

Instructions:
1. Open LM Studio.
2. Load the model you want to test (e.g., Qwen 4B, Phi-4).
3. Start the "Local Server" in LM Studio (default port is 1234).
4. Run this script: python scripts/evaluate_lm_studio.py --model_name "qwen-4b-q4" --dataset gsm8k

This script connects to LM Studio's OpenAI-compatible API and runs the temperature sweep.
"""

import argparse
import json
import os
import time
from pathlib import Path

try:
    import lm_eval
except ImportError:
    print("Missing lm-eval. Please install it: pip install lm-eval")
    exit(1)

def run_evaluation(model_name: str, dataset: str, max_samples: int, output_dir: str, base_url: str):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    print("
==========================================================")
    print(f"Starting LM Studio pipeline for: {model_name}")
    print(f"Connecting to server at: {base_url}")
    print("Ensure your model is loaded and the LM Studio server is running!")
    print("==========================================================
")

    temperatures = [round(i * 0.1, 1) for i in range(21)]
    results_summary = []

    for temp in temperatures:
        print(f"
--- Evaluating {model_name} at T={temp} ---")
        
        do_sample = temp > 0.0
        hf_temp = temp if do_sample else 0.0 # OpenAI/LM Studio API usually accepts 0.0 for greedy
        
        start_time = time.time()
        try:
            # We use the built-in 'local-chat-completions' or 'local-completions' 
            # model type which expects an OpenAI-compatible API.
            # Using local-chat-completions for instruct models is usually better.
            model_args = f"model={model_name},base_url={base_url}"
            
            # Since lm-eval's API model doesn't always accept dynamic kwargs in simple_evaluate easily,
            # we inject them via the model_args string for the OpenAI API wrapper.
            # E.g. temperature=0.5,top_p=1.0
            model_args += f",temperature={hf_temp},top_p=1.0"

            results = lm_eval.simple_evaluate(
                model="local-chat-completions",
                model_args=model_args,
                tasks=[dataset],
                num_fewshot=0,
                limit=max_samples if max_samples > 0 else None,
            )
            
            elapsed = time.time() - start_time
            task_results = results.get('results', {}).get(dataset, {})
            
            # Extract accuracy
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

        results_summary.append(record)
        
        # Save incremental results
        safe_model_name = model_name.replace('/', '_').replace('', '_')
        out_file = Path(output_dir) / f"LM_Studio_{safe_model_name}_{dataset}_results.json"
        with open(out_file, "w") as f:
            json.dump(results_summary, f, indent=2)
            
    print(f"
Evaluation sweep complete for {model_name}!")
    print(f"Results saved to: {out_file}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate a model loaded in LM Studio.")
    parser.add_argument("--model_name", type=str, required=True, help="A name to identify your run (e.g. qwen3-4b-q4)")
    parser.add_argument("--dataset", type=str, default="gsm8k", help="lm-eval task name (e.g., gsm8k, arc_easy)")
    parser.add_argument("--limit", type=int, default=10, help="Number of samples to evaluate (0 for full dataset)")
    parser.add_argument("--output_dir", type=str, default="data/results", help="Output directory")
    parser.add_argument("--base_url", type=str, default="http://127.0.0.1:1234/v1", help="LM Studio API base URL")
    args = parser.parse_args()

    run_evaluation(args.model_name, args.dataset, args.limit, args.output_dir, args.base_url)

if __name__ == "__main__":
    main()