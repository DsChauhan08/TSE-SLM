#!/usr/bin/env python3
"""
Evaluate a HuggingFace model with lm-eval over a sweep of temperatures.

Notes:
- This script tries to be robust but depends on the lm-eval harness API (HFLM).
- If lm-eval/HFLM changes, you may need to adjust how gen_kwargs are passed.
- For small tests use --limit small (e.g. 5 or 10). Full datasets will take long.
"""

import argparse
import json
import os
import time
from pathlib import Path

# torch is used to detect device; import early to avoid NameError
import torch

# lm-eval imports
try:
    import lm_eval
    from lm_eval.models.huggingface import HFLM
    from lm_eval import simple_evaluate
except Exception:
    print("Missing lm-eval or related packages. See the setup script to install dependencies.")
    raise

def run_evaluation(model_name: str, dataset: str, max_samples: int, output_dir: str):
    """
    Runs evaluation for a single model and dataset across a range of temperatures.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # 0.0 to 2.0 with 0.1 steps
    temperatures = [round(i * 0.1, 1) for i in range(21)]
    results_summary = []

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model: {model_name} on device: {device} ...")

    # We'll initialize the HFLM once and update its gen_kwargs inside the loop.
    # If HFLM doesn't accept dynamic updates to gen_kwargs, we'll re-init inside the loop (handled below).
    lm = None

    for temp in temperatures:
        print("\n======================================")
        print(f"Evaluating T={temp}")
        print("======================================")

        # At T=0.0, do greedy decoding. Otherwise, use sampling.
        do_sample = temp > 0.0
        # When do_sample=False, temperature must be 1.0 or None for HF; use 1.0 to ensure greedy behavior.
        hf_temp = temp if do_sample else 1.0

        gen_kwargs = {"temperature": hf_temp, "do_sample": do_sample, "top_p": 1.0}

        start_time = time.time()
        try:
            # Try to reuse the same HFLM instance and set gen_kwargs attribute if supported
            if lm is None:
                lm = HFLM(pretrained=model_name, device=device)
            # Try to set a gen_kwargs attribute on the HFLM wrapper. This may or may not be used by the harness.
            try:
                setattr(lm, "gen_kwargs", gen_kwargs)
            except Exception:
                # If we cannot set it, fallback to re-initialize per-temp with a gen_kwargs constructor arg (some versions accept it).
                lm = HFLM(pretrained=model_name, device=device, gen_kwargs=gen_kwargs)
        except Exception as e:
            print("Failed to initialize or update HFLM model. Error:")
            print(e)
            print("Attempting to re-initialize HFLM for this temperature...")
            try:
                lm = HFLM(pretrained=model_name, device=device, gen_kwargs=gen_kwargs)
            except Exception as e2:
                print("Re-initialization also failed. Aborting this temperature with an error.")
                print(e2)
                elapsed = time.time() - start_time
                results_summary.append({
                    "temperature": temp,
                    "error": str(e2),
                    "time_seconds": elapsed
                })
                # Save partial results and continue
                out_file = Path(output_dir) / f"{model_name.replace('/', '_')}_{dataset}_results.json"
                with open(out_file, "w") as f:
                    json.dump(results_summary, f, indent=2)
                continue

        # Now run lm_eval.simple_evaluate for the given dataset
        try:
            results = lm_eval.simple_evaluate(
                model=lm,
                tasks=[dataset],
                num_fewshot=0,
                limit=max_samples,
            )
        except Exception as e:
            # If the harness needs gen_kwargs passed differently, capture the error and save partial results
            print(f"lm_eval.simple_evaluate failed at T={temp} with error:")
            print(e)
            elapsed = time.time() - start_time
            results_summary.append({
                "temperature": temp,
                "error": str(e),
                "time_seconds": elapsed
            })
            out_file = Path(output_dir) / f"{model_name.replace('/', '_')}_{dataset}_results.json"
            with open(out_file, "w") as f:
                json.dump(results_summary, f, indent=2)
            continue

        elapsed = time.time() - start_time

        # Extract accuracy (the metric name can vary by task)
        task_results = results.get("results", {}).get(dataset, {})
        acc = task_results.get("acc,none", task_results.get("exact_match,none", None))

        print(f"Result for T={temp}: {acc} (Time: {elapsed:.2f}s)")

        record = {
            "temperature": temp,
            "accuracy": acc,
            "time_seconds": elapsed,
            "raw_results": task_results
        }
        results_summary.append(record)

        # Save incremental results (overwrites each loop but keeps full list)
        timestamp = int(time.time())
        out_file = Path(output_dir) / f"{model_name.replace('/', '_')}_{dataset}_results.json"
        with open(out_file, "w") as f:
            json.dump(results_summary, f, indent=2)

    print("Evaluation sweep complete.")
    return results_summary

def main():
    parser = argparse.ArgumentParser(description="Evaluate SLMs at varying temperatures.")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B", help="Hugging Face model ID")
    parser.add_argument("--dataset", type=str, default="gsm8k", help="lm-eval task name (e.g., gsm8k, arc_easy)")
    parser.add_argument("--limit", type=int, default=10, help="Number of samples to evaluate (for testing)")
    parser.add_argument("--output_dir", type=str, default="data/results", help="Output directory")
    args = parser.parse_args()

    print("Starting evaluation pipeline...")
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"Limit: {args.limit} samples per temperature")
    print(f"Output dir: {args.output_dir}")

    run_evaluation(args.model, args.dataset, args.limit, args.output_dir)

if __name__ == "__main__":
    main()
