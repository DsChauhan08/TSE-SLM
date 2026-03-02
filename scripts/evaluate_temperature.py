#!/usr/bin/env python3
"""
Evaluate a HuggingFace model with lm-eval over a sweep of temperatures.
Supports 4-bit and 8-bit quantization for larger models.
"""

import argparse
import json
import os
import time
from pathlib import Path
import torch

try:
    import lm_eval
    from lm_eval.models.huggingface import HFLM
except ImportError:
    print("Missing lm-eval. Please install it: pip install lm-eval")
    exit(1)

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

def run_evaluation(model_name: str, dataset: str, max_samples: int, output_dir: str, quantization: str, temperatures):
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    results_summary = []

    print(f"Loading model: {model_name} with quantization: {quantization} ...")

    # Setup kwargs for model initialization
    kwargs = {
        "pretrained": model_name,
        "backend": "causal",
        "trust_remote_code": True,
    }

    model_name_lower = model_name.lower()
    is_prequantized_4bit_repo = ("4bit" in model_name_lower) or ("bnb" in model_name_lower)

    if quantization == "4bit":
        if is_prequantized_4bit_repo:
            print("Detected pre-quantized 4-bit model repo; skipping explicit load_in_4bit flag.")
        else:
            kwargs["load_in_4bit"] = True
    elif quantization == "8bit":
        kwargs["load_in_8bit"] = True
    else:
        kwargs["device"] = "cuda" if torch.cuda.is_available() else "cpu"

    # Initialize the HFLM once
    try:
        lm = HFLM(**kwargs)
    except Exception as e:
        print(f"Failed to load model with initial kwargs, retrying with minimal kwargs. Error: {e}")
        fallback_kwargs = {
            "pretrained": model_name,
            "backend": "causal",
            "trust_remote_code": True,
        }
        if quantization == "none":
            fallback_kwargs["device"] = "cuda" if torch.cuda.is_available() else "cpu"
        lm = HFLM(**fallback_kwargs)

    for temp in temperatures:
        print("\n======================================")
        print(f"Evaluating T={temp}")
        print("======================================")

        do_sample = temp > 0.0
        hf_temp = temp if do_sample else 1.0
        gen_kwargs = {"temperature": hf_temp, "do_sample": do_sample, "top_p": 1.0}

        # Override the generation kwargs directly on the model instance
        if hasattr(lm, 'model_generate_args'):
            lm.model_generate_args = gen_kwargs
        else:
             lm._model_generate_args = gen_kwargs

        start_time = time.time()
        
        try:
            results = lm_eval.simple_evaluate(
                model=lm,
                tasks=[dataset],
                num_fewshot=0,
                limit=max_samples if max_samples > 0 else None,
                gen_kwargs=f"temperature={hf_temp},do_sample={do_sample},top_p=1.0"
            )
        except Exception as e:
            print(f"Evaluation failed at T={temp} with error: {e}")
            elapsed = time.time() - start_time
            results_summary.append({
                "temperature": temp,
                "error": str(e),
                "time_seconds": elapsed
            })
            continue

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
        results_summary.append(record)

        # Save incremental results
        out_file = Path(output_dir) / f"{model_name.replace('/', '_')}_{dataset}_results.json"
        with open(out_file, "w") as f:
            json.dump(results_summary, f, indent=2)

    print(f"\nEvaluation sweep complete for {model_name}!")
    return results_summary

def main():
    parser = argparse.ArgumentParser(description="Evaluate SLMs at varying temperatures.")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B", help="Hugging Face model ID")
    parser.add_argument("--dataset", type=str, default="gsm8k", help="lm-eval task name (e.g., gsm8k, arc_easy)")
    parser.add_argument("--limit", type=int, default=0, help="Number of samples to evaluate (0 for full dataset)")
    parser.add_argument("--output_dir", type=str, default="data/results", help="Output directory")
    parser.add_argument("--quantization", type=str, choices=["none", "4bit", "8bit"], default="none", help="Quantization level")
    parser.add_argument("--temperatures", type=str, default="", help="Comma-separated temperature list (e.g. 0.0,0.7,1.4)")
    args = parser.parse_args()
    temperatures = parse_temperatures(args.temperatures)

    print("Starting evaluation pipeline...")
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"Limit: {'Full Dataset' if args.limit <= 0 else args.limit}")
    print(f"Quantization: {args.quantization}")
    print(f"Output dir: {args.output_dir}")

    run_evaluation(args.model, args.dataset, args.limit, args.output_dir, args.quantization, temperatures)

if __name__ == "__main__":
    main()
