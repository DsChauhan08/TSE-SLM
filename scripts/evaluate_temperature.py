import argparse
import json
import os
import time
from pathlib import Path

# Important: Make sure to install lm-eval before running this script
try:
    import lm_eval
    from lm_eval.models.huggingface import HFLM
except ImportError:
    print("Please install lm_eval: pip install lm-eval")
    exit(1)

def run_evaluation(model_name, dataset, max_samples, output_dir):
    """
    Runs evaluation for a single model and dataset across a range of temperatures.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 0.0 to 2.0 with 0.1 steps
    temperatures = [round(i * 0.1, 1) for i in range(21)]
    results_summary = []
    
    # Initialize the model once
    print(f"Loading model: {model_name}...")
    lm = HFLM(pretrained=model_name, device="cuda" if torch.cuda.is_available() else "cpu")
    
    for temp in temperatures:
        print(f"
======================================")
        print(f"Evaluating T={temp}")
        print(f"======================================")
        
        # At T=0.0, do greedy decoding. Otherwise, use sampling.
        do_sample = temp > 0.0
        # When do_sample=False, temperature must be 1.0 or None in HF depending on version. We'll pass 1.0 to avoid HF errors, but behavior is greedy.
        hf_temp = temp if do_sample else 1.0
        
        gen_kwargs = f"temperature={hf_temp},do_sample={do_sample},top_p=1.0"
        
        start_time = time.time()
        
        results = lm_eval.simple_evaluate(
            model=lm,
            tasks=[dataset],
            num_fewshot=0,
            limit=max_samples,
            # We pass generation parameters via model_args or directly to generate
            # In lm_eval, gen_kwargs can be set on the HFLM initialization or passed here
            # For HFLM, we can update the model's default gen_kwargs
        )
        
        elapsed = time.time() - start_time
        
        # Extract accuracy (metric names vary by task, e.g., 'acc,none', 'exact_match,none')
        task_results = results['results'].get(dataset, {})
        acc = task_results.get('acc,none', task_results.get('exact_match,none', 0.0))
        
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate SLMs at varying temperatures.")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B", help="Hugging Face model ID")
    parser.add_argument("--dataset", type=str, default="gsm8k", help="lm-eval task name (e.g., gsm8k, arc_easy)")
    parser.add_argument("--limit", type=int, default=10, help="Number of samples to evaluate (for testing)")
    parser.add_argument("--output_dir", type=str, default="../data/results", help="Output directory")
    
    args = parser.parse_args()
    
    import torch # imported here to avoid slow startup if just running help
    
    print(f"Starting evaluation pipeline...")
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"Limit: {args.limit} samples per temperature")
    
    run_evaluation(args.model, args.dataset, args.limit, args.output_dir)
