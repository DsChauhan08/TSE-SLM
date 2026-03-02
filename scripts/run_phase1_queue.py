#!/usr/bin/env python3
"""
Phase 1 local experiment orchestrator.

Workflow:
1) Prefetch/download model N
2) Evaluate model N
3) While N is evaluating, prefetch model N+1

This supports a mixed list of:
- GGUF repos (evaluated via scripts/evaluate_gguf_local.py)
- Hugging Face 4-bit repos (evaluated via scripts/evaluate_temperature.py --quantization 4bit)
"""

import argparse
import json
import subprocess
import sys
import time
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from huggingface_hub import list_repo_files, snapshot_download


DEFAULT_MODELS = [
    "unsloth/llama-3-8b-bnb-4bit",
    "TeichAI/Qwen3-4B-Thinking-2507-Claude-4.5-Opus-High-Reasoning-Distill-GGUF",
    "unsloth/Phi-4-mini-reasoning-GGUF",
    "unsloth/gemma-3-4b-it-unsloth-bnb-4bit",
]

CPU_FALLBACK_REPO_MAP = {
    "unsloth/llama-3-8b-bnb-4bit": "unsloth/Llama-3.1-8B-Instruct-GGUF",
    "unsloth/gemma-3-4b-it-unsloth-bnb-4bit": "unsloth/gemma-3-4b-it-GGUF",
}


def has_nvidia_gpu() -> bool:
    return shutil.which("nvidia-smi") is not None


def is_gguf_model(model_id: str) -> bool:
    return "GGUF" in model_id.upper()


def is_four_bit_target(model_id: str) -> bool:
    model_upper = model_id.upper()
    return ("4BIT" in model_upper) or ("Q4" in model_upper) or is_gguf_model(model_id)


def pick_q4_gguf_file(repo_id: str) -> str:
    files = list_repo_files(repo_id)
    q4_files = [f for f in files if f.lower().endswith(".gguf") and "Q4" in f.upper()]
    if not q4_files:
        raise RuntimeError(f"No Q4 GGUF file found in {repo_id}")

    preferred_markers = ["Q4_K_M", "Q4_K_S", "Q4_0", "Q4_1"]
    for marker in preferred_markers:
        for candidate in q4_files:
            if marker in candidate.upper():
                return candidate
    return q4_files[0]


def prefetch_model(model_id: str) -> str:
    if is_gguf_model(model_id):
        selected_file = pick_q4_gguf_file(model_id)
        snapshot_download(repo_id=model_id, allow_patterns=[selected_file])
        return f"Prefetched GGUF: {selected_file}"

    snapshot_download(repo_id=model_id)
    return "Prefetched repository snapshot"


def build_eval_command(
    model_id: str,
    dataset: str,
    limit: int,
    output_dir: str,
    temperatures: str,
    startup_timeout: int,
    n_threads: int,
    n_gpu_layers: int,
):
    script_dir = Path(__file__).resolve().parent
    if is_gguf_model(model_id):
        cmd = [
            sys.executable,
            "-u",
            str(script_dir / "evaluate_gguf_local.py"),
            "--model",
            model_id,
            "--dataset",
            dataset,
            "--limit",
            str(limit),
            "--output_dir",
            output_dir,
        ]
        if temperatures:
            cmd.extend(["--temperatures", temperatures])
        if startup_timeout > 0:
            cmd.extend(["--startup_timeout", str(startup_timeout)])
        if n_threads > 0:
            cmd.extend(["--n_threads", str(n_threads)])
        if n_gpu_layers != 0:
            cmd.extend(["--n_gpu_layers", str(n_gpu_layers)])
        return cmd

    cmd = [
        sys.executable,
        "-u",
        str(script_dir / "evaluate_temperature.py"),
        "--model",
        model_id,
        "--quantization",
        "4bit",
        "--dataset",
        dataset,
        "--limit",
        str(limit),
        "--output_dir",
        output_dir,
    ]
    if temperatures:
        cmd.extend(["--temperatures", temperatures])
    return cmd


def run_command_with_log(cmd, cwd: Path, log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log_file.write(line)
        return process.wait()


def resolve_model_for_environment(model_id: str):
    if has_nvidia_gpu():
        return model_id, None
    fallback = CPU_FALLBACK_REPO_MAP.get(model_id)
    if fallback:
        note = f"No NVIDIA GPU detected; using CPU-compatible 4-bit GGUF fallback: {fallback}"
        return fallback, note
    return model_id, None


def main():
    parser = argparse.ArgumentParser(description="Run Phase 1 model queue with rolling prefetch + evaluation.")
    parser.add_argument("--dataset", type=str, default="gsm8k", help="lm-eval task name")
    parser.add_argument("--limit", type=int, default=10, help="Number of samples per temperature (0 for full dataset)")
    parser.add_argument("--output_dir", type=str, default="data/results", help="Results directory")
    parser.add_argument("--log_dir", type=str, default="logs/phase1", help="Log output directory")
    parser.add_argument("--model", action="append", help="Model repo ID to evaluate (repeat for multiple)")
    parser.add_argument("--temperatures", type=str, default="", help="Comma-separated temperatures for debug sweeps")
    parser.add_argument("--startup_timeout", type=int, default=300, help="GGUF server startup timeout in seconds")
    parser.add_argument("--n_threads", type=int, default=0, help="llama.cpp CPU threads for GGUF runs (0 = default)")
    parser.add_argument("--n_gpu_layers", type=int, default=0, help="llama.cpp GPU layers for GGUF runs (-1 = all layers)")
    parser.add_argument("--no_prefetch", action="store_true", help="Disable background prefetching")
    args = parser.parse_args()

    models = args.model if args.model else DEFAULT_MODELS
    model_plan = []
    for requested in models:
        resolved, note = resolve_model_for_environment(requested)
        model_plan.append({"requested": requested, "resolved": resolved, "note": note})

    invalid_models = [m["resolved"] for m in model_plan if not is_four_bit_target(m["resolved"])]
    if invalid_models:
        raise ValueError(f"Non-4bit model IDs are not allowed for this Phase 1 runner: {invalid_models}")

    project_root = Path(__file__).resolve().parents[1]
    summary = []

    print("=== Local Experiment Queue Runner ===")
    print(f"Requested models: {[m['requested'] for m in model_plan]}")
    print(f"Resolved models: {[m['resolved'] for m in model_plan]}")
    for model_entry in model_plan:
        if model_entry["note"]:
            print(f"[Model Map] {model_entry['requested']} -> {model_entry['resolved']} ({model_entry['note']})")
    print(f"Dataset: {args.dataset}")
    print(f"Limit: {args.limit}")
    print(f"Temperatures: {args.temperatures if args.temperatures else '0.0..2.0 step 0.1'}")
    print(f"Startup timeout: {args.startup_timeout}s")
    print(f"n_threads: {args.n_threads if args.n_threads > 0 else 'default'}")
    print(f"n_gpu_layers: {args.n_gpu_layers if args.n_gpu_layers != 0 else 'default/cpu'}")
    print(f"Prefetch enabled: {not args.no_prefetch}")

    with ThreadPoolExecutor(max_workers=1) as executor:
        prefetch_future = None
        if not args.no_prefetch and model_plan:
            print(f"\n[Prefetch] Starting initial prefetch for {model_plan[0]['resolved']}")
            prefetch_future = executor.submit(prefetch_model, model_plan[0]["resolved"])

        for idx, model_entry in enumerate(model_plan):
            requested_model = model_entry["requested"]
            model_id = model_entry["resolved"]
            prefetch_error = None
            prefetch_message = None

            if prefetch_future is not None:
                try:
                    prefetch_message = prefetch_future.result()
                    print(f"[Prefetch] Ready for {model_id}: {prefetch_message}")
                except Exception as exc:
                    prefetch_error = str(exc)
                    print(f"[Prefetch] Warning for {model_id}: {prefetch_error}")

            next_prefetch = None
            if not args.no_prefetch and idx + 1 < len(model_plan):
                next_model = model_plan[idx + 1]["resolved"]
                print(f"[Prefetch] Starting background prefetch for next model: {next_model}")
                next_prefetch = executor.submit(prefetch_model, next_model)

            cmd = build_eval_command(
                model_id,
                args.dataset,
                args.limit,
                args.output_dir,
                args.temperatures,
                args.startup_timeout,
                args.n_threads,
                args.n_gpu_layers,
            )
            log_name = model_id.replace("/", "__").replace("\\", "__") + ".log"
            log_path = project_root / args.log_dir / log_name

            start = time.time()
            print(f"\n[Run] Evaluating {model_id}")
            print(f"[Run] Command: {' '.join(cmd)}")
            return_code = run_command_with_log(cmd, project_root, log_path)
            elapsed = time.time() - start

            status = "success" if return_code == 0 else "failed"
            summary.append(
                {
                    "requested_model": requested_model,
                    "model": model_id,
                    "status": status,
                    "return_code": return_code,
                    "elapsed_seconds": elapsed,
                    "log_file": str(log_path),
                    "prefetch_message": prefetch_message,
                    "prefetch_error": prefetch_error,
                }
            )
            print(f"[Run] Finished {model_id} with status={status} in {elapsed:.2f}s")

            prefetch_future = next_prefetch

    output_dir = project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "phase1_queue_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved queue summary: {summary_path}")
    failures = [row for row in summary if row["status"] != "success"]
    if failures:
        print(f"Completed with {len(failures)} failed model runs.")
        raise SystemExit(1)
    print("Completed all model runs successfully.")


if __name__ == "__main__":
    main()
