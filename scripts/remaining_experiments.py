#!/usr/bin/env python3
"""
Run only the remaining Phase 4 experiments.

This script orchestrates the unfinished experimental blocks by calling:
    scripts/run_phase1_queue.py

It is resume-safe:
- It checks existing result files first.
- With --skip_done (default), fully completed blocks are skipped.
- Partially completed blocks are resumed by the downstream evaluator.
"""

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Set, Tuple


DEFAULT_MODELS = [
    "unsloth/Llama-3.1-8B-Instruct-GGUF",
    "TeichAI/Qwen3-4B-Thinking-2507-Claude-4.5-Opus-High-Reasoning-Distill-GGUF",
    "unsloth/Phi-4-mini-reasoning-GGUF",
    "unsloth/gemma-3-4b-it-GGUF",
]

MAIN_DATASETS = ["arc_easy", "arc_challenge", "commonsense_qa", "piqa", "hellaswag"]


@dataclass(frozen=True)
class ExperimentBlock:
    name: str
    dataset: str
    run_tag: str
    top_p: float = 1.0
    top_k: int = 0
    seed: int = -1


def parse_temperatures(raw_temperatures: str) -> List[float]:
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


def sanitize_model_name(model_id: str) -> str:
    return model_id.replace("/", "_")


def result_path(output_dir: Path, model_id: str, dataset: str, run_tag: str) -> Path:
    tag_suffix = f"_{run_tag}" if run_tag else ""
    filename = f"{sanitize_model_name(model_id)}_{dataset}{tag_suffix}_results.json"
    return output_dir / filename


def load_rows(path: Path) -> List[dict]:
    if not path.exists():
        return []
    data = json.loads(path.read_text())
    if not isinstance(data, list):
        raise ValueError(f"Unexpected non-list JSON in {path}")
    return [row for row in data if isinstance(row, dict)]


def completed_temperature_set(rows: List[dict]) -> Set[float]:
    completed = set()
    for row in rows:
        if "error" in row:
            continue
        value = row.get("temperature")
        if value is None:
            continue
        completed.add(round(float(value), 1))
    return completed


def block_status(
    block: ExperimentBlock,
    models: List[str],
    output_dir: Path,
    expected_temps: Set[float],
) -> Dict[str, dict]:
    details: Dict[str, dict] = {}
    for model_id in models:
        path = result_path(output_dir, model_id, block.dataset, block.run_tag)
        rows = load_rows(path)
        errors = sum(1 for row in rows if "error" in row)
        completed = completed_temperature_set(rows)
        covered = sorted(expected_temps.intersection(completed))
        is_complete = len(covered) == len(expected_temps) and errors == 0
        details[model_id] = {
            "file": str(path),
            "exists": path.exists(),
            "records": len(rows),
            "error_records": errors,
            "covered_temperatures": covered,
            "covered_count": len(covered),
            "expected_count": len(expected_temps),
            "complete": is_complete,
        }
    return details


def build_queue_command(
    project_root: Path,
    python_bin: str,
    models: List[str],
    block: ExperimentBlock,
    args: argparse.Namespace,
) -> List[str]:
    cmd = [
        python_bin,
        "-u",
        str(project_root / "scripts" / "run_phase1_queue.py"),
    ]
    for model in models:
        cmd.extend(["--model", model])
    cmd.extend(
        [
            "--dataset",
            block.dataset,
            "--limit",
            str(args.limit),
            "--output_dir",
            args.output_dir,
            "--log_dir",
            args.log_dir,
            "--startup_timeout",
            str(args.startup_timeout),
            "--n_threads",
            str(args.n_threads),
            "--n_gpu_layers",
            str(args.n_gpu_layers),
            "--top_p",
            str(block.top_p),
            "--top_k",
            str(block.top_k),
            "--run_tag",
            block.run_tag,
        ]
    )
    if args.temperatures:
        cmd.extend(["--temperatures", args.temperatures])
    if block.seed >= 0:
        cmd.extend(["--seed", str(block.seed)])
    if args.no_prefetch:
        cmd.append("--no_prefetch")
    return cmd


def any_incomplete(status_by_model: Dict[str, dict]) -> bool:
    return any(not row["complete"] for row in status_by_model.values())


def planned_blocks(suite: str) -> List[ExperimentBlock]:
    main = [ExperimentBlock(name=f"main-{ds}", dataset=ds, run_tag="phase4-main") for ds in MAIN_DATASETS]
    controls = [
        ExperimentBlock(name="control-top_p_0p9", dataset="gsm8k", run_tag="phase4-control-top_p_0p9", top_p=0.9),
        ExperimentBlock(name="control-top_k_40", dataset="gsm8k", run_tag="phase4-control-top_k_40", top_k=40),
    ]
    repeats = [
        ExperimentBlock(name="repeat-seed42", dataset="gsm8k", run_tag="phase4-repeat-seed42", seed=42),
        ExperimentBlock(name="repeat-seed314", dataset="gsm8k", run_tag="phase4-repeat-seed314", seed=314),
    ]

    if suite == "main":
        return main
    if suite == "controls":
        return controls
    if suite == "repeats":
        return repeats
    return main + controls + repeats


def main():
    parser = argparse.ArgumentParser(description="Run remaining Phase 4 experiments (resume-safe).")
    parser.add_argument("--suite", choices=["main", "controls", "repeats", "all"], default="all")
    parser.add_argument("--model", action="append", help="Model repo ID (repeatable). Overrides defaults if provided.")
    parser.add_argument("--temperatures", type=str, default="", help="Optional CSV temperatures (default: 0.0..2.0).")
    parser.add_argument("--limit", type=int, default=10, help="Samples per temperature (0 for full dataset).")
    parser.add_argument("--output_dir", type=str, default="data/results")
    parser.add_argument("--log_dir", type=str, default="logs/phase1")
    parser.add_argument("--startup_timeout", type=int, default=300)
    parser.add_argument("--n_threads", type=int, default=4)
    parser.add_argument("--n_gpu_layers", type=int, default=-1)
    parser.add_argument("--python_bin", type=str, default=sys.executable, help="Python executable for subprocess calls.")
    parser.add_argument("--no_prefetch", action="store_true", help="Disable model prefetch in queue runner.")
    parser.add_argument("--skip_done", action="store_true", default=True, help="Skip blocks that are already complete.")
    parser.add_argument("--run", action="store_true", help="Execute runs (without this flag, script performs a dry plan).")
    parser.add_argument("--status_file", type=str, default="data/results/remaining_experiments_status.json")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    output_dir = project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    status_file = project_root / args.status_file

    models = args.model if args.model else DEFAULT_MODELS
    expected_temps = set(parse_temperatures(args.temperatures))
    blocks = planned_blocks(args.suite)
    run_results = []
    had_failures = False

    print("=== Remaining Experiments Planner ===")
    print(f"Mode: {'execute' if args.run else 'dry-run'}")
    print(f"Suite: {args.suite}")
    print(f"Models: {models}")
    print(f"Expected temperatures: {sorted(expected_temps)}")

    for block in blocks:
        before = block_status(block, models, output_dir, expected_temps)
        incomplete_before = any_incomplete(before)
        cmd = build_queue_command(project_root, args.python_bin, models, block, args)
        will_run = incomplete_before or not args.skip_done
        exit_code = None

        print("\n--------------------------------------")
        print(f"Block: {block.name} | dataset={block.dataset} | run_tag={block.run_tag}")
        print(f"Incomplete before run: {incomplete_before}")
        print(f"Command: {' '.join(cmd)}")

        if args.run and will_run:
            completed = subprocess.run(cmd, cwd=str(project_root))
            exit_code = completed.returncode
            if exit_code != 0:
                had_failures = True
        elif args.run and not will_run:
            print("Skipping block: already complete.")
        else:
            print("Dry-run: command not executed.")

        after = block_status(block, models, output_dir, expected_temps)
        run_results.append(
            {
                "block": block.name,
                "dataset": block.dataset,
                "run_tag": block.run_tag,
                "command": cmd,
                "incomplete_before": incomplete_before,
                "incomplete_after": any_incomplete(after),
                "executed": bool(args.run and will_run),
                "exit_code": exit_code,
                "status_before": before,
                "status_after": after,
            }
        )

    status_payload = {
        "suite": args.suite,
        "mode": "execute" if args.run else "dry-run",
        "models": models,
        "expected_temperatures": sorted(expected_temps),
        "blocks": run_results,
        "had_failures": had_failures,
    }
    status_file.write_text(json.dumps(status_payload, indent=2))
    print(f"\nWrote status file: {status_file}")

    if had_failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
