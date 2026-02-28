# Phase 1: Project Plan & Setup Validation

## Hardware Assessment
- **CPU**: AMD Ryzen 7 7730U
- **RAM**: 13 GB (8.2 GB Available)
- **GPU**: Integrated AMD Radeon Graphics (No NVIDIA CUDA available)
- **Verdict**: Local inference for small models (0.5B - 1.5B) using CPU is completely viable for pilot testing, script creation, and debugging. The pilot test demonstrated ~10-14 seconds per 100-token generation.
- **Action**: Full-scale data collection (Phase 2) **must** be executed on cloud platforms like Google Colab (T4 GPU) or Kaggle to iterate across all 20 temperature states, 4 models, and multiple benchmarks within a reasonable timeframe.

## Pilot Test Results
Successfully verified `Qwen/Qwen2.5-0.5B-Instruct` locally. Temperature variations caused observable behavior changes:
- `T=0.1`: Deterministic, low entropy.
- `T=1.0`: Creative variation in reasoning paths.
- `T=1.8`: High entropy, resulted in severe hallucination (e.g., introducing imaginary entities like "Emma" and "Billy").

## Completed Setup
- [x] Initialized virtual environment (`venv`).
- [x] Created `requirements.txt` incorporating PyTorch, Hugging Face `transformers`, and `lm-eval`.
- [x] Written `literature_review.md`.
- [x] Written `scripts/pilot_test.py` to confirm local environment execution.
- [x] Created `scripts/evaluate_temperature.py`, a robust wrapper around `lm-eval` designed for running the full experiment loops in Phase 2 on Colab.

## Next Steps (Phase 2)
1. Upload this workspace to a private GitHub repository.
2. Clone repository into Google Colab.
3. Run `scripts/evaluate_temperature.py` on a GPU instance against GSM8K and ARC.
