# Literature Review: Temperature Effects on Language Models

This document tracks key papers and findings related to temperature scaling in Large Language Models (LLMs) and Small Language Models (SLMs).

## Core Concepts
- **Temperature Scaling ($T$)**: Modifies the softmax distribution over the vocabulary. 
  - $T 	o 0$: Greedy decoding, deterministic, low variance.
  - $T = 1.0$: Default probability distribution.
  - $T > 1.0$: Flatter distribution, higher entropy, higher variance.

## Key Literature from Proposal
1. **"The Effect of Sampling Temperature on Problem Solving in Large Language Models" (Renze, 2024)**
   - *Hypothesis*: Explores the relationship between temperature and reasoning/problem-solving in models >7B parameters.
   - *Key Finding*: U-shaped or distinct optimal temperature bands typically exist for reasoning tasks, suggesting higher temperatures can unlock more diverse reasoning paths, while too high degrades coherence.

2. **"Exploring the Impact of Temperature on Large Language Models: Hot or Cold?" (2025)**
   - *Scope*: Systematically evaluated temperature across 6 distinct capabilities on large-scale models.
   - *Note*: Provides the methodological basis for this study's factorial design but exclusively focused on LLMs.

3. **"On the Role of Temperature Sampling in Test-Time Scaling"**
   - *Scope*: Demonstrates how generating multiple samples (n > 1) at higher temperatures and picking the best one (Pass@k or Best-of-N) improves reasoning capabilities.
   - *Gap*: Requires high computational budget, which contradicts the low-latency, low-energy constraints of edge-deployed SLMs.

## Identified Gap
- **SLM Architecture**: Models under 3B parameters (e.g., Qwen2.5 0.5B, TinyLLaMA 1.1B, Gemma-2 2B) have fundamentally less capacity to store world knowledge and represent uncertainty.
- **Objective**: Does the temperature-performance curve in SLMs mirror that of LLMs, or is it brittle to stochasticity? This research aims to answer whether SLMs possess a usable spectrum of reasoning variation under elevated temperatures.
