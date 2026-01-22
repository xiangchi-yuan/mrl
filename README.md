# Reinforced Agentic Model Merge (RAM)


## Requirements

Ensure you have the following Python libraries installed:

```bash
pip install torch transformers
```

## Evaluation

We evaluate RAM across three distinct agentic domains: **Tool Using**, **Coding**, and **Memory**. To reproduce the results, please refer to the respective repositories linked below for environment setup and evaluation scripts.

### 1. Tool Using
We utilize the **Berkeley Function Call Leaderboard (BFCL)** to assess tool-use capabilities.
- **Benchmark & Setup:** [Berkeley Function Call Leaderboard](https://github.com/ShishirPatil/gorilla/blob/main/berkeley-function-call-leaderboard/README.md#installation--setup)

### 2. Coding
We employ **CURE** to evaluate code generation and reasoning performance.
- **Benchmark & Setup:** [CURE Repository](https://github.com/Gen-Verse/CURE?tab=readme-ov-file)

### 3. Memory
We use **MemAgent** to test long-context retention and retrieval capabilities.
- **Benchmark & Setup:** [MemAgent Repository](https://github.com/BytedTsinghua-SIA/MemAgent)

## Overview of RAM

<img src="intro.png" width="150%" alt="RAM Framework">
