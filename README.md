# Agentic Reinforcement Merge (ARM) for LLMs

This is a specialized tool for merging multiple task-specific Large Language Models (LLMs) back into a base model. It focuses on identifying **element-wise parameter changes** and analyzing **parameter overlap** to mitigate catastrophic forgetting and interference between models.

The script provides detailed pre-merge statistics and implements several advanced merging strategies (ARM, ARM-Rescale, TIES, etc.) to optimize the combination of distinct model capabilities (e.g., Coding, Math, Tool-use).

## ✨ Key Features

* **Change Detection**: Automatically detects which parameters have changed relative to the base model based on a configurable `--threshold`, ignoring static parameters.
* **Overlap Analysis**: Generates a detailed statistical report before merging, showing how many parameters conflict or overlap between different task models.
* **Advanced Merge Strategies**:
    * **ARM**: Averages task vectors only where changes actually occurred, reducing noise from unchanged models.
    * **ARM-Rescale (v1 & v2)**: Dynamically scales (boosts) the weights of non-overlapping parameters based on the model's global overlap ratio. 
    * **ARM-TIES**: Implements the TIES-Merging algorithm (Trim, Elect Sign, Disjoint Merge) to handle interference.
    * **ARM-Unique**: Extracts and merges parameter changes that are unique to a specific target model.
* **Auto-Save**: Automatically saves the merged model weights and the corresponding tokenizer.

## 🛠️ Requirements

Ensure you have the following Python libraries installed:

```bash
pip install torch transformers
