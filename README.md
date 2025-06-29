# Auto LLM Merging

## Overview

This is a framework that supports automatic merging of language models, consisting of various components.

### Key Components

- **Merge Methods**: Supports multiple merging algorithms including Slerp, linear merge, Ties, and task arithmetic
- **Loader**: Manages input/output operations for model loading and saving
- **Utils**: Contains utilities for merging, logging, caching, and other helper functions
- **Tokenizer**: Handles alignment of tokenizers across different models
- **Merge Strategies**: Implements diverse merging approaches:
  - Normal model merging
  - Normal slice merging
  - Depth-wise Integration Strategy (DIS)
  - Layer-wise Fusion Strategy (LFS)
  - Layer-wise Fusion Strategy with Multi-objective optimization (LFS_MO)
  - Layer Pruning Strategy
- **Evaluation**: Powered by [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness/tree/15ffb0dafa9c869c7436ba9a3cf3067ac4c9d846) for comprehensive model assessment

## Installation

To run the code, please ensure you have the required dependencies installed:

```bash
# Clone the repository
git clone https://github.com/Guinan-Su/auto-merge-llm
cd auto-merge-llm

# Install dependencies using conda
conda env create -f environment.yml
```
## Usage

The framework uses SMAC optimizer and supports both single-objective and multi-objective optimization approaches.

### 1. Layer-wise Fusion Strategy (LFS)

#### 1.1 Single-objective Optimization

```bash
python3 merge.py --config ./exp_config/config_lfs.yaml
```
#### 1.2 Multi-objective Optimization

```bash
python3 merge.py --config ./exp_config/config_lfs_mo.yaml
```
### 2. Depth-wise Integration (DIS)

```bash
python3 merge.py --config ./exp_config/config_dis.yaml
```
### 3. Layer Pruning

```bash
python3 merge.py --config ./exp_config/config_prune.yaml
```
We've placed the tasks used in our experiments in the `custom_tasks` directory. You can also design your own optimization objectives.

### Reproducing Our Results

We've placed the search configurations from our paper in the `search_config` directory. You can evaluate these configurations using the provided configs.

### Acknowledgements

We appreciate the excellent open-source projects [MergeKit](https://github.com/arcee-ai/mergekit) and [MergeLM](https://github.com/yule-BUAA/MergeLM). We have referenced or utilized portions of code from these projects.

## Citation

We hope this code will be helpful for your research. If you find it useful, please consider citing our work. 

```bibtex
@article{su2024fine,
  title={Fine, I'll Merge It Myself: A Multi-Fidelity Framework for Automated Model Merging},
  author={Su, Guinan and Geiping, Jonas},
  journal={arXiv preprint arXiv:2502.04030},
  year={2024}
}
@article{su2025gptailor,
  title={GPTailor: Large Language Model Pruning Through Layer Cutting and Stitching},
  author={Su, Guinan and Shen, Li and Yin, Lu and Liu, Shiwei and Yang, Yanwu and Geiping, Jonas},
  journal={arXiv preprint arXiv:2506.20480},
  year={2025}
}
