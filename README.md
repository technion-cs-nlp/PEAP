# Position-aware Automatic Circuit Discovery

This repository implements **Position-Aware Edge Attribution Patching (PEAP)**, 

Arxiv: [https://arxiv.org/abs/2502.04577](https://arxiv.org/abs/2502.04577)

## Overview

PEAP extends Edge Attribution PAtching (EAP) by incorporating positional information, allowing researchers to understand how different parts of the input sequence interact with various model components (attention heads, MLPs, residual connections). The method computes attribution scores for both:

- **Non-crossing edges**: Connections within the same residual stream position
- **Crossing edges**: Attention connections between different sequence positions

## Key Features

- **Position-aware analysis**: Segment input sequences into meaningful spans and analyze interactions between them
- **Comprehensive edge attribution**: Compute attribution scores for all types of connections in transformer models
- **Circuit discovery**: Automatically discover computational circuits of varying sizes using different search strategies
- **Faithfulness evaluation**: Assess how well discovered circuits preserve model behavior through ablation studies
- **Multiple tasks supported**: Includes implementations for Indirect Object Identification (IOI), WinoBias, and Greater-Than comparison tasks

## Repository Structure

```
src/
├── pos_aware_edge_attribution_patching.py  # Core PEAP implementation
├── eval_utils.py                           # Circuit discovery and evaluation utilities
├── eval.py                                 # Main evaluation pipeline
├── exp.py                                  # Experiment classes for different tasks
├── data_generation.py                      # Dataset generation for supported tasks
├── input_attribution.py                    # Input attribution analysis methods
├── schema_generation.py                    # Automatic span schema generation using LLMs
└── environment.yml                         # Conda environment specification
```

## Core Components

### PEAP Algorithm (`pos_aware_edge_attribution_patching.py`)
- Computes position-aware attribution scores using gradient-based methods
- Handles both counterfactual and mean ablation strategies
- Supports multiple aggregation methods (sum, average, max absolute value)

### Circuit Discovery (`eval_utils.py`)
- Implements algorithms to find circuits of specified sizes
- Supports threshold-based and top-k circuit discovery
- Provides both forward (logits→embeddings) and reverse (embeddings→logits) search

### Faithfulness Evaluation (`eval.py`)
- Evaluates discovered circuits through mean ablation
- Computes faithfulness metrics and accuracy preservation
- Generates comprehensive evaluation reports

### Dataset Generation (`data_generation.py`)
- Creates datasets for IOI (ABBA/BABA patterns), WinoBias, and Greater-Than tasks
- Includes automatic model evaluation on generated datasets
- Supports multiple random seeds for reproducibility

### Schema Generation (`schema_generation.py`)
- Automatically generates span schemas using large language models
- Supports multiple LLM backends (GPT-4, Claude, Llama)
- Includes input attribution methods to identify important tokens

## Supported Tasks

1. **[Indirect Object Identification (IOI)](https://arxiv.org/abs/2211.00593)**
2. **[WinoBias](https://uclanlp.github.io/corefBias/overview)**
3. **[Greater-Than](https://arxiv.org/abs/2305.00586)**

## Installation

```bash
conda env create -f src/environment.yml
conda activate peap
```

## Usage

### 1. Generate Datasets
```bash
python src/data_generation.py --model_name gpt2 --save_dir ./data --task ioi_baba --seed 42
```

### 2. Compute PEAP Scores
```bash
python src/pos_aware_edge_attribution_patching.py \
    -e ioi -m gpt2 -cl clean_data.csv -co counter_data.csv \
    -sp span1 span2 span3 length -ds 1000 -p results.pkl
```

### 3. Discover and Evaluate Circuits
```bash
python src/eval.py \
    -e ioi -m gpt2 -cl clean_data.csv -co counter_data.csv \
    -sp span1 span2 span3 length -n 100 -tk 10 20 50 \
    -p peap_results.pkl -sp results.pkl
```

## How PEAP Works

### 1. Span Definition
PEAP segments input sequences into meaningful spans based on:
- Syntactic structure (subjects, objects, verbs)
- Semantic roles (professions, names, actions)
- Task-specific elements (important tokens identified through attribution)

### 2. Attribution Computation
For each edge between model components, PEAP computes:
- **Non-crossing edges**: Direct connections within spans
- **Crossing edges**: Attention-mediated connections between different spans through query-key-value interactions

### 3. Circuit Discovery
Using computed attribution scores, we discovers circuits through:
- **Top-k selection**: Select k highest-scoring edges


### 4. Faithfulness Evaluation
Discovered circuits are evaluated by:
- **Mean ablation**: Replace non-circuit components with mean activations
- **Performance preservation**: Measure how well the circuit maintains original model behavior
- **Size-performance tradeoffs**: Analyze circuit efficiency across different sizes


## Citation

If you use this code in your research, please cite the associated paper (citation details to be added upon publication).

