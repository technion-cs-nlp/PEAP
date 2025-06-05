# Position-aware Automatic Circuit Discovery

This repository implements **Position-Aware Edge Attribution Patching (PEAP)**, 

Arxiv: [https://arxiv.org/abs/2502.04577](https://arxiv.org/abs/2502.04577)

## Overview

PEAP extends [Edge Attribution Patching](https://arxiv.org/abs/2310.10348) (EAP) by incorporating positional edges, enabling researchers to understand how components at different token positions interact with each other. The method computes attribution scores for both:

- **Non-crossing edges**: Connections within the same token position (such as attention head -> mlp, mlp -> attention head, embedding -> mlp and more)
- **Crossing edges**: Connections between attention heads at different token positions.


## Key Features

- **Position-aware analysis**: Segment input sequences into meaningful spans and analyze interactions between them
- **Position-aware edge attribution**: Compute attribution scores for many types of connections in transformer models
- **Circuit discovery**: Automatically discover circuits of varying sizes 
- **Faithfulness evaluation**: Evaluate how faithfully the circuits preserve model behavior using ablation studies.
- **Multiple tasks supported**: Includes implementations for Indirect Object Identification (IOI), WinoBias, and Greater-Than comparison tasks

## Repository Structure

```
src/
├── pos_aware_edge_attribution_patching.py  # Core PEAP implementation
├── eval_utils.py                           # Circuit discovery and evaluation utilities
├── eval.py                                 # Full evaluation pipeline
├── exp.py                                  # Experiment classes for different tasks
├── data_generation.py                      # Dataset generation for supported tasks
├── input_attribution.py                    # Input attribution analysis methods (for finding the Schema)
├── schema_generation.py                    # Automatic span schema generation using LLMs
└── environment.yml                         # Conda environment specification
```

## Core Components

### PEAP Algorithm (`pos_aware_edge_attribution_patching.py`)
- Computes position-aware attribution scores using gradient-based methods
- Handles both counterfactual and mean ablation strategies
-Supports multiple aggregation methods to handle spans of varying lengths. (sum, average, max absolute value)

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

### Schema Generation (`schema_generation.py`)
- Automatically generates span schemas using large language models
- Supports multiple LLM backends (GPT-4, Claude, Llama)
- Includes input attribution methods to identify important tokens

## Supported Tasks

1. **[Indirect Object Identification (IOI)](https://arxiv.org/abs/2211.00593)**
2. **[WinoBias](https://uclanlp.github.io/corefBias/overview)**
3. **[Greater-Than](https://arxiv.org/abs/2305.00586)**

## How to Add New Tasks

1. Create a customized `Experiment` object, as defined in `Experiment.py`.

2. Prepare a DataFrame that follows these guidelines:
   - It must include a `"prompt"` column.
   - It should have one column per span, where each column contains the index of the **first token** in that span.
     - The **end** of span *t* is **one index before** the starting index of span *t+1*. This means every token is included in exactly one span.
   - Empty spans are allowed — just set the start index equal to the start index of the next span.
   - The DataFrame must also include a "length" column indicating the total number of tokens in the prompt. This helps handle prompts of varying lengths and will also serve as the boundary for the final span, ensuring it includes all remaining tokens.
   - A Beginning-of-Sequence (BOS) token is automatically added when running the pipeline.
     - Do **not** include it in the `"prompt"` text.
     - However, make sure to account for it when setting span indices.  
       For example, in the prompt `"I love you"`, the token `"I"` should have index `1` in the DataFrame, since the BOS token will be added at position `0`.

### Example

For the prompt `"I love you"` (which is tokenized as `["I", "love", "you"]` and becomes `["<BOS>", "I", "love", "you"]`), the DataFrame might look like this:

| prompt        | span_0 | span_1 | span_2 | length |
|---------------|--------|--------|--------|--------|
| I love you    |   1    |   2    |   3    |   4    |

- `span_0` starts at index 1 (token `"I"`)
- `span_1` starts at index 2 (token `"love"`)  
  → So `span_0` includes only `"I"`
- `span_2` starts at index 3 (token `"you"`)  
  → So `span_1` includes only `"love"`
- Since `length = 4` (because the BOS token will be added), `span_2` includes all tokens from index 3 up to (but not including) index 4 → just `"you"`




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
Make sure to add "length" as the last span. 
```bash
python src/pos_aware_edge_attribution_patching.py \
    -e ioi -m gpt2 -cl data/gpt2/ioi_ABBA/human_baseline/IOI_data_clean.csv -co data/gpt2/ioi_ABBA/human_baseline/IOI_data_counter_abc.csv \
    -sp prefix IO and S1 S1+1 action1 S2 action2 to length -ds 10 -p ioi_results.pkl
```

### 3. Discover and Evaluate Circuits
Make sure to add "length" as the last span. 
```bash
python src/eval.py \
    -e ioi -m gpt2 -cl data/gpt2/ioi_ABBA/human_baseline/IOI_data_clean.csv -co data/gpt2/ioi_ABBA/human_baseline/IOI_data_counter_abc.csv \
    -sp prefix IO and S1 S1+1 action1 S2 action2 to length -n 10 -tk 100 200 300 \
    -p ioi_results.pkl -sp results.pkl
```

## How PEAP Works

### 1. Span Definition
PEAP segments input sequences into meaningful spans based on:
- Syntactic structure (subjects, objects, verbs)
- Semantic roles (professions, names, actions)
- Task-specific elements (important tokens identified through attribution)

### 2. Attribution Computation
 PEAP computes attribution scors for both:
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


