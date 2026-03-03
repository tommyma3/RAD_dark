# Compressed Context Algorithm Distillation (CCCAD)

Implementation of Algorithm Distillation (AD) with a compression mechanism for handling long sequences in in-context reinforcement learning.

## Overview

This project implements:
1. **Algorithm Distillation (AD)**: A decoder-only transformer that learns to imitate RL algorithms from their learning histories
2. **Compressed AD (CAD)**: Extends AD with a compression transformer that compresses long sequences into latent tokens

## Architecture

### Compressed Algorithm Distillation

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Compressed Algorithm Distillation                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  When sequence length < max_seq_length:                                      │
│    Input: [(s,a,r,s')_1, ..., (s,a,r,s')_t, s_query] → AD Transformer       │
│                                                                              │
│  When sequence length >= max_seq_length:                                     │
│    1. Compress: [history] + query_tokens → Compression TF → latent_tokens   │
│    2. Continue: [latent_tokens, new_transitions, s_query] → AD Transformer  │
│    3. Repeat compression as needed                                          │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Installation

```bash
pip install torch accelerate transformers einops h5py stable-baselines3 tensorboard tqdm pyyaml
```

## Usage

### 1. Collect Training Data

First, collect RL training histories using PPO:

```bash
python collect.py
```

### 2. Train Original AD (Optional Baseline)

```bash
# Single GPU
python train.py

# Multi-GPU
accelerate launch --multi_gpu --num_processes=2 train.py
```

### 3. Train Compressed AD

#### Step 1: Pre-train Compression Transformer

```bash
# Single GPU
python train_pretrain_compression.py

# Multi-GPU
accelerate launch --multi_gpu --num_processes=2 train_pretrain_compression.py
```

#### Step 2: Fine-tune Full CAD System

```bash
# Single GPU
python train_cad.py

# Multi-GPU
accelerate launch --multi_gpu --num_processes=2 train_cad.py

# With custom pre-trained checkpoint
python train_cad.py --pretrain_ckpt ./runs/CAD-pretrain-darkroom-seed0/pretrain-final.pt

# Without curriculum learning
python train_cad.py --no_curriculum
```

### 4. Evaluate

```bash
# Evaluate original AD
python evaluate.py

# Evaluate CAD
python evaluate_cad.py --ckpt_dir ./runs/CAD-darkroom-seed0 --eval_episodes 500
```

## Configuration

### Key CAD Config Parameters (`config/model/cad_dr.yaml`)

```yaml
# Compression Settings
n_compress_tokens: 40        # Number of latent tokens (compression ratio)
compress_n_layers: 2         # Compression transformer depth
compress_n_heads: 4          # Compression attention heads

# Stability Settings
max_gradient_rounds: 2       # Detach gradients after N compression rounds
use_recon_reg: True          # Enable reconstruction regularization
recon_reg_weight: 0.1        # Weight for reconstruction loss

# Curriculum (start with fewer compressions, gradually increase)
max_compressions: null       # null = unlimited
```

### Curriculum Schedule

The default curriculum gradually increases compression complexity:

| Training Step | Max Compressions |
|---------------|-----------------|
| 0             | 1               |
| 10,000        | 2               |
| 25,000        | 3               |
| 40,000        | Unlimited       |

## Multi-GPU Training

### Configure Accelerate (First Time)

```bash
accelerate config
```

Or use the provided config:

```bash
accelerate launch --config_file accelerate_config_multigpu.yaml train_cad.py
```

### Manual Multi-GPU Launch

```bash
# 2 GPUs
accelerate launch --multi_gpu --num_processes=2 train_cad.py

# 4 GPUs
accelerate launch --multi_gpu --num_processes=4 train_cad.py
```

## Project Structure

```
CCCAD/
├── model/
│   ├── ad.py              # Original Algorithm Distillation
│   ├── compressed_ad.py   # Compressed AD with rolling compression
│   └── compression.py     # Compression Transformer module
├── config/model/
│   ├── ad_dr.yaml         # Original AD config
│   ├── cad_dr.yaml        # Compressed AD config
│   └── cad_pretrain.yaml  # Pre-training config
├── train.py               # Train original AD
├── train_pretrain_compression.py  # Pre-train compression
├── train_cad.py           # Fine-tune full CAD
├── evaluate.py            # Evaluate AD
├── evaluate_cad.py        # Evaluate CAD
├── dataset.py             # Dataset classes
└── README.md
```

## Key Design Decisions

### Handling the Moving Target Problem

Since the compression transformer processes its own outputs (latent tokens from previous compressions), we employ:

1. **Pre-training**: Train compression with reconstruction loss before fine-tuning
2. **Gradient Truncation**: Detach gradients after `max_gradient_rounds` compression rounds
3. **Curriculum Learning**: Start with few compressions, gradually increase
4. **Reconstruction Regularization**: Light reconstruction loss during fine-tuning

### Compression Trigger

Compression is triggered when:
- Sequence length reaches `max_seq_length`
- After compression, the buffer is cleared and accumulation restarts
- This repeats indefinitely for arbitrarily long episodes

## Citation

If you use this code, please cite the original Algorithm Distillation paper:

```bibtex
@article{laskin2022context,
  title={In-context reinforcement learning with algorithm distillation},
  author={Laskin, Michael and Wang, Luyu and Oh, Junhyuk and Parisotto, Emilio and Spencer, Stephen and Steiber, Richie and Strouse, DJ and Hansen, Steven and Fiez, Angelos and Simchowitz, Max and others},
  journal={arXiv preprint arXiv:2210.14215},
  year={2022}
}
```