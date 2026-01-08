# Early Exit LLM Training & Inference

Train auxiliary prediction heads on intermediate layers of LLMs for early exit / speculative decoding.

## Repository Structure

```
branchy_hook/
├── src/early_exit/          # Core library
│   ├── model_loader.py      # Model loading with early exit heads
│   ├── model_config.py      # Configuration dataclasses
│   └── debug_logger.py      # Debug utilities
│
├── scripts/                 # Entry points
│   ├── train.py             # Train auxiliary heads
│   ├── calibrate.py         # Calibrate thresholds
│   ├── inference.py         # Run inference with speculative decoding
│   └── evaluate.py          # Evaluate head accuracy
│
├── checkpoints/             # Trained models + calibration
│   └── llama3-8b-4bit/      # Llama-3 8B checkpoint
│       ├── aux_heads.pt     # Trained head weights
│       ├── config.json      # Model configuration
│       └── calibration.json # Calibrated thresholds
│
├── configs/                 # Accelerate/training configs
│   └── ddp_config.yaml
│
└── experiments/             # Experiment outputs (gitignored)
    ├── wandb/
    └── results/
```

## Quick Start

### 0. Installation

Install the package in development mode so all scripts can import `src.early_exit`:

```bash
pip install -e .
# Or with all optional dependencies:
pip install -e ".[train,calibrate]"
```

### 1. Training Auxiliary Heads

Training supports two modes: **online** and **offline**.

#### Online Mode (Default)
Runs backbone + head training together each step:

```bash
accelerate launch --config_file configs/ddp_config.yaml scripts/train.py \
    --mode online \
    --model_name meta-llama/Meta-Llama-3-8B \
    --num_heads 3 \
    --quantization 4bit \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --output_dir ./checkpoints/my_model \
    --max_steps 3000
```

#### Offline Mode
Caches hidden states first, then trains heads from cache (faster for multiple experiments):

```bash
# Step 1: Extract hidden states (can set max_steps=0 to only extract)
accelerate launch --config_file configs/ddp_config.yaml scripts/train.py \
    --mode offline \
    --cache_dir ./hidden_state_cache \
    --cache_steps 5000 \
    --max_steps 0 \
    --model_name meta-llama/Meta-Llama-3-8B \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --output_dir ./checkpoints/my_model

# Step 2: Train from cache (much faster, no backbone needed)
python scripts/train.py \
    --mode offline \
    --cache_dir ./hidden_state_cache \
    --max_steps 3000 \
    --output_dir ./checkpoints/my_model
```

Cache is automatically reused if model/dataset match. Use `--cache_steps` to control extraction size.

### 2. Calibration

After training, calibrate thresholds for each head:

```bash
python scripts/calibrate.py \
    --config_path ./checkpoints/llama3-8b-4bit/config.json \
    --dataset_name wikitext \
    --dataset_config wikitext-2-raw-v1 \
    --num_samples 1000 \
    --output_path ./checkpoints/llama3-8b-4bit/calibration.json
```

### 3. Inference with Speculative Decoding

Run inference using early exit heads for self-speculative decoding:

```bash
# Dynamic early exit (adaptive)
python scripts/inference.py \
    --config_path ./checkpoints/llama3-8b-4bit/config.json \
    --calibration_path ./checkpoints/llama3-8b-4bit/calibration.json \
    --prompt "The quick brown fox" \
    --max_tokens 100 \
    --mode adaptive_early_exit \
    --accuracy_level 0.75

# Compare all methods
python scripts/inference.py \
    --config_path ./checkpoints/llama3-8b-4bit/config.json \
    --calibration_path ./checkpoints/llama3-8b-4bit/calibration.json \
    --prompt "The quick brown fox" \
    --mode comprehensive
```

### 4. Evaluate Head Accuracy

```bash
python scripts/evaluate.py \
    --config_path ./checkpoints/llama3-8b-4bit/config.json \
    --num_samples 100 \
    --output_path ./experiments/results/accuracy.json
```

## Pre-trained Checkpoints

| Model | Heads | Layers | Download |
|-------|-------|--------|----------|
| Llama-3 8B (4-bit) | 3 | 8, 16, 24 | `checkpoints/llama3-8b-4bit/` |

## How It Works

1. **Training**: Auxiliary heads are attached to intermediate decoder layers. Each head learns to predict the final token from intermediate hidden states.

2. **Calibration**: Run inference on a calibration set to determine entropy thresholds for each head at various target accuracy levels.

3. **Inference**: Use self-speculative decoding:
   - **Draft phase**: Generate tokens using early exit heads (faster, using fewer layers)
   - **Verify phase**: Run full model on drafted tokens to verify correctness
   - **Accept**: Keep tokens that match, correct first mismatch

## Requirements

```
torch>=2.0
transformers>=4.35
accelerate
bitsandbytes
datasets
wandb (optional)
scikit-learn (for calibration)
```

## Uploading to HuggingFace Hub

To share your trained models and use them with the DSSD Demo, upload the artifacts to the HuggingFace Hub.

### 1. Create a Repository
Create a new model repository on HuggingFace (e.g., `username/DSSD-MyModel`).

### 2. Upload Artifacts
You need to upload the following files from your checkpoint folder:
- `aux_heads.pt`: The trained weights
- `config.json`: Model configuration
- `calibration.json`: Calibration thresholds
- `README.md`: Model card

You can use the `huggingface-cli` or a Python script:

```bash
huggingface-cli upload username/DSSD-MyModel checkpoints/my_model/ . --exclude "*.bin" "*.pth"
```

Or using python:

```python
from huggingface_hub import HfApi
api = HfApi()

files = ["aux_heads.pt", "config.json", "calibration.json", "README.md"]
repo_id = "username/DSSD-MyModel"

for filename in files:
    api.upload_file(
        path_or_fileobj=f"checkpoints/my_model/{filename}",
        path_in_repo=filename,
        repo_id=repo_id,
        repo_type="model"
    )
```

