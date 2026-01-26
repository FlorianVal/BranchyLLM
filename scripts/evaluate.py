#!/usr/bin/env python3
"""
Evaluate Head Accuracy Script.

Evaluates the accuracy of each auxiliary head and the main lm_head on wikitext.

Usage:
    python evaluate_head_accuracy.py \
        --config_path ./Llama3-8B-Quantized/config.json \
        --num_samples 100 \
        --output_path ./head_accuracy_results.json
"""

import argparse
import json
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict

import torch
from datasets import load_dataset
from tqdm import tqdm

from src.early_exit.model_config import ModelConfig
from src.early_exit.model_loader import load_model_with_heads

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


@dataclass
class HeadAccuracyResult:
    """Accuracy results for a single head."""

    head_idx: int
    layer_idx: int
    correct: int
    total: int
    accuracy: float
    top5_accuracy: float
    # Accuracy against lm_head (how often head matches main model prediction)
    accuracy_vs_lm_head: float
    top5_accuracy_vs_lm_head: float


@dataclass
class EvaluationResults:
    """Full evaluation results."""

    model_name: str
    num_samples: int
    total_tokens: int
    main_head_accuracy: float
    main_head_top5_accuracy: float
    head_results: List[HeadAccuracyResult]


def get_evaluation_data(
    dataset_name: str,
    dataset_config: Optional[str],
    tokenizer,
    num_samples: int,
    max_length: int = 512,
    split: str = "test",
) -> List[Dict]:
    """Load and tokenize evaluation dataset."""
    logger.info(f"Loading dataset {dataset_name}/{dataset_config} split={split}")

    # Try to load the specified split, fall back to validation or train
    try:
        dataset = load_dataset(
            dataset_name, dataset_config, split=split, streaming=True
        )
    except ValueError:
        try:
            dataset = load_dataset(
                dataset_name, dataset_config, split="validation", streaming=True
            )
            logger.info("Using validation split instead of test")
        except ValueError:
            dataset = load_dataset(
                dataset_name, dataset_config, split="train", streaming=True
            )
            logger.info("Using train split instead of test")

    samples = []
    for i, example in enumerate(dataset):
        if i >= num_samples:
            break

        # Get text
        if "text" in example:
            text = example["text"]
        elif "content" in example:
            text = example["content"]
        else:
            for key, value in example.items():
                if isinstance(value, str) and len(value) > 10:
                    text = value
                    break
            else:
                continue

        if not text or len(text.strip()) < 10:
            continue

        tokens = tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors="pt",
        )

        samples.append(
            {
                "input_ids": tokens["input_ids"][0],  # Remove batch dim [1, L] -> [L]
                "attention_mask": tokens["attention_mask"][0],
            }
        )

    logger.info(f"Loaded {len(samples)} evaluation samples")
    return samples


def evaluate_head_accuracy(
    model,
    samples: List[Dict],
    model_config: ModelConfig,
    device: str = "cuda",
    batch_size: int = 8,
) -> EvaluationResults:
    """
    Evaluate the accuracy of each head and the main lm_head.
    Supports batched inference and vectorized metric calculation.
    """
    from torch.utils.data import DataLoader
    from transformers import default_data_collator

    num_heads = model_config.num_heads
    head_layer_indices = model_config.head_layer_indices

    # Create DataLoader for batching
    dataloader = DataLoader(
        samples,
        batch_size=batch_size,
        collate_fn=default_data_collator,
        shuffle=False,
    )

    # Counters
    total_tokens = 0
    main_correct = 0
    main_top5_correct = 0

    # Per-head counters
    head_correct = torch.zeros(num_heads, device=device, dtype=torch.long)
    head_top5_correct = torch.zeros(num_heads, device=device, dtype=torch.long)
    head_matches_lm = torch.zeros(num_heads, device=device, dtype=torch.long)
    head_top5_matches_lm = torch.zeros(num_heads, device=device, dtype=torch.long)

    for batch in tqdm(dataloader, desc="Evaluating"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        with torch.no_grad():
            main_logits, head_logits_list, _ = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

        # Labels are next tokens (shift input_ids by 1)
        labels = input_ids[:, 1:]  # [B, L-1]
        valid_mask = attention_mask[:, 1:] == 1  # [B, L-1]
        num_valid_tokens = valid_mask.sum().item()
        total_tokens += num_valid_tokens

        # --- Main Model Evaluation ---
        # Shift logits: predict token t+1 from hidden state at t
        main_logits_shifted = main_logits[:, :-1, :]  # [B, L-1, V]
        
        # Top-1
        main_preds = torch.argmax(main_logits_shifted, dim=-1)  # [B, L-1]
        main_correct_mask = (main_preds == labels) & valid_mask
        main_correct += main_correct_mask.sum().item()

        # Top-5
        # labels.unsqueeze(-1) -> [B, L-1, 1]
        # topk.indices -> [B, L-1, 5]
        main_top5 = torch.topk(main_logits_shifted, k=5, dim=-1).indices
        main_top5_mask = (main_top5 == labels.unsqueeze(-1)).any(dim=-1) & valid_mask
        main_top5_correct += main_top5_mask.sum().item()

        # --- Aux Heads Evaluation ---
        for head_idx, head_logits in enumerate(head_logits_list):
            head_logits_shifted = head_logits[:, :-1, :]  # [B, L-1, V]
            
            # Head Predictions
            head_preds = torch.argmax(head_logits_shifted, dim=-1)  # [B, L-1]
            head_top5 = torch.topk(head_logits_shifted, k=5, dim=-1).indices

            # 1. Accuracy vs Ground Truth
            head_correct_mask = (head_preds == labels) & valid_mask
            head_correct[head_idx] += head_correct_mask.sum()

            head_top5_mask = (head_top5 == labels.unsqueeze(-1)).any(dim=-1) & valid_mask
            head_top5_correct[head_idx] += head_top5_mask.sum()

            # 2. Accuracy vs Main Model (Fidelity)
            # Matches if head prediction == main model prediction
            match_mask = (head_preds == main_preds) & valid_mask
            head_matches_lm[head_idx] += match_mask.sum()

            # Top-5 Match: Does main model's pred appear in head's top 5?
            # Or: Does head's pred appear in main model's top 5? 
            # Original code: "if main_pred in head_top5"
            match_top5_mask = (head_top5 == main_preds.unsqueeze(-1)).any(dim=-1) & valid_mask
            head_top5_matches_lm[head_idx] += match_top5_mask.sum()

    # Move metrics to CPU
    head_correct = head_correct.cpu().numpy()
    head_top5_correct = head_top5_correct.cpu().numpy()
    head_matches_lm = head_matches_lm.cpu().numpy()
    head_top5_matches_lm = head_top5_matches_lm.cpu().numpy()

    # Compute Accuracies
    main_accuracy = main_correct / total_tokens if total_tokens > 0 else 0.0
    main_top5_accuracy = main_top5_correct / total_tokens if total_tokens > 0 else 0.0

    head_results = []
    for head_idx in range(num_heads):
        accuracy = head_correct[head_idx] / total_tokens if total_tokens > 0 else 0.0
        top5_accuracy = (
            head_top5_correct[head_idx] / total_tokens if total_tokens > 0 else 0.0
        )
        accuracy_vs_lm = (
            head_matches_lm[head_idx] / total_tokens if total_tokens > 0 else 0.0
        )
        top5_acc_vs_lm = (
            head_top5_matches_lm[head_idx] / total_tokens if total_tokens > 0 else 0.0
        )

        result = HeadAccuracyResult(
            head_idx=head_idx,
            layer_idx=head_layer_indices[head_idx],
            correct=int(head_correct[head_idx]),
            total=total_tokens,
            accuracy=float(accuracy),
            top5_accuracy=float(top5_accuracy),
            accuracy_vs_lm_head=float(accuracy_vs_lm),
            top5_accuracy_vs_lm_head=float(top5_acc_vs_lm),
        )
        head_results.append(result)

    return EvaluationResults(
        model_name=model_config.model_name,
        num_samples=len(samples),
        total_tokens=total_tokens,
        main_head_accuracy=main_accuracy,
        main_head_top5_accuracy=main_top5_accuracy,
        head_results=head_results,
    )


def print_results(results: EvaluationResults):
    """Print results in a nice table format."""
    print("\n" + "=" * 80)
    print("HEAD ACCURACY EVALUATION RESULTS")
    print("=" * 80)
    print(f"Model: {results.model_name}")
    print(f"Samples: {results.num_samples}")
    print(f"Total tokens evaluated: {results.total_tokens}")
    print()

    # Table 1: Accuracy vs ground truth (next token)
    print("Accuracy vs Ground Truth (next token prediction):")
    print(
        f"{'Head':<12} {'Layer':<10} {'Accuracy':<15} {'Top-5 Acc':<15} {'Correct/Total':<20}"
    )
    print("-" * 72)

    # Print main head first
    print(
        f"{'Main (lm)':<12} "
        f"{'Final':<10} "
        f"{results.main_head_accuracy:<15.2%} "
        f"{results.main_head_top5_accuracy:<15.2%} "
        f"{int(results.main_head_accuracy * results.total_tokens)}/{results.total_tokens}"
    )
    print("-" * 72)

    # Print auxiliary heads
    for hr in results.head_results:
        print(
            f"{'Head ' + str(hr.head_idx):<12} "
            f"{hr.layer_idx:<10} "
            f"{hr.accuracy:<15.2%} "
            f"{hr.top5_accuracy:<15.2%} "
            f"{hr.correct}/{hr.total}"
        )

    print()

    # Table 2: Accuracy vs lm_head (how often heads match main model)
    print("Accuracy vs lm_head (how often head matches main model prediction):")
    print(f"{'Head':<12} {'Layer':<10} {'Match Rate':<15} {'Top-5 Match':<15}")
    print("-" * 52)

    for hr in results.head_results:
        print(
            f"{'Head ' + str(hr.head_idx):<12} "
            f"{hr.layer_idx:<10} "
            f"{hr.accuracy_vs_lm_head:<15.2%} "
            f"{hr.top5_accuracy_vs_lm_head:<15.2%}"
        )

    print()


def save_results(results: EvaluationResults, output_path: str):
    """Save results to JSON."""
    output = {
        "model_name": results.model_name,
        "num_samples": results.num_samples,
        "total_tokens": results.total_tokens,
        "main_head": {
            "accuracy": results.main_head_accuracy,
            "top5_accuracy": results.main_head_top5_accuracy,
        },
        "auxiliary_heads": [asdict(hr) for hr in results.head_results],
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    logger.info(f"Saved results to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate head accuracy on wikitext")
    parser.add_argument(
        "--config_path",
        type=str,
        required=True,
        help="Path to model config JSON",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Path to aux_heads.pt (if None, inferred from config_path dir)",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="wikitext",
        help="Evaluation dataset name",
    )
    parser.add_argument(
        "--dataset_config",
        type=str,
        default="wikitext-2-raw-v1",
        help="Dataset configuration",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=100,
        help="Number of samples for evaluation",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Maximum sequence length",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./head_accuracy_results.json",
        help="Path to save results",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for evaluation",
    )
    args = parser.parse_args()

    # Load model
    logger.info(f"Loading model from {args.config_path}")
    model, tokenizer, model_config = load_model_with_heads(
        config_path=args.config_path,
        checkpoint_path=args.checkpoint_path,
        device=args.device,
        match_training_quantization=True,
    )

    # Load evaluation data
    samples = get_evaluation_data(
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        tokenizer=tokenizer,
        num_samples=args.num_samples,
        max_length=args.max_length,
        split=args.split,
    )

    # Run evaluation
    results = evaluate_head_accuracy(
        model=model,
        samples=samples,
        model_config=model_config,
        device=args.device,
        batch_size=args.batch_size,
    )

    # Print results
    print_results(results)

    # Save results
    save_results(results, args.output_path)


if __name__ == "__main__":
    main()
