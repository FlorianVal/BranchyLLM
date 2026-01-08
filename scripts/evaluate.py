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
                "input_ids": tokens["input_ids"],
                "attention_mask": tokens["attention_mask"],
            }
        )

    logger.info(f"Loaded {len(samples)} evaluation samples")
    return samples


def evaluate_head_accuracy(
    model,
    samples: List[Dict],
    model_config: ModelConfig,
    device: str = "cuda",
) -> EvaluationResults:
    """
    Evaluate the accuracy of each head and the main lm_head.

    For next-token prediction, we compare each head's predictions to the
    main model's predictions (treating main model as ground truth).
    """
    num_heads = model_config.num_heads
    head_layer_indices = model_config.head_layer_indices

    # Counters for main head
    main_correct = 0
    main_top5_correct = 0
    total_tokens = 0

    # Counters per auxiliary head
    head_correct = [0] * num_heads
    head_top5_correct = [0] * num_heads
    # Counters for accuracy vs lm_head (how often head matches main model)
    head_matches_lm = [0] * num_heads
    head_top5_matches_lm = [0] * num_heads

    for sample in tqdm(samples, desc="Evaluating"):
        input_ids = sample["input_ids"].to(device)
        attention_mask = sample["attention_mask"].to(device)

        with torch.no_grad():
            main_logits, head_logits_list, _ = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

        # Create labels by shifting input_ids by 1 (next token prediction)
        # Label at position t is the token at position t+1
        labels = input_ids[:, 1:].clone()

        # Predictions are made at positions 0 to seq_len-2 to predict position 1 to seq_len-1
        main_logits_shifted = main_logits[:, :-1, :]

        # Main model predictions and accuracy
        main_preds = torch.argmax(main_logits_shifted, dim=-1)
        main_top5_preds = torch.topk(main_logits_shifted, k=5, dim=-1).indices

        # Get valid positions (where attention mask is 1 and we have a label)
        valid_mask = attention_mask[:, 1:] == 1

        batch_size, seq_len = labels.shape

        for b in range(batch_size):
            for t in range(seq_len):
                if not valid_mask[b, t]:
                    continue

                total_tokens += 1
                label = labels[b, t].item()
                main_pred = main_preds[b, t].item()
                main_top5 = main_top5_preds[b, t].tolist()

                # Main head accuracy
                if main_pred == label:
                    main_correct += 1
                if label in main_top5:
                    main_top5_correct += 1

                # Auxiliary head accuracy
                for head_idx in range(num_heads):
                    head_logits = head_logits_list[head_idx][:, :-1, :]
                    head_pred = torch.argmax(head_logits[b, t, :]).item()
                    head_top5 = torch.topk(head_logits[b, t, :], k=5).indices.tolist()

                    # Accuracy vs ground truth (next token)
                    if head_pred == label:
                        head_correct[head_idx] += 1
                    if label in head_top5:
                        head_top5_correct[head_idx] += 1

                    # Accuracy vs lm_head (how often head matches main model)
                    if head_pred == main_pred:
                        head_matches_lm[head_idx] += 1
                    if main_pred in head_top5:
                        head_top5_matches_lm[head_idx] += 1

    # Compute accuracies
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
            correct=head_correct[head_idx],
            total=total_tokens,
            accuracy=accuracy,
            top5_accuracy=top5_accuracy,
            accuracy_vs_lm_head=accuracy_vs_lm,
            top5_accuracy_vs_lm_head=top5_acc_vs_lm,
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
    )

    # Print results
    print_results(results)

    # Save results
    save_results(results, args.output_path)


if __name__ == "__main__":
    main()
