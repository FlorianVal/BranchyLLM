#!/usr/bin/env python3
"""
Calibration script for early exit heads.

Runs inference on a calibration dataset to compute entropy thresholds
for each head at various target accuracy levels.

Usage:
    python calibrate.py \
        --config_path ./Llama3-8B-Quantized/config.json \
        --dataset_name wikitext \
        --dataset_config wikitext-2-raw-v1 \
        --num_samples 1000 \
        --output_path ./calibration_results.json
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

from src.early_exit.model_config import ModelConfig, CalibrationResult
from src.early_exit.model_loader import (
    load_model_with_heads,
    compute_entropy,
    compute_confidence,
    get_uncertainty_fn,
)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def get_calibration_data(
    dataset_name: str,
    dataset_config: Optional[str],
    tokenizer: AutoTokenizer,
    num_samples: int,
    max_length: int = 512,
    split: str = "train",
) -> List[Dict]:
    """Load and tokenize calibration dataset."""
    logger.info(f"Loading dataset {dataset_name}/{dataset_config}")

    # Load dataset
    dataset = load_dataset(dataset_name, dataset_config, split=split, streaming=True)

    samples = []
    for i, example in enumerate(dataset):
        if i >= num_samples:
            break

        # Get text - handle different dataset formats
        if "text" in example:
            text = example["text"]
        elif "content" in example:
            text = example["content"]
        else:
            # Try first string field
            for key, value in example.items():
                if isinstance(value, str) and len(value) > 10:
                    text = value
                    break
            else:
                continue

        # Skip empty texts
        if not text or len(text.strip()) < 10:
            continue

        # Tokenize
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

    logger.info(f"Loaded {len(samples)} calibration samples")
    return samples


def calibrate_heads(
    model,
    tokenizer: AutoTokenizer,
    samples: List[Dict],
    uncertainty_fn,
    device: str = "cuda",
) -> Tuple[Dict[int, List], Dict[int, List], Dict[int, List]]:
    """
    Run calibration to collect uncertainty scores and correctness for each head.

    Returns:
        uncertainties: Dict[head_idx -> List[uncertainty_scores]]
        correctness: Dict[head_idx -> List[is_correct]]
        main_predictions: Dict[head_idx -> List[main_model_token_id]]
    """
    uncertainties = defaultdict(list)
    correctness = defaultdict(list)
    head_predictions = defaultdict(list)

    logger.info("Running calibration inference...")

    for sample in tqdm(samples, desc="Calibrating"):
        input_ids = sample["input_ids"].to(device)
        attention_mask = sample["attention_mask"].to(device)

        # Forward through model
        with torch.no_grad():
            main_logits, head_logits_list, _ = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

        # Get main model predictions (ground truth for heads)
        main_preds = torch.argmax(main_logits, dim=-1)  # (batch, seq_len)

        # For each head, collect uncertainty and correctness
        for head_idx, head_logits in enumerate(head_logits_list):
            # Compute uncertainty for each token position
            # Shape: (batch, seq_len)
            uncertainty = uncertainty_fn(head_logits, dim=-1)

            # Get head predictions
            head_preds = torch.argmax(head_logits, dim=-1)

            # Check correctness (does head agree with main model?)
            is_correct = (head_preds == main_preds).float()

            # Only consider non-padded positions
            mask = attention_mask.bool()

            # Flatten and filter by mask
            uncertainty_masked = uncertainty[mask].float().cpu().numpy()
            is_correct_masked = is_correct[mask].float().cpu().numpy()

            uncertainties[head_idx].extend(uncertainty_masked.tolist())
            correctness[head_idx].extend(is_correct_masked.tolist())

    return dict(uncertainties), dict(correctness), dict(head_predictions)


def compute_thresholds_max_coverage(
    unc: np.ndarray,
    corr: np.ndarray,
    acc_level: float,
) -> Tuple[float, float]:
    """
    Maximum coverage strategy: Find the LARGEST threshold that achieves target accuracy.
    Maximizes early exit rate.
    """
    sorted_indices = np.argsort(unc)
    sorted_unc = unc[sorted_indices]
    sorted_corr = corr[sorted_indices]

    cumsum_correct = np.cumsum(sorted_corr)
    cumsum_total = np.arange(1, len(sorted_corr) + 1)
    cumulative_acc = cumsum_correct / cumsum_total

    valid_indices = np.where(cumulative_acc >= acc_level)[0]

    if len(valid_indices) == 0:
        threshold = float(sorted_unc[0]) if len(sorted_unc) > 0 else 0.0
        coverage = 0.0
    else:
        best_idx = valid_indices[-1]
        threshold = float(sorted_unc[best_idx])
        coverage = (best_idx + 1) / len(sorted_unc)

    return threshold, coverage


def compute_thresholds_min_coverage(
    unc: np.ndarray,
    corr: np.ndarray,
    acc_level: float,
) -> Tuple[float, float]:
    """
    Minimum coverage strategy: Find the SMALLEST threshold that achieves target accuracy.
    Most conservative, best for safety-critical applications.
    """
    sorted_indices = np.argsort(unc)
    sorted_unc = unc[sorted_indices]
    sorted_corr = corr[sorted_indices]

    cumsum_correct = np.cumsum(sorted_corr)
    cumsum_total = np.arange(1, len(sorted_corr) + 1)
    cumulative_acc = cumsum_correct / cumsum_total

    valid_indices = np.where(cumulative_acc >= acc_level)[0]

    if len(valid_indices) == 0:
        threshold = float(sorted_unc[0]) if len(sorted_unc) > 0 else 0.0
        coverage = 0.0
    else:
        # Use the FIRST valid index (smallest threshold)
        best_idx = valid_indices[0]
        threshold = float(sorted_unc[best_idx])
        coverage = (best_idx + 1) / len(sorted_unc)

    return threshold, coverage


def compute_thresholds_midpoint(
    unc: np.ndarray,
    corr: np.ndarray,
    acc_level: float,
) -> Tuple[float, float]:
    """
    Midpoint strategy: Take the midpoint between the last valid and first invalid threshold.
    More robust to small perturbations in the data.
    """
    sorted_indices = np.argsort(unc)
    sorted_unc = unc[sorted_indices]
    sorted_corr = corr[sorted_indices]

    cumsum_correct = np.cumsum(sorted_corr)
    cumsum_total = np.arange(1, len(sorted_corr) + 1)
    cumulative_acc = cumsum_correct / cumsum_total

    valid_indices = np.where(cumulative_acc >= acc_level)[0]

    if len(valid_indices) == 0:
        threshold = float(sorted_unc[0]) if len(sorted_unc) > 0 else 0.0
        coverage = 0.0
    else:
        last_valid_idx = valid_indices[-1]
        last_valid_unc = float(sorted_unc[last_valid_idx])

        # Get first invalid (if exists)
        if last_valid_idx + 1 < len(sorted_unc):
            first_invalid_unc = float(sorted_unc[last_valid_idx + 1])
            threshold = (last_valid_unc + first_invalid_unc) / 2
        else:
            # All are valid, use the max
            threshold = last_valid_unc

        coverage = (last_valid_idx + 1) / len(sorted_unc)

    return threshold, coverage


def compute_thresholds_platt_scaling(
    unc: np.ndarray,
    corr: np.ndarray,
    acc_level: float,
) -> Tuple[float, float, Optional[object]]:
    """
    Platt scaling strategy: Fit a logistic regression to predict P(correct | entropy).
    Then find the threshold where P(correct) >= target_accuracy.

    Returns threshold, coverage, and the fitted calibrator.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.exceptions import ConvergenceWarning
    import warnings

    # Fit logistic regression
    X = unc.reshape(-1, 1)
    y = corr.astype(int)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        calibrator = LogisticRegression(max_iter=1000, solver="lbfgs")
        calibrator.fit(X, y)

    # Find threshold where predicted probability >= acc_level
    # P(correct=1 | entropy) = sigmoid(w * entropy + b)
    # We want P(correct=1) >= acc_level
    # sigmoid(w * threshold + b) = acc_level
    # w * threshold + b = logit(acc_level)
    # threshold = (logit(acc_level) - b) / w

    w = calibrator.coef_[0][0]
    b = calibrator.intercept_[0]

    # logit(p) = log(p / (1-p))
    logit_acc = np.log(acc_level / (1 - acc_level + 1e-10))

    if abs(w) < 1e-10:
        # Weight is essentially zero, can't determine threshold
        threshold = float(np.median(unc))
    else:
        threshold = (logit_acc - b) / w
        # Clamp to observed range
        threshold = float(np.clip(threshold, unc.min(), unc.max()))

    # Compute coverage at this threshold
    coverage = float(np.mean(unc <= threshold))

    return threshold, coverage, calibrator


def compute_thresholds(
    uncertainties: Dict[int, List],
    correctness: Dict[int, List],
    accuracy_levels: List[float],
    strategy: str = "max_coverage",
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Dict], Optional[Dict]]:
    """
    Compute entropy thresholds for each head at each accuracy level.

    Args:
        uncertainties: Dict[head_idx -> List[uncertainty_scores]]
        correctness: Dict[head_idx -> List[is_correct]]
        accuracy_levels: List of target accuracy levels
        strategy: One of "max_coverage", "min_coverage", "midpoint", "platt_scaling"

    Returns:
        thresholds: Dict[accuracy_level -> Dict[head_idx -> threshold]]
        statistics: Dict[head_idx -> Dict of stats]
        calibrators: Dict[head_idx -> fitted calibrator] (only for platt_scaling)
    """
    thresholds = {}
    statistics = {}
    calibrators = {} if strategy == "platt_scaling" else None

    logger.info(f"Using threshold strategy: {strategy}")

    for head_idx in sorted(uncertainties.keys()):
        unc = np.array(uncertainties[head_idx])
        corr = np.array(correctness[head_idx])

        # Compute statistics
        mean_unc = float(np.mean(unc))
        std_unc = float(np.std(unc))
        overall_acc = float(np.mean(corr))

        statistics[str(head_idx)] = {
            "mean_uncertainty": mean_unc,
            "std_uncertainty": std_unc,
            "overall_accuracy": overall_acc,
            "num_samples": len(unc),
        }

        logger.info(
            f"Head {head_idx}: mean_unc={mean_unc:.4f}, std_unc={std_unc:.4f}, "
            f"overall_acc={overall_acc:.4f}"
        )

    # For each accuracy level, find thresholds
    for acc_level in accuracy_levels:
        level_key = f"{acc_level:.2f}"
        thresholds[level_key] = {}

        for head_idx in sorted(uncertainties.keys()):
            unc = np.array(uncertainties[head_idx])
            corr = np.array(correctness[head_idx])

            if strategy == "max_coverage":
                threshold, coverage = compute_thresholds_max_coverage(
                    unc, corr, acc_level
                )
            elif strategy == "min_coverage":
                threshold, coverage = compute_thresholds_min_coverage(
                    unc, corr, acc_level
                )
            elif strategy == "midpoint":
                threshold, coverage = compute_thresholds_midpoint(unc, corr, acc_level)
            elif strategy == "platt_scaling":
                threshold, coverage, calibrator = compute_thresholds_platt_scaling(
                    unc, corr, acc_level
                )
                if calibrators is not None:
                    calibrators[f"{head_idx}_{level_key}"] = {
                        "coef": float(calibrator.coef_[0][0]),
                        "intercept": float(calibrator.intercept_[0]),
                    }
            else:
                raise ValueError(f"Unknown strategy: {strategy}")

            thresholds[level_key][str(head_idx)] = threshold

            logger.debug(
                f"Head {head_idx}, acc_level={acc_level}: threshold={threshold:.4f}, "
                f"coverage={coverage:.2%}"
            )

    return thresholds, statistics, calibrators


def main():
    parser = argparse.ArgumentParser(description="Calibrate early exit heads")
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
        help="Path to aux_heads.pt (default: infer from config_path)",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="wikitext",
        help="HuggingFace dataset name",
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
        default="train",
        help="Dataset split to use",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1000,
        help="Number of samples for calibration",
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
        required=True,
        help="Path to save calibration results JSON",
    )
    parser.add_argument(
        "--uncertainty_metric",
        type=str,
        default="entropy",
        choices=["entropy", "confidence"],
        help="Uncertainty metric to use",
    )
    parser.add_argument(
        "--accuracy_levels",
        type=str,
        default="0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95,0.99",
        help="Comma-separated accuracy levels for threshold computation",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use",
    )
    parser.add_argument(
        "--threshold_strategy",
        type=str,
        default="max_coverage",
        choices=["max_coverage", "min_coverage", "midpoint", "platt_scaling"],
        help="Threshold selection strategy: max_coverage (default, maximizes early exits), "
        "min_coverage (conservative), midpoint (robust), platt_scaling (probabilistic)",
    )
    args = parser.parse_args()

    # Parse accuracy levels
    accuracy_levels = [float(x) for x in args.accuracy_levels.split(",")]
    logger.info(f"Target accuracy levels: {accuracy_levels}")

    # Load model
    logger.info(f"Loading model from {args.config_path}")
    model, tokenizer, model_config = load_model_with_heads(
        config_path=args.config_path,
        checkpoint_path=args.checkpoint_path,
        device=args.device,
        match_training_quantization=True,
    )

    # Get uncertainty function
    uncertainty_fn = get_uncertainty_fn(args.uncertainty_metric)
    logger.info(f"Using uncertainty metric: {args.uncertainty_metric}")

    # Load calibration data
    samples = get_calibration_data(
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        tokenizer=tokenizer,
        num_samples=args.num_samples,
        max_length=args.max_length,
        split=args.split,
    )

    # Run calibration
    uncertainties, correctness, _ = calibrate_heads(
        model=model,
        tokenizer=tokenizer,
        samples=samples,
        uncertainty_fn=uncertainty_fn,
        device=args.device,
    )

    # Compute thresholds
    thresholds, statistics, calibrators = compute_thresholds(
        uncertainties=uncertainties,
        correctness=correctness,
        accuracy_levels=accuracy_levels,
        strategy=args.threshold_strategy,
    )

    # Create calibration result
    calibration = CalibrationResult(
        model_config_path=args.config_path,
        calibration_dataset=f"{args.dataset_name}/{args.dataset_config}",
        calibration_samples=len(samples),
        uncertainty_metric=args.uncertainty_metric,
        accuracy_levels=accuracy_levels,
        thresholds=thresholds,
        statistics=statistics,
    )

    # Save
    calibration.to_json(args.output_path)
    logger.info(f"Saved calibration results to {args.output_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("CALIBRATION SUMMARY")
    print("=" * 60)

    for head_idx, stats in statistics.items():
        print(f"\nHead {head_idx}:")
        print(f"  Overall accuracy: {stats['overall_accuracy']:.2%}")
        print(f"  Mean uncertainty: {stats['mean_uncertainty']:.4f}")
        print(f"  Std uncertainty: {stats['std_uncertainty']:.4f}")

    print("\n" + "-" * 60)
    print("Thresholds by accuracy level:")
    print("-" * 60)

    for level_key in sorted(thresholds.keys(), key=float):
        print(f"\nAccuracy {level_key}:")
        for head_idx in sorted(thresholds[level_key].keys(), key=int):
            threshold = thresholds[level_key][head_idx]
            print(f"  Head {head_idx}: threshold = {threshold:.4f}")


if __name__ == "__main__":
    main()
