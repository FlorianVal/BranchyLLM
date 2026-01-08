#!/usr/bin/env python3
"""
Speculative Decoding with Early Exit Heads.

Implements LayerSkip-style speculative decoding using trained early exit
heads as the draft model. Supports:
1. Full model baseline (no speculation)
2. Fixed LayerSkip (single head, fixed draft length)
3. Dynamic early exit (entropy-based, variable draft length)

Key insight: We don't need to rerun the full model during verification
because we already computed the full forward pass during drafting - we just
use the main lm_head output to verify the drafted tokens.

Usage:
    python speculative_decoding.py \
        --config_path ./Llama3-8B-Quantized/config.json \
        --calibration_path ./calibration_results.json \
        --prompt "The quick brown fox" \
        --max_tokens 50 \
        --mode dynamic_early_exit \
        --accuracy_level 0.95
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from src.early_exit.model_config import ModelConfig, CalibrationResult
from src.early_exit.model_loader import (
    load_model_with_heads,
    get_uncertainty_fn,
    compute_entropy,
)
from src.early_exit.debug_logger import DebugLogger

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


@dataclass
class GenerationStats:
    """Statistics from text generation."""

    mode: str
    total_tokens_generated: int
    total_time_seconds: float
    tokens_per_second: float
    # Speculative decoding specific
    draft_attempts: int = 0
    tokens_drafted: int = 0
    tokens_accepted: int = 0
    acceptance_rate: float = 0.0
    average_draft_length: float = 0.0
    # Early exit specific
    exit_distribution: Dict[str, int] = None  # head_idx/layer -> count
    average_exit_layer: float = 0.0


class SpeculativeDecoder:
    """
    Speculative decoder using early exit heads.
    """

    def __init__(
        self,
        model,
        tokenizer: AutoTokenizer,
        model_config: ModelConfig,
        calibration: Optional[CalibrationResult] = None,
        device: str = "cuda",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.model_config = model_config
        self.calibration = calibration
        self.device = device

        # Get uncertainty function if calibration provided
        if calibration is not None:
            self.uncertainty_fn = get_uncertainty_fn(calibration.uncertainty_metric)
        else:
            self.uncertainty_fn = compute_entropy

    def generate_full_model(
        self,
        prompt: str,
        max_tokens: int,
    ) -> Tuple[str, GenerationStats]:
        """
        Baseline: Generate tokens using only the full model.
        """
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        generated_tokens = []

        start_time = time.time()

        for _ in range(max_tokens):
            with torch.no_grad():
                outputs = self.model.model(input_ids, use_cache=False)
                logits = outputs.logits

            # Get next token
            next_token_logits = logits[0, -1, :]
            next_token_id = torch.argmax(next_token_logits).item()

            # Check for EOS
            if next_token_id == self.tokenizer.eos_token_id:
                break

            generated_tokens.append(next_token_id)
            input_ids = torch.cat(
                [input_ids, torch.tensor([[next_token_id]], device=self.device)], dim=1
            )

        end_time = time.time()
        total_time = end_time - start_time

        # Decode output
        output_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        stats = GenerationStats(
            mode="full_model",
            total_tokens_generated=len(generated_tokens),
            total_time_seconds=total_time,
            tokens_per_second=len(generated_tokens) / total_time
            if total_time > 0
            else 0,
        )

        return prompt + output_text, stats

    def generate_fixed_layerskip(
        self,
        prompt: str,
        max_tokens: int,
        draft_head_idx: int = 0,
        draft_length: int = 4,
    ) -> Tuple[str, GenerationStats]:
        """
        Fixed LayerSkip: Use single head with fixed draft length.

        Key insight: During drafting, we already run the full forward pass
        (to get activations for the head). We capture the main_logits at each
        step and use them for verification, avoiding the need to re-run.
        """
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        generated_tokens = []

        draft_attempts = 0
        tokens_drafted = 0
        drafts_accepted = 0  # How many drafted tokens matched main model

        start_time = time.time()

        while len(generated_tokens) < max_tokens:
            # Draft phase: generate draft_length tokens using the specified head
            # Also capture main_logits for each position for verification
            drafted = []
            main_logits_history = []  # Store main model logits for verification
            current_ids = input_ids.clone()

            for _ in range(draft_length):
                with torch.no_grad():
                    # Full forward to get intermediate activations
                    main_logits, head_logits_list, activations = self.model(
                        input_ids=current_ids,
                    )

                # Store the main model's prediction for verification
                main_logits_history.append(main_logits[0, -1, :].clone())

                # Use the draft head for prediction
                head_logits = head_logits_list[draft_head_idx]
                next_token_id = torch.argmax(head_logits[0, -1, :]).item()

                if next_token_id == self.tokenizer.eos_token_id:
                    break

                drafted.append(next_token_id)
                current_ids = torch.cat(
                    [current_ids, torch.tensor([[next_token_id]], device=self.device)],
                    dim=1,
                )

            if not drafted:
                break

            draft_attempts += 1
            tokens_drafted += len(drafted)

            # Verify phase: use the main_logits we already captured during drafting
            # This ensures correct context for each verification
            accepted = []

            for i, (drafted_token, main_logits_at_pos) in enumerate(
                zip(drafted, main_logits_history[: len(drafted)])
            ):
                # Get the full model's prediction for this position
                verified_token = torch.argmax(main_logits_at_pos).item()

                if drafted_token == verified_token:
                    accepted.append(drafted_token)
                    drafts_accepted += 1
                else:
                    # Mismatch: accept the correct token and stop verification
                    accepted.append(verified_token)
                    break

            generated_tokens.extend(accepted)

            # Update input_ids with accepted tokens
            input_ids = torch.cat(
                [input_ids, torch.tensor([accepted], device=self.device)], dim=1
            )

            # Check for EOS
            if self.tokenizer.eos_token_id in accepted:
                break

        end_time = time.time()
        total_time = end_time - start_time

        output_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        stats = GenerationStats(
            mode="fixed_layerskip",
            total_tokens_generated=len(generated_tokens),
            total_time_seconds=total_time,
            tokens_per_second=len(generated_tokens) / total_time
            if total_time > 0
            else 0,
            draft_attempts=draft_attempts,
            tokens_drafted=tokens_drafted,
            tokens_accepted=drafts_accepted,  # How many drafts were correct
            acceptance_rate=drafts_accepted / tokens_drafted
            if tokens_drafted > 0
            else 0,
            average_draft_length=tokens_drafted / draft_attempts
            if draft_attempts > 0
            else 0,
        )

        return prompt + output_text, stats

    def generate_dynamic_early_exit(
        self,
        prompt: str,
        max_tokens: int,
        accuracy_level: float = 0.95,
        max_draft_length: int = 10,
    ) -> Tuple[str, GenerationStats]:
        """
        Dynamic Early Exit: Use entropy thresholds to determine:
        1. Which head to use for drafting each token
        2. When to stop drafting and verify

        Key insight: During the draft phase, we run the FULL forward pass
        but make predictions using early exit heads when they're confident.
        When no head is confident, we already have the main lm_head output,
        so we use it directly for verification without rerunning anything.
        """
        if self.calibration is None:
            raise ValueError("Calibration required for dynamic early exit")

        level_key = f"{accuracy_level:.2f}"
        if level_key not in self.calibration.thresholds:
            raise ValueError(f"No calibration for accuracy level {accuracy_level}")

        thresholds = {
            int(k): v for k, v in self.calibration.thresholds[level_key].items()
        }

        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        generated_tokens = []

        draft_attempts = 0
        tokens_drafted = 0
        tokens_accepted = 0
        exit_counts = defaultdict(int)  # head_idx -> count
        layer_sum = 0

        num_heads = self.model_config.num_heads
        head_layer_indices = self.model_config.head_layer_indices
        num_layers = self.model_config.num_hidden_layers

        start_time = time.time()

        while len(generated_tokens) < max_tokens:
            drafted = []
            draft_head_indices = []  # Which head was used for each drafted token
            current_ids = input_ids.clone()

            # Keep track of main model logits for verification
            main_logits_history = []

            # Draft phase
            for _ in range(max_draft_length):
                with torch.no_grad():
                    # Run full forward pass
                    main_logits, head_logits_list, _ = self.model(
                        input_ids=current_ids,
                    )

                main_logits_history.append(main_logits[0, -1, :].clone())

                # Try each head in order (earliest first)
                exited = False
                for head_idx in range(num_heads):
                    head_logits = head_logits_list[head_idx]
                    token_logits = head_logits[0, -1, :]

                    uncertainty = self.uncertainty_fn(
                        token_logits.unsqueeze(0), dim=-1
                    ).item()
                    threshold = thresholds[head_idx]

                    if uncertainty < threshold:
                        # Head is confident, use its prediction
                        next_token_id = torch.argmax(token_logits).item()

                        if next_token_id == self.tokenizer.eos_token_id:
                            break

                        drafted.append(next_token_id)
                        draft_head_indices.append(head_idx)
                        exit_counts[head_idx] += 1
                        layer_sum += head_layer_indices[head_idx]

                        current_ids = torch.cat(
                            [
                                current_ids,
                                torch.tensor([[next_token_id]], device=self.device),
                            ],
                            dim=1,
                        )
                        exited = True
                        break

                if not exited:
                    # No head confident enough
                    # We already have the main model output, use it directly!
                    # This IS the verification - no recomputation needed
                    next_token_id = torch.argmax(main_logits[0, -1, :]).item()
                    exit_counts["full"] = exit_counts.get("full", 0) + 1
                    layer_sum += num_layers

                    if next_token_id == self.tokenizer.eos_token_id:
                        break

                    # When main model is used, we must verify all drafted tokens
                    break

            if not drafted and not main_logits_history:
                break

            draft_attempts += 1
            tokens_drafted += len(drafted)

            # Verification phase
            if drafted:
                # Verify drafted tokens against main model predictions
                # We already have main_logits_history from the draft phase!
                accepted = []
                drafts_matched = 0  # Count how many drafts were correct

                for i, (drafted_token, main_logits_at_pos) in enumerate(
                    zip(drafted, main_logits_history[: len(drafted)])
                ):
                    verified_token = torch.argmax(main_logits_at_pos).item()

                    if drafted_token == verified_token:
                        accepted.append(drafted_token)
                        drafts_matched += 1
                    else:
                        # Mismatch: accept correct token and stop
                        accepted.append(verified_token)
                        break

                tokens_accepted += drafts_matched  # Only count matching drafts
                generated_tokens.extend(accepted)

                input_ids = torch.cat(
                    [input_ids, torch.tensor([accepted], device=self.device)], dim=1
                )
            else:
                # No drafts, just use the main model token
                next_token_id = torch.argmax(main_logits_history[-1]).item()
                if next_token_id != self.tokenizer.eos_token_id:
                    generated_tokens.append(next_token_id)
                    # Don't count main model tokens as "accepted drafts"
                    input_ids = torch.cat(
                        [
                            input_ids,
                            torch.tensor([[next_token_id]], device=self.device),
                        ],
                        dim=1,
                    )
                else:
                    break

            # Check for EOS
            if self.tokenizer.eos_token_id in generated_tokens[-max_draft_length:]:
                break

        end_time = time.time()
        total_time = end_time - start_time

        output_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        # Calculate average exit layer
        total_exits = sum(exit_counts.values())
        avg_exit_layer = layer_sum / total_exits if total_exits > 0 else num_layers

        stats = GenerationStats(
            mode="dynamic_early_exit",
            total_tokens_generated=len(generated_tokens),
            total_time_seconds=total_time,
            tokens_per_second=len(generated_tokens) / total_time
            if total_time > 0
            else 0,
            draft_attempts=draft_attempts,
            tokens_drafted=tokens_drafted,
            tokens_accepted=tokens_accepted,  # Now correctly counts only matching drafts
            acceptance_rate=tokens_accepted / tokens_drafted
            if tokens_drafted > 0
            else 0,
            average_draft_length=tokens_drafted / draft_attempts
            if draft_attempts > 0
            else 0,
            exit_distribution=dict(exit_counts),
            average_exit_layer=avg_exit_layer,
        )

        return prompt + output_text, stats

    def generate_adaptive_early_exit(
        self,
        prompt: str,
        max_tokens: int,
        accuracy_level: float = 0.75,
        max_draft_length: int = 5,
        debug_log_path: Optional[str] = None,
    ) -> Tuple[str, GenerationStats]:
        """
        Self-speculative decoding with adaptive early exit.

        This method GUARANTEES the same output as full model by:
        1. DRAFT: Generate K tokens using early exit heads (fast, partial compute)
        2. VERIFY: Run full model on all K tokens at once
        3. ACCEPT: Keep tokens that match full model, correct first mismatch

        Speedup comes from:
        - Drafting uses fewer layers (early exit)
        - Verification is batched (1 full pass for K tokens)
        - High acceptance rate = more tokens per full pass

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            accuracy_level: Target for draft thresholds
            max_draft_length: Maximum tokens to draft per cycle
            debug_log_path: Optional path for debug logs
        """
        if self.calibration is None:
            raise ValueError("Calibration required for adaptive early exit")

        level_key = f"{accuracy_level:.2f}"
        if level_key not in self.calibration.thresholds:
            raise ValueError(f"No calibration for accuracy level {accuracy_level}")

        thresholds = {
            int(k): v for k, v in self.calibration.thresholds[level_key].items()
        }

        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        generated_tokens = []

        head_layer_indices = self.model_config.head_layer_indices
        num_layers = self.model_config.num_hidden_layers

        # Stats tracking
        draft_attempts = 0
        tokens_drafted = 0
        tokens_accepted = 0  # Drafts that matched full model
        total_layers_in_drafts = 0
        exit_counts = defaultdict(int)

        # Debug logging
        debug_logger = None
        if debug_log_path:
            debug_logger = DebugLogger(
                debug_log_path, method_name=f"adaptive_spec_{accuracy_level}"
            )
            debug_logger.set_metadata(
                model=self.model_config.model_name,
                accuracy_level=accuracy_level,
                thresholds=thresholds,
            )

        start_time = time.time()

        while len(generated_tokens) < max_tokens:
            # ============================================================
            # DRAFT PHASE: Generate tokens using early exit heads
            # ============================================================
            drafted = []
            draft_exit_layers = []
            current_ids = input_ids.clone()

            for draft_idx in range(max_draft_length):
                # Use adaptive forward (partial compute)
                token_id, _, exit_layer, uncertainty, debug_info = (
                    self.model.forward_adaptive(
                        input_ids=current_ids,
                        past_key_values=None,  # No cache for drafting (need full context)
                        thresholds=thresholds,
                        uncertainty_fn=self.uncertainty_fn,
                    )
                )

                layers_executed = debug_info.get("layers_executed", num_layers)
                total_layers_in_drafts += layers_executed

                # Track which head was used
                exit_head = None
                for head_idx, head_layer in enumerate(head_layer_indices):
                    if exit_layer == head_layer:
                        exit_head = head_idx
                        break

                if exit_head is not None:
                    exit_counts[exit_head] += 1
                else:
                    exit_counts["full"] += 1

                if token_id == self.tokenizer.eos_token_id:
                    break

                drafted.append(token_id)
                draft_exit_layers.append(exit_layer)
                current_ids = torch.cat(
                    [current_ids, torch.tensor([[token_id]], device=self.device)], dim=1
                )

            if not drafted:
                break

            draft_attempts += 1
            tokens_drafted += len(drafted)

            # ============================================================
            # VERIFY PHASE: Run full model on all drafted tokens
            # ============================================================
            with torch.no_grad():
                # Run full model on input + drafted tokens
                outputs = self.model.model(current_ids, use_cache=False)
                verify_logits = outputs.logits

            # Verify each drafted token
            accepted = []
            start_pos = input_ids.shape[1] - 1  # Position before drafting

            for i, drafted_token in enumerate(drafted):
                verify_pos = start_pos + i
                verified_token = torch.argmax(verify_logits[0, verify_pos, :]).item()

                if drafted_token == verified_token:
                    accepted.append(drafted_token)
                    tokens_accepted += 1
                else:
                    # Mismatch: use full model's token and stop
                    accepted.append(verified_token)
                    break

            # Debug logging for each token
            if debug_logger:
                for i, token_id in enumerate(accepted):
                    was_drafted = drafted[i] if i < len(drafted) else None
                    matched = i < len(drafted) and drafted[i] == accepted[i]
                    token_text = self.tokenizer.decode([token_id])
                    debug_logger.log_token(
                        position=len(generated_tokens) + i,
                        method="draft_accepted" if matched else "corrected",
                        token_id=token_id,
                        token_text=token_text,
                        exit_layer=draft_exit_layers[i]
                        if i < len(draft_exit_layers)
                        else num_layers,
                        extra_info=f"drafted={was_drafted}, accepted={token_id}",
                    )

            generated_tokens.extend(accepted)

            # Update input_ids
            input_ids = torch.cat(
                [input_ids, torch.tensor([accepted], device=self.device)], dim=1
            )

            if self.tokenizer.eos_token_id in accepted:
                break

        end_time = time.time()
        total_time = end_time - start_time

        if debug_logger:
            debug_logger.save()

        output_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        # Calculate stats
        total_tokens = len(generated_tokens)
        avg_layers_per_draft = (
            total_layers_in_drafts / tokens_drafted
            if tokens_drafted > 0
            else num_layers
        )
        acceptance_rate = tokens_accepted / tokens_drafted if tokens_drafted > 0 else 0

        # Average exit layer from distribution
        layer_sum = 0
        total_exits = sum(exit_counts.values())
        for head_idx, count in exit_counts.items():
            if head_idx == "full":
                layer_sum += count * num_layers
            else:
                layer_sum += count * head_layer_indices[head_idx]
        avg_exit_layer = layer_sum / total_exits if total_exits > 0 else num_layers

        stats = GenerationStats(
            mode="adaptive_early_exit",
            total_tokens_generated=total_tokens,
            total_time_seconds=total_time,
            tokens_per_second=total_tokens / total_time if total_time > 0 else 0,
            draft_attempts=draft_attempts,
            tokens_drafted=tokens_drafted,
            tokens_accepted=tokens_accepted,
            acceptance_rate=acceptance_rate,
            average_draft_length=tokens_drafted / draft_attempts
            if draft_attempts > 0
            else 0,
            exit_distribution=dict(exit_counts),
            average_exit_layer=avg_exit_layer,
        )

        return prompt + output_text, stats


def compare_methods(
    decoder: SpeculativeDecoder,
    prompt: str,
    max_tokens: int,
    accuracy_level: float = 0.95,
    layerskip_head: int = 0,
    layerskip_draft_length: int = 4,
) -> Dict[str, Tuple[str, GenerationStats]]:
    """Run all methods and compare results."""
    results = {}

    logger.info("Running full model baseline...")
    text, stats = decoder.generate_full_model(prompt, max_tokens)
    results["full_model"] = (text, stats)

    logger.info(
        f"Running fixed LayerSkip (head={layerskip_head}, draft_len={layerskip_draft_length})..."
    )
    text, stats = decoder.generate_fixed_layerskip(
        prompt, max_tokens, layerskip_head, layerskip_draft_length
    )
    results["fixed_layerskip"] = (text, stats)

    logger.info(f"Running dynamic early exit (accuracy_level={accuracy_level})...")
    text, stats = decoder.generate_dynamic_early_exit(
        prompt, max_tokens, accuracy_level
    )
    results["dynamic_early_exit"] = (text, stats)

    return results


def print_comparison(results: Dict[str, Tuple[str, GenerationStats]]):
    """Print comparison of generation methods."""
    print("\n" + "=" * 80)
    print("SPECULATIVE DECODING COMPARISON")
    print("=" * 80)

    baseline_tps = results.get(
        "full_model",
        (
            None,
            GenerationStats(
                mode="",
                total_tokens_generated=0,
                total_time_seconds=1,
                tokens_per_second=1,
            ),
        ),
    )[1].tokens_per_second

    print(
        f"\n{'Method':<35} {'Tokens/s':<10} {'Speedup':<8} {'Accept%':<10} {'AvgDraft':<8}"
    )
    print("-" * 75)

    for method, (text, stats) in results.items():
        speedup = stats.tokens_per_second / baseline_tps if baseline_tps > 0 else 0
        accept_rate = (
            f"{stats.acceptance_rate:.1%}" if stats.acceptance_rate > 0 else "N/A"
        )
        avg_draft = (
            f"{stats.average_draft_length:.1f}"
            if stats.average_draft_length > 0
            else "N/A"
        )

        print(
            f"{method:<35} "
            f"{stats.tokens_per_second:<10.2f} "
            f"{speedup:<8.2f}x "
            f"{accept_rate:<10} "
            f"{avg_draft:<8}"
        )

    # Print generated texts
    print("\n" + "-" * 80)
    print("GENERATED TEXTS")
    print("-" * 80)

    for method, (text, stats) in results.items():
        print(f"\n[{method}] ({stats.total_tokens_generated} tokens):")
        # Truncate long texts
        display_text = text if len(text) < 300 else text[:300] + "..."
        print(display_text)

    # Print exit distribution for dynamic methods
    for method, (text, stats) in results.items():
        if stats.exit_distribution:
            print(f"\n{'-' * 40}")
            print(f"EXIT DISTRIBUTION: {method}")
            print("-" * 40)
            total = sum(stats.exit_distribution.values())
            for key in sorted(stats.exit_distribution.keys(), key=lambda x: str(x)):
                count = stats.exit_distribution[key]
                pct = count / total * 100 if total > 0 else 0
                label = f"Head {key}" if key != "full" else "Full Model"
                print(f"  {label}: {count} ({pct:.1f}%)")
            print(f"  Average exit layer: {stats.average_exit_layer:.1f}")


def comprehensive_compare(
    decoder: SpeculativeDecoder,
    prompt: str,
    max_tokens: int,
    heads: List[int],
    draft_lengths: List[int],
    accuracy_levels: List[float],
) -> Dict[str, Tuple[str, GenerationStats]]:
    """
    Run comprehensive comparison of all configurations:
    - Full model baseline
    - Fixed LayerSkip for each head × each draft length
    - Dynamic early exit for each accuracy level
    """
    results = {}

    # Full model baseline
    logger.info("Running full model baseline...")
    text, stats = decoder.generate_full_model(prompt, max_tokens)
    results["full_model"] = (text, stats)
    baseline_text = text

    # Fixed LayerSkip for all combinations
    for head_idx in heads:
        for draft_len in draft_lengths:
            key = f"layerskip_h{head_idx}_d{draft_len}"
            logger.info(f"Running {key}...")
            text, stats = decoder.generate_fixed_layerskip(
                prompt, max_tokens, head_idx, draft_len
            )
            results[key] = (text, stats)

    # Dynamic early exit for each accuracy level
    for acc_level in accuracy_levels:
        key = f"dynamic_acc{acc_level:.2f}"
        logger.info(f"Running {key}...")
        try:
            text, stats = decoder.generate_dynamic_early_exit(
                prompt, max_tokens, acc_level
            )
            results[key] = (text, stats)
        except ValueError as e:
            logger.warning(f"Skipping {key}: {e}")

    return results, baseline_text


def print_comprehensive_comparison(
    results: Dict[str, Tuple[str, GenerationStats]],
    baseline_text: str,
    model_config: ModelConfig,
):
    """Print comprehensive comparison results."""
    print("\n" + "=" * 110)
    print("COMPREHENSIVE SPECULATIVE DECODING ANALYSIS")
    print("=" * 110)
    print(f"\nModel: {model_config.model_name}")
    print(f"Heads at layers: {model_config.head_layer_indices}")

    baseline_stats = results.get("full_model", (None, None))[1]
    baseline_tps = baseline_stats.tokens_per_second if baseline_stats else 1.0

    # Section 1: Fixed LayerSkip Results
    print("\n" + "-" * 110)
    print("FIXED LAYERSKIP CONFIGURATIONS")
    print("-" * 110)
    print(
        f"\n{'Config':<25} {'Tokens/s':<10} {'Speedup':<8} {'Drafted':<10} {'Accepted':<10} {'Accept%':<10} {'Match?':<8}"
    )
    print("-" * 85)

    # First print baseline
    if baseline_stats:
        print(
            f"{'full_model':<25} {baseline_stats.tokens_per_second:<10.2f} {'1.00x':<8} {'N/A':<10} {'N/A':<10} {'N/A':<10} {'✓':<8}"
        )

    layerskip_results = [
        (k, v) for k, v in results.items() if k.startswith("layerskip")
    ]
    layerskip_results.sort(key=lambda x: x[0])

    for method, (text, stats) in layerskip_results:
        speedup = stats.tokens_per_second / baseline_tps if baseline_tps > 0 else 0
        drafted = str(stats.tokens_drafted)
        accepted = str(stats.tokens_accepted)
        accept_rate = f"{stats.acceptance_rate:.1%}"
        match = "✓" if text == baseline_text else "✗"

        print(
            f"{method:<25} "
            f"{stats.tokens_per_second:<10.2f} "
            f"{speedup:<8.2f}x "
            f"{drafted:<10} "
            f"{accepted:<10} "
            f"{accept_rate:<10} "
            f"{match:<8}"
        )

    # Section 2: Dynamic Early Exit Results
    print("\n" + "-" * 110)
    print("DYNAMIC EARLY EXIT CONFIGURATIONS")
    print("-" * 110)
    print(
        f"\n{'Config':<25} {'Tokens/s':<10} {'Speedup':<8} {'Drafted':<10} {'Accepted':<10} {'Accept%':<10} {'AvgLayer':<10} {'Match?':<8}"
    )
    print("-" * 95)

    dynamic_results = [(k, v) for k, v in results.items() if k.startswith("dynamic")]
    dynamic_results.sort(key=lambda x: x[0])

    for method, (text, stats) in dynamic_results:
        speedup = stats.tokens_per_second / baseline_tps if baseline_tps > 0 else 0
        drafted = str(stats.tokens_drafted) if stats.tokens_drafted > 0 else "0"
        accepted = str(stats.tokens_accepted) if stats.tokens_drafted > 0 else "0"
        accept_rate = (
            f"{stats.acceptance_rate:.1%}" if stats.tokens_drafted > 0 else "N/A"
        )
        avg_layer = f"{stats.average_exit_layer:.1f}"
        match = "✓" if text == baseline_text else "✗"

        print(
            f"{method:<25} "
            f"{stats.tokens_per_second:<10.2f} "
            f"{speedup:<8.2f}x "
            f"{drafted:<10} "
            f"{accepted:<10} "
            f"{accept_rate:<10} "
            f"{avg_layer:<10} "
            f"{match:<8}"
        )

    # Section 3: Exit distribution for dynamic methods
    print("\n" + "-" * 100)
    print("DYNAMIC EXIT DISTRIBUTIONS")
    print("-" * 100)

    for method, (text, stats) in dynamic_results:
        if stats.exit_distribution:
            total = sum(stats.exit_distribution.values())
            dist_str = ", ".join(
                f"H{k}:{v / total * 100:.0f}%"
                if k != "full"
                else f"Full:{v / total * 100:.0f}%"
                for k, v in sorted(
                    stats.exit_distribution.items(), key=lambda x: str(x[0])
                )
            )
            print(f"{method}: {dist_str}")

    # Section 4: Sample outputs
    print("\n" + "-" * 100)
    print("SAMPLE OUTPUTS (showing first 200 chars)")
    print("-" * 100)

    print(f"\n[full_model]:\n{baseline_text[:200]}...")

    # Show one LayerSkip that differs
    for method, (text, stats) in layerskip_results[:3]:
        if text != baseline_text:
            print(f"\n[{method}] (DIFFERS):\n{text[:200]}...")
            break

    # Show dynamic at different accuracy levels
    for method, (text, stats) in dynamic_results:
        match_status = "MATCHES" if text == baseline_text else "DIFFERS"
        print(f"\n[{method}] ({match_status}):\n{text[:200]}...")


def main():
    parser = argparse.ArgumentParser(
        description="Speculative decoding with early exit heads"
    )
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
        help="Path to aux_heads.pt",
    )
    parser.add_argument(
        "--calibration_path",
        type=str,
        default=None,
        help="Path to calibration results JSON (required for dynamic mode)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="The quick brown fox",
        help="Input prompt for generation",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=50,
        help="Maximum tokens to generate",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="compare",
        choices=[
            "full",
            "fixed_layerskip",
            "dynamic_early_exit",
            "adaptive_early_exit",
            "compare",
            "comprehensive",
            "test_adaptive",
        ],
        help="Generation mode. 'adaptive_early_exit' uses TRUE early exit with real speedup",
    )
    parser.add_argument(
        "--accuracy_level",
        type=float,
        default=0.95,
        help="Accuracy level for dynamic early exit",
    )
    parser.add_argument(
        "--layerskip_head",
        type=int,
        default=0,
        help="Head index for fixed LayerSkip mode",
    )
    parser.add_argument(
        "--layerskip_draft_length",
        type=int,
        default=4,
        help="Draft length for fixed LayerSkip mode",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Path to save results JSON",
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

    # Load calibration if provided
    calibration = None
    if args.calibration_path:
        logger.info(f"Loading calibration from {args.calibration_path}")
        calibration = CalibrationResult.from_json(args.calibration_path)
    elif args.mode in ["dynamic_early_exit", "compare", "comprehensive"]:
        raise ValueError("Calibration required for dynamic early exit mode")

    # Create decoder
    decoder = SpeculativeDecoder(
        model=model,
        tokenizer=tokenizer,
        model_config=model_config,
        calibration=calibration,
        device=args.device,
    )

    # Run generation
    if args.mode == "compare":
        results = compare_methods(
            decoder=decoder,
            prompt=args.prompt,
            max_tokens=args.max_tokens,
            accuracy_level=args.accuracy_level,
            layerskip_head=args.layerskip_head,
            layerskip_draft_length=args.layerskip_draft_length,
        )
        print_comparison(results)

        # Save results
        if args.output_path:
            output = {
                method: {
                    "text": text,
                    "stats": asdict(stats),
                }
                for method, (text, stats) in results.items()
            }
            with open(args.output_path, "w") as f:
                json.dump(output, f, indent=2)
            logger.info(f"Saved results to {args.output_path}")

    elif args.mode == "full":
        text, stats = decoder.generate_full_model(args.prompt, args.max_tokens)
        print(f"\nGenerated text:\n{text}")
        print(f"\nStats: {stats.tokens_per_second:.2f} tokens/s")

    elif args.mode == "fixed_layerskip":
        text, stats = decoder.generate_fixed_layerskip(
            args.prompt,
            args.max_tokens,
            args.layerskip_head,
            args.layerskip_draft_length,
        )
        print(f"\nGenerated text:\n{text}")
        print(
            f"\nStats: {stats.tokens_per_second:.2f} tokens/s, "
            f"acceptance rate: {stats.acceptance_rate:.1%}"
        )

    elif args.mode == "dynamic_early_exit":
        text, stats = decoder.generate_dynamic_early_exit(
            args.prompt, args.max_tokens, args.accuracy_level
        )
        print(f"\nGenerated text:\n{text}")
        print(
            f"\nStats: {stats.tokens_per_second:.2f} tokens/s, "
            f"acceptance rate: {stats.acceptance_rate:.1%}, "
            f"avg exit layer: {stats.average_exit_layer:.1f}"
        )

    elif args.mode == "comprehensive":
        # All heads and draft lengths for LayerSkip
        heads = list(range(model_config.num_heads))
        draft_lengths = [5, 10, 15]

        # All available accuracy levels from calibration
        accuracy_levels = (
            calibration.accuracy_levels if calibration else [0.70, 0.75, 0.80]
        )

        logger.info(f"Running comprehensive analysis:")
        logger.info(f"  Heads: {heads}")
        logger.info(f"  Draft lengths: {draft_lengths}")
        logger.info(f"  Accuracy levels: {accuracy_levels}")

        results, baseline_text = comprehensive_compare(
            decoder=decoder,
            prompt=args.prompt,
            max_tokens=args.max_tokens,
            heads=heads,
            draft_lengths=draft_lengths,
            accuracy_levels=accuracy_levels,
        )
        print_comprehensive_comparison(results, baseline_text, model_config)

        # Save results
        if args.output_path:
            output = {
                method: {
                    "text": text,
                    "stats": asdict(stats),
                }
                for method, (text, stats) in results.items()
            }
            with open(args.output_path, "w") as f:
                json.dump(output, f, indent=2)
            logger.info(f"Saved results to {args.output_path}")

    elif args.mode == "adaptive_early_exit":
        # TRUE adaptive early exit with real speedup
        debug_log_path = None
        if args.output_path:
            debug_log_path = args.output_path.replace(".json", "_debug.json")

        text, stats = decoder.generate_adaptive_early_exit(
            args.prompt,
            args.max_tokens,
            args.accuracy_level,
            debug_log_path=debug_log_path,
        )
        print(f"\nGenerated text:\n{text}")
        print(
            f"\nStats: {stats.tokens_per_second:.2f} tokens/s, "
            f"avg layers: {stats.average_draft_length * model_config.num_hidden_layers:.1f}/{model_config.num_hidden_layers}, "
            f"avg exit layer: {stats.average_exit_layer:.1f}"
        )
        print(f"Exit distribution: {stats.exit_distribution}")

    elif args.mode == "test_adaptive":
        # Test that adaptive matches full model when no thresholds are met
        logger.info("Testing adaptive forward vs full model...")

        # Generate with full model
        full_text, full_stats = decoder.generate_full_model(
            args.prompt, args.max_tokens
        )

        # Generate with adaptive (high accuracy = use full model mostly)
        debug_log_path = "./debug_logs/test_adaptive_full.json"
        adaptive_text, adaptive_stats = decoder.generate_adaptive_early_exit(
            args.prompt,
            args.max_tokens,
            accuracy_level=0.99,
            debug_log_path=debug_log_path,
        )

        # Compare
        print("\n" + "=" * 60)
        print("ADAPTIVE FORWARD TEST")
        print("=" * 60)
        print(f"\nFull model output (first 200 chars):\n{full_text[:200]}...")
        print(f"\nAdaptive output (first 200 chars):\n{adaptive_text[:200]}...")

        if full_text == adaptive_text:
            print("\n✓ MATCH: Adaptive produces same output as full model!")
        else:
            print("\n✗ MISMATCH: Outputs differ!")
            # Find first difference
            for i, (c1, c2) in enumerate(zip(full_text, adaptive_text)):
                if c1 != c2:
                    print(f"First difference at char {i}: '{c1}' vs '{c2}'")
                    break

        print(f"\nFull model: {full_stats.tokens_per_second:.2f} tokens/s")
        print(f"Adaptive: {adaptive_stats.tokens_per_second:.2f} tokens/s")
        print(f"Adaptive exit distribution: {adaptive_stats.exit_distribution}")


if __name__ == "__main__":
    main()
