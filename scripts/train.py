import argparse
import logging
import os
from datetime import datetime
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.logging import get_logger
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    default_data_collator,
    get_cosine_schedule_with_warmup,
)

from src.early_exit.hidden_state_cache import (
    CacheMetadata,
    HiddenStateCache,
    CachedHiddenStateDataset,
    extract_hidden_states,
    compute_config_hash,
)

# Initialize logger
logger = get_logger(__name__)


class AuxiliaryHead(nn.Module):
    """
    Auxiliary Head that mimics the architecture of the model's lm_head.
    It typically consists of a Normalization layer (if applicable) and a Linear layer.
    """

    def __init__(
        self, hidden_size: int, vocab_size: int, norm_layer: Optional[nn.Module] = None
    ):
        super().__init__()
        self.norm = norm_layer if norm_layer is not None else nn.Identity()
        self.linear = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: (batch_size, seq_len, hidden_size)
        Returns:
            logits: (batch_size, seq_len, vocab_size)
        """
        hidden_states = self.norm(hidden_states)
        logits = self.linear(hidden_states)
        return logits


class ModelWithAuxiliaryHeads(nn.Module):
    """
    Wrapper class to handle the main model, auxiliary heads, and hooks.
    """

    def __init__(
        self,
        model_name: str,
        num_heads: int,
        quantization: str = "none",
        device_map: str = "auto",
    ):
        super().__init__()
        self.model_name = model_name
        self.num_heads = num_heads

        # Quantization config
        quantization_config = None
        if quantization == "4bit":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16
                if torch.cuda.is_bf16_supported()
                else torch.float32,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        elif quantization == "8bit":
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
            )

        # Load main model
        self.config = AutoConfig.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            config=self.config,
            quantization_config=quantization_config,
            torch_dtype=torch.bfloat16
            if torch.cuda.is_bf16_supported()
            else torch.float32,
            # We trust accelerate to handle device placement if no quantization,
            # but for quantization we might need to rely on HF accelerate integration or device_map="auto"??
            # Actually, standard accelerate launch handles this.
        )

        # Freeze main model
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        self.aux_heads = nn.ModuleList()
        self.head_layer_indices = self._calculate_head_layer_indices()
        self.hook_handles = []
        self.intermediate_activations = {}

        self._initialize_heads()
        self._register_hooks()

    def _calculate_head_layer_indices(self) -> List[int]:
        """
        Calculate indices for equal spacing.
        We exclude 0 (embedding) and final layer (main head).
        """
        num_layers = self.config.num_hidden_layers
        # linspace from 0 to num_layers, take internal points
        # e.g. 32 layers, 1 head -> [0, 16, 32] -> index 16
        # e.g. 32 layers, 3 heads -> [0, 8, 16, 24, 32] -> indices 8, 16, 24
        indices = np.linspace(0, num_layers, self.num_heads + 2, dtype=int)[1:-1]
        logger.info(f"Placing heads at layers: {indices}")
        return indices.tolist()

    def _initialize_heads(self):
        """
        Initialize auxiliary heads to mimic the main model's head structure.
        Llama 3 has a Model.norm (RMSNorm) before the lm_head.
        """
        hidden_size = self.config.hidden_size
        vocab_size = self.config.vocab_size

        for _ in range(self.num_heads):
            # Create a copy of the normalization layer logic
            # We instantiate new metrics, we don't share the main model's norm parameters
            # because the statistics at intermediate layers might be different,
            # but we want the same architecture (RMSNorm).
            # We can copy the class from the main model.

            # Llama model usually has `model.norm`
            if hasattr(self.model, "model") and hasattr(self.model.model, "norm"):
                original_norm = self.model.model.norm
                # Deepcopy or re-instantiate similar norm
                import copy

                norm_layer = copy.deepcopy(original_norm)
            elif hasattr(self.model, "transformer") and hasattr(
                self.model.transformer, "ln_f"
            ):
                # GPT-2
                original_norm = self.model.transformer.ln_f
                import copy

                norm_layer = copy.deepcopy(original_norm)
            else:
                norm_layer = None
                logger.warning(
                    "Could not find normalization layer in model. Using Identity."
                )

            head = AuxiliaryHead(hidden_size, vocab_size, norm_layer)
            # Cast head to model dtype
            head.to(self.model.dtype)
            self.aux_heads.append(head)

    def _hook_fn(self, module, input, output, layer_idx):
        # Llama decoder layer output is a tuple (hidden_states, ...)
        if isinstance(output, tuple):
            hidden_states = output[0]
        else:
            hidden_states = output

        # Check for batch dimension to avoid storage issues if needed,
        # but here we just store reference.
        # We need to ensure we don't keep graph if not needed, but we DO need graph for training heads.
        self.intermediate_activations[layer_idx] = hidden_states

    def _register_hooks(self):
        """
        Register forward hooks on the specified layers.
        """
        # Clear any existing hooks
        for h in self.hook_handles:
            h.remove()
        self.hook_handles = []

        # Identify layers.
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            # Llama, Mistral, etc.
            layers = self.model.model.layers
        elif hasattr(self.model, "transformer") and hasattr(
            self.model.transformer, "h"
        ):
            # GPT-2, GPT-J
            layers = self.model.transformer.h
        elif hasattr(self.model, "layers"):
            # Some other models
            layers = self.model.layers
        else:
            # Try to find ModuleList by inspecting
            found = False
            for name, module in self.model.named_modules():
                if isinstance(module, nn.ModuleList) and len(module) > 0:
                    layers = module
                    found = True
                    break
            if not found:
                raise ValueError(f"Unknown model structure for {self.model_name}")

        for idx in self.head_layer_indices:
            # Register hook on the layer
            handle = layers[idx].register_forward_hook(
                lambda m, i, o, layer_idx=idx: self._hook_fn(m, i, o, layer_idx)
            )
            self.hook_handles.append(handle)

    def forward(self, input_ids, attention_mask=None, **kwargs):
        """
        Forward pass.
        1. Run main model (frozen) to populate intermediate activations and get teacher logits.
        2. Run auxiliary heads on intermediate activations.
        """
        self.intermediate_activations = {}

        with torch.no_grad():
            outputs = self.model(
                input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                **kwargs,
            )
            teacher_logits = outputs.logits

        # Collect head outputs
        head_outputs = []
        for idx, head_module in zip(self.head_layer_indices, self.aux_heads):
            hidden = self.intermediate_activations[idx]
            # Verify hidden state requires grad (it shouldn't from model, but we need it for input?)
            # Actually, since model is frozen, 'hidden' has requires_grad=False.
            # But the Head parameters have requires_grad=True.
            # So the loss will backprop into Head parameters.
            # We detach hidden just to be safe we don't try to backprop into main model
            # (though requires_grad=False should prevent it).
            hidden = hidden.detach()

            head_logits = head_module(hidden)
            head_outputs.append(head_logits)

        return teacher_logits, head_outputs


def get_dataloaders(
    dataset_name: str,
    dataset_config_name: Optional[str],
    tokenizer: AutoTokenizer,
    batch_size: int,
    max_length: int = 512,
):
    """
    Load and process dataset.
    Returns train_dataloader and eval_dataloader.
    """
    # Load train and validation datasets
    # Try to load 'validation', if not exists, try 'test', else split?
    # For streaming, splitting is hard. Let's try to load 'validation' split.
    try:
        train_dataset = load_dataset(
            dataset_name, dataset_config_name, split="train", streaming=True
        )
        try:
            eval_dataset = load_dataset(
                dataset_name, dataset_config_name, split="validation", streaming=True
            )
        except Exception:
            # Fallback to test if validation not found
            eval_dataset = load_dataset(
                dataset_name, dataset_config_name, split="test", streaming=True
            )
    except Exception as e:
        logger.warning(
            f"Could not load validation/test split: {e}. Using a slice of train for eval (if possible) or skipping."
        )
        eval_dataset = None

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )

    # Train
    train_tokenized_dataset = train_dataset.map(
        tokenize_function, batched=True, remove_columns=["text"]
    )
    train_dataloader = DataLoader(
        train_tokenized_dataset, batch_size=batch_size, collate_fn=default_data_collator
    )

    # Eval
    eval_dataloader = None
    if eval_dataset is not None:
        eval_tokenized_dataset = eval_dataset.map(
            tokenize_function, batched=True, remove_columns=["text"]
        )
        eval_dataloader = DataLoader(
            eval_tokenized_dataset,
            batch_size=batch_size,
            collate_fn=default_data_collator,
        )

    return train_dataloader, eval_dataloader


def compute_kl_loss(student_logits, teacher_logits):
    """
    KL Divergence Loss.
    """
    # Softmax on both
    temp = 1.0
    teacher_probs = F.softmax(teacher_logits / temp, dim=-1)
    student_log_probs = F.log_softmax(student_logits / temp, dim=-1)

    # KLDivLoss: input should be log-probs, target should be probs
    loss = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean")
    return loss


def compute_accuracy(student_logits, teacher_logits):
    """
    Compute accuracy: fraction of times student top-1 matches teacher top-1.
    """
    student_preds = torch.argmax(student_logits, dim=-1)
    teacher_preds = torch.argmax(teacher_logits, dim=-1)
    return (student_preds == teacher_preds).float().mean()


def evaluate(model_wrapper, eval_dataloader, accelerator, num_steps=100):
    """
    Run evaluation on the validation dataset.
    """
    model_wrapper.eval()
    losses = []
    accuracies = []

    # We might not have len(eval_dataloader) for streaming, so we iterate for num_steps
    data_iter = iter(eval_dataloader)

    logger.info("Starting evaluation...")

    for _ in tqdm(
        range(num_steps),
        disable=not accelerator.is_local_main_process,
        desc="Evaluating",
    ):
        try:
            batch = next(data_iter)
        except StopIteration:
            break

        with torch.no_grad():
            teacher_logits, head_logits_list = model_wrapper(**batch)

            batch_loss = 0.0

            batch_acc = 0.0

            for i, head_logits in enumerate(head_logits_list):
                loss = compute_kl_loss(head_logits, teacher_logits)
                acc = compute_accuracy(head_logits, teacher_logits)

                batch_loss += loss.item()
                batch_acc += acc.item()

                # We can track per-head accuracy, but for simplicity let's track mean accuracy
                # or maybe just log everything at end?
                # Let's just store the mean accuracy for now for simple metric
                # If we want detailed, we'd need a list of list

            batch_loss /= len(head_logits_list)
            batch_acc /= len(head_logits_list)

            losses.append(batch_loss)
            accuracies.append(batch_acc)

    avg_loss = np.mean(losses) if losses else 0.0
    avg_acc = np.mean(accuracies) if accuracies else 0.0

    model_wrapper.train()

    return {"eval_loss": avg_loss, "eval_accuracy": avg_acc}


def train_from_cache(
    aux_heads: nn.ModuleList,
    cache_dir: str,
    layer_indices: List[int],
    max_steps: int,
    optimizer,
    lr_scheduler,
    accelerator,
    args,
    eval_dataloader=None,
    model_wrapper=None,  # Needed for evaluation
):
    """Train auxiliary heads from cached hidden states.

    This is Phase 2 of offline training - loads cached hidden states
    and trains heads without running the backbone.
    """
    logger.info(f"Training from cached hidden states at {cache_dir}")

    # Load cached dataset
    cache_dataset = CachedHiddenStateDataset(cache_dir, layer_indices)

    # Create dataloader that iterates over cached batches
    # Note: each item is already a full batch, so batch_size=1
    cache_dataloader = DataLoader(
        cache_dataset,
        batch_size=1,  # Each cached item is already a batch
        shuffle=True,  # Can shuffle since we're training from cache
        num_workers=0,
    )

    # Prepare with accelerator
    aux_heads, optimizer, cache_dataloader = accelerator.prepare(
        aux_heads, optimizer, cache_dataloader
    )

    aux_heads.train()

    progress_bar = tqdm(range(max_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0

    # Metrics tracking
    running_metrics = {}
    accumulation_counter = 0
    best_eval_loss = float("inf")

    while completed_steps < max_steps:
        for batch in cache_dataloader:
            if completed_steps >= max_steps:
                break

            with accelerator.accumulate(aux_heads):
                # Get teacher logits (squeeze batch dim since we use batch_size=1 in dataloader)
                teacher_logits = batch["teacher_logits"].squeeze(0)

                # Compute loss for each head
                total_loss = 0.0
                metrics = {}

                for head_idx, (layer_idx, head_module) in enumerate(
                    zip(layer_indices, aux_heads)
                ):
                    hidden_states = batch[f"hidden_states_{layer_idx}"].squeeze(0)

                    # Forward through head
                    head_logits = head_module(hidden_states)

                    # Compute KL loss
                    loss = compute_kl_loss(head_logits, teacher_logits)
                    acc = compute_accuracy(head_logits, teacher_logits)

                    total_loss += loss
                    metrics[f"head_{head_idx}_loss"] = loss.item()
                    metrics[f"head_{head_idx}_accuracy"] = acc.item()

                # Mean loss
                total_loss = total_loss / len(aux_heads)
                metrics["train_loss"] = total_loss.item()

                # Accumulate metrics
                for k, v in metrics.items():
                    running_metrics[k] = running_metrics.get(k, 0.0) + v
                accumulation_counter += 1

                # Backward
                accelerator.backward(total_loss)
                optimizer.step()
                optimizer.zero_grad()

                # Log on sync
                if accelerator.sync_gradients:
                    avg_metrics = {
                        k: v / accumulation_counter for k, v in running_metrics.items()
                    }

                    progress_bar.update(1)
                    completed_steps += 1
                    lr_scheduler.step()

                    if accelerator.is_local_main_process:
                        current_lr = lr_scheduler.get_last_lr()[0]
                        avg_metrics["lr"] = current_lr
                        accelerator.log(avg_metrics, step=completed_steps)

                        if completed_steps % 50 == 0:
                            logger.info(
                                f"Step {completed_steps} - Train Loss: {avg_metrics['train_loss']:.4f}"
                            )

                        progress_bar.set_postfix(
                            loss=f"{avg_metrics['train_loss']:.4f}",
                            lr=f"{current_lr:.2e}",
                        )

                    # Reset accumulation
                    running_metrics = {}
                    accumulation_counter = 0

                    # Evaluation
                    if (
                        completed_steps % args.eval_steps == 0
                        and eval_dataloader is not None
                        and model_wrapper is not None
                    ):
                        eval_metrics = evaluate(
                            model_wrapper,
                            eval_dataloader,
                            accelerator,
                            num_steps=args.eval_max_steps,
                        )
                        if accelerator.is_local_main_process:
                            accelerator.log(eval_metrics, step=completed_steps)
                            logger.info(
                                f"Step {completed_steps} - Eval Loss: {eval_metrics['eval_loss']:.4f}"
                            )

                            if eval_metrics["eval_loss"] < best_eval_loss:
                                best_eval_loss = eval_metrics["eval_loss"]
                                torch.save(
                                    aux_heads.state_dict(),
                                    os.path.join(args.output_dir, "best_aux_heads.pt"),
                                )

                    # Checkpointing
                    if completed_steps % args.save_steps == 0:
                        if accelerator.is_local_main_process:
                            torch.save(
                                aux_heads.state_dict(),
                                os.path.join(
                                    args.output_dir,
                                    f"aux_heads_step_{completed_steps}.pt",
                                ),
                            )

    return aux_heads


def _save_config(args, unwrapped_model):
    """Save model configuration to JSON for inference."""
    import json

    config_data = {
        "model_name": args.model_name,
        "num_heads": args.num_heads,
        "head_layer_indices": unwrapped_model.head_layer_indices,
        "quantization": args.quantization,
        "hidden_size": unwrapped_model.config.hidden_size,
        "vocab_size": unwrapped_model.config.vocab_size,
        "num_hidden_layers": unwrapped_model.config.num_hidden_layers,
        "training_config": {
            "dataset_name": args.dataset_name,
            "dataset_config_name": args.dataset_config_name,
            "batch_size": args.batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "max_steps": args.max_steps,
            "lr": args.lr,
            "max_length": args.max_length,
            "mode": args.mode,
        },
    }
    config_path = os.path.join(args.output_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config_data, f, indent=2)
    logger.info(f"Saved config to {config_path}")


def main():
    parser = argparse.ArgumentParser(description="Train Auxiliary Heads on Llama 3")
    parser.add_argument(
        "--model_name",
        type=str,
        default="meta-llama/Meta-Llama-3-8B",
        help="Model name",
    )
    parser.add_argument("--dataset_name", type=str, required=True, help="Dataset name")
    parser.add_argument(
        "--dataset_config_name", type=str, default=None, help="Dataset config name"
    )
    parser.add_argument(
        "--num_heads", type=int, default=3, help="Number of auxiliary heads"
    )
    parser.add_argument(
        "--batch_size", type=int, default=4, help="Batch size per device"
    )
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument(
        "--max_steps", type=int, default=1000, help="Max training steps"
    )
    parser.add_argument(
        "--output_dir", type=str, default="aux_heads_output", help="Output directory"
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="auxiliary_heads",
        help="WandB project name",
    )
    parser.add_argument("--run_name", type=str, default=None, help="WandB run name")
    parser.add_argument(
        "--quantization",
        type=str,
        default="none",
        choices=["none", "4bit", "8bit"],
        help="Quantization: none, 4bit, 8bit",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Maximum sequence length (context length).",
    )
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=500,
        help="Run evaluation every X steps.",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=1000,
        help="Save checkpoint every X steps.",
    )
    parser.add_argument(
        "--eval_max_steps",
        type=int,
        default=100,
        help="Number of steps to run for evaluation.",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=0,
        help="Number of steps for the warmup phase.",
    )
    # Offline mode arguments
    parser.add_argument(
        "--mode",
        type=str,
        default="online",
        choices=["online", "offline"],
        help="Training mode: online (run backbone+heads together) or offline (cache hidden states first)",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="./hidden_state_cache",
        help="Directory to store cached hidden states for offline mode",
    )
    parser.add_argument(
        "--cache_steps",
        type=int,
        default=None,
        help="Number of steps to cache in offline mode (default: same as max_steps)",
    )
    args = parser.parse_args()

    if args.run_name is None:
        model_basename = args.model_name.split("/")[-1]
        args.run_name = f"{model_basename}-{datetime.now().strftime('%Y-%m-%d/%H:%M')}"

    accelerator = Accelerator(
        log_with="wandb", gradient_accumulation_steps=args.gradient_accumulation_steps
    )
    # Initialize trackers
    if accelerator.is_local_main_process:
        accelerator.init_trackers(
            project_name=args.wandb_project,
            config=vars(args),
            init_kwargs={"wandb": {"name": args.run_name}} if args.run_name else {},
        )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    if accelerator.is_local_main_process:
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load Model Wrapper
    model_wrapper = ModelWithAuxiliaryHeads(
        args.model_name, args.num_heads, args.quantization
    )

    # Optimizer
    # Use bnb 8-bit optimizer if quantization is enabled to save memory
    if args.quantization != "none":
        try:
            import bitsandbytes as bnb

            optimizer = bnb.optim.AdamW8bit(
                model_wrapper.aux_heads.parameters(), lr=args.lr
            )
            logger.info("Using 8-bit AdamW optimizer")
        except ImportError:
            logger.warning(
                "bitsandbytes not installed, falling back to torch.optim.AdamW"
            )
            optimizer = torch.optim.AdamW(
                model_wrapper.aux_heads.parameters(), lr=args.lr
            )
    else:
        optimizer = torch.optim.AdamW(model_wrapper.aux_heads.parameters(), lr=args.lr)

    # Scheduler
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.max_steps,
        num_cycles=0.5,
    )

    # Dataloader
    train_dataloader, eval_dataloader = get_dataloaders(
        args.dataset_name,
        args.dataset_config_name,
        tokenizer,
        args.batch_size,
        args.max_length,
    )

    # =========================================================================
    # MODE BRANCHING: Online vs Offline training
    # =========================================================================

    if args.mode == "offline":
        logger.info("Running in OFFLINE mode")

        # Determine cache steps
        cache_steps = (
            args.cache_steps if args.cache_steps is not None else args.max_steps
        )

        # Create cache metadata for validation
        cache_metadata = CacheMetadata(
            model_name=args.model_name,
            dataset_name=args.dataset_name,
            dataset_config=args.dataset_config_name,
            sample_start=0,
            sample_end=cache_steps,
            layer_indices=model_wrapper.head_layer_indices,
            hidden_size=model_wrapper.config.hidden_size,
            max_length=args.max_length,
            config_hash=compute_config_hash(
                args.model_name, model_wrapper.head_layer_indices
            ),
        )

        # Check if cache exists and is valid
        cache = HiddenStateCache(args.cache_dir)
        is_valid, reason = cache.is_valid_for(cache_metadata)

        if is_valid:
            logger.info(f"Found valid cache at {args.cache_dir}")
        else:
            logger.info(f"Cache invalid or missing: {reason}")
            logger.info(f"Extracting {cache_steps} batches of hidden states...")

            # Prepare for extraction (need model on device)
            model_wrapper, train_dataloader = accelerator.prepare(
                model_wrapper, train_dataloader
            )

            # Extract hidden states
            extract_hidden_states(
                model_wrapper=model_wrapper,
                dataloader=train_dataloader,
                cache_dir=args.cache_dir,
                num_steps=cache_steps,
                layer_indices=model_wrapper.head_layer_indices,
                metadata=cache_metadata,
                accelerator=accelerator,
            )

            logger.info("Cache extraction complete!")

        # Phase 2: Train from cache
        if args.max_steps > 0:
            logger.info(f"Training heads from cache for {args.max_steps} steps...")

            # Train from cached hidden states
            train_from_cache(
                aux_heads=model_wrapper.aux_heads,
                cache_dir=args.cache_dir,
                layer_indices=model_wrapper.head_layer_indices,
                max_steps=args.max_steps,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                accelerator=accelerator,
                args=args,
                eval_dataloader=eval_dataloader,
                model_wrapper=model_wrapper,
            )
        else:
            logger.info("max_steps=0, skipping training (extraction only)")

        # Save final model
        accelerator.end_training()
        if accelerator.is_local_main_process:
            logger.info("Saving auxiliary heads...")
            unwrapped_model = accelerator.unwrap_model(model_wrapper)
            torch.save(
                unwrapped_model.aux_heads.state_dict(),
                os.path.join(args.output_dir, "aux_heads.pt"),
            )
            _save_config(args, unwrapped_model)

        return  # Exit after offline training

    # =========================================================================
    # ONLINE MODE (existing training loop)
    # =========================================================================
    logger.info("Running in ONLINE mode")

    # Prepare with Accelerator
    # Note: model_wrapper includes the main model. Accelerate might try to handle it.
    # Since main model is frozen, it should be fine.
    # Prepare eval_dataloader only if it exists
    if eval_dataloader is not None:
        model_wrapper, optimizer, train_dataloader, eval_dataloader = (
            accelerator.prepare(
                model_wrapper,
                optimizer,
                train_dataloader,
                eval_dataloader,
            )
        )
    else:
        model_wrapper, optimizer, train_dataloader = accelerator.prepare(
            model_wrapper, optimizer, train_dataloader
        )

    model_wrapper.train()

    progress_bar = tqdm(
        range(args.max_steps), disable=not accelerator.is_local_main_process
    )
    completed_steps = 0

    data_iter = iter(train_dataloader)

    # Metrics accumulation
    running_metrics = {}
    accumulation_counter = 0
    best_eval_loss = float("inf")

    while completed_steps < args.max_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_dataloader)
            batch = next(data_iter)

        with accelerator.accumulate(model_wrapper):
            # Forward
            # batch contains input_ids, attention_mask, etc.
            teacher_logits, head_logits_list = model_wrapper(**batch)

            # Compute Loss and Metrics
            total_loss = 0.0
            metrics = {}

            for i, head_logits in enumerate(head_logits_list):
                loss = compute_kl_loss(head_logits, teacher_logits)
                acc = compute_accuracy(head_logits, teacher_logits)

                # Add to total loss (sum first, then divide later for mean)
                total_loss += loss

                metrics[f"head_{i}_loss"] = loss.item()
                metrics[f"head_{i}_accuracy"] = acc.item()

            # Mean usage: "instead of a sum, please use the mean of the losses of each head"
            total_loss = total_loss / len(head_logits_list)
            metrics["train_loss"] = total_loss.item()

            # Accumulate metrics
            for k, v in metrics.items():
                running_metrics[k] = running_metrics.get(k, 0.0) + v
            accumulation_counter += 1

            # Update progress bar description with accumulation info
            if accelerator.is_local_main_process:
                progress_bar.set_description(
                    f"Step {completed_steps + 1} (Accum {accumulation_counter}/{args.gradient_accumulation_steps})"
                )

            # Backward
            accelerator.backward(total_loss)
            optimizer.step()
            optimizer.zero_grad()

            # Log to WandB and Console only on sync
            if accelerator.sync_gradients:
                # Average metrics over accumulation steps
                avg_metrics = {
                    k: v / accumulation_counter for k, v in running_metrics.items()
                }

                progress_bar.update(1)
                completed_steps += 1
                lr_scheduler.step()
                progress_bar.set_description(f"Step {completed_steps}")

                if accelerator.is_local_main_process:
                    # Log LR
                    current_lr = lr_scheduler.get_last_lr()[0]
                    avg_metrics["lr"] = current_lr

                    accelerator.log(avg_metrics, step=completed_steps)
                    if completed_steps % 50 == 0:
                        logger.info(
                            f"Step {completed_steps} - Train Loss: {avg_metrics['train_loss']:.4f}"
                        )
                        # Optional: Log detailed head info occasionally
                        head_0_loss = avg_metrics.get("head_0_loss", 0)
                        logger.info(
                            f"Step {completed_steps} - Head 0 Loss: {head_0_loss:.4f}"
                        )

                # Update progress bar with metrics
                if accelerator.is_local_main_process:
                    current_lr = lr_scheduler.get_last_lr()[0]
                    progress_bar.set_postfix(
                        loss=f"{avg_metrics['train_loss']:.4f}", lr=f"{current_lr:.2e}"
                    )

                # Reset accumulation
                # Reset accumulation
                running_metrics = {}
                accumulation_counter = 0

                # Evaluation
                if completed_steps % args.eval_steps == 0:
                    if eval_dataloader is not None:
                        eval_metrics = evaluate(
                            model_wrapper,
                            eval_dataloader,
                            accelerator,
                            num_steps=args.eval_max_steps,
                        )
                        if accelerator.is_local_main_process:
                            accelerator.log(eval_metrics, step=completed_steps)
                            logger.info(
                                f"Step {completed_steps} - Eval Loss: {eval_metrics['eval_loss']:.4f} - Eval Acc: {eval_metrics['eval_accuracy']:.4f}"
                            )

                            # Save Best
                            if eval_metrics["eval_loss"] < best_eval_loss:
                                best_eval_loss = eval_metrics["eval_loss"]
                                logger.info(f"New best eval loss! Saving checkpoint...")
                                unwrapped_model = accelerator.unwrap_model(
                                    model_wrapper
                                )
                                torch.save(
                                    unwrapped_model.aux_heads.state_dict(),
                                    os.path.join(args.output_dir, "best_aux_heads.pt"),
                                )
                    else:
                        logger.info("No eval dataset available, skipping evaluation.")

                # Checkpointing
                if completed_steps % args.save_steps == 0:
                    if accelerator.is_local_main_process:
                        logger.info(f"Saving checkpoint at step {completed_steps}...")
                        unwrapped_model = accelerator.unwrap_model(model_wrapper)
                        # Save entire aux_heads state
                        torch.save(
                            unwrapped_model.aux_heads.state_dict(),
                            os.path.join(
                                args.output_dir, f"aux_heads_step_{completed_steps}.pt"
                            ),
                        )

    # End logging
    accelerator.end_training()

    # Save heads and config
    if accelerator.is_local_main_process:
        logger.info("Saving auxiliary heads...")
        # Since model_wrapper is wrapped by Activate, unwrap it
        unwrapped_model = accelerator.unwrap_model(model_wrapper)
        torch.save(
            unwrapped_model.aux_heads.state_dict(),
            os.path.join(args.output_dir, "aux_heads.pt"),
        )
        logger.info(f"Saved to {os.path.join(args.output_dir, 'aux_heads.pt')}")

        # Save model configuration for inference
        import json

        config_data = {
            "model_name": args.model_name,
            "num_heads": args.num_heads,
            "head_layer_indices": unwrapped_model.head_layer_indices,
            "quantization": args.quantization,
            "hidden_size": unwrapped_model.config.hidden_size,
            "vocab_size": unwrapped_model.config.vocab_size,
            "num_hidden_layers": unwrapped_model.config.num_hidden_layers,
            "training_config": {
                "dataset_name": args.dataset_name,
                "dataset_config_name": args.dataset_config_name,
                "batch_size": args.batch_size,
                "gradient_accumulation_steps": args.gradient_accumulation_steps,
                "max_steps": args.max_steps,
                "lr": args.lr,
                "max_length": args.max_length,
            },
        }
        config_path = os.path.join(args.output_dir, "config.json")
        with open(config_path, "w") as f:
            json.dump(config_data, f, indent=2)
        logger.info(f"Saved config to {config_path}")


if __name__ == "__main__":
    main()
