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
        self.quantization = quantization
        self.device_map = device_map
        self.model = None
        self.config = None

        self._load_model()

        self.aux_heads = nn.ModuleList()
        self.head_layer_indices = self._calculate_head_layer_indices()
        self.hook_handles = []
        self.intermediate_activations = {}

        self._initialize_heads()
        self._register_hooks()

    def _build_quantization_config(self) -> Optional[BitsAndBytesConfig]:
        if self.quantization == "4bit":
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16
                if torch.cuda.is_bf16_supported()
                else torch.float32,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        if self.quantization == "8bit":
            return BitsAndBytesConfig(
                load_in_8bit=True,
            )
        return None

    def _load_model(self) -> None:
        if self.model is not None:
            return

        quantization_config = self._build_quantization_config()
        self.config = AutoConfig.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            config=self.config,
            quantization_config=quantization_config,
            torch_dtype=torch.bfloat16
            if torch.cuda.is_bf16_supported()
            else torch.float32,
        )

        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    def load_backbone(self) -> None:
        if self.model is not None:
            return

        logger.info("Loading backbone model...")
        self._load_model()
        self._register_hooks()

    def unload_backbone(self) -> None:
        if self.model is None:
            return

        logger.info("Unloading backbone model to free memory...")
        for h in self.hook_handles:
            h.remove()
        self.hook_handles = []

        del self.model
        self.model = None
        self.intermediate_activations = {}

        import gc

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

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
        Initialize auxiliary heads.
        """
        self.aux_heads = nn.ModuleList()
        vocab_size = self.config.vocab_size
        hidden_size = self.config.hidden_size

        # Try to find norm layer
        norm_layer = None
        if hasattr(self.model, "model") and hasattr(self.model.model, "norm"):
            norm_layer = self.model.model.norm
        elif hasattr(self.model, "transformer") and hasattr(
            self.model.transformer, "ln_f"
        ):
            norm_layer = self.model.transformer.ln_f

        for idx in self.head_layer_indices:
            # Create a new head
            # Note: We should ideally copy the norm layer if it's layer-specific,
            # but usually it's the same for all layers in Llama.
            # We also need to ensure the head is on the correct device/dtype.
            head = AuxiliaryHead(hidden_size, vocab_size, norm_layer=norm_layer)
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
    num_workers: int = 0,
    pin_memory: bool = True,
    prefetch_factor: Optional[int] = None,
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
        train_tokenized_dataset,
        batch_size=batch_size,
        collate_fn=default_data_collator,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=num_workers > 0,
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
            num_workers=num_workers,
            pin_memory=pin_memory,
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
            persistent_workers=num_workers > 0,
        )

    return train_dataloader, eval_dataloader


def compute_kl_loss(student_logits, teacher_logits, top_k=0):
    """
    KL Divergence Loss with optional top-k optimization.

    Args:
        student_logits: (batch, seq_len, vocab_size)
        teacher_logits: (batch, seq_len, vocab_size)
        top_k: If > 0, only compute KL on top-K teacher tokens (much faster for large vocabs)

    Returns:
        KL loss scalar
    """
    temp = 1.0

    if top_k > 0:
        # Top-K optimization: only consider top-K tokens from teacher
        # This significantly reduces computation when vocab_size is large (e.g., 128K)
        k = min(top_k, teacher_logits.size(-1))
        top_k_values, top_k_indices = torch.topk(teacher_logits, k, dim=-1)

        # Softmax only on the top-K values
        teacher_probs_top_k = F.softmax(top_k_values / temp, dim=-1)

        # Gather student logits for only top-K positions
        batch_size, seq_len, vocab_size = student_logits.shape
        batch_indices = torch.arange(batch_size, device=student_logits.device).view(
            -1, 1, 1
        )
        seq_indices = torch.arange(seq_len, device=student_logits.device).view(1, -1, 1)

        student_logits_top_k = student_logits[batch_indices, seq_indices, top_k_indices]

        # Log-softmax on student top-K
        student_log_probs_top_k = F.log_softmax(student_logits_top_k / temp, dim=-1)

        # KLDivLoss: input should be log-probs, target should be probs
        loss = F.kl_div(
            student_log_probs_top_k, teacher_probs_top_k, reduction="batchmean"
        )
    else:
        # Full vocabulary KL loss
        teacher_probs = F.softmax(teacher_logits / temp, dim=-1)
        student_log_probs = F.log_softmax(student_logits / temp, dim=-1)
        loss = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean")

    return loss


def compute_accuracy(student_logits, teacher_logits):
    """
    Compute accuracy: fraction of times student top-1 matches teacher top-1.
    """
    student_preds = torch.argmax(student_logits, dim=-1)
    teacher_preds = torch.argmax(teacher_logits, dim=-1)
    return (student_preds == teacher_preds).float().mean()


def evaluate(model_wrapper, eval_dataloader, accelerator, num_steps=100, top_k_kl=0):
    """
    Run evaluation on the validation dataset.

    Args:
        model_wrapper: Model with auxiliary heads
        eval_dataloader: Evaluation dataloader
        accelerator: Accelerate instance
        num_steps: Number of evaluation steps
        top_k_kl: Top-K for KL loss optimization
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
                loss = compute_kl_loss(head_logits, teacher_logits, top_k=top_k_kl)
                acc = compute_accuracy(head_logits, teacher_logits)

                batch_loss += loss.item()
                batch_acc += acc.item()

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
    global_step_offset=0,
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
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
        persistent_workers=args.num_workers > 0,
    )

    # Prepare with accelerator
    aux_heads, optimizer, cache_dataloader = accelerator.prepare(
        aux_heads, optimizer, cache_dataloader
    )

    # Unwrap aux_heads for iteration and optimization
    unwrapped_aux_heads = accelerator.unwrap_model(aux_heads)

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

                # Parallelize forward passes through all heads
                # Stack all hidden states to process them in parallel (conceptually)
                hidden_states_list = [
                    batch[f"hidden_states_{layer_idx}"].squeeze(0)
                    for layer_idx in layer_indices
                ]

                # Forward through all heads
                head_logits_list = []
                for head_module, hidden_states in zip(
                    unwrapped_aux_heads, hidden_states_list
                ):
                    head_logits_list.append(head_module(hidden_states))

                # Compute loss and metrics for each head
                total_loss = 0.0
                metrics = {}

                for head_idx, head_logits in enumerate(head_logits_list):
                    # Compute KL loss with optional top-k optimization
                    loss = compute_kl_loss(
                        head_logits, teacher_logits, top_k=args.top_k_kl
                    )
                    acc = compute_accuracy(head_logits, teacher_logits)

                    total_loss += loss
                    metrics[f"head_{head_idx}_loss"] = loss.item()
                    metrics[f"head_{head_idx}_accuracy"] = acc.item()

                # Mean loss
                total_loss = total_loss / len(unwrapped_aux_heads)
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
                    current_global_step = global_step_offset + completed_steps
                    lr_scheduler.step()

                    if accelerator.is_local_main_process:
                        current_lr = lr_scheduler.get_last_lr()[0]
                        avg_metrics["lr"] = current_lr
                        accelerator.log(avg_metrics, step=current_global_step)

                        if current_global_step % 50 == 0:
                            logger.info(
                                f"Step {current_global_step} - Train Loss: {avg_metrics['train_loss']:.4f}"
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
                        current_global_step % args.eval_steps == 0
                        and eval_dataloader is not None
                        and model_wrapper is not None
                    ):
                        model_wrapper.load_backbone()
                        model_wrapper.to(accelerator.device)
                        eval_metrics = evaluate(
                            model_wrapper,
                            eval_dataloader,
                            accelerator,
                            num_steps=args.eval_max_steps,
                            top_k_kl=args.top_k_kl,
                        )
                        if accelerator.is_local_main_process:
                            accelerator.log(eval_metrics, step=current_global_step)
                            logger.info(
                                f"Step {current_global_step} - Eval Loss: {eval_metrics['eval_loss']:.4f}"
                            )

                            if eval_metrics["eval_loss"] < best_eval_loss:
                                best_eval_loss = eval_metrics["eval_loss"]
                                torch.save(
                                    unwrapped_aux_heads.state_dict(),
                                    os.path.join(args.output_dir, "best_aux_heads.pt"),
                                )
                        model_wrapper.unload_backbone()

                    # Checkpointing
                    if current_global_step % args.save_steps == 0:
                        if accelerator.is_local_main_process:
                            torch.save(
                                unwrapped_aux_heads.state_dict(),
                                os.path.join(
                                    args.output_dir,
                                    f"aux_heads_step_{current_global_step}.pt",
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
    # DataLoader optimization arguments
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Number of DataLoader workers. Use 0 for single-process loading (safer for large batches). Default: 1",
    )
    parser.add_argument(
        "--pin_memory",
        type=lambda x: x.lower() in ["true", "1", "yes"],
        default=True,
        help="Pin memory for faster GPU transfer. Default: True",
    )
    parser.add_argument(
        "--prefetch_factor",
        type=int,
        default=2,
        help="Number of batches to prefetch per worker. Default: 2",
    )
    parser.add_argument(
        "--top_k_kl",
        type=int,
        default=0,
        help="Top-K for KL loss computation. If > 0, only compute KL on top-K teacher tokens. 0 = full vocab. Default: 0 (full vocab)",
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
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=1,
        help="Number of epochs to train (for offline mode)",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=None,
        help="Number of steps to process in a single chunk (extract -> train loop) to save disk space. If None, processes all at once.",
    )
    parser.add_argument(
        "--cache_batch_size",
        type=int,
        default=None,
        help="Batch size for hidden state extraction in offline mode (default: same as batch_size)",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=None,
        help="Batch size for training heads from cache in offline mode (default: same as batch_size)",
    )
    args = parser.parse_args()

    # Set default batch sizes for offline mode
    if args.cache_batch_size is None:
        args.cache_batch_size = args.batch_size
    if args.train_batch_size is None:
        args.train_batch_size = args.batch_size

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

    # Save config immediately so we have it for evaluation even if training crashes/runs
    if accelerator.is_local_main_process:
        _save_config(args, model_wrapper)

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
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        prefetch_factor=args.prefetch_factor,
    )

    # =========================================================================
    # MODE BRANCHING: Online vs Offline training
    # =========================================================================

    if args.mode == "offline":
        logger.info("Running in OFFLINE mode")

        # Determine cache steps
        total_cache_steps = (
            args.cache_steps if args.cache_steps is not None else args.max_steps
        )

        # Save these before accelerator.prepare() wraps the model in DDP
        head_layer_indices = model_wrapper.head_layer_indices
        model_config = model_wrapper.config
        aux_heads_ref = model_wrapper.aux_heads

        # Check if we are using batched/chunked execution
        if args.chunk_size is not None and args.chunk_size > 0:
            logger.info(f"Using CHUNKED execution with chunk_size={args.chunk_size}")

            # Prepare dataloader
            train_dataloader = accelerator.prepare(train_dataloader)
            if eval_dataloader is not None:
                eval_dataloader = accelerator.prepare(eval_dataloader)

            # Persistent iterator for extraction
            extraction_iter = iter(train_dataloader)
            total_trained_steps = 0

            while total_trained_steps < args.max_steps:
                logger.info(
                    f"--- Chunk Loop: {total_trained_steps}/{args.max_steps} steps ---"
                )
                current_chunk_train_steps = min(
                    args.chunk_size, args.max_steps - total_trained_steps
                )

                # Extraction batches needed
                samples_needed = (
                    current_chunk_train_steps
                    * args.train_batch_size
                    * args.gradient_accumulation_steps
                    * accelerator.num_processes
                )
                extraction_batches_needed = (
                    samples_needed + args.cache_batch_size - 1
                ) // args.cache_batch_size

                # Cache metadata
                cache_metadata = CacheMetadata(
                    model_name=args.model_name,
                    dataset_name=args.dataset_name,
                    dataset_config=args.dataset_config_name,
                    sample_start=total_trained_steps
                    * args.train_batch_size
                    * args.gradient_accumulation_steps
                    * accelerator.num_processes,
                    sample_end=None,
                    layer_indices=head_layer_indices,
                    hidden_size=model_config.hidden_size,
                    max_length=args.max_length,
                    batch_size=args.cache_batch_size,
                    config_hash=compute_config_hash(
                        args.model_name, head_layer_indices
                    ),
                )

                # Extract
                model_wrapper.load_backbone()
                if model_wrapper.model is not None:
                    model_wrapper.model.to(accelerator.device)
                extract_hidden_states(
                    model_wrapper=model_wrapper,
                    dataloader=train_dataloader,
                    cache_dir=args.cache_dir,
                    num_steps=extraction_batches_needed,
                    layer_indices=head_layer_indices,
                    metadata=cache_metadata,
                    accelerator=accelerator,
                    data_iter=extraction_iter,
                )
                model_wrapper.unload_backbone()

                # Train
                train_from_cache(
                    aux_heads=aux_heads_ref,
                    cache_dir=args.cache_dir,
                    layer_indices=head_layer_indices,
                    max_steps=current_chunk_train_steps,
                    optimizer=optimizer,
                    lr_scheduler=lr_scheduler,
                    accelerator=accelerator,
                    args=args,
                    eval_dataloader=None,  # Disable internal eval
                    model_wrapper=model_wrapper,
                    global_step_offset=total_trained_steps,
                )
                total_trained_steps += current_chunk_train_steps

                # Optional: Eval (requires reloading backbone)
                if eval_dataloader is not None and (
                    total_trained_steps % args.eval_steps < args.chunk_size
                ):
                    logger.info("Reloading backbone for evaluation...")
                    model_wrapper.load_backbone()
                    model_wrapper.to(accelerator.device)
                    eval_metrics = evaluate(
                        model_wrapper,
                        eval_dataloader,
                        accelerator,
                        num_steps=args.eval_max_steps,
                        top_k_kl=args.top_k_kl,
                    )
                    if accelerator.is_local_main_process:
                        accelerator.log(eval_metrics, step=total_trained_steps)
                        logger.info(f"Chunk Eval Loss: {eval_metrics['eval_loss']:.4f}")
                    model_wrapper.unload_backbone()
        else:
            # ORIGINAL NON-CHUNKED LOGIC
            cache_metadata = CacheMetadata(
                model_name=args.model_name,
                dataset_name=args.dataset_name,
                dataset_config=args.dataset_config_name,
                sample_start=0,
                sample_end=total_cache_steps,
                layer_indices=head_layer_indices,
                hidden_size=model_config.hidden_size,
                max_length=args.max_length,
                batch_size=args.cache_batch_size,
                config_hash=compute_config_hash(args.model_name, head_layer_indices),
            )

            # Check if cache exists and is valid
            cache = HiddenStateCache(args.cache_dir)
            is_valid, reason = cache.is_valid_for(cache_metadata)

            if not is_valid:
                logger.info(f"Cache invalid or missing: {reason}")
                model_wrapper.load_backbone()
                if model_wrapper.model is not None:
                    model_wrapper.model.to(accelerator.device)
                extract_hidden_states(
                    model_wrapper=model_wrapper,
                    dataloader=train_dataloader,
                    cache_dir=args.cache_dir,
                    num_steps=total_cache_steps,
                    layer_indices=head_layer_indices,
                    metadata=cache_metadata,
                    accelerator=accelerator,
                )
                model_wrapper.unload_backbone()

            if args.max_steps > 0:
                train_from_cache(
                    aux_heads=aux_heads_ref,
                    cache_dir=args.cache_dir,
                    layer_indices=head_layer_indices,
                    max_steps=args.max_steps,
                    optimizer=optimizer,
                    lr_scheduler=lr_scheduler,
                    accelerator=accelerator,
                    args=args,
                    eval_dataloader=eval_dataloader,
                    model_wrapper=model_wrapper,
                )

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
                loss = compute_kl_loss(head_logits, teacher_logits, top_k=args.top_k_kl)
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
                            top_k_kl=args.top_k_kl,
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
