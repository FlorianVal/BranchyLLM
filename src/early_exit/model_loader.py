"""
Model loading utilities for inference with early exit heads.
Reuses AuxiliaryHead from train_heads.py but provides a clean inference interface.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Dict, Callable
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    BitsAndBytesConfig,
)

from .model_config import ModelConfig


# ============================================================================
# Uncertainty Metrics (Modular)
# ============================================================================


def compute_entropy(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Compute entropy of the probability distribution.
    Lower entropy = more confident.

    Args:
        logits: (..., vocab_size) tensor of logits
        dim: dimension to compute entropy over

    Returns:
        entropy: (...) tensor of entropy values
    """
    probs = F.softmax(logits, dim=dim)
    log_probs = F.log_softmax(logits, dim=dim)
    entropy = -torch.sum(probs * log_probs, dim=dim)
    return entropy


def compute_confidence(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Compute max softmax probability (confidence).
    Higher confidence = more confident.

    Note: For threshold comparison, you may want to use 1 - confidence
    to have the same "lower is better" semantics as entropy.

    Args:
        logits: (..., vocab_size) tensor of logits
        dim: dimension to compute confidence over

    Returns:
        confidence: (...) tensor of max probability values
    """
    probs = F.softmax(logits, dim=dim)
    confidence = torch.max(probs, dim=dim).values
    return confidence


def get_uncertainty_fn(metric: str = "entropy") -> Callable:
    """
    Get the uncertainty function based on metric name.

    Args:
        metric: "entropy" or "confidence"

    Returns:
        Function that computes uncertainty from logits
    """
    if metric == "entropy":
        return compute_entropy
    elif metric == "confidence":
        # Return negative confidence so lower = more confident (like entropy)
        return lambda logits, dim=-1: -compute_confidence(logits, dim)
    else:
        raise ValueError(f"Unknown uncertainty metric: {metric}")


# ============================================================================
# Auxiliary Head (copied from train_heads.py for independence)
# ============================================================================


class AuxiliaryHead(nn.Module):
    """
    Auxiliary Head that mimics the architecture of the model's lm_head.
    Consists of a Normalization layer (if applicable) and a Linear layer.
    """

    def __init__(
        self, hidden_size: int, vocab_size: int, norm_layer: Optional[nn.Module] = None
    ):
        super().__init__()
        self.norm = norm_layer if norm_layer is not None else nn.Identity()
        self.linear = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.norm(hidden_states)
        logits = self.linear(hidden_states)
        return logits


# ============================================================================
# Model with Early Exit Heads for Inference
# ============================================================================


class EarlyExitModel(nn.Module):
    """
    Wrapper for inference with early exit heads.
    Supports both early exit inference and full model inference.
    """

    def __init__(
        self,
        model: AutoModelForCausalLM,
        aux_heads: nn.ModuleList,
        head_layer_indices: List[int],
        config: AutoConfig,
    ):
        super().__init__()
        self.model = model
        self.aux_heads = aux_heads
        self.head_layer_indices = head_layer_indices
        self.config = config

        # Hook storage
        self.hook_handles = []
        self.intermediate_activations = {}

        # Register hooks
        self._register_hooks()

    def _hook_fn(self, module, input, output, layer_idx):
        """Capture intermediate activations from decoder layers."""
        if isinstance(output, tuple):
            hidden_states = output[0]
        else:
            hidden_states = output
        self.intermediate_activations[layer_idx] = hidden_states

    def _register_hooks(self):
        """Register forward hooks on the specified layers."""
        for h in self.hook_handles:
            h.remove()
        self.hook_handles = []

        # Find the layers module
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            layers = self.model.model.layers
        elif hasattr(self.model, "transformer") and hasattr(
            self.model.transformer, "h"
        ):
            layers = self.model.transformer.h
        elif hasattr(self.model, "layers"):
            layers = self.model.layers
        else:
            raise ValueError("Unknown model structure")

        for idx in self.head_layer_indices:
            handle = layers[idx].register_forward_hook(
                lambda m, i, o, layer_idx=idx: self._hook_fn(m, i, o, layer_idx)
            )
            self.hook_handles.append(handle)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, List[torch.Tensor], Dict[int, torch.Tensor]]:
        """
        Full forward pass through model and all heads.

        Returns:
            main_logits: (batch, seq_len, vocab_size) from main lm_head
            head_logits: List of (batch, seq_len, vocab_size) from each aux head
            hidden_states: Dict mapping layer_idx -> hidden_states
        """
        self.intermediate_activations = {}

        with torch.no_grad():
            outputs = self.model(
                input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                **kwargs,
            )
            main_logits = outputs.logits

        head_logits = []
        for idx, head in zip(self.head_layer_indices, self.aux_heads):
            hidden = self.intermediate_activations[idx]
            logits = head(hidden)
            head_logits.append(logits)

        return main_logits, head_logits, self.intermediate_activations

    def forward_to_layer(
        self,
        input_ids: torch.Tensor,
        target_layer: int,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[Tuple]]:
        """
        Forward pass up to a specific layer.

        This is used for early exit inference where we want to stop
        computation at an intermediate layer.

        Note: Currently this runs the full forward and just returns
        the intermediate activation. True early exit would require
        modifying the model's forward pass.

        Returns:
            hidden_states: (batch, seq_len, hidden_size) at target layer
            past_key_values: Updated KV cache if using generation
        """
        self.intermediate_activations = {}

        with torch.no_grad():
            outputs = self.model(
                input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                output_hidden_states=True,
                use_cache=True,
                **kwargs,
            )

        hidden_states = self.intermediate_activations.get(target_layer)
        if hidden_states is None:
            # If target layer is not in head_layer_indices, use full hidden states
            hidden_states = outputs.hidden_states[
                target_layer + 1
            ]  # +1 because 0 is embedding

        return hidden_states, outputs.past_key_values

    def get_head_output(
        self, head_idx: int, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        """Get logits from a specific head given hidden states."""
        return self.aux_heads[head_idx](hidden_states)

    def forward_adaptive(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple] = None,
        thresholds: Optional[Dict[int, float]] = None,
        uncertainty_fn: Optional[Callable] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[int, Optional[Tuple], int, float, Dict]:
        """
        Adaptive forward pass with TRUE early exit.

        Only executes layers up to the first confident head, saving computation.
        KV cache grows as we go deeper - if early layers were cached but we need
        to go deeper, we continue from where the cache ends.

        Args:
            input_ids: (batch, seq_len) input token IDs
            attention_mask: Optional attention mask
            past_key_values: KV cache from previous tokens.
                             Can be partial (only up to layer N) or full.
            thresholds: Dict mapping head_idx -> confidence threshold
                       If None or empty, runs full model
            uncertainty_fn: Function to compute uncertainty from logits
            position_ids: Optional position IDs for RoPE

        Returns:
            token_id: Predicted next token ID
            new_past_key_values: Updated KV cache (may be partial or full)
            exit_layer: Layer index where we exited (or num_layers for full)
            uncertainty: Uncertainty value at exit point
            debug_info: Dict with head predictions/uncertainties for debugging
        """
        device = input_ids.device
        batch_size, seq_len = input_ids.shape

        # Default to running full model if no thresholds
        if thresholds is None:
            thresholds = {}
        if uncertainty_fn is None:
            uncertainty_fn = compute_entropy

        # Access model internals (Llama architecture)
        # For non-quantized models
        if hasattr(self.model, "model"):
            embed_tokens = self.model.model.embed_tokens
            layers = self.model.model.layers
            norm = self.model.model.norm
            lm_head = self.model.lm_head
            rotary_emb = getattr(self.model.model, "rotary_emb", None)
        else:
            raise ValueError("Unsupported model architecture for adaptive forward")

        num_layers = len(layers)

        # Determine how many layers are already cached
        cached_layers = 0
        if past_key_values is not None:
            cached_layers = len(past_key_values)

        # Compute position IDs
        if position_ids is None:
            past_length = past_key_values[0][0].shape[2] if past_key_values else 0
            position_ids = torch.arange(
                past_length, past_length + seq_len, dtype=torch.long, device=device
            ).unsqueeze(0)

        # Embedding
        hidden_states = embed_tokens(input_ids)

        # Compute rotary embeddings (RoPE) for Llama
        # This is required by newer transformers versions
        position_embeddings = None
        if rotary_emb is not None:
            # rotary_emb expects hidden_states to determine dtype
            cos, sin = rotary_emb(hidden_states, position_ids)
            position_embeddings = (cos, sin)

        # Prepare attention mask
        if attention_mask is not None:
            # Create causal mask
            # For simplicity, we'll let the layer handle it
            pass

        # Store new KV cache entries
        new_past_key_values = list(past_key_values) if past_key_values else []

        # Track which heads we've checked
        debug_info = {
            "head_predictions": {},
            "head_uncertainties": {},
            "layers_executed": 0,
        }

        # Sort head layer indices for proper ordering
        sorted_heads = sorted(enumerate(self.head_layer_indices), key=lambda x: x[1])

        # Run through layers
        for layer_idx, layer in enumerate(layers):
            # Check if we have cache for this layer
            layer_past = None
            if layer_idx < cached_layers and past_key_values is not None:
                layer_past = past_key_values[layer_idx]

            # Execute this layer
            with torch.no_grad():
                layer_outputs = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=layer_past,
                    use_cache=True,
                    position_embeddings=position_embeddings,  # Pass RoPE embeddings
                )

            hidden_states = layer_outputs[0]
            new_kv = layer_outputs[1] if len(layer_outputs) > 1 else None

            # Update or extend KV cache
            if layer_idx < len(new_past_key_values):
                # Update existing cache entry
                if new_kv is not None:
                    new_past_key_values[layer_idx] = new_kv
            else:
                # Extend cache
                if new_kv is not None:
                    new_past_key_values.append(new_kv)

            debug_info["layers_executed"] = layer_idx + 1

            # Check if this is a head checkpoint layer
            for head_idx, head_layer in sorted_heads:
                if layer_idx == head_layer:
                    # Run the head on current hidden states (last position only)
                    head_hidden = hidden_states[:, -1:, :]
                    head_logits = self.aux_heads[head_idx](head_hidden)

                    # Compute uncertainty
                    uncertainty = uncertainty_fn(head_logits[:, -1, :], dim=-1).item()
                    head_token = torch.argmax(head_logits[0, -1, :]).item()

                    # Store for debugging
                    debug_info["head_predictions"][head_idx] = head_token
                    debug_info["head_uncertainties"][head_idx] = uncertainty

                    # Check against threshold
                    if head_idx in thresholds and uncertainty < thresholds[head_idx]:
                        # Early exit!
                        return (
                            head_token,
                            tuple(new_past_key_values),
                            layer_idx,
                            uncertainty,
                            debug_info,
                        )

        # No early exit - run final norm and lm_head
        hidden_states = norm(hidden_states)
        logits = lm_head(hidden_states[:, -1:, :])
        token_id = torch.argmax(logits[0, -1, :]).item()

        # Store full model prediction for debugging
        debug_info["full_model_token"] = token_id

        return (
            token_id,
            tuple(new_past_key_values),
            num_layers,
            0.0,  # No uncertainty for full model
            debug_info,
        )

    def forward_full_with_cache(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[int, Tuple, Dict]:
        """
        Full forward pass with KV cache support and debug info.
        Used as baseline for comparison with adaptive.

        Returns:
            token_id: Predicted next token
            past_key_values: Updated KV cache
            debug_info: Dict with all head predictions for debugging
        """
        device = input_ids.device
        seq_len = input_ids.shape[1]

        # Compute position IDs
        if position_ids is None:
            past_length = past_key_values[0][0].shape[2] if past_key_values else 0
            position_ids = torch.arange(
                past_length, past_length + seq_len, dtype=torch.long, device=device
            ).unsqueeze(0)

        # Full forward pass
        self.intermediate_activations = {}

        with torch.no_grad():
            outputs = self.model(
                input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=True,
                output_hidden_states=True,
            )

        main_logits = outputs.logits
        token_id = torch.argmax(main_logits[0, -1, :]).item()

        # Get all head predictions for debugging
        debug_info = {
            "head_predictions": {},
            "head_uncertainties": {},
            "full_model_token": token_id,
            "layers_executed": self.config.num_hidden_layers,
        }

        for head_idx, layer_idx in enumerate(self.head_layer_indices):
            hidden = self.intermediate_activations.get(layer_idx)
            if hidden is not None:
                head_logits = self.aux_heads[head_idx](hidden[:, -1:, :])
                head_token = torch.argmax(head_logits[0, -1, :]).item()
                head_unc = compute_entropy(head_logits[:, -1, :], dim=-1).item()
                debug_info["head_predictions"][head_idx] = head_token
                debug_info["head_uncertainties"][head_idx] = head_unc

        return token_id, outputs.past_key_values, debug_info

    @property
    def device(self):
        return next(self.model.parameters()).device

    @property
    def dtype(self):
        return next(self.model.parameters()).dtype


def load_model_with_heads(
    config_path: str,
    checkpoint_path: Optional[str] = None,
    device: str = "auto",
    match_training_quantization: bool = True,
) -> Tuple[EarlyExitModel, AutoTokenizer, ModelConfig]:
    """
    Load a model with trained early exit heads.

    Args:
        config_path: Path to the model config JSON
        checkpoint_path: Path to aux_heads.pt (if None, inferred from config_path dir)
        device: Device to load model on ("auto", "cuda", "cpu")
        match_training_quantization: If True, use same quantization as training

    Returns:
        model: EarlyExitModel ready for inference
        tokenizer: Tokenizer for the model
        config: ModelConfig with metadata
    """
    import os
    from pathlib import Path
    import copy

    # Load config
    model_config = ModelConfig.from_json(config_path)

    # Infer checkpoint path if not provided
    if checkpoint_path is None:
        config_dir = Path(config_path).parent
        checkpoint_path = str(config_dir / "aux_heads.pt")

    # Determine quantization
    quantization = model_config.quantization if match_training_quantization else "none"

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
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)

    # Device map
    device_map = device if device != "auto" else "auto"

    # Load base model
    hf_config = AutoConfig.from_pretrained(model_config.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_config.model_name,
        config=hf_config,
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32,
        device_map=device_map,
    )
    model.eval()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Create auxiliary heads
    aux_heads = nn.ModuleList()

    for _ in range(model_config.num_heads):
        # Get norm layer architecture from model
        if hasattr(model, "model") and hasattr(model.model, "norm"):
            norm_layer = copy.deepcopy(model.model.norm)
        elif hasattr(model, "transformer") and hasattr(model.transformer, "ln_f"):
            norm_layer = copy.deepcopy(model.transformer.ln_f)
        else:
            norm_layer = None

        head = AuxiliaryHead(
            model_config.hidden_size,
            model_config.vocab_size,
            norm_layer,
        )
        aux_heads.append(head)

    # Load trained weights
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    aux_heads.load_state_dict(state_dict)

    # Move heads to model device and dtype
    model_device = next(model.parameters()).device
    model_dtype = next(model.parameters()).dtype
    aux_heads = aux_heads.to(device=model_device, dtype=model_dtype)
    aux_heads.eval()

    # Create wrapper
    early_exit_model = EarlyExitModel(
        model=model,
        aux_heads=aux_heads,
        head_layer_indices=model_config.head_layer_indices,
        config=hf_config,
    )

    return early_exit_model, tokenizer, model_config
