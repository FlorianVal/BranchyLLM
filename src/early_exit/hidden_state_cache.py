"""
Hidden state caching for offline training mode.

Caches intermediate hidden states from the backbone model to disk,
enabling efficient re-training of auxiliary heads without re-running
the backbone.
"""

import hashlib
import json
import logging
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


@dataclass
class CacheMetadata:
    """Metadata for cache validation.

    Used to verify that cached data matches the current configuration.
    If any field differs, the cache is invalid and must be regenerated.
    """

    model_name: str
    dataset_name: str
    dataset_config: Optional[str]
    sample_start: int
    sample_end: int  # exclusive
    layer_indices: List[int]
    hidden_size: int
    max_length: int
    # Hash of model config for extra validation
    config_hash: str

    def to_json(self, path: str) -> None:
        """Save metadata to JSON file."""
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def from_json(cls, path: str) -> "CacheMetadata":
        """Load metadata from JSON file."""
        with open(path, "r") as f:
            data = json.load(f)
        return cls(**data)

    def matches(self, other: "CacheMetadata") -> Tuple[bool, str]:
        """Check if this metadata matches another.

        Returns:
            (matches, reason) - True if matches, or (False, explanation)
        """
        if self.model_name != other.model_name:
            return False, f"Model mismatch: {self.model_name} vs {other.model_name}"
        if self.dataset_name != other.dataset_name:
            return (
                False,
                f"Dataset mismatch: {self.dataset_name} vs {other.dataset_name}",
            )
        if self.dataset_config != other.dataset_config:
            return (
                False,
                f"Dataset config mismatch: {self.dataset_config} vs {other.dataset_config}",
            )
        if self.layer_indices != other.layer_indices:
            return (
                False,
                f"Layer indices mismatch: {self.layer_indices} vs {other.layer_indices}",
            )
        if self.hidden_size != other.hidden_size:
            return (
                False,
                f"Hidden size mismatch: {self.hidden_size} vs {other.hidden_size}",
            )
        if self.max_length != other.max_length:
            return (
                False,
                f"Max length mismatch: {self.max_length} vs {other.max_length}",
            )
        return True, "OK"


def compute_config_hash(model_name: str, layer_indices: List[int]) -> str:
    """Compute a hash of the configuration for validation."""
    config_str = f"{model_name}:{sorted(layer_indices)}"
    return hashlib.md5(config_str.encode()).hexdigest()[:16]


class HiddenStateCache:
    """Manager for reading/writing hidden state cache to disk.

    Cache format:
        cache_dir/
        ├── metadata.json       # CacheMetadata
        ├── batch_000000.pt     # {hidden_states, teacher_logits}
        ├── batch_000001.pt
        └── ...

    Each batch file contains:
        - hidden_states: Dict[layer_idx -> Tensor(batch, seq, hidden)]
        - teacher_logits: Tensor(batch, seq, vocab)
    """

    def __init__(self, cache_dir: str):
        self.cache_dir = Path(cache_dir)
        self.metadata_path = self.cache_dir / "metadata.json"

    def exists(self) -> bool:
        """Check if cache directory exists with metadata."""
        return self.metadata_path.exists()

    def get_metadata(self) -> Optional[CacheMetadata]:
        """Load existing metadata if available."""
        if not self.exists():
            return None
        try:
            return CacheMetadata.from_json(str(self.metadata_path))
        except Exception as e:
            logger.warning(f"Failed to load cache metadata: {e}")
            return None

    def is_valid_for(self, required_metadata: CacheMetadata) -> Tuple[bool, str]:
        """Check if existing cache is valid for the required config.

        Returns:
            (valid, reason)
        """
        existing = self.get_metadata()
        if existing is None:
            return False, "No existing cache"

        matches, reason = existing.matches(required_metadata)
        if not matches:
            return False, reason

        # Check if cache covers the required range
        if existing.sample_start > required_metadata.sample_start:
            return (
                False,
                f"Cache starts at {existing.sample_start}, need {required_metadata.sample_start}",
            )
        if existing.sample_end < required_metadata.sample_end:
            return (
                False,
                f"Cache ends at {existing.sample_end}, need {required_metadata.sample_end}",
            )

        return True, "Cache valid"

    def initialize(self, metadata: CacheMetadata) -> None:
        """Initialize cache directory with metadata."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        metadata.to_json(str(self.metadata_path))
        logger.info(f"Initialized cache at {self.cache_dir}")

    def save_batch(
        self,
        batch_idx: int,
        hidden_states: Dict[int, torch.Tensor],
        teacher_logits: torch.Tensor,
    ) -> None:
        """Save a batch of hidden states and teacher logits.

        Args:
            batch_idx: Index of this batch
            hidden_states: Dict mapping layer_idx -> hidden state tensor
            teacher_logits: Teacher model logits
        """
        batch_path = self.cache_dir / f"batch_{batch_idx:06d}.pt"

        # Convert to CPU and save
        data = {
            "hidden_states": {k: v.cpu() for k, v in hidden_states.items()},
            "teacher_logits": teacher_logits.cpu(),
        }
        torch.save(data, batch_path)

    def load_batch(
        self, batch_idx: int
    ) -> Tuple[Dict[int, torch.Tensor], torch.Tensor]:
        """Load a batch from cache.

        Returns:
            (hidden_states, teacher_logits)
        """
        batch_path = self.cache_dir / f"batch_{batch_idx:06d}.pt"
        data = torch.load(batch_path, map_location="cpu")
        return data["hidden_states"], data["teacher_logits"]

    def num_batches(self) -> int:
        """Count number of cached batches."""
        return len(list(self.cache_dir.glob("batch_*.pt")))

    def clear(self) -> None:
        """Remove all cached data."""
        if self.cache_dir.exists():
            import shutil

            shutil.rmtree(self.cache_dir)
            logger.info(f"Cleared cache at {self.cache_dir}")


class CachedHiddenStateDataset(Dataset):
    """PyTorch Dataset that loads hidden states from cache.

    This allows training heads without running the backbone model.
    """

    def __init__(self, cache_dir: str, layer_indices: List[int]):
        """
        Args:
            cache_dir: Path to cache directory
            layer_indices: Which layer indices to load
        """
        self.cache = HiddenStateCache(cache_dir)
        self.layer_indices = layer_indices

        if not self.cache.exists():
            raise ValueError(f"Cache not found at {cache_dir}")

        self.num_batches = self.cache.num_batches()
        if self.num_batches == 0:
            raise ValueError(f"Cache at {cache_dir} is empty")

        logger.info(f"Loaded cache with {self.num_batches} batches from {cache_dir}")

    def __len__(self) -> int:
        return self.num_batches

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Load a cached batch.

        Returns:
            Dict with:
                - hidden_states_{layer_idx}: Tensor for each layer
                - teacher_logits: Teacher model logits
        """
        hidden_states, teacher_logits = self.cache.load_batch(idx)

        result = {"teacher_logits": teacher_logits}
        for layer_idx in self.layer_indices:
            result[f"hidden_states_{layer_idx}"] = hidden_states[layer_idx]

        return result


def extract_hidden_states(
    model_wrapper,
    dataloader,
    cache_dir: str,
    num_steps: int,
    layer_indices: List[int],
    metadata: CacheMetadata,
    accelerator=None,
) -> None:
    """Extract hidden states from backbone and save to cache.

    Args:
        model_wrapper: ModelWithAuxiliaryHeads instance
        dataloader: Data loader (must be deterministic, no shuffle!)
        cache_dir: Where to save cache
        num_steps: Number of batches to extract
        layer_indices: Which layers to cache
        metadata: Cache metadata for validation
        accelerator: Optional accelerator for distributed
    """
    cache = HiddenStateCache(cache_dir)

    # Clear and reinitialize
    cache.clear()
    cache.initialize(metadata)

    model_wrapper.eval()
    data_iter = iter(dataloader)

    is_main = accelerator is None or accelerator.is_local_main_process

    from tqdm import tqdm

    progress = tqdm(
        range(num_steps), desc="Extracting hidden states", disable=not is_main
    )

    for batch_idx in progress:
        try:
            batch = next(data_iter)
        except StopIteration:
            logger.info(f"Dataset exhausted after {batch_idx} batches")
            break

        # Move to device if needed
        if accelerator is not None:
            # Already prepared by accelerator
            pass

        with torch.no_grad():
            # Run backbone to populate intermediate_activations
            input_ids = batch["input_ids"]
            attention_mask = batch.get("attention_mask", None)

            # Clear activations
            model_wrapper.intermediate_activations = {}

            # Forward through backbone
            outputs = model_wrapper.model(
                input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            teacher_logits = outputs.logits

            # Collect hidden states for specified layers
            hidden_states = {}
            for layer_idx in layer_indices:
                if layer_idx in model_wrapper.intermediate_activations:
                    hidden_states[layer_idx] = model_wrapper.intermediate_activations[
                        layer_idx
                    ].detach()
                else:
                    logger.warning(f"Layer {layer_idx} not found in activations")

            # Save to cache
            cache.save_batch(batch_idx, hidden_states, teacher_logits.detach())

    # Update metadata with actual count
    actual_metadata = CacheMetadata(
        model_name=metadata.model_name,
        dataset_name=metadata.dataset_name,
        dataset_config=metadata.dataset_config,
        sample_start=metadata.sample_start,
        sample_end=batch_idx + 1,  # Actual number extracted
        layer_indices=metadata.layer_indices,
        hidden_size=metadata.hidden_size,
        max_length=metadata.max_length,
        config_hash=metadata.config_hash,
    )
    actual_metadata.to_json(str(cache.metadata_path))

    logger.info(f"Extracted {batch_idx + 1} batches to {cache_dir}")
