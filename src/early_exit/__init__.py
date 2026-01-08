# Inference module for early exit heads
from .model_config import ModelConfig, CalibrationResult, load_config, save_config
from .model_loader import (
    load_model_with_heads,
    compute_entropy,
    compute_confidence,
    get_uncertainty_fn,
    EarlyExitModel,
)
from .debug_logger import DebugLogger, compare_logs, print_mismatch_report
from .hidden_state_cache import (
    CacheMetadata,
    HiddenStateCache,
    CachedHiddenStateDataset,
    extract_hidden_states,
    compute_config_hash,
)

__all__ = [
    "ModelConfig",
    "CalibrationResult",
    "load_config",
    "save_config",
    "load_model_with_heads",
    "compute_entropy",
    "compute_confidence",
    "get_uncertainty_fn",
    "EarlyExitModel",
    "DebugLogger",
    "compare_logs",
    "print_mismatch_report",
    "CacheMetadata",
    "HiddenStateCache",
    "CachedHiddenStateDataset",
    "extract_hidden_states",
    "compute_config_hash",
]
