"""
Model configuration and calibration result dataclasses.
"""
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class ModelConfig:
    """Configuration for a trained early exit model."""
    model_name: str
    num_heads: int
    head_layer_indices: List[int]
    quantization: str  # "none", "4bit", "8bit"
    hidden_size: int
    vocab_size: int
    num_hidden_layers: int
    training_config: Optional[Dict] = None
    
    @classmethod
    def from_json(cls, path: str) -> "ModelConfig":
        """Load config from JSON file."""
        with open(path, "r") as f:
            data = json.load(f)
        return cls(
            model_name=data["model_name"],
            num_heads=data["num_heads"],
            head_layer_indices=data["head_layer_indices"],
            quantization=data["quantization"],
            hidden_size=data["hidden_size"],
            vocab_size=data["vocab_size"],
            num_hidden_layers=data["num_hidden_layers"],
            training_config=data.get("training_config"),
        )
    
    def to_json(self, path: str) -> None:
        """Save config to JSON file."""
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)


@dataclass 
class CalibrationResult:
    """
    Calibration results containing thresholds per head per accuracy level.
    
    thresholds: Dict[str, Dict[str, float]]
        Maps accuracy_level -> head_index -> entropy_threshold
        e.g., {"0.95": {"0": 0.5, "1": 0.3, "2": 0.2}}
    
    statistics: Dict[str, Dict[str, Dict]]
        Per-head statistics from calibration
        e.g., {"0": {"mean_entropy": 1.2, "accuracy": 0.85, ...}}
    """
    model_config_path: str
    calibration_dataset: str
    calibration_samples: int
    uncertainty_metric: str  # "entropy" or "confidence"
    accuracy_levels: List[float]
    thresholds: Dict[str, Dict[str, float]] = field(default_factory=dict)
    statistics: Dict[str, Dict] = field(default_factory=dict)
    
    @classmethod
    def from_json(cls, path: str) -> "CalibrationResult":
        """Load calibration from JSON file."""
        with open(path, "r") as f:
            data = json.load(f)
        return cls(**data)
    
    def to_json(self, path: str) -> None:
        """Save calibration to JSON file."""
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)
    
    def get_threshold(self, accuracy_level: float, head_idx: int) -> float:
        """Get threshold for a specific accuracy level and head."""
        level_key = f"{accuracy_level:.2f}"
        head_key = str(head_idx)
        return self.thresholds[level_key][head_key]


def load_config(path: str) -> ModelConfig:
    """Convenience function to load model config."""
    return ModelConfig.from_json(path)


def save_config(config: ModelConfig, path: str) -> None:
    """Convenience function to save model config."""
    config.to_json(path)
