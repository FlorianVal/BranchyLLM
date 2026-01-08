"""
Debug logging for early exit inference.
Logs token-by-token decisions for debugging mismatches between methods.
"""

import json
import logging
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class TokenDecision:
    """Record of a single token generation decision."""

    position: int
    method: str  # "full", "head_0", "head_1", "head_2", "adaptive", etc.
    token_id: int
    token_text: str
    exit_layer: int
    uncertainty: float
    threshold: float
    all_head_predictions: Optional[Dict[str, int]] = None  # head_idx -> token_id
    all_head_uncertainties: Optional[Dict[str, float]] = None  # head_idx -> uncertainty
    layers_executed: int = 0
    extra_info: Optional[str] = None


class DebugLogger:
    """
    Logs detailed token-by-token decisions for debugging mismatches.

    Usage:
        debug_log = DebugLogger("./debug_logs/run_001.json")
        debug_log.log_token(...)
        debug_log.save()
    """

    def __init__(self, log_path: str, method_name: str = "unknown"):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.method_name = method_name
        self.entries: List[TokenDecision] = []
        self.metadata: Dict = {}

    def set_metadata(self, **kwargs):
        """Set metadata for the run (model name, config, etc.)."""
        self.metadata.update(kwargs)

    def log_token(
        self,
        position: int,
        method: str,
        token_id: int,
        token_text: str,
        exit_layer: int,
        uncertainty: float = 0.0,
        threshold: float = 0.0,
        all_head_predictions: Optional[Dict[str, int]] = None,
        all_head_uncertainties: Optional[Dict[str, float]] = None,
        layers_executed: int = 0,
        extra_info: Optional[str] = None,
    ):
        """Log a single token decision."""
        entry = TokenDecision(
            position=position,
            method=method,
            token_id=token_id,
            token_text=token_text,
            exit_layer=exit_layer,
            uncertainty=uncertainty,
            threshold=threshold,
            all_head_predictions=all_head_predictions,
            all_head_uncertainties=all_head_uncertainties,
            layers_executed=layers_executed,
            extra_info=extra_info,
        )
        self.entries.append(entry)

        # Also log to console at debug level
        logger.debug(
            f"[{method}] pos={position} token={token_id} ({token_text!r}) "
            f"exit_layer={exit_layer} unc={uncertainty:.4f} thr={threshold:.4f}"
        )

    def save(self):
        """Save all entries to JSON file."""
        output = {
            "method": self.method_name,
            "metadata": self.metadata,
            "num_tokens": len(self.entries),
            "tokens": [asdict(e) for e in self.entries],
        }

        with open(self.log_path, "w") as f:
            json.dump(output, f, indent=2)

        logger.info(f"Saved debug log to {self.log_path}")

    def summary(self) -> Dict:
        """Get summary statistics from the log."""
        if not self.entries:
            return {}

        exit_layer_sum = sum(e.exit_layer for e in self.entries)
        layers_executed_sum = sum(e.layers_executed for e in self.entries)

        # Count exits by layer
        exit_counts = {}
        for e in self.entries:
            layer = str(e.exit_layer)
            exit_counts[layer] = exit_counts.get(layer, 0) + 1

        return {
            "num_tokens": len(self.entries),
            "avg_exit_layer": exit_layer_sum / len(self.entries),
            "avg_layers_executed": layers_executed_sum / len(self.entries)
            if layers_executed_sum > 0
            else 0,
            "exit_distribution": exit_counts,
        }


def compare_logs(log1_path: str, log2_path: str) -> Dict:
    """
    Compare two debug logs to find mismatches.

    Returns:
        Dict with comparison results including list of mismatches.
    """
    with open(log1_path) as f:
        log1 = json.load(f)
    with open(log2_path) as f:
        log2 = json.load(f)

    tokens1 = log1["tokens"]
    tokens2 = log2["tokens"]

    mismatches = []
    min_len = min(len(tokens1), len(tokens2))

    for i in range(min_len):
        t1, t2 = tokens1[i], tokens2[i]
        if t1["token_id"] != t2["token_id"]:
            mismatches.append(
                {
                    "position": i,
                    "log1_method": log1["method"],
                    "log1_token": t1["token_id"],
                    "log1_text": t1["token_text"],
                    "log1_exit_layer": t1["exit_layer"],
                    "log1_uncertainty": t1["uncertainty"],
                    "log2_method": log2["method"],
                    "log2_token": t2["token_id"],
                    "log2_text": t2["token_text"],
                    "log2_exit_layer": t2["exit_layer"],
                    "log2_uncertainty": t2["uncertainty"],
                }
            )

    # Check for length differences
    if len(tokens1) != len(tokens2):
        mismatches.append(
            {
                "type": "length_mismatch",
                "log1_length": len(tokens1),
                "log2_length": len(tokens2),
            }
        )

    result = {
        "log1": log1_path,
        "log2": log2_path,
        "log1_method": log1["method"],
        "log2_method": log2["method"],
        "total_tokens_compared": min_len,
        "num_mismatches": len([m for m in mismatches if m.get("position") is not None]),
        "match_rate": 1.0
        - len([m for m in mismatches if m.get("position") is not None])
        / max(min_len, 1),
        "mismatches": mismatches,
    }

    return result


def print_mismatch_report(comparison: Dict):
    """Print a human-readable mismatch report."""
    print(f"\n{'=' * 60}")
    print(
        f"MISMATCH REPORT: {comparison['log1_method']} vs {comparison['log2_method']}"
    )
    print(f"{'=' * 60}")
    print(f"Tokens compared: {comparison['total_tokens_compared']}")
    print(f"Mismatches: {comparison['num_mismatches']}")
    print(f"Match rate: {comparison['match_rate']:.2%}")

    if comparison["mismatches"]:
        print(f"\n{'-' * 60}")
        print("First 10 mismatches:")
        print(f"{'-' * 60}")

        for m in comparison["mismatches"][:10]:
            if m.get("type") == "length_mismatch":
                print(f"  LENGTH MISMATCH: {m['log1_length']} vs {m['log2_length']}")
            else:
                print(
                    f"  pos={m['position']}: "
                    f"{m['log1_method']}:{m['log1_token']}({m['log1_text']!r}) "
                    f"vs {m['log2_method']}:{m['log2_token']}({m['log2_text']!r})"
                )
