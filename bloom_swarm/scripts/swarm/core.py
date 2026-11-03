#!/usr/bin/env python3
"""
Core data loading and math primitives for swarm aggregation.

Data representation:
  weeks: sorted list of week strings (length T)
  agents: sorted list of agent names (length K)
  P: T by K float array of probabilities, NaN where missing or invalid
  y: length T array of 0 or 1
  meta: per week, per agent dict of reason and evidence
"""

import json
from pathlib import Path
from typing import Tuple, Dict, List, Optional, Any

import numpy as np

# Load config
_CONFIG_PATH = Path(__file__).parent / "config.json"
with open(_CONFIG_PATH) as f:
    CONFIG = json.load(f)

CLIP_LO, CLIP_HI = CONFIG["clip"]
SEED = CONFIG["seed"]


def set_seed(seed: int = SEED) -> None:
    """Set numpy random seed for reproducibility."""
    np.random.seed(seed)


def clip(p: np.ndarray) -> np.ndarray:
    """Clip probabilities to configured bounds, NaN-aware."""
    return np.clip(p, CLIP_LO, CLIP_HI)


def logit(p: np.ndarray) -> np.ndarray:
    """Logit transform: log(p / (1 - p)), NaN-aware."""
    p_clipped = clip(p)
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.log(p_clipped / (1 - p_clipped))
    return result


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Sigmoid transform: 1 / (1 + exp(-x)), handles large values."""
    # Clip to avoid overflow
    x_safe = np.clip(x, -500, 500)
    return 1.0 / (1.0 + np.exp(-x_safe))


def brier(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Brier score: (p - y)^2, NaN-aware."""
    return (p - y) ** 2


def log_loss(p: np.ndarray, y: np.ndarray, eps: float = 1e-15) -> np.ndarray:
    """Log loss: -y*log(p) - (1-y)*log(1-p), NaN-aware."""
    p_clipped = np.clip(p, eps, 1 - eps)
    return -(y * np.log(p_clipped) + (1 - y) * np.log(1 - p_clipped))


def nanmean(arr: np.ndarray, axis: Optional[int] = None) -> np.ndarray:
    """Mean ignoring NaN values."""
    return np.nanmean(arr, axis=axis)


def nansum(arr: np.ndarray, axis: Optional[int] = None) -> np.ndarray:
    """Sum ignoring NaN values."""
    return np.nansum(arr, axis=axis)


def load_data(
    weeks_dir: Path = Path("data/weeks"),
    outputs_dir: Path = Path("outputs"),
) -> Tuple[List[str], List[str], np.ndarray, np.ndarray, Dict[str, Dict[str, Dict[str, Any]]]]:
    """
    Load all data into the core representation.

    Returns:
        weeks: sorted list of week strings (length T)
        agents: sorted list of agent names (length K)
        P: T by K float array of probabilities, NaN where missing or invalid
        y: length T array of 0 or 1
        meta: {week: {agent: {"reason": str, "evidence": list}}}
    """
    # Discover all weeks
    week_files = sorted(weeks_dir.glob("*.json"))
    weeks = []
    y_list = []
    week_data = {}

    for wf in week_files:
        with open(wf) as f:
            data = json.load(f)
        week_str = data["week"]
        weeks.append(week_str)
        y_list.append(data["actual"])
        week_data[week_str] = data

    y = np.array(y_list, dtype=float)

    # Discover all agents from output files
    agent_set = set()
    output_files = sorted(outputs_dir.glob("*.json"))
    outputs = {}

    for of in output_files:
        parts = of.stem.rsplit("_", 1)
        if len(parts) != 2:
            continue
        week_str, agent_name = parts
        agent_set.add(agent_name)

        with open(of) as f:
            data = json.load(f)

        if week_str not in outputs:
            outputs[week_str] = {}
        outputs[week_str][agent_name] = data

    agents = sorted(agent_set)
    T = len(weeks)
    K = len(agents)

    # Build P matrix and meta dict
    P = np.full((T, K), np.nan, dtype=float)
    meta = {}

    for t, week_str in enumerate(weeks):
        meta[week_str] = {}
        if week_str not in outputs:
            continue

        for k, agent_name in enumerate(agents):
            if agent_name not in outputs[week_str]:
                continue

            agent_data = outputs[week_str][agent_name]
            p_val = agent_data.get("p")

            # Validate p
            if p_val is None or not isinstance(p_val, (int, float)):
                continue
            if np.isnan(p_val) or np.isinf(p_val):
                continue
            if p_val < 0 or p_val > 1:
                continue

            # Clip and store
            P[t, k] = clip(np.array([p_val]))[0]

            meta[week_str][agent_name] = {
                "reason": agent_data.get("reason", ""),
                "evidence": agent_data.get("evidence", []),
            }

    return weeks, agents, P, y, meta


def get_agent_index(agents: List[str], name: str) -> int:
    """Get index of agent by name."""
    return agents.index(name)


def get_week_index(weeks: List[str], week_str: str) -> int:
    """Get index of week by string."""
    return weeks.index(week_str)


def valid_mask(P: np.ndarray) -> np.ndarray:
    """Return boolean mask of valid (non-NaN) entries."""
    return ~np.isnan(P)


def get_valid_agents_for_week(P: np.ndarray, t: int) -> np.ndarray:
    """Return indices of agents with valid predictions for week t."""
    return np.where(~np.isnan(P[t, :]))[0]


def compute_brier_per_agent(P: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Compute mean Brier score per agent across all weeks where they have data.

    Returns: length K array of Brier scores
    """
    T, K = P.shape
    scores = np.zeros(K)

    for k in range(K):
        valid = ~np.isnan(P[:, k])
        if valid.sum() > 0:
            scores[k] = np.mean(brier(P[valid, k], y[valid]))
        else:
            scores[k] = 0.5  # Default for missing

    return scores


def compute_weights_from_brier(brier_scores: np.ndarray, valid_agents: np.ndarray) -> np.ndarray:
    """
    Convert Brier scores to normalized weights: w_i = (1/brier_i) / sum(1/brier_j).

    Args:
        brier_scores: length K array of Brier scores
        valid_agents: indices of valid agents for this prediction

    Returns: length K array of weights (0 for invalid agents)
    """
    K = len(brier_scores)
    weights = np.zeros(K)

    if len(valid_agents) == 0:
        return weights

    # Compute raw weights for valid agents
    raw = np.zeros(len(valid_agents))
    for i, k in enumerate(valid_agents):
        # Add small epsilon to avoid division by zero
        raw[i] = 1.0 / (brier_scores[k] + 1e-6)

    # Normalize
    raw_sum = raw.sum()
    if raw_sum > 0:
        raw = raw / raw_sum
    else:
        raw = np.ones(len(valid_agents)) / len(valid_agents)

    # Assign back
    for i, k in enumerate(valid_agents):
        weights[k] = raw[i]

    return weights


def pool_logodds(P_row: np.ndarray, weights: np.ndarray) -> float:
    """
    Pool probabilities in log-odds space using given weights.

    Args:
        P_row: length K array of probabilities (may have NaN)
        weights: length K array of weights (should be 0 for invalid agents)

    Returns: pooled probability
    """
    valid = ~np.isnan(P_row) & (weights > 0)
    if not valid.any():
        return 0.5

    logits = logit(P_row[valid])
    w = weights[valid]

    # Renormalize weights for valid agents
    w = w / w.sum()

    pooled_logit = np.sum(w * logits)
    return float(sigmoid(pooled_logit))


def pool_linear(P_row: np.ndarray, weights: np.ndarray) -> float:
    """
    Pool probabilities linearly (probability space) using given weights.

    Args:
        P_row: length K array of probabilities (may have NaN)
        weights: length K array of weights (should be 0 for invalid agents)

    Returns: pooled probability
    """
    valid = ~np.isnan(P_row) & (weights > 0)
    if not valid.any():
        return 0.5

    probs = P_row[valid]
    w = weights[valid]

    # Renormalize weights for valid agents
    w = w / w.sum()

    pooled = np.sum(w * probs)
    return float(clip(np.array([pooled]))[0])
