#!/usr/bin/env python3
"""
Cost-loss economic value analysis for forecast decisions.

Models a lake manager deciding whether to issue a bloom advisory.
- Closing/warning costs C
- Missing a bloom costs L
- Cost ratio r = C/L is in (0, 1)

Issue warning when p > r.

Relative Economic Value V(r) = (E_clim - E_forecast) / (E_clim - E_perfect)

where:
- E_clim = expense using climatology (best of always/never warning)
- E_perfect = expense with perfect forecasts
- E_forecast = expense using probabilistic forecast

Interpretation: "at cost ratio 0.2, the swarm captures X% of perfect forecast value"
"""

from typing import Dict, List, Any
import numpy as np

from .core import CONFIG


COST_LOSS_GRID_N = CONFIG["cost_loss_grid_n"]


def compute_expenses(
    preds: np.ndarray,
    y: np.ndarray,
    r: float,
    L: float = 1.0,
) -> float:
    """
    Compute total expense for a forecast at cost ratio r.

    Issue warning when p > r.
    Expense = C * n_warnings + L * n_missed_blooms
    where C = r * L.

    Args:
        preds: probability forecasts
        y: actual outcomes (0 or 1)
        r: cost ratio C/L
        L: loss from missed bloom (default 1.0)

    Returns:
        Total expense
    """
    C = r * L
    warnings = (preds > r).astype(int)

    # Missed blooms: actual bloom but no warning
    missed = ((y == 1) & (warnings == 0)).sum()

    # Cost of warnings
    warning_cost = warnings.sum() * C

    # Cost of missed blooms
    miss_cost = missed * L

    return warning_cost + miss_cost


def compute_climatology_expense(
    y: np.ndarray,
    r: float,
    L: float = 1.0,
) -> float:
    """
    Compute expense for climatology strategy.

    Climatology: take the better of always warning and never warning.

    Always warn: pay C for each period = n * r * L
    Never warn: pay L for each bloom = n_blooms * L
    """
    n = len(y)
    n_blooms = y.sum()
    C = r * L

    always_warn = n * C
    never_warn = n_blooms * L

    return min(always_warn, never_warn)


def compute_perfect_expense(
    y: np.ndarray,
    r: float,
    L: float = 1.0,
) -> float:
    """
    Compute expense for perfect forecasts.

    Perfect: warn only when there will be a bloom.
    Expense = C * n_blooms
    """
    n_blooms = y.sum()
    C = r * L
    return n_blooms * C


def relative_economic_value(
    preds: np.ndarray,
    y: np.ndarray,
    r: float,
) -> float:
    """
    Compute relative economic value at cost ratio r.

    V(r) = (E_clim - E_forecast) / (E_clim - E_perfect)

    Returns value in [0, 1] where 1 is perfect.
    Can be negative if forecast does worse than climatology.
    """
    E_clim = compute_climatology_expense(y, r)
    E_perfect = compute_perfect_expense(y, r)
    E_forecast = compute_expenses(preds, y, r)

    denominator = E_clim - E_perfect
    if denominator == 0:
        return 1.0 if E_forecast <= E_clim else 0.0

    value = (E_clim - E_forecast) / denominator
    return float(value)


def compute_value_curve(
    preds: np.ndarray,
    y: np.ndarray,
    n_points: int = COST_LOSS_GRID_N,
) -> Dict[str, List[float]]:
    """
    Compute economic value curve over range of cost ratios.
    """
    r_grid = np.linspace(0.02, 0.98, n_points)
    values = [relative_economic_value(preds, y, r) for r in r_grid]

    return {
        "r": r_grid.tolist(),
        "values": values,
    }


def find_peak_value(curve: Dict[str, List[float]]) -> Dict[str, float]:
    """
    Find peak value and where it occurs.
    """
    values = np.array(curve["values"])
    r_grid = np.array(curve["r"])

    peak_idx = np.argmax(values)
    return {
        "r": float(r_grid[peak_idx]),
        "value": float(values[peak_idx]),
    }


def get_value_at_r(curve: Dict[str, List[float]], target_r: float) -> float:
    """
    Get value at a specific cost ratio (interpolated if needed).
    """
    r_grid = np.array(curve["r"])
    values = np.array(curve["values"])

    # Find nearest r
    idx = np.argmin(np.abs(r_grid - target_r))
    return float(values[idx])


def compute_economic_analysis(
    forecasts: Dict[str, np.ndarray],
    y: np.ndarray,
) -> Dict[str, Any]:
    """
    Compute economic value analysis for multiple forecast sources.

    Args:
        forecasts: {name: predictions array}
        y: outcomes

    Returns:
        Economic analysis results
    """
    results = {
        "r_grid": np.linspace(0.02, 0.98, COST_LOSS_GRID_N).tolist(),
        "curves": {},
        "peaks": {},
        "values_at_r": {},
    }

    target_rs = [0.1, 0.2, 0.5]

    for name, preds in forecasts.items():
        valid = ~np.isnan(preds)
        if not valid.all():
            # Skip forecasts with missing values
            continue

        curve = compute_value_curve(preds, y)
        results["curves"][name] = curve["values"]

        peak = find_peak_value(curve)
        results["peaks"][name] = peak

        results["values_at_r"][name] = {
            f"r={r}": get_value_at_r(curve, r)
            for r in target_rs
        }

    return results


def print_economic_summary(analysis: Dict[str, Any]):
    """Print economic value analysis summary."""
    print("\n" + "=" * 80)
    print("PHASE 4: ECONOMIC VALUE ANALYSIS")
    print("=" * 80)

    print("\nCost-loss model: lake manager deciding whether to issue bloom advisory")
    print("  - Closing/warning costs C")
    print("  - Missing a bloom costs L")
    print("  - Cost ratio r = C/L")
    print("  - Issue warning when predicted probability p > r")

    print("\nPeak economic values:")
    print(f"{'Forecast':<25} {'Peak Value':<12} {'At r':<10}")
    print("-" * 47)

    sorted_peaks = sorted(
        analysis["peaks"].items(),
        key=lambda x: x[1]["value"],
        reverse=True
    )

    for name, peak in sorted_peaks:
        print(f"{name:<25} {peak['value']*100:<12.1f}% {peak['r']:<10.2f}")

    print("\nValues at specific cost ratios:")
    print(f"{'Forecast':<25} {'r=0.1':<12} {'r=0.2':<12} {'r=0.5':<12}")
    print("-" * 61)

    for name in analysis["values_at_r"].keys():
        vals = analysis["values_at_r"][name]
        print(f"{name:<25} {vals['r=0.1']*100:<12.1f}% {vals['r=0.2']*100:<12.1f}% {vals['r=0.5']*100:<12.1f}%")

    print("\nInterpretation:")
    if "inverse_brier_logodds" in analysis["values_at_r"]:
        val_02 = analysis["values_at_r"]["inverse_brier_logodds"]["r=0.2"]
        print(f"  At a cost ratio of 0.2, the swarm captures {val_02*100:.0f}% of perfect forecast value.")
    print("  Higher values mean more economic benefit from using the forecast.")
    print("=" * 80)
