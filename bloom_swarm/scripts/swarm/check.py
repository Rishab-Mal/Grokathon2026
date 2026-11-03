#!/usr/bin/env python3
"""
Integration validator for the swarm system.

Checks run at 1:15 when real data arrives:
a. Validate week and output files against schemas
b. Print weeks x agents grid with status symbols
c. Flag stuck agents, repeated reasons, one-sided predictions, leakage hints
d. Flag missing actuals, class balance warnings
e. Exit code 1 if >20% cells are bad
"""

import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple

import numpy as np

# Add scripts to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from swarm.core import CONFIG, load_data


# Schema validation (hand-written, no jsonschema dependency)
def validate_week_file(data: Dict, path: Path) -> List[str]:
    """Validate a week input file. Returns list of errors."""
    errors = []

    # Required fields
    required = ["week", "weather", "runoff", "satellite", "history", "actual"]
    for field in required:
        if field not in data:
            errors.append(f"{path}: missing required field '{field}'")

    # Week format
    if "week" in data:
        if not re.match(r"^\d{4}-\d{2}-\d{2}$", data["week"]):
            errors.append(f"{path}: invalid week format '{data['week']}'")

    # Actual must be 0 or 1
    if "actual" in data:
        if data["actual"] not in [0, 1]:
            errors.append(f"{path}: actual must be 0 or 1, got {data['actual']}")

    # Weather structure
    if "weather" in data and data["weather"] is not None:
        w = data["weather"]
        if not isinstance(w, dict):
            errors.append(f"{path}: weather must be an object")
        else:
            for key in ["max_temp_avg", "rain_total_mm", "hot_days"]:
                if key not in w:
                    errors.append(f"{path}: weather missing '{key}'")

    return errors


def validate_output_file(data: Dict, path: Path) -> List[str]:
    """Validate an agent output file. Returns list of errors."""
    errors = []

    # Required fields
    required = ["name", "p", "reason", "evidence"]
    for field in required:
        if field not in data:
            errors.append(f"{path}: missing required field '{field}'")

    # p must be between 0 and 1
    if "p" in data:
        p = data["p"]
        if not isinstance(p, (int, float)):
            errors.append(f"{path}: p must be a number")
        elif p < 0 or p > 1:
            errors.append(f"{path}: p must be between 0 and 1, got {p}")

    # evidence must be a list
    if "evidence" in data:
        if not isinstance(data["evidence"], list):
            errors.append(f"{path}: evidence must be a list")

    return errors


def validate_all_files(
    weeks_dir: Path,
    outputs_dir: Path,
) -> Tuple[List[str], Dict[str, Any]]:
    """
    Validate all week and output files.

    Returns:
        errors: list of error messages
        stats: validation statistics
    """
    errors = []
    stats = {
        "weeks_valid": 0,
        "weeks_invalid": 0,
        "outputs_valid": 0,
        "outputs_invalid": 0,
    }

    # Validate week files
    for path in sorted(weeks_dir.glob("*.json")):
        try:
            with open(path) as f:
                data = json.load(f)
            file_errors = validate_week_file(data, path)
            if file_errors:
                errors.extend(file_errors)
                stats["weeks_invalid"] += 1
            else:
                stats["weeks_valid"] += 1
        except json.JSONDecodeError as e:
            errors.append(f"{path}: invalid JSON - {e}")
            stats["weeks_invalid"] += 1

    # Validate output files
    for path in sorted(outputs_dir.glob("*.json")):
        try:
            with open(path) as f:
                data = json.load(f)
            file_errors = validate_output_file(data, path)
            if file_errors:
                errors.extend(file_errors)
                stats["outputs_invalid"] += 1
            else:
                stats["outputs_valid"] += 1
        except json.JSONDecodeError as e:
            errors.append(f"{path}: invalid JSON - {e}")
            stats["outputs_invalid"] += 1

    return errors, stats


def print_status_grid(
    weeks: List[str],
    agents: List[str],
    P: np.ndarray,
    meta: Dict,
) -> Dict[str, int]:
    """
    Print weeks x agents grid with status symbols.

    Symbols:
    - OK: valid prediction
    - XX: missing
    - !!: invalid (NaN or out of range)
    - CC: clipped (at boundary)
    """
    T, K = P.shape
    CLIP_LO, CLIP_HI = CONFIG["clip"]

    counts = {"ok": 0, "missing": 0, "invalid": 0, "clipped": 0}

    print("\n" + "=" * 80)
    print("DATA STATUS GRID")
    print("=" * 80)

    # Header
    header = f"{'Week':<12}"
    for agent in agents:
        header += f" {agent[:8]:>8}"
    print(header)
    print("-" * (12 + 9 * K))

    for t, week in enumerate(weeks):
        row = f"{week:<12}"
        for k, agent in enumerate(agents):
            p = P[t, k]
            if np.isnan(p):
                if agent in meta.get(week, {}):
                    row += "       XX"  # File exists but invalid
                    counts["invalid"] += 1
                else:
                    row += "       --"  # Missing file
                    counts["missing"] += 1
            elif p == CLIP_LO or p == CLIP_HI:
                row += "       CC"
                counts["clipped"] += 1
            else:
                row += "       OK"
                counts["ok"] += 1
        print(row)

    print("-" * (12 + 9 * K))
    print(f"OK: {counts['ok']}, Missing: {counts['missing']}, Invalid: {counts['invalid']}, Clipped: {counts['clipped']}")

    return counts


def check_stuck_agents(
    P: np.ndarray,
    agents: List[str],
    threshold: float = 0.05,
) -> List[str]:
    """Flag agents with low prediction variance (stuck)."""
    warnings = []
    for k, agent in enumerate(agents):
        valid = ~np.isnan(P[:, k])
        if valid.sum() > 1:
            std = np.std(P[valid, k])
            if std < threshold:
                warnings.append(f"  - {agent}: std = {std:.4f} (threshold: {threshold})")
    return warnings


def check_repeated_reasons(
    meta: Dict,
    threshold: int = 2,
) -> List[str]:
    """Flag agents with repeated reasons."""
    warnings = []
    reason_counts: Dict[str, Dict[str, int]] = {}

    for week, agents in meta.items():
        for agent, data in agents.items():
            reason = data.get("reason", "")
            if agent not in reason_counts:
                reason_counts[agent] = {}
            reason_counts[agent][reason] = reason_counts[agent].get(reason, 0) + 1

    for agent, reasons in reason_counts.items():
        for reason, count in reasons.items():
            if count > threshold:
                warnings.append(f"  - {agent}: '{reason[:50]}...' used {count} times")

    return warnings


def check_one_sided(
    P: np.ndarray,
    agents: List[str],
) -> List[str]:
    """Flag agents whose predictions are always on one side of 0.5."""
    warnings = []
    for k, agent in enumerate(agents):
        valid = ~np.isnan(P[:, k])
        if valid.sum() > 0:
            all_above = np.all(P[valid, k] >= 0.5)
            all_below = np.all(P[valid, k] <= 0.5)
            if all_above or all_below:
                side = "above" if all_above else "below"
                warnings.append(f"  - {agent}: all predictions {side} 0.5")
    return warnings


def check_leakage_hints(meta: Dict) -> List[str]:
    """Flag reasons that mention temporal info (potential leakage)."""
    warnings = []
    patterns = [
        r"\b20\d{2}\b",  # Year mentions
        r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\b",
        r"\badvisory\b",  # Direct advisory mentions
    ]

    for week, agents in meta.items():
        for agent, data in agents.items():
            reason = data.get("reason", "").lower()
            evidence = " ".join(data.get("evidence", [])).lower()
            text = reason + " " + evidence

            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    match = re.search(pattern, text, re.IGNORECASE)
                    warnings.append(f"  - {week}/{agent}: leakage hint '{match.group()}'")

    return warnings


def check_class_balance(y: np.ndarray) -> List[str]:
    """Flag class balance issues."""
    warnings = []
    n_pos = y.sum()
    n_neg = len(y) - n_pos

    if n_pos < 3:
        warnings.append(f"  - Only {int(n_pos)} positive cases (AUC/calibration unreliable)")
    if n_neg < 3:
        warnings.append(f"  - Only {int(n_neg)} negative cases (AUC/calibration unreliable)")

    return warnings


def check_missing_actuals(
    weeks_dir: Path,
) -> List[str]:
    """Flag weeks with missing actual values."""
    warnings = []

    for path in sorted(weeks_dir.glob("*.json")):
        with open(path) as f:
            data = json.load(f)
        if "actual" not in data or data["actual"] is None:
            warnings.append(f"  - {path.stem}: missing actual")

    return warnings


def run_checks(
    weeks_dir: Path = Path("data/weeks"),
    outputs_dir: Path = Path("outputs"),
) -> int:
    """
    Run all validation checks.

    Returns exit code (0 = ok, 1 = too many bad cells).
    """
    print("=" * 80)
    print("INTEGRATION VALIDATION CHECK")
    print("=" * 80)

    # Validate files
    print("\n1. SCHEMA VALIDATION")
    errors, stats = validate_all_files(weeks_dir, outputs_dir)
    print(f"   Week files: {stats['weeks_valid']} valid, {stats['weeks_invalid']} invalid")
    print(f"   Output files: {stats['outputs_valid']} valid, {stats['outputs_invalid']} invalid")

    if errors:
        print("\n   Errors:")
        for e in errors[:10]:
            print(f"   - {e}")
        if len(errors) > 10:
            print(f"   ... and {len(errors) - 10} more")

    # Load data
    print("\n2. LOADING DATA")
    weeks, agents, P, y, meta = load_data(weeks_dir, outputs_dir)
    print(f"   Weeks: {len(weeks)}, Agents: {len(agents)}")

    # Status grid
    counts = print_status_grid(weeks, agents, P, meta)

    # Quality checks
    print("\n3. QUALITY CHECKS")

    print("\n   Stuck agents (std < 0.05):")
    stuck = check_stuck_agents(P, agents)
    if stuck:
        for w in stuck:
            print(w)
    else:
        print("   None")

    print("\n   Repeated reasons (>2 uses):")
    repeated = check_repeated_reasons(meta)
    if repeated:
        for w in repeated[:5]:
            print(w)
        if len(repeated) > 5:
            print(f"   ... and {len(repeated) - 5} more")
    else:
        print("   None")

    print("\n   One-sided predictions:")
    onesided = check_one_sided(P, agents)
    if onesided:
        for w in onesided:
            print(w)
    else:
        print("   None")

    print("\n   Leakage hints:")
    leakage = check_leakage_hints(meta)
    if leakage:
        for w in leakage[:5]:
            print(w)
        if len(leakage) > 5:
            print(f"   ... and {len(leakage) - 5} more")
    else:
        print("   None")

    print("\n   Missing actuals:")
    missing_actuals = check_missing_actuals(weeks_dir)
    if missing_actuals:
        for w in missing_actuals:
            print(w)
    else:
        print("   None")

    print("\n   Class balance:")
    balance = check_class_balance(y)
    if balance:
        for w in balance:
            print(w)
    else:
        print(f"   OK: {int(y.sum())} positives, {int(len(y) - y.sum())} negatives")

    # Final verdict
    print("\n" + "=" * 80)
    total_cells = P.shape[0] * P.shape[1]
    bad_cells = counts["missing"] + counts["invalid"]
    bad_rate = bad_cells / total_cells if total_cells > 0 else 0

    print(f"SUMMARY: {bad_cells}/{total_cells} cells bad ({bad_rate:.1%})")

    if bad_rate > 0.2:
        print("FAIL: More than 20% bad cells")
        return 1
    else:
        print("PASS: Data quality acceptable")
        return 0


def main():
    """Entry point for check.py."""
    exit_code = run_checks()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
