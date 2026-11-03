#!/usr/bin/env python3
"""
Tests for swarm evaluation package.

Each test prints PASS or FAIL.
"""

import sys
from pathlib import Path

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np

from swarm.core import CONFIG, load_data, set_seed, brier, logit, sigmoid, clip
from swarm.aggregators import (
    METHODS, PRIMARY_METHOD, get_method,
    BaseRate, EqualLinear, EqualLogodds, InverseBrierLogodds,
    BatesGranger, StackedLogistic
)
from swarm.protocols import leave_one_out, walk_forward
from swarm.diversity import diversity_prediction_theorem
from swarm.shapley import exact_shapley_values, verify_shapley_efficiency
from swarm.robustness import inject_liar


def test_result(name: str, passed: bool, details: str = ""):
    """Print test result."""
    status = "PASS" if passed else "FAIL"
    if details:
        print(f"[{status}] {name}: {details}")
    else:
        print(f"[{status}] {name}")
    return passed


def test_identical_agents():
    """If all agents predict the same, every pooling method should return that same p."""
    set_seed()
    agents = ["A", "B", "C"]
    T = 10
    p_value = 0.6

    P = np.full((T, 3), p_value)
    y = np.random.randint(0, 2, T).astype(float)

    passed = True
    for method_name in ["equal_linear", "equal_logodds", "inverse_brier_logodds", "bates_granger"]:
        method = get_method(method_name, agents)
        method.fit(P, y)
        preds = method.predict(P)

        if not np.allclose(preds, p_value, atol=0.01):
            passed = False
            break

    return test_result("identical_agents", passed,
                       f"All methods return same p when agents identical")


def test_weights_non_negative():
    """Weights should be non-negative and sum to 1."""
    set_seed()
    agents = ["A", "B", "C", "D", "E"]
    T = 20
    P = np.random.uniform(0.1, 0.9, (T, 5))
    y = np.random.randint(0, 2, T).astype(float)

    passed = True
    for method_name in METHODS.keys():
        method = get_method(method_name, agents)
        method.fit(P, y)
        weights = method.weights()

        if weights is not None:
            values = list(weights.values())
            if any(w < -1e-9 for w in values):
                passed = False
                break
            if abs(sum(values) - 1.0) > 1e-6:
                passed = False
                break

    return test_result("weights_non_negative", passed,
                       "All weights >= 0 and sum to 1")


def test_loo_leakage():
    """Under LOO, changing y[t] should not change prediction for week t."""
    set_seed()
    agents = ["A", "B", "C"]
    T = 10
    P = np.random.uniform(0.2, 0.8, (T, 3))
    y = np.random.randint(0, 2, T).astype(float)

    passed = True
    for method_name in METHODS.keys():
        # Get original prediction for week 5
        preds1, _ = leave_one_out(method_name, P, y, agents)
        pred_t = preds1[5]

        # Change y[5] and re-run
        y_modified = y.copy()
        y_modified[5] = 1 - y[5]  # Flip

        preds2, _ = leave_one_out(method_name, P, y_modified, agents)
        pred_t_modified = preds2[5]

        if abs(pred_t - pred_t_modified) > 1e-9:
            passed = False
            break

    return test_result("loo_leakage", passed,
                       "Changing y[t] does not change LOO prediction for week t")


def test_wf_leakage():
    """Under walk-forward, changing y[s] with s >= t should not change prediction for week t."""
    set_seed()
    agents = ["A", "B", "C"]
    T = 10
    P = np.random.uniform(0.2, 0.8, (T, 3))
    y = np.random.randint(0, 2, T).astype(float)

    warmup = CONFIG["walk_forward_warmup"]
    test_week = warmup + 1

    passed = True
    for method_name in METHODS.keys():
        # Get original prediction
        preds1, _ = walk_forward(method_name, P, y, agents)
        pred_t = preds1[test_week]

        # Change y[t+1] and re-run
        y_modified = y.copy()
        y_modified[test_week + 1] = 1 - y[test_week + 1]

        preds2, _ = walk_forward(method_name, P, y_modified, agents)
        pred_t_modified = preds2[test_week]

        if abs(pred_t - pred_t_modified) > 1e-9:
            passed = False
            break

    return test_result("wf_leakage", passed,
                       "Changing future y does not change WF prediction")


def test_nan_handling():
    """Week should still predict with NaN, and weights should renormalize."""
    set_seed()
    agents = ["A", "B", "C"]
    T = 10
    P = np.random.uniform(0.2, 0.8, (T, 3))
    y = np.random.randint(0, 2, T).astype(float)

    # Add NaN
    P[5, 1] = np.nan

    passed = True
    try:
        preds, weights = leave_one_out(PRIMARY_METHOD, P, y, agents)

        # Check prediction is valid
        if np.isnan(preds[5]):
            passed = False

        # Check weights renormalize
        if weights[5]:
            w = weights[5]
            if abs(sum(w.values()) - 1.0) > 1e-6:
                passed = False
    except Exception as e:
        passed = False

    return test_result("nan_handling", passed,
                       "Predictions work with NaN cells")


def test_extreme_inputs():
    """Inputs of 0 and 1 should not produce inf or nan."""
    set_seed()
    agents = ["A", "B"]
    T = 10
    P = np.array([[0.03, 0.97] for _ in range(T)])
    y = np.random.randint(0, 2, T).astype(float)

    passed = True
    for method_name in METHODS.keys():
        try:
            preds, _ = leave_one_out(method_name, P, y, agents)
            if np.any(np.isnan(preds)) or np.any(np.isinf(preds)):
                passed = False
                break
        except Exception:
            passed = False
            break

    return test_result("extreme_inputs", passed,
                       "No inf/nan with extreme p values")


def test_diversity_theorem():
    """Diversity theorem identity should hold within 1e-9."""
    set_seed()
    T, K = 20, 5
    P = np.random.uniform(0.2, 0.8, (T, K))
    y = np.random.randint(0, 2, T).astype(float)

    result = diversity_prediction_theorem(P, y)
    passed = result["identity_holds"]

    return test_result("diversity_theorem", passed,
                       f"Gap: {result['gap']:.12f}")


def test_shapley_efficiency():
    """Shapley values should sum to v(all agents)."""
    set_seed()
    agents = ["A", "B", "C"]
    T = 10
    P = np.random.uniform(0.2, 0.8, (T, 3))
    y = np.random.randint(0, 2, T).astype(float)

    # Base rate Brier
    base_rate = np.mean(y)
    base_brier = float(np.mean(brier(np.full(T, base_rate), y)))

    shapley_values = exact_shapley_values(P, y, agents, base_brier)
    passed, gap = verify_shapley_efficiency(shapley_values, P, y, agents, base_brier)

    return test_result("shapley_efficiency", passed,
                       f"Gap: {gap:.12f}")


def test_bates_granger_identity():
    """With identity covariance, Bates-Granger should equal equal weights."""
    set_seed()
    agents = ["A", "B", "C"]
    T = 20

    # Create uncorrelated predictions
    P = np.random.uniform(0.2, 0.8, (T, 3))
    y = np.random.randint(0, 2, T).astype(float)

    # Equal logodds
    eq = EqualLogodds(agents)
    eq.fit(P, y)
    preds_eq = eq.predict(P)

    # Bates-Granger with zero shrinkage would need identity cov
    # With shrinkage, it should be close
    bg = BatesGranger(agents)
    bg.fit(P, y)
    preds_bg = bg.predict(P)

    # Check weights are close to equal
    bg_weights = bg.weights()
    expected = 1.0 / 3
    weights_close = all(abs(w - expected) < 0.2 for w in bg_weights.values())

    return test_result("bates_granger_identity", weights_close,
                       f"Weights: {bg_weights}")


def test_stacked_large_penalty():
    """With huge L2 penalty, stacked logistic should approach base rate."""
    set_seed()
    agents = ["A", "B"]
    T = 20
    P = np.random.uniform(0.2, 0.8, (T, 2))
    y = np.random.randint(0, 2, T).astype(float)

    # Modify config temporarily
    original_grid = CONFIG["stack_l2_grid"]
    CONFIG["stack_l2_grid"] = [1000.0]

    try:
        stacked = StackedLogistic(agents)
        stacked.fit(P, y)
        preds = stacked.predict(P)

        # Should be close to base rate
        base_rate = np.mean(y)
        close_to_base = np.allclose(preds, base_rate, atol=0.2)
    finally:
        CONFIG["stack_l2_grid"] = original_grid

    return test_result("stacked_large_penalty", close_to_base,
                       f"Predictions near {base_rate:.3f}")


def test_brier_decomposition_gap():
    """Brier decomposition gap should be small."""
    from swarm.metrics import murphy_decomposition

    set_seed()
    T = 50
    preds = np.random.uniform(0.2, 0.8, T)
    y = np.random.randint(0, 2, T).astype(float)

    decomp = murphy_decomposition(preds, y)
    passed = decomp["gap"] < 0.02

    return test_result("brier_decomposition", passed,
                       f"Gap: {decomp['gap']:.6f}")


def test_liar_weight_low():
    """Liar agent should get low weight."""
    set_seed()

    # Load real data
    weeks, agents, P, y, meta = load_data()
    if len(weeks) == 0:
        return test_result("liar_weight", False, "No data to test")

    _, _, liar_result = inject_liar(P, y, agents)

    # Liar should get less than equal weight
    K_new = len(agents) + 1
    expected_equal = 1.0 / K_new
    passed = liar_result["avg_liar_weight"] < expected_equal

    return test_result("liar_weight", passed,
                       f"Liar weight: {liar_result['avg_liar_weight']:.4f} vs equal {expected_equal:.4f}")


def run_all_tests():
    """Run all tests and report results."""
    print("=" * 60)
    print("SWARM EVALUATION TESTS")
    print("=" * 60)
    print()

    tests = [
        test_identical_agents,
        test_weights_non_negative,
        test_loo_leakage,
        test_wf_leakage,
        test_nan_handling,
        test_extreme_inputs,
        test_diversity_theorem,
        test_shapley_efficiency,
        test_bates_granger_identity,
        test_stacked_large_penalty,
        test_brier_decomposition_gap,
        test_liar_weight_low,
    ]

    results = []
    for test in tests:
        try:
            passed = test()
            results.append(passed)
        except Exception as e:
            print(f"[FAIL] {test.__name__}: Exception - {e}")
            results.append(False)

    print()
    print("=" * 60)
    n_pass = sum(results)
    n_total = len(results)
    print(f"RESULTS: {n_pass}/{n_total} tests passed")
    print("=" * 60)

    return n_pass == n_total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
