#!/usr/bin/env python3
"""
Aggregation methods for combining agent forecasts.

Each aggregator is a class with:
  fit(P_train, y_train) -> self
  predict(P_test) -> probabilities (length T_test array)
  weights() -> dict of agent to weight, or None if not a weighting method

All methods handle NaN cells by dropping that agent for that week
and renormalizing weights.
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional, List, Tuple
import numpy as np

from .core import (
    CONFIG, clip, logit, sigmoid, brier, nanmean,
    pool_logodds, pool_linear, valid_mask
)


class Aggregator(ABC):
    """Base class for all aggregation methods."""

    def __init__(self, agents: List[str]):
        self.agents = agents
        self.K = len(agents)
        self._weights: Optional[np.ndarray] = None

    @abstractmethod
    def fit(self, P_train: np.ndarray, y_train: np.ndarray) -> "Aggregator":
        """Fit the aggregator on training data."""
        pass

    @abstractmethod
    def predict(self, P_test: np.ndarray) -> np.ndarray:
        """Generate predictions for test data."""
        pass

    def weights(self) -> Optional[Dict[str, float]]:
        """Return agent weights as dict, or None if not a weighting method."""
        if self._weights is None:
            return None
        return {self.agents[k]: float(self._weights[k]) for k in range(self.K)}


class BaseRate(Aggregator):
    """Predicts the mean of y_train for every week."""

    def __init__(self, agents: List[str]):
        super().__init__(agents)
        self.base_rate = 0.5

    def fit(self, P_train: np.ndarray, y_train: np.ndarray) -> "BaseRate":
        if len(y_train) > 0:
            self.base_rate = float(np.mean(y_train))
        else:
            self.base_rate = 0.5
        # Clip base rate
        self.base_rate = float(clip(np.array([self.base_rate]))[0])
        return self

    def predict(self, P_test: np.ndarray) -> np.ndarray:
        T_test = P_test.shape[0]
        return np.full(T_test, self.base_rate)

    def weights(self) -> None:
        return None


class EqualLinear(Aggregator):
    """Plain mean of probabilities (linear pooling)."""

    def __init__(self, agents: List[str]):
        super().__init__(agents)

    def fit(self, P_train: np.ndarray, y_train: np.ndarray) -> "EqualLinear":
        self._weights = np.ones(self.K) / self.K
        return self

    def predict(self, P_test: np.ndarray) -> np.ndarray:
        T_test = P_test.shape[0]
        preds = np.zeros(T_test)
        for t in range(T_test):
            preds[t] = pool_linear(P_test[t, :], self._weights)
        return clip(preds)


class EqualLogodds(Aggregator):
    """Sigmoid of the mean logit (log-odds pooling with equal weights)."""

    def __init__(self, agents: List[str]):
        super().__init__(agents)

    def fit(self, P_train: np.ndarray, y_train: np.ndarray) -> "EqualLogodds":
        self._weights = np.ones(self.K) / self.K
        return self

    def predict(self, P_test: np.ndarray) -> np.ndarray:
        T_test = P_test.shape[0]
        preds = np.zeros(T_test)
        for t in range(T_test):
            preds[t] = pool_logodds(P_test[t, :], self._weights)
        return clip(preds)


class InverseBrierLogodds(Aggregator):
    """
    Weight_i proportional to 1 / brier_i on training weeks, pool in log odds.
    Falls back to equal weights if fewer than 3 training weeks.

    This is the PRIMARY method (pre-registered).
    """

    MIN_TRAIN_WEEKS = 3

    def __init__(self, agents: List[str]):
        super().__init__(agents)

    def fit(self, P_train: np.ndarray, y_train: np.ndarray) -> "InverseBrierLogodds":
        T_train = P_train.shape[0]

        if T_train < self.MIN_TRAIN_WEEKS:
            self._weights = np.ones(self.K) / self.K
            return self

        # Compute per-agent Brier scores on training data
        brier_scores = np.zeros(self.K)
        for k in range(self.K):
            valid = ~np.isnan(P_train[:, k])
            if valid.sum() > 0:
                brier_scores[k] = np.mean(brier(P_train[valid, k], y_train[valid]))
            else:
                brier_scores[k] = 0.5  # High score for missing

        # Convert to weights: w_i = (1/brier_i) / sum(1/brier_j)
        raw = 1.0 / (brier_scores + 1e-6)
        self._weights = raw / raw.sum()
        return self

    def predict(self, P_test: np.ndarray) -> np.ndarray:
        T_test = P_test.shape[0]
        preds = np.zeros(T_test)
        for t in range(T_test):
            preds[t] = pool_logodds(P_test[t, :], self._weights)
        return clip(preds)


class BatesGranger(Aggregator):
    """
    Covariance-aware weighting (Bates-Granger).

    Build the K by K covariance matrix of errors (p_i - y) on training weeks.
    Shrink toward diagonal with intensity bg_shrinkage.
    Weights proportional to inverse covariance times vector of ones.
    Clip negative weights to zero and renormalize.
    Pool in log odds.

    Falls back to inverse_brier if matrix is singular.

    This method accounts for agents that make correlated mistakes, which
    matters because the Skeptic sees the same data as everyone else.
    """

    def __init__(self, agents: List[str]):
        super().__init__(agents)
        self.shrinkage = CONFIG["bg_shrinkage"]

    def fit(self, P_train: np.ndarray, y_train: np.ndarray) -> "BatesGranger":
        T_train = P_train.shape[0]

        if T_train < 3:
            self._weights = np.ones(self.K) / self.K
            return self

        # Compute errors: e_t,k = p_t,k - y_t
        errors = np.zeros((T_train, self.K))
        for t in range(T_train):
            for k in range(self.K):
                if np.isnan(P_train[t, k]):
                    errors[t, k] = np.nan
                else:
                    errors[t, k] = P_train[t, k] - y_train[t]

        # Compute covariance matrix of errors
        # Use pairwise complete observations
        cov = np.zeros((self.K, self.K))
        for i in range(self.K):
            for j in range(self.K):
                valid = ~np.isnan(errors[:, i]) & ~np.isnan(errors[:, j])
                n = valid.sum()
                if n > 1:
                    ei = errors[valid, i]
                    ej = errors[valid, j]
                    cov[i, j] = np.mean((ei - ei.mean()) * (ej - ej.mean()))
                else:
                    cov[i, j] = 0.0 if i != j else 0.25  # Default variance

        # Shrink toward diagonal
        diag = np.diag(np.diag(cov))
        cov_shrunk = (1 - self.shrinkage) * cov + self.shrinkage * diag

        # Ensure positive definiteness by adding small ridge
        cov_shrunk += np.eye(self.K) * 1e-6

        # Try to compute weights from inverse covariance
        try:
            cov_inv = np.linalg.inv(cov_shrunk)
            ones = np.ones(self.K)
            raw_weights = cov_inv @ ones

            # Clip negative weights to zero
            raw_weights = np.maximum(raw_weights, 0)

            if raw_weights.sum() > 0:
                self._weights = raw_weights / raw_weights.sum()
            else:
                # Fall back to equal weights
                self._weights = np.ones(self.K) / self.K

        except np.linalg.LinAlgError:
            # Singular matrix: fall back to inverse Brier
            fallback = InverseBrierLogodds(self.agents)
            fallback.fit(P_train, y_train)
            self._weights = fallback._weights

        return self

    def predict(self, P_test: np.ndarray) -> np.ndarray:
        T_test = P_test.shape[0]
        preds = np.zeros(T_test)
        for t in range(T_test):
            preds[t] = pool_logodds(P_test[t, :], self._weights)
        return clip(preds)


class StackedLogistic(Aggregator):
    """
    Logistic regression of y on agent logits, with intercept and L2 penalty on slopes only.
    Fitted by IRLS. Choose penalty from stack_l2_grid by inner LOO on training weeks.
    Impute missing logits with that agent's training mean.
    """

    def __init__(self, agents: List[str]):
        super().__init__(agents)
        self.l2_grid = CONFIG["stack_l2_grid"]
        self.intercept = 0.0
        self.coefs = None

    def fit(self, P_train: np.ndarray, y_train: np.ndarray) -> "StackedLogistic":
        T_train, K = P_train.shape

        if T_train < 3:
            self._weights = np.ones(K) / K
            self.intercept = logit(np.array([np.mean(y_train) if len(y_train) > 0 else 0.5]))[0]
            self.coefs = np.zeros(K)
            return self

        # Convert to logits and impute missing with column means
        P_logits = logit(P_train)
        col_means = np.zeros(K)
        for k in range(K):
            valid = ~np.isnan(P_logits[:, k])
            if valid.sum() > 0:
                col_means[k] = np.mean(P_logits[valid, k])
            else:
                col_means[k] = 0.0

        X = P_logits.copy()
        for k in range(K):
            X[np.isnan(X[:, k]), k] = col_means[k]

        # Inner LOO to select best lambda
        best_lambda = self.l2_grid[0]
        best_loss = np.inf

        for lam in self.l2_grid:
            loo_loss = 0.0
            for t in range(T_train):
                # Fit on all except t
                mask = np.ones(T_train, dtype=bool)
                mask[t] = False
                X_train_inner = X[mask]
                y_train_inner = y_train[mask]

                coefs, intercept = self._fit_irls(X_train_inner, y_train_inner, lam)

                # Predict t
                pred_logit = intercept + X[t] @ coefs
                pred = float(sigmoid(pred_logit))
                loo_loss += brier(np.array([pred]), np.array([y_train[t]]))[0]

            if loo_loss < best_loss:
                best_loss = loo_loss
                best_lambda = lam

        # Fit on full training data with best lambda
        self.coefs, self.intercept = self._fit_irls(X, y_train, best_lambda)

        # Store column means for prediction
        self._col_means = col_means

        # Weights are proportional to |coefs|
        if np.sum(np.abs(self.coefs)) > 0:
            self._weights = np.abs(self.coefs) / np.sum(np.abs(self.coefs))
        else:
            self._weights = np.ones(K) / K

        return self

    def _fit_irls(
        self, X: np.ndarray, y: np.ndarray, lam: float, max_iter: int = 100, tol: float = 1e-6
    ) -> Tuple[np.ndarray, float]:
        """Fit logistic regression with L2 penalty on slopes using IRLS."""
        T, K = X.shape

        # Add intercept column
        X_aug = np.column_stack([np.ones(T), X])

        # Initialize coefficients
        beta = np.zeros(K + 1)

        for _ in range(max_iter):
            # Compute predictions
            eta = X_aug @ beta
            mu = sigmoid(eta)

            # Clip mu to avoid numerical issues
            mu = np.clip(mu, 1e-10, 1 - 1e-10)

            # Working response
            z = eta + (y - mu) / (mu * (1 - mu))

            # Weights (diagonal of W)
            w = mu * (1 - mu)

            # Penalty matrix: penalize slopes but not intercept
            penalty = np.zeros((K + 1, K + 1))
            for k in range(1, K + 1):
                penalty[k, k] = lam

            # Weighted least squares update
            W = np.diag(w)
            XtWX = X_aug.T @ W @ X_aug + penalty
            XtWz = X_aug.T @ (w * z)

            try:
                beta_new = np.linalg.solve(XtWX, XtWz)
            except np.linalg.LinAlgError:
                # Add ridge for stability
                XtWX += np.eye(K + 1) * 1e-6
                beta_new = np.linalg.solve(XtWX, XtWz)

            if np.max(np.abs(beta_new - beta)) < tol:
                beta = beta_new
                break
            beta = beta_new

        return beta[1:], beta[0]

    def predict(self, P_test: np.ndarray) -> np.ndarray:
        T_test = P_test.shape[0]
        preds = np.zeros(T_test)

        P_logits = logit(P_test)

        for t in range(T_test):
            x = P_logits[t, :].copy()
            # Impute missing with training column means
            for k in range(len(x)):
                if np.isnan(x[k]):
                    x[k] = self._col_means[k]

            pred_logit = self.intercept + x @ self.coefs
            preds[t] = float(sigmoid(pred_logit))

        return clip(preds)


class ExtremizedPrimary(Aggregator):
    """
    The primary method's pooled logit multiplied by factor a.
    Factor a chosen from extremize_grid by inner LOO on training weeks.

    Extremizing explanation:
    Combining agents that share information tends to make the pool underconfident.
    When agents access overlapping data sources, the raw average doesn't account
    for shared information, so it stays closer to 0.5 than warranted. Multiplying
    the pooled logit by a factor > 1 corrects this by making predictions more extreme.
    """

    def __init__(self, agents: List[str]):
        super().__init__(agents)
        self.extremize_grid = CONFIG["extremize_grid"]
        self.primary = None
        self.best_a = 1.0

    def fit(self, P_train: np.ndarray, y_train: np.ndarray) -> "ExtremizedPrimary":
        T_train = P_train.shape[0]

        # Fit primary method
        self.primary = InverseBrierLogodds(self.agents)
        self.primary.fit(P_train, y_train)
        self._weights = self.primary._weights.copy()

        if T_train < 3:
            self.best_a = 1.0
            return self

        # Inner LOO to select best a
        best_a = 1.0
        best_loss = np.inf

        for a in self.extremize_grid:
            loo_loss = 0.0
            for t in range(T_train):
                # Fit primary on all except t
                mask = np.ones(T_train, dtype=bool)
                mask[t] = False
                inner_primary = InverseBrierLogodds(self.agents)
                inner_primary.fit(P_train[mask], y_train[mask])

                # Predict t with extremization
                P_row = P_train[t, :]
                pooled = inner_primary.predict(P_row.reshape(1, -1))[0]
                pooled_logit = logit(np.array([pooled]))[0]
                extremized = sigmoid(a * pooled_logit)

                loo_loss += brier(np.array([extremized]), np.array([y_train[t]]))[0]

            if loo_loss < best_loss:
                best_loss = loo_loss
                best_a = a

        self.best_a = best_a
        return self

    def predict(self, P_test: np.ndarray) -> np.ndarray:
        T_test = P_test.shape[0]
        preds = np.zeros(T_test)

        for t in range(T_test):
            pooled = pool_logodds(P_test[t, :], self._weights)
            pooled_logit = logit(np.array([pooled]))[0]
            preds[t] = float(sigmoid(self.best_a * pooled_logit))

        return clip(preds)


class BestAgentSelected(Aggregator):
    """
    Picks the single agent with lowest training Brier and uses only that agent.

    This is the realistic "just trust the best agent" baseline, because you
    don't know in advance which agent is best.
    """

    def __init__(self, agents: List[str]):
        super().__init__(agents)
        self.best_agent_idx = 0

    def fit(self, P_train: np.ndarray, y_train: np.ndarray) -> "BestAgentSelected":
        T_train = P_train.shape[0]

        if T_train == 0:
            self.best_agent_idx = 0
            self._weights = np.zeros(self.K)
            self._weights[0] = 1.0
            return self

        # Compute per-agent Brier scores
        brier_scores = np.full(self.K, np.inf)
        for k in range(self.K):
            valid = ~np.isnan(P_train[:, k])
            if valid.sum() > 0:
                brier_scores[k] = np.mean(brier(P_train[valid, k], y_train[valid]))

        self.best_agent_idx = int(np.argmin(brier_scores))
        self._weights = np.zeros(self.K)
        self._weights[self.best_agent_idx] = 1.0
        return self

    def predict(self, P_test: np.ndarray) -> np.ndarray:
        T_test = P_test.shape[0]
        preds = np.zeros(T_test)

        for t in range(T_test):
            p = P_test[t, self.best_agent_idx]
            if np.isnan(p):
                # Fall back to mean of available agents
                valid = ~np.isnan(P_test[t, :])
                if valid.any():
                    preds[t] = np.mean(P_test[t, valid])
                else:
                    preds[t] = 0.5
            else:
                preds[t] = p

        return clip(preds)


class HedgeOnline(Aggregator):
    """
    Multiplicative weights where weight_i is proportional to
    exp(-eta * cumulative_brier_loss_i) on PAST weeks only.

    Valid only under walk-forward protocol. Under LOO, this reduces to
    a static method fitted on past weeks.
    """

    def __init__(self, agents: List[str]):
        super().__init__(agents)
        self.eta = CONFIG["hedge_eta"]
        self.cumulative_losses = None

    def fit(self, P_train: np.ndarray, y_train: np.ndarray) -> "HedgeOnline":
        """For Hedge, fit computes cumulative losses on training data."""
        T_train = P_train.shape[0]

        self.cumulative_losses = np.zeros(self.K)

        for t in range(T_train):
            for k in range(self.K):
                if not np.isnan(P_train[t, k]):
                    loss = brier(np.array([P_train[t, k]]), np.array([y_train[t]]))[0]
                    self.cumulative_losses[k] += loss

        # Compute weights from cumulative losses
        raw = np.exp(-self.eta * self.cumulative_losses)
        self._weights = raw / raw.sum()
        return self

    def predict(self, P_test: np.ndarray) -> np.ndarray:
        T_test = P_test.shape[0]
        preds = np.zeros(T_test)
        for t in range(T_test):
            preds[t] = pool_logodds(P_test[t, :], self._weights)
        return clip(preds)


# Registry of all methods
METHODS: Dict[str, type] = {
    "base_rate": BaseRate,
    "equal_linear": EqualLinear,
    "equal_logodds": EqualLogodds,
    "inverse_brier_logodds": InverseBrierLogodds,
    "bates_granger": BatesGranger,
    "stacked_logistic": StackedLogistic,
    "extremized_primary": ExtremizedPrimary,
    "best_agent_selected": BestAgentSelected,
    "hedge_online": HedgeOnline,
}

PRIMARY_METHOD = CONFIG["primary_method"]


def get_method(name: str, agents: List[str]) -> Aggregator:
    """Get an aggregator instance by name."""
    if name not in METHODS:
        raise ValueError(f"Unknown method: {name}. Available: {list(METHODS.keys())}")
    return METHODS[name](agents)
