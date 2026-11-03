# Methodology Appendix

*One-page technical reference for judge questions*

## Aggregation Methods

1. **Base Rate**: Predicts historical bloom frequency (climatological baseline)
2. **Equal Log-Odds**: Pools agent probabilities in log-odds space with equal weights
3. **Inverse Brier (PRIMARY)**: Weights agents by inverse of their past Brier scores, pools in log-odds
4. **Bates-Granger**: Uses error covariance matrix to down-weight correlated agents
5. **Stacked Logistic**: Logistic regression on agent logits with L2 regularization
6. **Extremized**: Multiplies pooled logit by factor >1 to correct underconfidence
7. **Best Agent Selected**: Uses only the agent with best training performance
8. **Hedge Online**: Multiplicative weights based on cumulative past losses

## Why Log-Odds Pooling?

Linear pooling (averaging probabilities) can produce poorly calibrated forecasts when
agents are correlated. Log-odds pooling (averaging logits) better handles extremes
and naturally weights confident predictions appropriately.

## Evaluation Protocols

- **Leave-One-Out (LOO)**: For week t, train on all other weeks, predict week t
- **Walk-Forward (WF)**: For week t, train only on weeks before t (realistic deployment)

Both protocols prevent data leakage: predictions never use future information.

## Statistical Testing

- **Diebold-Mariano**: Tests whether loss differences are significant (with HLN correction for small samples)
- **Block Bootstrap**: Preserves autocorrelation in weekly data using blocks of 3 weeks
- **Permutation Test**: Non-parametric significance via sign flips
- **Holm Correction**: Controls family-wise error rate across all comparisons

## Shapley Values

Shapley values fairly attribute the swarm's improvement over base rate to individual agents.
Unlike leave-one-out ablation, Shapley considers each agent's contribution in all possible
coalitions, giving fairer credit when agents have overlapping information.

## Pre-Registration

**The primary method (inverse_brier_logodds) was fixed before seeing results.**
This prevents cherry-picking the best-looking method after evaluation. All other methods
are secondary comparisons for context.

## Known Limitations

1. Sample size (~16 weeks) limits statistical power
2. High agent correlation (effective N ~ 1.2) limits diversity benefits
3. Runoff data is a proxy for actual nutrient measurements
4. No external validation set (using leave-one-out instead)
5. Cost-loss model assumes fixed decision thresholds

