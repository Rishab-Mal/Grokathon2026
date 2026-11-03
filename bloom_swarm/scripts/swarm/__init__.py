"""Swarm forecast aggregation and evaluation package."""

from .core import (
    CONFIG,
    load_data,
    clip,
    logit,
    sigmoid,
    brier,
    log_loss,
    set_seed,
)

from .aggregators import (
    METHODS,
    PRIMARY_METHOD,
    get_method,
    Aggregator,
)

from .protocols import (
    leave_one_out,
    walk_forward,
    evaluate_all_methods,
    evaluate_agents,
    print_leaderboard,
)

__all__ = [
    "CONFIG",
    "load_data",
    "clip",
    "logit",
    "sigmoid",
    "brier",
    "log_loss",
    "set_seed",
    "METHODS",
    "PRIMARY_METHOD",
    "get_method",
    "Aggregator",
    "leave_one_out",
    "walk_forward",
    "evaluate_all_methods",
    "evaluate_agents",
    "print_leaderboard",
]
