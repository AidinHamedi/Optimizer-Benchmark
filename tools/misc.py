"""Utility functions for computing optimizer rankings."""

from typing import Any


def get_afr_ranks(
    results: dict[str, Any],
) -> tuple[list[tuple[str, float]], dict[str, dict[str, int]]]:
    """Compute rankings based on average rank across test functions."""
    # Collect all function names across all optimizers
    functions = {fn for data in results.values() for fn in data.get("error_rates", {})}

    function_ranks: dict[str, dict[str, int]] = {opt: {} for opt in results}

    # Rank optimizers within each function
    for fn in functions:
        errors = [
            (opt, data["error_rates"][fn])
            for opt, data in results.items()
            if fn in data["error_rates"]
        ]

        errors.sort(key=lambda x: x[1])

        # Competition ranking: tied optimizers get the same rank
        rank = 1
        prev_val = None
        for i, (opt, val) in enumerate(errors, start=1):
            if val != prev_val:
                rank = i
            function_ranks[opt][fn] = rank
            prev_val = val

    # Average each optimizer's ranks across all functions
    avg_ranks: dict[str, float] = {}
    for opt, ranks in function_ranks.items():
        avg_ranks[opt] = (sum(ranks.values()) / len(ranks)) if ranks else float("inf")

    return sorted(avg_ranks.items(), key=lambda x: x[1]), function_ranks


def get_aer_ranks(
    optimizers: dict[str, Any], weights: dict[str, float]
) -> list[tuple[str, float]]:
    """Compute rankings based on weighted average error rate."""
    optimizer_scores: list[tuple[str, float]] = []

    for opt, data in optimizers.items():
        error_rates = data.get("error_rates", {})
        if not error_rates:
            optimizer_scores.append((opt, float("inf")))
            continue

        total_weight = 0.0
        weighted_sum = 0.0

        for func, error in error_rates.items():
            w = weights.get(func, 1.0)
            weighted_sum += error * w
            total_weight += w

        avg_error = weighted_sum / total_weight if total_weight > 0 else float("inf")
        optimizer_scores.append((opt, avg_error))

    return sorted(optimizer_scores, key=lambda x: x[1])


def ranks_to_dict(rank_output: list[tuple[str, float]]) -> dict[str, int]:
    """Convert a sorted ranking list to a {name: rank} lookup dict."""
    return {opt: i for i, (opt, _) in enumerate(rank_output, start=1)}
