"""Utility functions for computing optimizer rankings."""

from typing import Any


def get_function_ranks(
    results: dict[str, Any],
) -> dict[str, dict[str, int]]:
    """Compute ranking of optimizers for each function based on error rate.

    Returns:
        A dictionary mapping optimizer names to a dict of {function_name: rank}.
    """
    # Collect all function names across all optimizers
    functions = {fn for data in results.values() for fn in data.get("error_rates", {})}

    function_ranks: dict[str, dict[str, int]] = {opt: {} for opt in results}

    # Rank optimizers within each function
    for fn in functions:
        # Get list of (optimizer, error) for this function
        # Filter out optimizers that don't have results for this function
        errors = [
            (opt, data["error_rates"][fn])
            for opt, data in results.items()
            if fn in data["error_rates"]
        ]

        # Sort by error (lower is better)
        errors.sort(key=lambda x: x[1])

        # Competition ranking: tied optimizers get the same rank
        current_rank = 1
        prev_val = None
        for i, (opt, val) in enumerate(errors, start=1):
            if val != prev_val:
                current_rank = i
            function_ranks[opt][fn] = current_rank
            prev_val = val

    return function_ranks


def get_weighted_avg_ranks(
    function_ranks: dict[str, dict[str, int]], weights: dict[str, float]
) -> list[tuple[str, float]]:
    """Compute rankings based on weighted average rank.

    Returns:
        A sorted list of (optimizer_name, weighted_avg_rank).
    """
    optimizer_scores: list[tuple[str, float]] = []

    for opt, ranks in function_ranks.items():
        if not ranks:
            optimizer_scores.append((opt, float("inf")))
            continue

        total_weight = 0.0
        weighted_rank_sum = 0.0

        for func, rank in ranks.items():
            w = weights.get(func, 1.0)
            weighted_rank_sum += rank * w
            total_weight += w

        # Calculate weighted average
        avg_rank = (
            weighted_rank_sum / total_weight if total_weight > 0 else float("inf")
        )
        optimizer_scores.append((opt, avg_rank))

    return sorted(optimizer_scores, key=lambda x: x[1])


def get_avg_ranks(
    function_ranks: dict[str, dict[str, int]],
) -> list[tuple[str, float]]:
    """Compute rankings based on simple average rank (unweighted).

    Returns:
        A sorted list of (optimizer_name, avg_rank).
    """
    avg_ranks: dict[str, float] = {}
    for opt, ranks in function_ranks.items():
        avg_ranks[opt] = (sum(ranks.values()) / len(ranks)) if ranks else float("inf")

    return sorted(avg_ranks.items(), key=lambda x: x[1])


def ranks_to_dict(rank_output: list[tuple[str, float]]) -> dict[str, int]:
    """Convert a sorted ranking list to a {name: rank_position} lookup dict."""
    return {opt: i for i, (opt, _) in enumerate(rank_output, start=1)}
