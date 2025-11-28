def get_afr_ranks(results: dict) -> tuple[list, dict]:
    """Get Ranks Based On Average Rank On Functions"""
    functions = {fn for data in results.values() for fn in data.get("error_rates", {})}

    function_ranks = {opt: {} for opt in results}

    for fn in functions:
        errors = [
            (opt, data["error_rates"][fn])
            for opt, data in results.items()
            if fn in data["error_rates"]
        ]

        errors.sort(key=lambda x: x[1])

        rank = 1
        prev_val = None
        for i, (opt, val) in enumerate(errors, start=1):
            if val != prev_val:
                rank = i
            function_ranks[opt][fn] = rank
            prev_val = val

    avg_ranks = {}
    for opt, ranks in function_ranks.items():
        avg_ranks[opt] = (sum(ranks.values()) / len(ranks)) if ranks else float("inf")

    return sorted(avg_ranks.items(), key=lambda x: x[1]), function_ranks


def get_aer_ranks(optimizers: dict, weights: dict) -> list:
    """Get Ranks Based On Average Error Rate On Functions"""
    optimizer_scores = []

    for opt, data in optimizers.items():
        error_rates = data.get("error_rates", {})
        if not error_rates:
            optimizer_scores.append((opt, float("inf")))
            continue

        weighted_sum = sum(
            error * weights.get(func, 1.0) for func, error in error_rates.items()
        )
        avg_error = weighted_sum / len(error_rates)
        optimizer_scores.append((opt, avg_error))

    return sorted(optimizer_scores, key=lambda x: x[1])


def ranks_to_dict(rank_output):
    """
    Convert output of get_afr_ranks() or get_aer_ranks()
    to a dictionary {optimizer_name: rank}.
    """
    return {opt: i for i, (opt, _) in enumerate(rank_output, start=1)}
