"""Generate the optimizer comparison ranking JSON file."""

import json

from .core import RANKS_FILE, VIS_WEBPAGE_BASE_URL, Console, load_results
from .misc import get_avg_ranks, get_function_ranks, get_weighted_avg_ranks

TOOL_NAME = "COMP-GEN"


def get_ranks_json(
    avg_rankings: list[tuple[str, float]],
    weighted_rankings: list[tuple[str, float]],
    console: Console,
) -> str:
    """Generate JSON string containing both ranking systems."""

    def _create_list(ranks: list[tuple[str, float]]) -> list[dict[str, str | int]]:
        entries = []

        for rank, (optimizer, value) in enumerate(ranks, start=1):
            if isinstance(value, float) and value != float("inf"):
                display_value = f"{value:.4f}".rstrip("0").rstrip(".")
            else:
                display_value = "Failed ⚠️"
                console.error(f"Optimizer {optimizer} invalid rank value: {value}")

            entries.append(
                {
                    "rank": rank,
                    "optimizer": optimizer,
                    "value": display_value,
                    "vis": f"{VIS_WEBPAGE_BASE_URL}{optimizer}",
                }
            )

        return entries

    return json.dumps(
        {
            "rankingByAvgRank": _create_list(avg_rankings),
            "rankingByWeightedRank": _create_list(weighted_rankings),
        }
    )


def main(console: Console) -> None:
    """Generate and save the comparison rankings JSON file."""
    console.info("Generating comparison...")

    try:
        data = load_results()
        optimizers = data["optimizers"]
        weights = data["functions"]["weights"]
    except Exception as e:
        console.error(f"Failed to load results: {e}")
        return None

    # Calculate raw ranks first
    function_ranks = get_function_ranks(optimizers)

    # Calculate aggregated metrics
    avg_ranks = get_avg_ranks(function_ranks)
    weighted_ranks = get_weighted_avg_ranks(function_ranks, weights)

    json_data = get_ranks_json(avg_ranks, weighted_ranks, console)

    RANKS_FILE.write_text(json_data, encoding="utf-8")

    console.info("Done.")


if __name__ == "__main__":
    console = Console(TOOL_NAME)
    try:
        main(console)
    except Exception as e:
        console.error(f"An error occurred: {e}")
