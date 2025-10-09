import json

from .core import RANKS_FILE, VIS_WEBPAGE_BASE_URL, Console, load_results
from .misc import get_aer_ranks, get_afr_ranks

TOOL_NAME = "COMP-GEN"


def get_ranks_json(afr_rankings: list, aer_rankings: list, console: Console) -> str:
    def _create_list(ranks: list) -> list:
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
            "rankingByAvgRank": _create_list(afr_rankings),
            "rankingByErrorRate": _create_list(aer_rankings),
        }
    )


def main(console: Console):
    console.info("Generating comparison...")

    try:
        results = load_results()["optimizers"]
    except Exception:
        console.error("Failed to load optimizer results.")
        return None

    json_data = get_ranks_json(
        get_afr_ranks(results)[0], get_aer_ranks(results), console
    )

    RANKS_FILE.write_text(json_data, encoding="utf-8")

    console.info("Done.")


if __name__ == "__main__":
    console = Console(TOOL_NAME)
    try:
        main(console)
    except Exception as e:
        console.error(f"An error occurred: {e}")
