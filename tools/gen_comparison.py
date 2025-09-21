import json
from pathlib import Path

RESULTS_FILE = Path("./results/results.json")
DOCS_DATA_JSON_PATH = Path("./docs/ranks.json")
VIS_BASE_DOCS_URL = "https://aidinhamedi.github.io/Optimizer-Benchmark/vis/"


def load_results(file_path: Path) -> dict:
    """Safely load and parse the results.json file."""
    if not file_path.exists():
        raise FileNotFoundError(f"Results file not found at {file_path}")
    try:
        return json.loads(file_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in results file: {e}")


def calculate_and_sort_rankings(results: dict) -> tuple[list, list]:
    """
    Compute and sort optimizer rankings by average function rank and error rate.
    """
    optimizers_data = results.get("optimizers", {})
    if not optimizers_data:
        return [], []

    all_functions = {
        fn for data in optimizers_data.values() for fn in data.get("error_rates", {})
    }

    function_ranks = {}
    for fn in all_functions:
        errors = [
            (opt, data["error_rates"].get(fn))
            for opt, data in optimizers_data.items()
            if data["error_rates"].get(fn) is not None
        ]
        errors.sort(key=lambda x: x[1])

        ranks = {}
        rank_counter, prev_val = 1, None
        for i, (opt, val) in enumerate(errors):
            if val != prev_val:
                rank_counter = i + 1
            ranks[opt] = rank_counter
            prev_val = val
        function_ranks[fn] = ranks

    avg_ranks = {}
    for opt in optimizers_data:
        ranks = [
            function_ranks[fn][opt]
            for fn in all_functions
            if opt in function_ranks.get(fn, {})
        ]
        avg_ranks[opt] = sum(ranks) / len(ranks) if ranks else float("inf")

    avg_rank_sorted = sorted(avg_ranks.items(), key=lambda x: x[1])

    error_rates = [
        (opt, data.get("weighted_avg_error_rate", float("inf")))
        for opt, data in optimizers_data.items()
    ]
    error_rate_sorted = sorted(error_rates, key=lambda x: x[1])

    return avg_rank_sorted, error_rate_sorted


def generate_website_data(avg_rank_sorted: list, error_rate_sorted: list):
    """Generate and save the ranks.json file for the interactive website."""

    def create_rank_list(sorted_data: list) -> list:
        web_list = []

        for rank, (optimizer, value) in enumerate(sorted_data, start=1):
            if isinstance(value, float) and value != float("inf"):
                display_value = f"{value:.4f}".rstrip("0").rstrip(".")
            elif value == float("inf"):
                display_value = "Failed ⚠️"
            else:
                display_value = str(value)

            web_list.append(
                {
                    "rank": rank,
                    "optimizer": optimizer,
                    "value": display_value,
                    "vis": f"{VIS_BASE_DOCS_URL}{optimizer}",
                }
            )
        return web_list

    web_json_content = {
        "rankingByAvgRank": create_rank_list(avg_rank_sorted),
        "rankingByErrorRate": create_rank_list(error_rate_sorted),
    }

    DOCS_DATA_JSON_PATH.write_text(
        json.dumps(web_json_content, indent=2), encoding="utf-8"
    )
    print(f"Successfully generated website data at: {DOCS_DATA_JSON_PATH}")


def main():
    """Main execution function."""
    print("Starting documentation and web data generation...")
    try:
        results = load_results(RESULTS_FILE)
        avg_rank_sorted, error_rate_sorted = calculate_and_sort_rankings(results)

        generate_website_data(avg_rank_sorted, error_rate_sorted)

        print("Done.")
    except (FileNotFoundError, ValueError) as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
