import json
import re
from pathlib import Path

from tabulate import tabulate

RESULTS_FILE = Path("./results/results.json")
README_BASE_PATH = Path("./readme-base/README.md")
README_OUTPUT_PATH = Path("./README.md")
DOCS_DATA_JSON_PATH = Path("./docs/ranks.json")
VIS_BASE_DOCS_URL = "https://aidinhamedi.github.io/Optimizer-Benchmark/vis/"
README_TABLE_PLACEHOLDER = "{%comparison%}"


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


def update_readme_file(avg_rank_sorted: list, error_rate_sorted: list):
    """Generate markdown tables and update the root README.md file."""

    def create_markdown_table(sorted_data: list, headers: list) -> str:
        table_data = []

        for rank, (optimizer, value) in enumerate(sorted_data, start=1):
            if isinstance(value, float) and value != float("inf"):
                display_value = f"{value:.2f}"
            elif value == float("inf"):
                display_value = "Failed ⚠️"
            else:
                display_value = str(value)

            vis_link = f"[Open]({VIS_BASE_DOCS_URL}{optimizer})"
            table_data.append([rank, optimizer, display_value, vis_link])

        return tabulate(table_data, headers=headers, tablefmt="github")

    avg_table = create_markdown_table(
        avg_rank_sorted,
        ["Rank (Avg Function Rank)", "Optimizer", "Average Rank", "Vis"],
    )
    error_table = create_markdown_table(
        error_rate_sorted, ["Rank (Error Rate)", "Optimizer", "Avg Error Rate", "Vis"]
    )

    comparison_block = f"""<h4>
<details open>
<summary>Ranking by Avg Function Rank ⚡</summary>
<h6>

{avg_table}

</h6>
</details>
</h4>

<h4>
<details>
<summary>Ranking by Error Rate ⚡</summary>
<h6>

{error_table}

</h6>
</details>
</h4>"""

    base_readme_content = README_BASE_PATH.read_text(encoding="utf-8")
    final_readme = base_readme_content.replace(
        README_TABLE_PLACEHOLDER, comparison_block
    )

    README_OUTPUT_PATH.write_text(final_readme, encoding="utf-8")
    print(f"Successfully updated main README at: {README_OUTPUT_PATH}")


def main():
    """Main execution function."""
    print("Starting documentation and web data generation...")
    try:
        results = load_results(RESULTS_FILE)
        avg_rank_sorted, error_rate_sorted = calculate_and_sort_rankings(results)

        generate_website_data(avg_rank_sorted, error_rate_sorted)
        update_readme_file(avg_rank_sorted, error_rate_sorted)

        print("All tasks completed successfully.")
    except (FileNotFoundError, ValueError) as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
