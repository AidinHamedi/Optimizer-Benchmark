import json
from pathlib import Path

from tabulate import tabulate

RESULTS_FILE = Path("./results/results.json")
DOC_SAVE_PATH = Path("./docs-common/includes/comparison.md")
VIS_BASE_DOCS_URL = "https://aidinhamedi.github.io/Optimizer-Benchmark/vis/"


def calculate_ranks(results):
    """Compute per-function and average ranks for optimizers."""
    # Collect all functions
    all_functions = set()
    for data in results["optimizers"].values():
        all_functions.update(data["error_rates"].keys())

    # Function-wise ranks
    function_ranks = {}
    for fn in all_functions:
        errors = []
        for opt, data in results["optimizers"].items():
            if fn in data["error_rates"]:
                errors.append((opt, data["error_rates"][fn]))
        errors.sort(key=lambda x: x[1])

        ranks = {}
        current_rank, prev_val = 1, None
        for i, (opt, val) in enumerate(errors):
            if val != prev_val:
                current_rank = i + 1
            ranks[opt] = current_rank
            prev_val = val
        function_ranks[fn] = ranks

    # Average rank across functions
    avg_ranks = {}
    for opt in results["optimizers"]:
        ranks = [
            function_ranks[fn][opt] for fn in all_functions if opt in function_ranks[fn]
        ]
        avg_ranks[opt] = sum(ranks) / len(ranks) if ranks else float("inf")

    return avg_ranks


def main():
    print("Reading results...")
    if not RESULTS_FILE.exists():
        raise FileNotFoundError(f"Results file not found at {RESULTS_FILE}")

    try:
        results = json.loads(RESULTS_FILE.read_text(encoding="utf-8"))
    except Exception as e:
        raise Exception(f"Error occurred while opening the results file: {e}")

    print("Generating comparison doc...")

    error_sorted = sorted(
        [
            (
                opt,
                results["optimizers"][opt]["weighted_avg_error_rate"],
                f"[Open]({VIS_BASE_DOCS_URL + opt})",
            )
            for opt in results["optimizers"]
        ],
        key=lambda x: x[1],
    )

    error_table = tabulate(
        [
            (i, opt, round(rate, 4) if rate != float("inf") else "Failed ⚠️", link)
            for i, (opt, rate, link) in enumerate(error_sorted, start=1)
        ],
        headers=["Rank (Error Rate)", "Optimizer", "Avg Error Rate", "Vis"],
        tablefmt="github",
    )

    avg_ranks = calculate_ranks(results)
    avg_sorted = sorted(
        [
            (opt, avg_ranks[opt], f"[Open]({VIS_BASE_DOCS_URL + opt})")
            for opt in results["optimizers"]
        ],
        key=lambda x: x[1],
    )

    avg_table = tabulate(
        [
            (i, opt, round(rank, 2), link)
            for i, (opt, rank, link) in enumerate(avg_sorted, start=1)
        ],
        headers=["Rank (Avg Function Rank)", "Optimizer", "Average Rank", "Vis"],
        tablefmt="github",
    )

    markdown_out = (
        "### Ranking by Error Rate\n\n"
        + error_table
        + "\n\n### Ranking by Avg Function Rank\n\n"
        + avg_table
    )

    DOC_SAVE_PATH.parents[0].mkdir(parents=True, exist_ok=True)
    DOC_SAVE_PATH.write_text(markdown_out)

    print("Done.")


if __name__ == "__main__":
    main()
