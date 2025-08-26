import json
from pathlib import Path

from tabulate import tabulate

RESULTS_FILE = Path("./results/results.json")
DOC_SAVE_PATH = Path("./docs/comparison.md")
VIS_BASE_DOCS_URL = "https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/"


def main():
    print("Reading results...")
    if RESULTS_FILE.exists():
        try:
            with RESULTS_FILE.open("r", encoding="utf-8") as f:
                results = json.load(f)
        except Exception as e:
            raise Exception(f"Error occurred while opening the results file: {e}")
    else:
        raise FileNotFoundError(f"Results file not found at {RESULTS_FILE}")

    print("Generating comparison doc...")

    avg_error_rates = sorted(
        [
            (
                optimizer,
                results["optimizers"][optimizer]["avg_error_rate"],
                f"[Open]({VIS_BASE_DOCS_URL + optimizer})",
            )
            for optimizer in results["optimizers"]
        ],
        key=lambda x: x[1],
    )

    markdown_table = tabulate(
        [
            (i + 1, opt, rate, link)
            for i, (opt, rate, link) in enumerate(avg_error_rates)
        ],
        headers=["Rank", "Optimizer", "Average Error Rate", "Vis"],
        tablefmt="github",
    )

    DOC_SAVE_PATH.write_text(markdown_table)

    print("Done.")


if __name__ == "__main__":
    main()
