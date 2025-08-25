import json
from pathlib import Path

RESULTS_FILE = Path("./results/results.json")
DOCS_SAVE_PATH = Path("./docs/vis")
VIS_URL = "https://github.com/AidinHamedi/ML-Optimizer-Benchmark/raw/vis-ref/results"
FILE_FORMAT = ".jpg"


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

    print("Generating visualizations docs...")
    for optimizer in results["optimizers"]:
        print(f" - Generating visualizations for {optimizer}...")
        markdown = f"# {optimizer}\n\n"
        for function in results["optimizers"][optimizer]["error_rates"]:
            path = (VIS_URL + "/" + optimizer + "/" + function + FILE_FORMAT).replace(
                " ", "%20"
            )
            print(f"    - {function} (url: {path})")
            markdown += f"## {function}\n\n"
            markdown += f"![{function}]({path})\n\n"

        DOCS_SAVE_PATH.mkdir(parents=True, exist_ok=True)
        with (DOCS_SAVE_PATH / f"{optimizer}.md").open("w", encoding="utf-8") as f:
            f.write(markdown)

    print("Done.")


if __name__ == "__main__":
    main()
