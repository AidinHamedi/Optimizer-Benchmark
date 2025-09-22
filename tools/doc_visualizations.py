import json
from pathlib import Path
from typing import Any, Dict, Tuple

RESULTS_FILE = Path("./results/results.json")
DOCS_SAVE_PATH = Path("./docs/vis")
TEMPLATES_PATH = Path("./tools/templates")
STATIC_FILES = ("styles.css", "script.js")
VIS_BASE_URL = "https://github.com/AidinHamedi/Optimizer-Benchmark/raw/vis-ref/results"
FILE_FORMAT = ".jpg"


def load_template(template_name: str, templates_path: Path = TEMPLATES_PATH) -> str:
    """Load templates from filesystem."""
    template_file = templates_path / f"{template_name}.html"
    if not template_file.exists():
        raise FileNotFoundError(
            f"Template {template_name}.html not found at {template_file}"
        )
    return template_file.read_text(encoding="utf-8")


def render_template(template_name: str, **kwargs) -> str:
    """Render a template with the given variables."""
    template = load_template(template_name)
    return template.format(**kwargs)


def calculate_ranks(
    results: Dict[str, Any],
) -> Tuple[Dict[str, Dict[str, int]], Dict[str, int], Dict[str, int]]:
    """
    Calculates three types of ranks for optimizers:
    1.  function_ranks: Rank of each optimizer on each specific function.
    2.  error_rate_ranks: Overall rank based on the final weighted average error.
    3.  avg_rank_ranks: Overall rank based on the average rank across all functions.

    This dual-ranking system provides a more nuanced view of performance.
    """
    function_ranks = {}
    all_functions = set()

    # First, gather all unique function names from the results.
    for optimizer_data in results["optimizers"].values():
        all_functions.update(optimizer_data["error_rates"].keys())

    # 1. Calculate per-function ranks.
    for function_name in all_functions:
        function_errors = []
        for optimizer, data in results["optimizers"].items():
            if function_name in data["error_rates"]:
                error_rate = data["error_rates"][function_name]
                function_errors.append((optimizer, error_rate))

        # Sort optimizers by their error rate for the current function.
        function_errors.sort(key=lambda x: x[1])
        ranks = {}
        current_rank = 1
        prev_error = None
        # Assign ranks, correctly handling ties. If two optimizers have the same
        # error, they receive the same rank, and the next rank is incremented accordingly.
        for i, (optimizer, error) in enumerate(function_errors):
            if error != prev_error:
                current_rank = i + 1
            ranks[optimizer] = current_rank
            prev_error = error
        function_ranks[function_name] = ranks

    # 2. Calculate overall rank based on weighted average error rate.
    optimizers = [
        (opt, data["weighted_avg_error_rate"])
        for opt, data in results["optimizers"].items()
    ]
    optimizers.sort(key=lambda x: x[1])

    error_rate_ranks = {}
    current_rank = 1
    prev_error = None
    # Assign ranks with tie handling, same as above.
    for i, (optimizer, error) in enumerate(optimizers):
        if error != prev_error:
            current_rank = i + 1
        error_rate_ranks[optimizer] = current_rank
        prev_error = error

    # 3. Calculate overall rank based on the average of per-function ranks.
    avg_ranks = []
    for optimizer in results["optimizers"]:
        ranks = [
            function_ranks[f][optimizer]
            for f in all_functions
            if optimizer in function_ranks[f]
        ]
        # This provides a different perspective: an optimizer that is consistently
        # good (e.g., always 5th place) might rank higher than one that is brilliant
        # on a few functions but fails on others.
        avg_rank = sum(ranks) / len(ranks) if ranks else float("inf")
        avg_ranks.append((optimizer, avg_rank))

    avg_ranks.sort(key=lambda x: x[1])
    avg_rank_ranks = {}
    current_rank = 1
    prev_val = None
    # Assign final ranks with tie handling.
    for i, (optimizer, val) in enumerate(avg_ranks):
        if val != prev_val:
            current_rank = i + 1
        avg_rank_ranks[optimizer] = current_rank
        prev_val = val

    return function_ranks, error_rate_ranks, avg_rank_ranks


def load_results(results_file: Path = RESULTS_FILE) -> Dict[str, Any]:
    """Load and parse the results JSON file."""
    if not results_file.exists():
        raise FileNotFoundError(f"Results file not found at {results_file.absolute()}")

    try:
        with results_file.open("r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in results file: {e}")
    except Exception as e:
        raise Exception(f"Error reading results file: {e}")


def generate_image_url(optimizer: str, function_name: str) -> str:
    """Generate URL for optimizer-function visualization image."""
    url = f"{VIS_BASE_URL}/{optimizer}/{function_name}{FILE_FORMAT}"
    return url.replace(" ", "%20")


def copy_static_files(
    templates_path: Path = TEMPLATES_PATH, output_path: Path = DOCS_SAVE_PATH
):
    """Copy CSS and JS files to output directory."""
    output_path.mkdir(parents=True, exist_ok=True)

    for file_name in STATIC_FILES:
        src = templates_path / file_name
        dst = output_path / file_name
        if src.exists():
            dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
            print(f"  Copied {file_name} to output directory")


def generate_visualizations(verbose: bool = True):
    """Main function to generate all visualization pages."""
    if verbose:
        print("Starting visualization generation...")
        print(f"Results file: {RESULTS_FILE.absolute()}")
        print(f"Output directory: {DOCS_SAVE_PATH.absolute()}")

    results = load_results()

    if verbose:
        print("Calculating optimizer ranks...")

    function_ranks, error_rate_ranks, avg_rank_ranks = calculate_ranks(results)

    DOCS_SAVE_PATH.mkdir(parents=True, exist_ok=True)

    if verbose:
        print("Copying static files...")

    copy_static_files()

    if verbose:
        print(f"Generating pages for {len(results['optimizers'])} optimizers...")

    for i, optimizer in enumerate(results["optimizers"]):
        if verbose:
            print(
                f"  [{i + 1}/{len(results['optimizers'])}] Generating page for {optimizer}..."
            )

        optimizer_data = results["optimizers"][optimizer]

        cards_html = ""
        for function_name, error_rate in optimizer_data["error_rates"].items():
            function_rank = function_ranks[function_name].get(optimizer, "N/A")

            card_html = render_template(
                "card",
                url=generate_image_url(optimizer, function_name),
                name=function_name,
                optimizer_name=optimizer,
                function_rank=function_rank,
            )
            cards_html += card_html

        page_html = render_template(
            "page",
            title=optimizer,
            cards=cards_html,
            error_rate_rank=error_rate_ranks.get(optimizer, "N/A"),
            avg_rank=avg_rank_ranks.get(optimizer, "N/A"),
            avg_error_rate=round(optimizer_data["weighted_avg_error_rate"], 4),
            functions_count=len(optimizer_data["error_rates"]),
        )

        output_file = DOCS_SAVE_PATH / f"{optimizer}.html"
        output_file.write_text(page_html, encoding="utf-8")

    if verbose:
        print("Visualization generation completed successfully!")
        print(
            f"Generated {len(results['optimizers'])} pages in {DOCS_SAVE_PATH.absolute()}"
        )


def main():
    """Entry point for the script."""
    try:
        generate_visualizations(verbose=True)
    except Exception as e:
        print(f"Error generating visualizations: {e}")
        raise


if __name__ == "__main__":
    main()
