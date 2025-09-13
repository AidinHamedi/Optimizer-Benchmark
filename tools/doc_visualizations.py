import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

RESULTS_FILE = Path("./results/results.json")
DOCS_SAVE_PATH = Path("./docs/vis")
TEMPLATES_PATH = Path("./tools/templates")
VIS_BASE_URL = (
    "https://github.com/AidinHamedi/ML-Optimizer-Benchmark/raw/vis-ref/results"
)
FILE_FORMAT = ".jpg"


def load_template(template_name: str, templates_path: Path = TEMPLATES_PATH) -> str:
    """Load and cache templates from filesystem."""
    if not hasattr(load_template, "_cache"):
        load_template._cache = {}

    if template_name not in load_template._cache:
        template_file = templates_path / f"{template_name}.html"
        if not template_file.exists():
            raise FileNotFoundError(
                f"Template {template_name}.html not found at {template_file}"
            )
        load_template._cache[template_name] = template_file.read_text(encoding="utf-8")

    return load_template._cache[template_name]


def render_template(template_name: str, **kwargs) -> str:
    """Render a template with the given variables."""
    template = load_template(template_name)
    return template.format(**kwargs)


def calculate_ranks(
    results: Dict[str, Any],
) -> Tuple[Dict[str, Dict[str, int]], Dict[str, int]]:
    """
    Calculate both function-specific and overall ranks for optimizers.

    Returns:
        Tuple of (function_ranks, overall_ranks)
    """
    function_ranks = {}
    all_functions = set()

    for optimizer_data in results["optimizers"].values():
        all_functions.update(optimizer_data["error_rates"].keys())

    for function_name in all_functions:
        function_errors = []
        for optimizer, data in results["optimizers"].items():
            if function_name in data["error_rates"]:
                error_rate = data["error_rates"][function_name]
                function_errors.append((optimizer, error_rate))

        function_errors.sort(key=lambda x: x[1])

        ranks = {}
        current_rank = 1
        prev_error = None

        for i, (optimizer, error) in enumerate(function_errors):
            if error != prev_error:
                current_rank = i + 1
            ranks[optimizer] = current_rank
            prev_error = error

        function_ranks[function_name] = ranks

    optimizers = []
    for optimizer, data in results["optimizers"].items():
        avg_error = data["weighted_avg_error_rate"]
        optimizers.append((optimizer, avg_error))

    optimizers.sort(key=lambda x: x[1])

    overall_ranks = {}
    current_rank = 1
    prev_error = None

    for i, (optimizer, error) in enumerate(optimizers):
        if error != prev_error:
            current_rank = i + 1
        overall_ranks[optimizer] = current_rank
        prev_error = error

    return function_ranks, overall_ranks


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
    static_files = ["styles.css", "script.js"]
    output_path.mkdir(parents=True, exist_ok=True)

    for file_name in static_files:
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

    if verbose:
        print("Loading results data...")
    results = load_results()

    if verbose:
        print("Calculating optimizer ranks...")
    function_ranks, overall_ranks = calculate_ranks(results)

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
        total_optimizers = len(results["optimizers"])

        cards_html = ""
        for function_name, error_rate in optimizer_data["error_rates"].items():
            function_rank = function_ranks[function_name].get(optimizer, "N/A")

            card_html = render_template(
                "card",
                url=generate_image_url(optimizer, function_name),
                name=function_name,
                function_rank=function_rank,
                total_optimizers=total_optimizers,
                error_rate=round(error_rate, 4),
            )
            cards_html += card_html

        page_html = render_template(
            "page",
            title=optimizer,
            cards=cards_html,
            overall_rank=overall_ranks.get(optimizer, "N/A"),
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
