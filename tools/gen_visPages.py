from .core import (
    DOCS_VIS_DIR,
    FILE_FORMAT,
    TEMPLATES_PATH,
    VIS_PAGE_STATIC_FILES,
    VIS_REPO_URL,
    Console,
    load_results,
)
from .misc import get_aer_ranks, get_afr_ranks, ranks_to_dict

TOOL_NAME = "VIS-PAGE-GEN"


def copy_static_files(console: Console):
    """Copy CSS and JS files to output directory."""
    for file_name in VIS_PAGE_STATIC_FILES:
        src = TEMPLATES_PATH / file_name
        dst = DOCS_VIS_DIR / file_name
        if src.exists():
            dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
            console.info(f"  Copied {file_name} to output directory")


def get_page_template(template_name: str):
    """Load templates from filesystem."""
    template_file = TEMPLATES_PATH / f"{template_name}.html"
    if not template_file.exists():
        raise FileNotFoundError(
            f"Template {template_name}.html not found at {template_file}"
        )
    return template_file.read_text(encoding="utf-8")


def render_template(template_name: str, **kwargs) -> str:
    """Render a template with the given variables."""
    template = get_page_template(template_name)
    return template.format(**kwargs)


def generate_image_url(optimizer: str, function_name: str) -> str:
    """Generate URL for optimizer-function visualization image."""
    url = f"{VIS_REPO_URL}/{optimizer}/{function_name}{FILE_FORMAT}"
    return url.replace(" ", "%20")


def main(console: Console):
    console.info("Generating vis pages...")

    try:
        results = load_results()["optimizers"]
    except Exception:
        console.error("Failed to load optimizer results.")
        return None

    DOCS_VIS_DIR.mkdir(parents=True, exist_ok=True)
    copy_static_files(console)

    afr_rankings, func_rankings = get_afr_ranks(results)
    afr_rankings = ranks_to_dict(afr_rankings)
    aer_rankings = ranks_to_dict(get_aer_ranks(results))

    for i, optimizer in enumerate(func_rankings):
        console.info(f"Generating page for {optimizer}...")

        cards_html = ""

        sorted_functions = sorted(
            func_rankings[optimizer].items(), key=lambda item: item[1]
        )

        for function_name, rank in sorted_functions:
            cards_html += render_template(
                "card",
                url=generate_image_url(optimizer, function_name),
                name=function_name,
                optimizer_name=optimizer,
                function_rank=rank,
            )

        page_html = render_template(
            "page",
            title=optimizer,
            cards=cards_html,
            error_rate_rank=aer_rankings.get(optimizer, "-"),
            avg_rank=afr_rankings.get(optimizer, "-"),
            avg_error_rate=round(
                results[optimizer].get("weighted_avg_error_rate", 0), 4
            ),
            functions_count=len(results[optimizer].get("error_rates", {})),
        )

        output_file = DOCS_VIS_DIR / f"{optimizer}.html"
        output_file.write_text(page_html, encoding="utf-8")

    console.info("Done.")


if __name__ == "__main__":
    console = Console(TOOL_NAME)
    try:
        main(console)
    except ZeroDivisionError as e:
        console.error(f"An error occurred: {e}")
