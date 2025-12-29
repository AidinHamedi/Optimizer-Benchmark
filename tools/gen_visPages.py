"""Generate individual HTML pages for each optimizer's visualization results."""

from .core import (
    DOCS_VIS_DIR,
    FILE_FORMAT,
    TEMPLATES_PATH,
    VIS_PAGE_STATIC_FILES,
    VIS_REPO_URL,
    Console,
    load_results,
)
from .misc import (
    get_avg_ranks,
    get_function_ranks,
    get_weighted_avg_ranks,
    ranks_to_dict,
)

TOOL_NAME = "VIS-PAGE-GEN"


def copy_static_files(console: Console) -> None:
    """Copy static CSS and JS files to the output directory."""
    for file_name in VIS_PAGE_STATIC_FILES:
        src = TEMPLATES_PATH / file_name
        dst = DOCS_VIS_DIR / file_name
        if src.exists():
            dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
            console.info(f"  Copied {file_name} to output directory")


def get_page_template(template_name: str) -> str:
    """Load an HTML template from the templates directory."""
    template_file = TEMPLATES_PATH / f"{template_name}.html"
    if not template_file.exists():
        raise FileNotFoundError(
            f"Template {template_name}.html not found at {template_file}"
        )
    return template_file.read_text(encoding="utf-8")


def render_template(template_name: str, **kwargs) -> str:
    """Render a template with variable substitution."""
    template = get_page_template(template_name)
    return template.format(**kwargs)


def main(console: Console) -> None:
    """Generate HTML visualization pages for all optimizers."""
    console.info("Generating vis pages...")

    try:
        data = load_results()
        optimizers = data["optimizers"]
        weights = data["functions"]["weights"]
    except Exception as e:
        console.error(f"Failed to load results: {e}")
        return None

    DOCS_VIS_DIR.mkdir(parents=True, exist_ok=True)
    copy_static_files(console)

    # Compute rankings
    func_ranks = get_function_ranks(optimizers)
    avg_rank_list = get_avg_ranks(func_ranks)
    weighted_rank_list = get_weighted_avg_ranks(func_ranks, weights)

    # Create lookups
    avg_rank_lookup = ranks_to_dict(avg_rank_list)
    weighted_rank_pos_lookup = ranks_to_dict(weighted_rank_list)

    # Create lookup for the actual weighted rank score
    weighted_score_lookup = {opt: score for opt, score in weighted_rank_list}

    for _, optimizer in enumerate(optimizers):
        console.info(f"Generating page for {optimizer}...")

        cards_html = ""

        current_func_ranks = func_ranks.get(optimizer, {})
        sorted_functions = sorted(current_func_ranks.items(), key=lambda item: item[1])

        for function_name, rank in sorted_functions:
            safe_opt = optimizer.replace(" ", "%20")
            safe_func = function_name.replace(" ", "%20")

            main_image_url = (
                f"{VIS_REPO_URL}/{safe_opt}/{safe_func}/surface{FILE_FORMAT}"
            )

            cards_html += render_template(
                "card",
                main_image_url=main_image_url,
                name=function_name,
                optimizer_id=safe_opt,
                function_id=safe_func,
                function_rank=rank,
                base_url=VIS_REPO_URL,
                ext=FILE_FORMAT,
            )

        page_html = render_template(
            "page",
            title=optimizer,
            cards=cards_html,
            weighted_rank_pos=weighted_rank_pos_lookup.get(optimizer, "-"),
            avg_rank=avg_rank_lookup.get(optimizer, "-"),
            weighted_rank_score=round(weighted_score_lookup.get(optimizer, 0.0), 4),
            functions_count=len(current_func_ranks),
        )

        output_file = DOCS_VIS_DIR / f"{optimizer}.html"
        output_file.write_text(page_html, encoding="utf-8")

    console.info("Done.")


if __name__ == "__main__":
    console = Console(TOOL_NAME)
    try:
        main(console)
    except Exception as e:
        console.error(f"An error occurred: {e}")
