import json
import xml.etree.ElementTree as ET
from datetime import date
from pathlib import Path

RESULTS_FILE = Path("./results/results.json")
SITEMAP_FILE = Path("./docs/sitemap.xml")
MAIN_PAGE_URL = "https://aidinhamedi.github.io/Optimizer-Benchmark/"
VIS_BASE_DOCS_URL = "https://aidinhamedi.github.io/Optimizer-Benchmark/vis/"
URLSET = ET.Element("urlset", xmlns="http://www.sitemaps.org/schemas/sitemap/0.9")


def add_url(loc, lastmod=None, priority=None):
    url = ET.SubElement(URLSET, "url")
    ET.SubElement(url, "loc").text = loc
    if lastmod:
        ET.SubElement(url, "lastmod").text = lastmod
    if priority:
        ET.SubElement(url, "priority").text = str(priority)


def load_results(file_path: Path) -> dict:
    """Safely load and parse the results.json file."""
    if not file_path.exists():
        raise FileNotFoundError(f"Results file not found at {file_path}")
    try:
        return json.loads(file_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in results file: {e}")


def main():
    """Main execution function."""
    print("Generating sitemap...")
    try:
        results = load_results(RESULTS_FILE)

        add_url(MAIN_PAGE_URL, lastmod=str(date.today()), priority=1.0)

        for optimizer_name in results["optimizers"]:
            add_url(
                f"{VIS_BASE_DOCS_URL}{optimizer_name}",
                lastmod=str(date.today()),
                priority=0.8,
            )

        tree = ET.ElementTree(URLSET)
        tree.write(SITEMAP_FILE, encoding="utf-8", xml_declaration=True)

        print("Done.")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
