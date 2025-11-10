import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from datetime import date

import requests

from .core import (
    SITEMAP_FILE,
    VIS_WEBPAGE_BASE_URL,
    WEBSITE_URL,
    Console,
    load_results,
)

TOOL_NAME = "SITEMAP"
SITEMAP_ROOT = ET.Element("urlset", xmlns="http://www.sitemaps.org/schemas/sitemap/0.9")


def is_url_accessible(url: str, timeout: int = 5) -> bool:
    """Check if a URL is accessible (HTTP 200)."""
    try:
        response = requests.head(url, allow_redirects=True, timeout=timeout)
        return response.status_code == 200
    except requests.RequestException:
        return False


def add_url(loc, lastmod=None, priority=None):
    """Add a URL to the sitemap."""
    url = ET.SubElement(SITEMAP_ROOT, "url")
    ET.SubElement(url, "loc").text = loc
    if lastmod:
        ET.SubElement(url, "lastmod").text = lastmod
    if priority:
        ET.SubElement(url, "priority").text = str(priority)


def main(console: Console):
    console.info("Generating sitemap...")

    try:
        results = load_results()["optimizers"]
    except Exception:
        console.error("Failed to load optimizer results.")
        return None

    add_url(WEBSITE_URL, lastmod=str(date.today()), priority=1.0)

    with ThreadPoolExecutor(max_workers=32) as executor:
        executor.map(
            lambda name: (
                add_url(
                    f"{VIS_WEBPAGE_BASE_URL}{name}",
                    lastmod=str(date.today()),
                    priority=0.8,
                )
                if is_url_accessible(f"{VIS_WEBPAGE_BASE_URL}{name}")
                else console.warn(
                    f"URL {VIS_WEBPAGE_BASE_URL}{name} is not accessible."
                )
            ),
            results,
        )

    tree = ET.ElementTree(SITEMAP_ROOT)
    tree.write(SITEMAP_FILE, encoding="utf-8", xml_declaration=True)

    console.info("Done.")


if __name__ == "__main__":
    console = Console(TOOL_NAME)
    try:
        main(console)
    except Exception as e:
        console.error(f"An error occurred: {e}")
