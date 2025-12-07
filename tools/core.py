"""Core utilities and constants for the tools module."""

import json
from pathlib import Path
from typing import Any

DOCS_DIR = Path("./docs")
DOCS_VIS_DIR = Path("./docs/vis")
RANKS_FILE = Path("./docs/ranks.json")
RESULTS_FILE = Path("./results/results.json")
SITEMAP_FILE = Path("./docs/sitemap_index.xml")
WEBSITE_URL = "https://aidinhamedi.github.io/Optimizer-Benchmark/"
VIS_WEBPAGE_BASE_URL = WEBSITE_URL + "vis/"
VIS_REPO_URL = (
    "https://cdn.statically.io/gh/AidinHamedi/Optimizer-Benchmark@vis-base/results"
)
TEMPLATES_PATH = Path("./tools/templates")
VIS_PAGE_STATIC_FILES = ("styles.css", "script.js")
FILE_FORMAT = ".jpg"


def load_json(json_path: Path) -> dict[str, Any]:
    """Load and parse a JSON file."""
    if not json_path.exists():
        raise FileNotFoundError(f"JSON file not found at {json_path}")
    try:
        return json.loads(json_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in file: {e}")


def load_results(json_path: Path = RESULTS_FILE) -> dict[str, Any]:
    """Load benchmark results from the results JSON file."""
    return load_json(json_path)


class Console:
    """Simple console logger with prefixed output."""

    def __init__(self, name: str) -> None:
        self.name = name

    def info(self, message: object, end: str = "\n") -> None:
        print(f"[{self.name}] INFO: {message}", end=end)

    def warn(self, message: object, end: str = "\n") -> None:
        print(f"[{self.name}] WARNING: {message}", end=end)

    def error(self, message: object, end: str = "\n") -> None:
        print(f"[{self.name}] ERROR: {message}", end=end)
