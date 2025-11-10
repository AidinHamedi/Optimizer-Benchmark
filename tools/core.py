import json
from pathlib import Path

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


def load_json(json_path: Path) -> dict:
    """Load JSON data from a file."""
    if not json_path.exists():
        raise FileNotFoundError(f"JSON file not found at {json_path}")
    try:
        return json.loads(json_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in file: {e}")


def load_results(json_path: Path = RESULTS_FILE) -> dict:
    """Load results from a JSON file."""
    return load_json(json_path)


class Console:
    """A class for printing messages to the console."""

    def __init__(self, name: str):
        self.name = name

    def info(self, message: object, end: str = "\n"):
        print(f"[{self.name}] INFO: {message}", end=end)

    def warn(self, message: object, end: str = "\n"):
        print(f"[{self.name}] WARNING: {message}", end=end)

    def error(self, message: object, end: str = "\n"):
        print(f"[{self.name}] ERROR: {message}", end=end)
