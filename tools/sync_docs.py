import re
import shutil
from pathlib import Path

INCLUDE_RE = re.compile(r"\{%\s*include\s+([^\s%}]+)\s*%}")
INCLUDE_DIR = Path("./docs-common/includes")
BASE_README_FILE = Path("./docs-common/README.md")
README_FILE = Path("README.md")
GH_PAGES_DIR = Path("./docs")
REPO_MAIN_DIR = Path("./")


def main():
    print("Reading the base README file...")
    try:
        readme_content = BASE_README_FILE.read_text()
    except Exception as e:
        raise FileNotFoundError(f"Failed to read {BASE_README_FILE}: {e}")

    print("Writing the gh-pages README file...")
    GH_PAGES_DIR.joinpath(README_FILE).write_text(readme_content)
    shutil.copytree(INCLUDE_DIR, GH_PAGES_DIR.joinpath("_includes"), dirs_exist_ok=True)

    print("Resolving markdown includes...")
    for match in INCLUDE_RE.finditer(readme_content):
        filename = match.group(1)
        file_path = INCLUDE_DIR / filename
        if file_path.exists():
            try:
                file_content = file_path.read_text(encoding="utf-8")
            except Exception as e:
                raise FileNotFoundError(f"Failed to read {file_path}: {e}")
            print(f" - Resolved include: {filename}")
            readme_content = readme_content.replace(match.group(0), file_content)
        else:
            raise FileNotFoundError(f"Missing include: {filename}")

    print("Writing the main README file...")
    REPO_MAIN_DIR.joinpath(README_FILE).write_text(readme_content)

    print("Done.")


if __name__ == "__main__":
    raise Exception("This script does not work properly. (will be fixed soon)")
    main()
