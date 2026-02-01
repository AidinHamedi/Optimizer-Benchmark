
# Commit Message Standard & Changelog

We follow the [Conventional Commits](https://www.conventionalcommits.org/) specification.
This helps us maintain a clean history and automate changelog generation.

## Format

```text
<type>(<scope>): <description>

[optional body]

[optional footer]

```

## 1. Types

| Type | Use Case |
| --- | --- |
| **feat** | A new feature for the user (e.g., a new scoring metric). |
| **fix** | A bug fix (e.g., fixing a PyTorch device mismatch). |
| **docs** | Documentation changes only. |
| **style** | Formatting, missing semi-colons, etc. (No code changes). |
| **refactor** | Code change that neither fixes a bug nor adds a feature. |
| **perf** | Code change that improves performance. |
| **test** | Adding missing tests or correcting existing tests. |
| **chore** | Updating `uv` dependencies, `pyproject.toml`, or CI/CD. |

## 2. Scopes

Based on our `src/optimizer_vis` structure, please use one of the following scopes:

* **core**: Changes to the main engine or top-level `__init__.py`.
* **tuning**: Anything related to hyperparameter optimization or LR finders.
* **scoring**: Changes to path scoring, metrics, or trajectory analysis.
* **plotting**: Visualization, landscape plots, or matplotlib/plotly logic.
* **io**: Saving/loading logic and result handlers.
* **types**: Changes specifically to `_types.py` files across the library.
* **deps**: Used with `chore` for updating `pyproject.toml` or `uv.lock`.

## How to Document Changes

### 1. The Changelog

When submitting a Pull Request, you must update the `CHANGELOG.md` file:

* Place your changes under the `## [Unreleased]` header.
* Use one of the following sub-headers:
    * `Added` (for new features)
    * `Changed` (for changes in existing functionality)
    * `Deprecated` (for soon-to-be removed features)
    * `Removed` (for now removed features)
    * `Fixed` (for any bug fixes)
    * `Security` (in case of vulnerabilities)

### 2. Pull Request Descriptions

Your PR title should follow the [Commit Message Standard](#commit-message-standard). 
In the description, include:

* **Task:** What was the goal?
* **Solution:** How did you solve it?
* **Breaking Changes:** Does this break existing project workflows?
