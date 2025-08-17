import tomllib
from pathlib import Path

from pytorch_optimizer import create_optimizer, get_supported_optimizers

from benchmark.evaluate import benchmark_optimizer

CONFIG_DIR = Path("./config.toml")
OUTPUT_DIR = Path("./results")


def read_toml_config(path: Path) -> dict:
    """Read and parse a TOML configuration file into a dictionary."""
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    with file_path.open("rb") as f:
        return tomllib.load(f)


def get_hyperparameter_search_space(name: str, config: dict) -> dict:
    """Return the hyperparameter search space for a given optimizer name."""
    return config.get(name, config["default"])


def main():
    """The entry point of the app."""
    configs = read_toml_config(CONFIG_DIR)
    optimizers = [
        opt
        for opt in get_supported_optimizers()
        if opt not in configs["benchmark"]["ignore_optimizers"]
    ]

    eval_configs = {
        "num_iters": configs["function_iterations"],
        **configs["benchmark"],
    }
    eval_args = configs.get("optimizer_eval_args", {})

    for i, optimizer_name in enumerate(optimizers, start=1):
        search_space = get_hyperparameter_search_space(
            optimizer_name, configs["hyperparameters"]
        )
        print(
            f"({i}/{len(optimizers)}) Processing {optimizer_name}... (Params to tune: {', '.join(search_space.keys())})"
        )

        def get_optimizer(model, optimizer_config):
            return create_optimizer(model, optimizer_name, **optimizer_config)

        benchmark_optimizer(
            get_optimizer,
            optimizer_name,
            OUTPUT_DIR,
            search_space,
            eval_configs,
            eval_args,
        )

    print("Done!")


if __name__ == "__main__":
    main()
