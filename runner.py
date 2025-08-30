import tomllib
from pathlib import Path

from pytorch_optimizer import (
    get_optimizer_parameters,
    get_supported_optimizers,
    load_optimizer,
)

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

        def get_optimizer(model, optimizer_config, num_iters):
            if optimizer_name == "adashift":
                optimizer_config["keep_num"] = 1
            elif optimizer_name == "ranger21":
                optimizer_config["num_iterations"] = num_iters
            elif optimizer_name == "ranger25":
                optimizer_config["orthograd"] = False

            optimizer_class = load_optimizer(optimizer_name)

            if optimizer_name == "adammini":
                optimizer = optimizer_class(model, weight_decay=0.0, **optimizer_config)  # type: ignore
            elif optimizer_name in [
                "adagc",
                "adalite",
                "adammini",
                "adams",
                "bsam",
                "emofact",
                "emolynx",
                "emonavi",
                "emoneco",
                "emozeal",
                "fadam",
                "ranger21",
                "sgdsai",
                "soap",
                "splus",
                "tiger",
            ]:
                optimizer = optimizer_class(
                    get_optimizer_parameters(model, 0, []),  # type: ignore
                    weight_decay=0.0,  # type: ignore
                    **optimizer_config,
                )
            else:
                optimizer = optimizer_class(
                    get_optimizer_parameters(model, 0, []),  # type: ignore
                    **optimizer_config,
                )

            return optimizer

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
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted by user.")
