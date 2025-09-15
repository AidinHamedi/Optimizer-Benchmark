import tomllib
from pathlib import Path

import click
from pytorch_optimizer import (
    get_supported_optimizers,
    load_optimizer,
)

from benchmark.evaluate import benchmark_optimizer

CONFIG_PATH = Path("./config.toml")
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


@click.command()
@click.option(
    "--debug",
    is_flag=True,
    help="Enable debug mode",
)
@click.option(
    "--get_num",
    is_flag=True,
    help="Get the number of optimizers",
)
@click.option(
    "--range",
    nargs=2,
    type=int,
    default=[0, -1],
    help="Range of optimizer indices to benchmark (start end). Default: 0 -1 (all)",
)
@click.option(
    "--filter",
    default=None,
    multiple=True,
    help="Filter optimizers by name. Can be used multiple times",
)
@click.option(
    "--functions",
    default=None,
    multiple=True,
    help="Functions to benchmark. Can be used multiple times",
)
@click.option(
    "--results_dir",
    default=OUTPUT_DIR,
    help=f"Output directory for results. Default: {OUTPUT_DIR}",
    type=Path,
)
@click.option(
    "--config_path",
    default=CONFIG_PATH,
    help=f"Path to config file. Default: {CONFIG_PATH}",
    type=Path,
)
def main(
    debug: bool,
    get_num: bool,
    range: list,
    filter: str,
    functions: list,
    results_dir: Path,
    config_path: Path,
):
    """The entry point of the app."""
    configs = read_toml_config(config_path)
    optimizers = [
        opt
        for opt in get_supported_optimizers(filter or None)
        if opt not in configs["benchmark"]["ignore_optimizers"]
    ][range[0] : range[1]]

    if debug:
        print(f"Optimizers: {optimizers}")

    if get_num:
        print(len(optimizers))
        return

    function_iterations = {
        func_name: func_config["iterations"]
        for func_name, func_config in configs["functions"].items()
    }

    eval_configs = {
        "num_iters": function_iterations,
        "error_weights": {
            func_name: func_config["error_weight"]
            for func_name, func_config in configs["functions"].items()
        },
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
                    model.parameters(),  # type: ignore
                    weight_decay=0.0,  # type: ignore
                    **optimizer_config,
                )
            else:
                optimizer = optimizer_class(
                    model.parameters(),  # type: ignore
                    **optimizer_config,
                )

            return optimizer

        benchmark_optimizer(
            get_optimizer,
            optimizer_name,
            results_dir,
            search_space,
            eval_configs,
            eval_args=eval_args,
            functions=functions or None,
            debug=debug,
        )

    print("Done!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted by user.")
