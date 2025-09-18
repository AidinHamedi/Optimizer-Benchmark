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

OPTIMIZER_PATCHES = {
    "adashift": lambda cfg, iters: cfg.update({"keep_num": 1}),
    "ranger21": lambda cfg, iters: cfg.update({"num_iterations": iters}),
    "ranger25": lambda cfg, iters: cfg.update({"orthograd": False}),
}

SPECIAL_WEIGHT_DECAY_OPTIMIZERS = {
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
}


def read_toml_config(path: Path) -> dict:
    """Read and parse a TOML configuration file into a dictionary."""
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    with file_path.open("rb") as f:
        return tomllib.load(f)


def normalize_multi_option(values: tuple[str]) -> list[str] | None:
    """Normalize CLI multi-options: () -> None, ('x',) -> ['x'], ('x','y') -> ['x','y']."""
    if not values:
        return None
    return list(values)


def get_hyperparameter_search_space(name: str, config: dict) -> dict:
    """Return the hyperparameter search space for a given optimizer name."""
    base = config.get(name, config["base"])
    search_space = dict(base)
    search_space.pop("_iter_scale", None)
    return search_space


def get_optimizer_factory(optimizer_name: str, debug: bool = False):
    """Return a function that instantiates the optimizer."""

    def factory(model, optimizer_config: dict, num_iters: int):
        patch = OPTIMIZER_PATCHES.get(optimizer_name)
        if patch:
            if debug:
                print(f"Applying patch for {optimizer_name}")
            patch(optimizer_config, num_iters)

        optimizer_class = load_optimizer(optimizer_name)

        if optimizer_name == "adammini":
            return optimizer_class(model, weight_decay=0.0, **optimizer_config)  # type: ignore
        elif optimizer_name in SPECIAL_WEIGHT_DECAY_OPTIMIZERS:
            if debug:
                print(f"Creating {optimizer_name} (with weight decay mod)")
            return optimizer_class(
                model.parameters(),
                weight_decay=0.0,  # type: ignore
                **optimizer_config,
            )
        else:
            if debug:
                print(f"Creating {optimizer_name}")
            return optimizer_class(model.parameters(), **optimizer_config)

    return factory


def prepare_optimizers(
    configs: dict, filters: list[str] | None, opt_range: list[int]
) -> list[str]:
    """Prepare list of optimizers to benchmark, applying filters and ignoring config exclusions."""
    all_optimizers = get_supported_optimizers(filters)
    ignore_list = configs["benchmark"].get("ignore_optimizers", [])
    return [opt for opt in all_optimizers if opt not in ignore_list][
        opt_range[0] : opt_range[1]
    ]


def prepare_eval_configs(configs: dict, optimizer_name: str) -> dict:
    """Build evaluation configs for a given optimizer."""
    return {
        "num_iters": {
            func_name: func_config["iterations"]
            * configs["optimizers"].get(optimizer_name, {}).get("_iter_scale", 1)
            for func_name, func_config in configs["functions"].items()
        },
        "error_weights": {
            func_name: func_config["error_weight"]
            for func_name, func_config in configs["functions"].items()
        },
        **configs["benchmark"],
    }


@click.command()
@click.option("--debug", is_flag=True, help="Enable debug mode")
@click.option("--get_num", is_flag=True, help="Get the number of optimizers")
@click.option(
    "--opt_range",
    nargs=2,
    type=int,
    default=[0, None],
    help="Range of optimizer indices to benchmark (start end). Default: 0 None (all)",
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
def main(**kwargs):
    """The entry point of the app."""
    debug: bool = kwargs["debug"]
    get_num: bool = kwargs["get_num"]
    opt_range: list[int] = kwargs["opt_range"]
    results_dir: Path = kwargs["results_dir"]
    config_path: Path = kwargs["config_path"]

    filters = normalize_multi_option(kwargs["filter"])
    funcs = normalize_multi_option(kwargs["functions"])

    configs = read_toml_config(config_path)
    optimizers = prepare_optimizers(configs, filters, opt_range)

    if debug:
        print(f"Optimizers: {optimizers}")

    if get_num:
        print(len(optimizers))
        return

    eval_args = configs.get("optimizer_eval_args", {})

    for i, optimizer_name in enumerate(optimizers, start=1):
        eval_configs = prepare_eval_configs(configs, optimizer_name)
        search_space = get_hyperparameter_search_space(
            optimizer_name, configs["optimizers"]
        )

        print(
            f"({i}/{len(optimizers)}) Processing {optimizer_name}... "
            f"(Params to tune: {', '.join(search_space.keys())})"
        )

        get_optimizer = get_optimizer_factory(optimizer_name, debug)

        benchmark_optimizer(
            get_optimizer,
            optimizer_name,
            results_dir,
            search_space,
            eval_configs,
            eval_args=eval_args,
            functions=funcs,
            debug=debug,
        )

    print("Done!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted by user.")
