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

# Some optimizers require specific arguments to be set based on the benchmark context.
# For example, Ranger21 needs to know the total number of iterations for its internal scheduler.
# This dictionary provides a clean way to apply these "patches" at runtime.
OPTIMIZER_PATCHES = {
    "adashift": lambda cfg, iters: cfg.update({"keep_num": 1}),
    "ranger21": lambda cfg, iters: cfg.update({"num_iterations": iters}),
    "ranger25": lambda cfg, iters: cfg.update({"orthograd": False}),
    "bsam": lambda cfg, iters: cfg.update({"num_data": 1}),
}

# This set identifies optimizers from the pytorch_optimizer library that have a non-standard
# weight_decay implementation. They expect weight_decay to be passed during initialization
# rather than being applied to parameter groups. We handle this in the factory.
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
    """
    Normalizes a Click multi-option tuple.
    Click returns an empty tuple () for no options, ('val',) for one, and ('val1', 'val2') for many.
    This function converts these to a more standard format: None or a list of strings.
    """
    if not values:
        return None
    return list(values)


def get_hyperparameter_search_space(name: str, config: dict) -> dict:
    """Return the hyperparameter search space for a given optimizer name."""
    # Fallback to the 'base' configuration if a specific one is not defined.
    base = config.get(name, config["base"])
    search_space = dict(base)
    # The '_iter_scale' key is a custom config for this benchmark, not a real hyperparameter.
    # It must be removed before passing the config to Optuna or the optimizer.
    search_space.pop("_iter_scale", None)
    return search_space


def get_optimizer_factory(optimizer_name: str, debug: bool = False):
    """
    Returns a factory function for creating an optimizer instance.
    This approach allows us to encapsulate complex instantiation logic, including
    applying patches and handling special cases for different optimizers.
    """

    def factory(model, optimizer_config: dict, num_iters: int):
        # Apply a patch if the optimizer requires special configuration.
        patch = OPTIMIZER_PATCHES.get(optimizer_name)
        if patch:
            if debug:
                print(f"Applying patch for {optimizer_name}")
            patch(optimizer_config, num_iters)

        optimizer_class = load_optimizer(optimizer_name)

        # Handle optimizers with unique or non-standard parameter requirements.
        if optimizer_name == "adammini":
            # AdamMini requires weight_decay to be handled separately.
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
            # Standard optimizer instantiation.
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
    # Filter out optimizers listed in the config's ignore_list and apply CLI range.
    return [opt for opt in all_optimizers if opt not in ignore_list][
        opt_range[0] : opt_range[1]
    ]


def prepare_eval_configs(configs: dict, optimizer_name: str) -> dict:
    """
    Builds the evaluation configuration for a given optimizer.
    This includes setting the number of iterations per function (with scaling)
    and defining the weights for error calculation.
    """
    return {
        # Some optimizers converge faster or slower. '_iter_scale' allows us to adjust
        # the number of iterations for them without changing the base config.
        "num_iters": {
            func_name: func_config["iterations"]
            * configs["optimizers"].get(optimizer_name, {}).get("_iter_scale", 1)
            for func_name, func_config in configs["functions"].items()
        },
        # Error weights allow us to prioritize performance on certain functions.
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

    # The --get_num flag provides an easy way for wrapper scripts (like the Makefile)
    # to query the total number of optimizers for parallel processing.
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

        try:
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
        except Exception as e:
            print(f"Failed to benchmark {optimizer_name}: {e}")

    print("Done!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted by user.")
