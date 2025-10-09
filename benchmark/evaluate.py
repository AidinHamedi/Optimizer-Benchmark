import json
import os
import shutil
import time
from pathlib import Path
from typing import Callable

import optuna
from tqdm import tqdm

from .criterion import objective
from .functions import FUNC_DICT
from .utils.executor import execute_steps
from .utils.model import Pos2D
from .visualizer import plot_function

optuna.logging.set_verbosity(optuna.logging.ERROR)

# Optuna storage can be configured here.
# 'in-memory' is faster but not persistent across runs.
# 'sqlite' is persistent, allowing studies to be resumed.
OPTUNA_STORAGE_TYPE = "in-memory"
OPTUNA_STORAGE_PATH = "sqlite:///optuna_cache.db"
# Defines how Optuna studies are named, affecting caching.
# 'opt': One study per optimizer (caches across functions).
# 'func': One study per function (caches across optimizers).
# 'opt+func': One study per optimizer-function pair (no caching).
OPTUNA_CACHE_TYPE = "opt"


def _progress_bar_callback(total_trials: int):
    pbar = tqdm(total=total_trials, desc=" ├ Hyper Optimization")

    def callback(study: optuna.Study, trial: optuna.Trial):
        pbar.update(1)
        pbar.set_postfix(
            {
                "Best Value": f"{study.best_value:.4f}",
                "Best Trial": study.best_trial.number,
            }
        )

        if len(study.trials) >= total_trials:
            pbar.close()

    return callback


def _load_results(results_json_dir, optimizer_name, error_rates, max_retries=5):
    """Load results.json safely with retries and update error_rates."""
    for attempt in range(max_retries + 1):
        if results_json_dir.exists():
            try:
                with results_json_dir.open("r", encoding="utf-8") as f:
                    results = json.load(f)

                # If previous results for this optimizer exist, load them.
                prev_error_rates = (
                    results.get("optimizers", {})
                    .get(optimizer_name, {})
                    .get("error_rates", {})
                )

                # Merge previous results into the current error_rates dictionary.
                # This is important for rerunning benchmarks on a subset of functions,
                # ensuring old results for other functions are not lost.
                for func_name, error_rate in prev_error_rates.items():
                    error_rates.setdefault(func_name, error_rate)

                return results

            except json.JSONDecodeError:
                if attempt < max_retries:
                    time.sleep(0.2)
                    continue
                else:
                    return {"optimizers": {}}
        else:
            return {"optimizers": {}}

    return {"optimizers": {}}


def benchmark_optimizer(
    optimizer_maker: Callable,
    optimizer_name: str,
    output_dir: Path,
    hyper_search_spaces: dict,
    config: dict,
    functions: list | None = None,
    eval_args: dict = {},
    debug: bool = False,
) -> None:
    """
    Benchmarks a given optimizer across multiple test functions with hyperparameter tuning.

    Args:
        optimizer_maker (Callable): A factory function that returns an optimizer instance given a position and parameters.
        optimizer_name (str): Name of the optimizer being benchmarked.
        output_dir (Path): Directory where results and visualizations will be saved.
        hyper_search_spaces (dict): Dictionary defining the hyperparameter search space for the optimizer.
        config (dict): Configuration dictionary containing tuning parameters like number of iterations and trials.
        functions (list | None, optional): List of functions to benchmark. If None, all functions will be used. Defaults to None.
        eval_args (dict, optional): Additional arguments for evaluation, keyed by optimizer name. Defaults to {}.
        debug (bool, optional): Whether to enable debug mode. Defaults to False.

    Returns:
        None
    """

    results_dir = output_dir.joinpath(optimizer_name)
    results_json_dir = output_dir.joinpath("results.json")

    # If exist_pass is false, we clear previous results for this optimizer.
    if results_dir.exists() and not config["exist_pass"]:
        shutil.rmtree(results_dir)

    os.makedirs(results_dir, exist_ok=True)

    error_rates = {}
    run_metrics = {}

    for func_name, consts in FUNC_DICT.items():
        # Allow running the benchmark on a specific subset of functions.
        if functions is not None and func_name not in functions:
            continue

        vis_file = results_dir.joinpath(func_name + config["img_format"])

        # If exist_pass is true, we can skip functions that already have a result.
        if vis_file.exists() and config["exist_pass"]:
            continue

        print(f" ┌ Evaluating On {func_name}...")
        func = consts["func"]
        eval_size = consts["size"]
        start_pos = consts["pos"]
        gm_pos = consts["gm_pos"]

        # This inner function is what Optuna will optimize.
        def optuna_objective(trial: optuna.Trial) -> float:
            optimizer_params = {}
            # Dynamically suggest hyperparameters based on the search space from config.toml.
            for name, space in hyper_search_spaces.items():
                if isinstance(space, list) and len(space) == 2:
                    optimizer_params[name] = trial.suggest_float(
                        name, space[0], space[1]
                    )
                elif isinstance(space, list) and len(space) == 3:
                    if space[2] == "int":
                        optimizer_params[name] = trial.suggest_int(
                            name, space[0], space[1]
                        )
                    else:
                        raise ValueError("Invalid hyperparameter space")
                elif space == "bool":
                    optimizer_params[name] = trial.suggest_categorical(
                        name, [True, False]
                    )
                else:
                    raise ValueError("Invalid hyperparameter space")

            num_iters = config["num_iters"][func_name]  # type: ignore

            error, metrics = objective(
                func,
                optimizer_maker,
                optimizer_params,
                start_pos,
                gm_pos,
                eval_size,
                num_iters,
                **eval_args.get(optimizer_name, {}),
                debug=debug,
            )

            trial.set_user_attr("metrics", metrics)

            return error

        # Create an Optuna study. The study name determines caching behavior.
        study = optuna.create_study(
            study_name=f"{func_name}~{optimizer_name}"
            if OPTUNA_CACHE_TYPE == "opt+func"
            else (func_name if OPTUNA_CACHE_TYPE == "func" else optimizer_name),
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=config["seed"]),
            storage=OPTUNA_STORAGE_PATH if OPTUNA_STORAGE_TYPE == "sqlite" else None,
            load_if_exists=OPTUNA_STORAGE_TYPE == "sqlite",
        )

        study.optimize(
            optuna_objective,
            n_trials=config["hypertune_trials"],
            show_progress_bar=False,
            catch=ZeroDivisionError,  # Catch errors from unstable hyperparameters.
            n_jobs=1 if config["deterministic"] else 2,
            callbacks=[_progress_bar_callback(config["hypertune_trials"])],  # type: ignore
        )

        error_rates[func_name] = study.best_value
        run_metrics[func_name] = study.best_trial.user_attrs.get("metrics", {})

        print(" ├ Best Metrics:")
        if error_rates[func_name] != float("inf"):
            for i, (metric_name, metric_value) in enumerate(
                run_metrics[func_name].items()
            ):
                print(
                    f" {'└┬' if i == 0 else (' └' if i == len(run_metrics[func_name]) - 1 else ' ├')} {metric_name}: {metric_value}, contribution: {round(metric_value / error_rates[func_name] * 100)}%"
                )
        else:
            print(" └─ No metrics available")

        # After finding the best parameters, run the optimizer one last time to generate the visualization.
        pos = Pos2D(func, start_pos)
        plot_function(
            func,
            func_name,
            execute_steps(
                pos,
                optimizer_maker(pos, study.best_params, config["num_iters"][func_name]),
                config["num_iters"][func_name],
                **eval_args.get(optimizer_name, {}),
            ),
            os.path.join(results_dir, func_name + config["img_format"]),
            optimizer_name,
            study.best_params,
            study.best_value,
            gm_pos,
            eval_size,
            debug=debug,
        )

    # Load existing results and merge them with the new ones.
    results = _load_results(results_json_dir, optimizer_name, error_rates)

    weights = config.get("error_weights", {})
    weighted_errors = {
        func_name: error_rate * weights.get(func_name, 1.0)
        for func_name, error_rate in error_rates.items()
    }

    results["optimizers"][optimizer_name] = {
        "error_rates": error_rates,
        "weighted_error_rates": weighted_errors,
        "avg_error_rate": sum(error_rates.values()) / len(error_rates),
        "weighted_avg_error_rate": sum(weighted_errors.values()) / len(weighted_errors),
    }

    with results_json_dir.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
