import json
import os
import shutil
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


def _progress_bar_callback(total_trials: int):
    pbar = tqdm(total=total_trials, desc=" └ Hyper Optimization")

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


def benchmark_optimizer(
    optimizer_maker: Callable,
    optimizer_name: str,
    output_dir: Path,
    hyper_search_spaces: dict,
    config: dict,
    eval_args: dict = {},
) -> None:
    """
    Benchmarks a given optimizer across multiple test functions with hyperparameter tuning.

    Args:
        optimizer_maker (Callable): A factory function that returns an optimizer instance given a position and parameters.
        optimizer_name (str): Name of the optimizer being benchmarked.
        output_dir (Path): Directory where results and visualizations will be saved.
        hyper_search_spaces (dict): Dictionary defining the hyperparameter search space for the optimizer.
        config (dict): Configuration dictionary containing tuning parameters like number of iterations and trials.
        eval_args (dict, optional): Additional arguments for evaluation, keyed by optimizer name. Defaults to {}.

    Returns:
        None
    """

    results_dir = output_dir.joinpath(optimizer_name)
    results_json_dir = output_dir.joinpath("results.json")

    if results_dir.exists() and not config["exist_pass"]:
        shutil.rmtree(results_dir)

    os.makedirs(results_dir, exist_ok=True)

    error_rates = {}

    for func_name, consts in FUNC_DICT.items():
        vis_file = results_dir.joinpath(func_name + config["img_format"])

        if vis_file.exists() and config["exist_pass"]:
            continue

        print(f" ┌ Evaluating On {func_name}...")
        func = consts["func"]
        eval_size = consts["size"]
        start_pos = consts["pos"]
        gm_pos = consts["gm_pos"]

        def optuna_objective(trial: optuna.Trial) -> float:
            optimizer_params = {}
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

            return objective(
                func,
                optimizer_maker,
                optimizer_params,
                start_pos,
                gm_pos,
                eval_size,
                num_iters,
                **eval_args.get(optimizer_name, {}),
            )

        study = optuna.create_study(
            study_name=func_name,
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=config["seed"]),
        )

        study.optimize(
            optuna_objective,
            n_trials=config["hypertune_trials"],
            show_progress_bar=False,
            catch=(RuntimeError,),
            n_jobs=1 if config["deterministic"] else 2,
            callbacks=[_progress_bar_callback(config["hypertune_trials"])],  # type: ignore
        )

        error_rates[func_name] = study.best_value

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
        )

    if results_json_dir.exists():
        try:
            with results_json_dir.open("r", encoding="utf-8") as f:
                results = json.load(f)

                prev_error_rates = (
                    results.get("optimizers", {})
                    .get(optimizer_name, {})
                    .get("error_rates", {})
                )

                for func_name, error_rate in prev_error_rates.items():
                    error_rates.setdefault(func_name, error_rate)

        except json.JSONDecodeError:
            results = {"optimizers": {}}
    else:
        results = {"optimizers": {}}

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
