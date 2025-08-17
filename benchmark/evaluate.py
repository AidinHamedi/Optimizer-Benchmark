import os
import shutil
import time
from pathlib import Path
from typing import Callable

import optuna
import torch
from torch import nn

from .functions import FUNC_DICT
from .hypertune import objective
from .utils.executor import execute_steps
from .utils.model import Pos2D
from .visualizer import plot_function

optuna.logging.set_verbosity(optuna.logging.WARNING)


def benchmark_optimizer(
    optimizer_maker: Callable,
    optimizer_name: str,
    output_dir: Path,
    hyper_search_spaces: dict,
    config: dict,
    eval_args: dict = {},
    exist_pass: bool = False,
    img_format: str = ".png",
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
        exist_pass (bool, optional): If True, skips benchmarking if results already exist. Defaults to False.
        img_format (str, optional): Format for saving output images (e.g., ".png"). Defaults to ".png".

    Returns:
        None
    """
    results_dir = os.path.join(output_dir, optimizer_name)

    if os.path.exists(results_dir):
        if exist_pass:
            return None
        shutil.rmtree(results_dir)

    os.makedirs(results_dir)

    for func_name, consts in FUNC_DICT.items():
        print(f" - Evaluating {func_name}...")
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
            show_progress_bar=True,
            n_jobs=2,
        )

        pos = Pos2D(func, start_pos)
        plot_function(
            func,
            func_name,
            execute_steps(
                pos,
                optimizer_maker(pos, study.best_params),
                config["num_iters"][func_name],
                **eval_args.get(optimizer_name, {}),
            ),
            os.path.join(results_dir, func_name + img_format),
            optimizer_name,
            study.best_params,
            gm_pos,
            eval_size,
        )
