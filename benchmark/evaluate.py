import json
import os
import shutil
import time
import warnings
from pathlib import Path
from typing import Any, Callable

import optuna
from optuna.samplers import CmaEsSampler, GPSampler, TPESampler
from tqdm import tqdm

from .criterion import objective
from .functions import FUNC_DICT
from .utils.executor import optimize
from .visualizer import plot_function

optuna.logging.set_verbosity(optuna.logging.ERROR)
warnings.filterwarnings("ignore")

# Storage: "in-memory" (fast) or "sqlite" (persistent, allows resuming and monitoring)
OPTUNA_STORAGE_TYPE = "in-memory"
OPTUNA_STORAGE_PATH = "sqlite:///optuna_cache.db"
# Cache type affects study naming: "opt", "func", or "opt+func"
OPTUNA_CACHE_TYPE = "opt"


def _progress_bar_callback(total_trials: int):
    """Create a tqdm progress bar callback for Optuna optimization."""
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


def _load_json(
    path: Path, default: dict[str, Any], max_retries: int = 10
) -> dict[str, Any]:
    """Load JSON file with retry logic for concurrent access."""
    for attempt in range(max_retries + 1):
        try:
            with path.open("r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            if attempt < max_retries:
                time.sleep(0.1)
                continue
            else:
                break

    return default


def _choose_sampler(search_space, config, debug=False):
    """
    Choose an Optuna sampler based on search space types and budget.
    """
    has_categorical = any(dist == "bool" for dist in search_space.values())

    if has_categorical:
        sampler = TPESampler(
            seed=config["seed"],
            multivariate=len(search_space) > 1,
            group=len(search_space) > 1,
            n_startup_trials=140,
            n_ei_candidates=max(int(100 / max(len(search_space), 1)), 20),
            consider_prior=True,
            prior_weight=0.9,
            consider_endpoints=True,
        )
        if debug:
            print("Using TPESampler (categorical present)")

        return sampler

    # if config["hypertune_trials"] < 250:
    #     sampler = GPSampler(
    #         n_startup_trials=20,
    #     )
    #     if debug:
    #         print("Using GPSampler (<250 trials, numeric)")

    #     return sampler

    sampler = CmaEsSampler(
        seed=config["seed"],
        restart_strategy="ipop",
    )
    if debug:
        print("Using CmaEsSampler (numeric, large budget)")

    return sampler


def benchmark_optimizer(
    optimizer_maker: Callable[..., Any],
    optimizer_name: str,
    output_dir: Path,
    hyper_search_spaces: dict[str, Any],
    config: dict[str, Any],
    functions: list[str] | None = None,
    eval_args: dict[str, dict[str, Any]] | None = None,
    debug: bool = False,
) -> None:
    """Benchmark an optimizer across multiple test functions with hyperparameter tuning.

    Args:
        optimizer_maker: Factory function that creates optimizer instances.
        optimizer_name: Name of the optimizer being benchmarked.
        output_dir: Directory for saving results and visualizations.
        hyper_search_spaces: Hyperparameter search space for Optuna.
        config: Configuration with tuning parameters and settings.
        functions: List of functions to benchmark. If None, uses all functions.
        eval_args: Additional arguments keyed by optimizer name.
        debug: Enable debug output.
    """
    if eval_args is None:
        eval_args = {}

    results_dir = output_dir.joinpath(optimizer_name)
    results_json_path = output_dir.joinpath("results.json")

    # Skip if exist_pass enabled and complete results already exist
    if config["exist_pass"]:
        _num_expected_files = (
            len(functions) if functions is not None else len(FUNC_DICT)
        )
        _results = _load_json(results_json_path, {"optimizers": {}}).get(
            "optimizers", {}
        )
        _images = list(results_dir.glob(f"*{config['img_format']}"))

        _results_complete = (
            optimizer_name in _results
            and results_dir.is_dir()
            and len(_images) == _num_expected_files
        )

        if _results_complete:
            print(f"Skipping {optimizer_name}: Complete results already exist.")
            return None

    if results_dir.exists():
        shutil.rmtree(results_dir)

    os.makedirs(results_dir, exist_ok=True)

    error_rates = {}
    train_metrics = {}
    eval_metrics = {}
    run_hyperparams = {}

    for func_name, consts in FUNC_DICT.items():
        if functions is not None and func_name not in functions:
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

            steps = optimize(
                func,
                optimizer_maker,
                optimizer_params,
                start_pos,
                config["num_iters"][func_name],
                eval_args.get(optimizer_name, {}),
            )

            error, metrics = objective(
                steps,
                func,
                start_pos,
                gm_pos,
                eval_size,
                "train",
                debug=debug,
            )

            trial.set_user_attr("train_metrics", metrics)

            return error

        sampler = _choose_sampler(hyper_search_spaces, config, debug=debug)

        study = optuna.create_study(
            study_name=f"{func_name}~{optimizer_name}"
            if OPTUNA_CACHE_TYPE == "opt+func"
            else (func_name if OPTUNA_CACHE_TYPE == "func" else optimizer_name),
            direction="minimize",
            sampler=sampler,
            storage=OPTUNA_STORAGE_PATH if OPTUNA_STORAGE_TYPE == "sqlite" else None,
            load_if_exists=OPTUNA_STORAGE_TYPE == "sqlite",
        )

        study.optimize(
            optuna_objective,
            n_trials=config["hypertune_trials"],
            show_progress_bar=False,
            catch=(ZeroDivisionError,),  # Catch errors from unstable hyperparameters
            n_jobs=1,
            callbacks=[_progress_bar_callback(config["hypertune_trials"])],  # type: ignore
        )

        run_hyperparams[func_name] = study.best_params
        train_metrics[func_name] = study.best_trial.user_attrs.get("train_metrics", {})

        func_optim_steps = optimize(
            func,
            optimizer_maker,
            study.best_params,
            start_pos,
            config["num_iters"][func_name],
            eval_args.get(optimizer_name, {}),
        )
        error_rates[func_name], eval_metrics[func_name] = objective(
            func_optim_steps,
            func,
            start_pos,
            gm_pos,
            eval_size,
            "eval",
            debug=debug,
        )

        print(" ├ Best Train Metrics:")
        if error_rates[func_name] != float("inf"):
            for i, (metric_name, metric_value) in enumerate(
                train_metrics[func_name].items()
            ):
                print(
                    f"{' └┬' if i == 0 else ('  └' if i == len(train_metrics[func_name]) - 1 else '  ├')} "
                    f"{metric_name}: {metric_value}, "
                    f"contribution: {round(metric_value / sum(train_metrics[func_name].values()) * 100)}%"
                )
        else:
            print(" └─ No metrics available")

        plot_function(
            func,
            func_name,
            func_optim_steps,
            os.path.join(results_dir, func_name + config["img_format"]),
            optimizer_name,
            study.best_params,
            eval_metrics[func_name],
            error_rates[func_name],
            gm_pos,
            eval_size,
            debug=debug,
        )

        print("")

    weights = config.get("error_weights", {})

    results = _load_json(
        results_json_path, {"optimizers": {}, "functions": {"weights": weights}}
    )

    results["optimizers"][optimizer_name] = {
        "hyperparameters": run_hyperparams,
        "error_rates": error_rates,
        "train_metrics": train_metrics,
    }

    with results_json_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
