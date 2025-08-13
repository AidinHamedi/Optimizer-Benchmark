import os
import shutil
from pathlib import Path
from typing import Callable

import torch
from torch import nn

from .functions import FUNC_DICT
from .utils.executor import execute_steps
from .utils.model import Pos2D


def benchmark_optimizer(
    optimizer_maker: Callable,
    optimizer_name: str,
    output_dir: Path,
    hypr_search_spaces: dict,
    config: dict,
):
    results_dir = os.path.join(output_dir, optimizer_name)

    if os.path.exists(results_dir):
        shutil.rmtree(results_dir)

    os.makedirs(results_dir)

    for name, data in FUNC_DICT.items():
        func = data["func"]
        eval_size = data["size"]
        start_pos = data["pos"]
        print(name, eval_size, start_pos)
