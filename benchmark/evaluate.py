from typing import Callable

import torch
from torch import nn

from . import functions
from .utils.executor import execute_steps
from .utils.model import Pos2D


def benchmark_optimizer(optimizer_maker: Callable, optimizer_name: str):
    pass
