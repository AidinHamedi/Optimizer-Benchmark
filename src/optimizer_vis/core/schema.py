from dataclasses import dataclass
from typing import Any

from .types import (
    FuncConstants,
    GlobalMinima,
    NormParams,
    Pos2D,
    ProblemBounds,
)
from .utils import to_tuple


@dataclass(frozen=True)
class OptimizationProblemConf:
    """Configuration settings for an optimization problem instance.

    Attributes:
        func_name: The display name of the function.
        start_pos: The initial coordinates for the optimizer.
        bounds: The spatial limits for the search space.
        global_minima: Collection of known global minimum coordinates.
        constants: Function hyperparameters (e.g., coefficients).
        norm: Normalization parameters for the function.
        vectorized: Whether the function supports vectorized inputs.
    """

    func_name: str
    start_pos: Pos2D
    bounds: ProblemBounds
    global_minima: GlobalMinima
    constants: FuncConstants
    norm: NormParams
    vectorized: bool

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OptimizationProblemConf":
        """Creates a configuration instance from a dictionary."""
        _required = {"name", "domain", "evaluation"}
        if not _required.issubset(data.keys()):
            missing = _required - data.keys()
            raise ValueError(f"Missing keys: {missing}")

        domain = data["domain"]
        evaluation = data["evaluation"]
        internal = data.get("internal", {})

        if not isinstance(evaluation["vectorized"], bool):
            raise ValueError("Invalid 'vectorized' value")

        return cls(
            func_name=data["name"],
            start_pos=Pos2D.from_pair(to_tuple(domain["start"])),
            bounds=ProblemBounds.from_pair(to_tuple(domain["bounds"])),
            global_minima=GlobalMinima.from_list(to_tuple(domain["global_min"])),
            constants=FuncConstants(internal),
            norm=NormParams.from_pair(evaluation["norm"]),
            vectorized=evaluation["vectorized"],
        )
