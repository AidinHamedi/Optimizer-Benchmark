import pytest
import torch

from optimizer_vis.core.base_problem import OptimizationProblem2d
from optimizer_vis.core.schema import (
    FuncConstants,
    GlobalMinima,
    OptimizationProblemConf,
    ProblemBounds,
)
from optimizer_vis.core.types import HeightZ, LearnablePos2D, Pos2D


def test_pos2d_constraints():
    """Ensures Pos2D strictly enforces 2D shape and type."""
    valid_tensor = torch.tensor([1.0, 2.0])
    pos = Pos2D(valid_tensor)
    assert pos.x == 1.0
    assert pos.y == 2.0

    with pytest.raises(ValueError):
        Pos2D(torch.tensor([1.0, 2.0, 3.0]))


def test_height_z_constraints():
    """Ensures HeightZ is strictly a scalar-like tensor."""
    with pytest.raises(ValueError):
        HeightZ(torch.tensor([1.0, 2.0]))

    hz = HeightZ(torch.tensor([5.0]))
    assert hz.item() == 5.0


def test_problem_bounds_validation():
    """Checks that invalid spatial ranges (min >= max) trigger errors."""
    with pytest.raises(ValueError):
        ProblemBounds(x_range=(10.0, 0.0), y_range=(0.0, 5.0))


def test_in_bounds_check():
    """Verifies the point-in-bounds inclusion logic."""
    bounds = ProblemBounds(x_range=(-5.0, 5.0), y_range=(-5.0, 5.0))
    assert bounds.in_bounds(Pos2D(torch.tensor([0.0, 0.0]))) is True
    assert bounds.in_bounds(Pos2D(torch.tensor([6.0, 0.0]))) is False


def test_bounds_scaling():
    """Tests if scaling boundaries expands correctly from the center."""
    bounds = ProblemBounds(x_range=(0.0, 10.0), y_range=(0.0, 10.0))
    scaled = bounds.scale(2.0)  # Should double the width to (-5, 15)
    assert scaled.x_min == -5.0
    assert scaled.x_max == 15.0


## --- PyTorch Integration Tests ---


class SimpleProblem(OptimizationProblem2d):
    """Concrete implementation for testing base class behavior."""

    def func(self, pos: LearnablePos2D) -> HeightZ:
        # Objective: f(x, y) = x^2 + y^2
        return HeightZ(torch.sum(pos**2))


def test_optimization_problem_gradient_flow():
    """Verifies that the learnable position tracks gradients correctly."""
    conf = OptimizationProblemConf(
        func_name="Test",
        start_pos=Pos2D(torch.tensor([1.0, 1.0])),
        bounds=ProblemBounds((-5, 5), (-5, 5)),
        global_minima=GlobalMinima.from_tensor(torch.tensor([0.0, 0.0])),
        constants=FuncConstants(),
    )

    problem = SimpleProblem(conf)
    output = problem.forward()
    output.backward()

    # Ensure the raw parameter received a gradient
    assert problem._raw_pos.grad is not None
    assert torch.all(problem._raw_pos.grad > 0)  # Gradient of x^2 at 1.0 is 2.0
