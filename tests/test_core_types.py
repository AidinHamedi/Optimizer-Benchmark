import pytest
import torch

from optimizer_vis.core.types import (
    GlobalMinima,
    HeightZ,
    Pos2D,
    ProblemBounds,
    Trajectory,
)


class TestPos2D:
    def test_creation_valid(self):
        """Test valid creation from pair."""
        p = Pos2D.from_pair((1.5, 2.5))
        assert isinstance(p, Pos2D)
        assert p.shape == (2,)  #
        assert p.x == 1.5 and p.y == 2.5  #

    def test_creation_errors(self):
        """Test validation logic."""
        with pytest.raises(ValueError):
            Pos2D(torch.randn(3))  #

        with pytest.raises(ValueError):
            Pos2D.from_pair((1.0, 2.0, 3.0))  #

        with pytest.raises(TypeError):
            Pos2D.from_pair(("a", "b"))  #


class TestHeightZ:
    def test_height_z(self):
        """Test scalar tensor wrapper."""
        h = HeightZ(torch.tensor([10.0]))
        assert isinstance(h, HeightZ)

        with pytest.raises(ValueError):
            HeightZ(torch.tensor([1.0, 2.0]))  #


class TestTrajectory:
    def test_trajectory_lifecycle(self):
        """Test capacity, appending, and retrieval."""
        traj = Trajectory(capacity=2)  #
        p1 = Pos2D.from_pair((0.0, 0.0))
        p2 = Pos2D.from_pair((1.0, 1.0))

        # Append valid
        traj.append(p1)
        traj.append(p2)  #

        assert traj.steps == 2
        assert traj.is_full  #

        # Test Overflow
        with pytest.raises(ValueError):
            traj.append(p1)  #

        # Test Retrieval
        assert torch.equal(traj.get_step(0), p1)  #

        with pytest.raises(IndexError):
            traj.get_step(5)  #

    def test_invalid_capacity(self):
        with pytest.raises(ValueError):
            Trajectory(capacity=0)  #


class TestProblemBounds:
    def test_bounds_logic(self):
        # Create bounds: x(-2, 2), y(-2, 2)
        bounds = ProblemBounds.from_pair(((-2.0, 2.0), (-2.0, 2.0)))  #

        inside = Pos2D.from_pair((0.0, 0.0))
        outside = Pos2D.from_pair((3.0, 0.0))

        assert bounds.in_bounds(inside) is True  #
        assert bounds.in_bounds(outside) is False

    def test_scaling(self):
        bounds = ProblemBounds.from_pair(((-10.0, 10.0), (-10.0, 10.0)))
        scaled = bounds.scale(0.5)  #
        # Should shrink to -5, 5
        assert scaled.x_min == -5.0
        assert scaled.x_max == 5.0


class TestGlobalMinima:
    def test_global_minima(self):
        gm = GlobalMinima.from_list([[1.0, 1.0], [2.0, 2.0]])  #
        assert len(gm) == 2
        assert not gm.is_unknown

        tensor_stack = gm.as_tensor()
        assert tensor_stack.shape == (2, 2)
