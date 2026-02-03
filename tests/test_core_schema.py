import pytest

from optimizer_vis.core.schema import OptimizationProblemConf


@pytest.fixture
def valid_config_dict():
    """Mock dictionary based on test.toml structure."""
    return {
        "name": "TestFunc",
        "domain": {
            "start": [-2.0, 2.0],
            "bounds": [[-2.1, 2.1], [-1.1, 3.1]],
            "global_min": [[1.0, 1.0]],
        },
        "evaluation": {"norm": [0.0, 100.0], "vectorized": False},
        "internal": {"a": 1.0},
    }


def test_config_loading(valid_config_dict):
    """Ensure dict converts to dataclass correctly."""
    conf = OptimizationProblemConf.from_dict(valid_config_dict)

    assert conf.func_name == "TestFunc"
    assert conf.start_pos.x == -2.0
    assert conf.bounds.y_max == 3.1
    assert conf.vectorized is False
    assert conf.constants.a == 1.0


def test_missing_keys(valid_config_dict):
    """Ensure missing keys raise ValueError."""
    del valid_config_dict["domain"]

    with pytest.raises(ValueError) as exc:
        OptimizationProblemConf.from_dict(valid_config_dict)

    assert "Missing keys" in str(exc.value)
