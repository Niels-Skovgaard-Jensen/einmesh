import torch

from einmesh import einmesh
from einmesh.spaces import LinSpace


def test_repeated_space():
    """Test that einmesh handles repeated spaces correctly."""
    x_space = LinSpace(0.0, 1.0, 5)
    result = einmesh("* x x", x=x_space)
    assert isinstance(result, torch.Tensor)
    assert result.shape == (2, 5, 5)


def test_repeated_space_in_parentheses():
    """Test that einmesh handles repeated spaces in parentheses correctly."""
    x_space = LinSpace(0.0, 1.0, 5)
    result = einmesh("(x x)", x=x_space)
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert isinstance(result[0], torch.Tensor)
    assert isinstance(result[1], torch.Tensor)
    assert result[0].shape == (5, 5)
    assert result[1].shape == (5, 5)


def test_repeated_space_in_parentheses_and_star():
    """Test that einmesh handles repeated spaces in parentheses and star correctly."""
    x_space = LinSpace(0.0, 1.0, 5)
    result = einmesh("* (x x)", x=x_space)
    assert isinstance(result, torch.Tensor)
    assert result.shape == (2, 5, 5)


def test_repeated_space_with_non_repeated_space():
    """Test that einmesh handles repeated spaces with non-repeated spaces correctly."""
    x_space = LinSpace(0.0, 1.0, 5)
    y_space = LinSpace(0.0, 1.0, 3)
    z_space = LinSpace(0.0, 1.0, 2)
    result = einmesh("* x (y y) z", x=x_space, y=y_space, z=z_space)
    assert isinstance(result, torch.Tensor)
    assert result.shape == (3, 5, 3 * 3, 2)


def test_very_repeated_space():
    """Test that einmesh handles very repeated spaces correctly."""
    x_space = LinSpace(0.0, 1.0, 5)
    y_space = LinSpace(0.0, 1.0, 3)
    results = einmesh("x x (x x) x (y x) x x", x=x_space, y=y_space)
    assert isinstance(results, tuple)
    assert len(results) == 7
    for result in results:
        assert isinstance(result, torch.Tensor)
        assert result.shape == (5, 5, 5 * 5, 5, 3 * 5, 5, 5)


def test_multi_repeated_space():
    """Test that einmesh handles very repeated spaces with star correctly."""
    x_space = LinSpace(0.0, 1.0, 5)
    y_space = LinSpace(0.0, 1.0, 3)
    z_space = LinSpace(0.0, 1.0, 2)
    results = einmesh("* x y z x y z x y z", x=x_space, y=y_space, z=z_space)
    assert isinstance(results, torch.Tensor)
    assert results.shape == (3, 5, 3, 2, 5, 3, 2, 5, 3, 2)
