import torch

from einmesh import einmesh
from einmesh.spaces import LinSpace
from tests.conftest import parametrize_backends


@parametrize_backends
def test_repeated_space(backend):
    """Test that einmesh handles repeated spaces correctly."""
    x_space = LinSpace(0.0, 1.0, 5)
    result = einmesh("* x x", x=x_space, backend=backend)
    assert backend.is_appropriate_type(result)
    assert result.shape == (2, 5, 5)


@parametrize_backends
def test_repeated_space_in_parentheses(backend):
    """Test that einmesh handles repeated spaces in parentheses correctly."""
    x_space = LinSpace(0.0, 1.0, 5)
    result = einmesh("(x x)", x=x_space, backend=backend)
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert backend.is_appropriate_type(result[0])
    assert backend.is_appropriate_type(result[1])
    assert result[0].shape == torch.Size([25])
    assert result[1].shape == torch.Size([25])


@parametrize_backends
def test_repeated_space_in_parentheses_and_star(backend):
    """Test that einmesh handles repeated spaces in parentheses and star correctly."""
    x_space = LinSpace(0.0, 1.0, 5)
    result = einmesh("* (x x)", x=x_space, backend=backend)
    assert backend.is_appropriate_type(result)
    assert result.shape == (2, 25)


@parametrize_backends
def test_repeated_space_with_non_repeated_space(backend):
    """Test that einmesh handles repeated spaces with non-repeated spaces correctly."""
    x_space = LinSpace(0.0, 1.0, 5)
    y_space = LinSpace(0.0, 1.0, 3)
    z_space = LinSpace(0.0, 1.0, 2)
    result = einmesh("* x (y y) z", x=x_space, y=y_space, z=z_space, backend=backend)
    assert backend.is_appropriate_type(result)
    assert result.shape == (4, 5, 3 * 3, 2)


@parametrize_backends
def test_very_repeated_space(backend):
    """Test that einmesh handles very repeated spaces correctly."""
    x_space = LinSpace(0.0, 1.0, 5)
    y_space = LinSpace(0.0, 1.0, 3)
    results = einmesh("x x (x x) x (y x) x x", x=x_space, y=y_space, backend=backend)
    assert isinstance(results, tuple)
    assert len(results) == 9
    for result in results:
        assert backend.is_appropriate_type(result)
        assert result.shape == (5, 5, 5 * 5, 5, 3 * 5, 5, 5)


@parametrize_backends
def test_multi_repeated_space(backend):
    """Test that einmesh handles very repeated spaces with star correctly."""
    x_space = LinSpace(0.0, 1.0, 5)
    y_space = LinSpace(0.0, 1.0, 3)
    z_space = LinSpace(0.0, 1.0, 2)
    results = einmesh("* x y z x y z x y z", x=x_space, y=y_space, z=z_space, backend=backend)
    assert backend.is_appropriate_type(results)
    assert results.shape == (9, 5, 3, 2, 5, 3, 2, 5, 3, 2)
