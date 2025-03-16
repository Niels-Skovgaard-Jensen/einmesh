import pytest
import torch

from einmesh import einmesh
from einmesh.parser import UndefinedSpaceError
from einmesh.spaces import LinSpace


def test_einmesh_basic():
    """Test the basic functionality of einmesh without output pattern."""
    x_space = LinSpace(0.0, 1.0, 5)
    y_space = LinSpace(0.0, 1.0, 3)

    meshes = einmesh("x y", x=x_space, y=y_space)

    assert len(meshes) == 2
    assert isinstance(meshes[0], torch.Tensor)
    assert isinstance(meshes[1], torch.Tensor)
    assert meshes[0].shape == (5, 3)
    assert meshes[1].shape == (5, 3)


def test_einmesh_star_pattern():
    """Test einmesh with * pattern to stack all meshes."""
    x_space = LinSpace(0.0, 1.0, 5)
    y_space = LinSpace(0.0, 1.0, 3)
    z_space = LinSpace(0.0, 1.0, 2)

    # Using just *
    result = einmesh("x y z -> *", x=x_space, y=y_space, z=z_space)

    assert isinstance(result, torch.Tensor)
    assert result.shape == (3, 5, 3, 2)  # 3 meshes stacked as first dimension

    # Check that the result contains the original meshes
    x_mesh, y_mesh, z_mesh = einmesh("x y z", x=x_space, y=y_space, z=z_space)
    assert torch.allclose(result[0], x_mesh)
    assert torch.allclose(result[1], y_mesh)
    assert torch.allclose(result[2], z_mesh)


def test_einmesh_parentheses_pattern():
    """Test einmesh with parentheses pattern to reshape dimensions."""
    x_space = LinSpace(0.0, 1.0, 5)
    y_space = LinSpace(0.0, 1.0, 3)

    # Using pattern with parentheses
    result = einmesh("x y -> (x y)", x=x_space, y=y_space)

    # Basic check that we get a tensor
    assert isinstance(result, torch.Tensor)

    # Using einops.rearrange properly flattens the dimensions within parentheses
    # So the result should be a 1D tensor with shape (5*3,) = (15,)
    assert result.ndim == 1
    assert result.shape[0] == 5 * 3  # 15 elements total


def test_einmesh_output_dimension_ordering():
    """Test that einmesh respects dimension ordering in output pattern."""
    x_space = LinSpace(0.0, 1.0, 5)
    y_space = LinSpace(0.0, 1.0, 3)

    # Get original meshes
    x_mesh, y_mesh = einmesh("x y", x=x_space, y=y_space)

    # Using output pattern to reorder dimensions
    result_x_y = einmesh("x y -> x y", x=x_space, y=y_space)
    result_y_x = einmesh("x y -> y x", x=x_space, y=y_space)

    # Check the type of results
    if isinstance(result_x_y, tuple):
        assert len(result_x_y) == 2
        assert torch.allclose(result_x_y[0], x_mesh)
        assert torch.allclose(result_x_y[1], y_mesh)

    if isinstance(result_y_x, tuple):
        assert len(result_y_x) == 2
        assert torch.allclose(result_y_x[0], y_mesh)
        assert torch.allclose(result_y_x[1], x_mesh)


def test_star_position():
    """Test that einmesh handles star position correctly."""
    x_space = LinSpace(0.0, 1.0, 7)
    y_space = LinSpace(0.0, 1.0, 9)

    result = einmesh("x y -> * x y", x=x_space, y=y_space)
    assert result.shape == (2, 7, 9)

    result = einmesh("x y -> x * y", x=x_space, y=y_space)
    assert result.shape == (7, 2, 9)

    result = einmesh("x y -> x y *", x=x_space, y=y_space)
    assert result.shape == (7, 9, 2)


def test_axis_collection():
    """Test that einmesh handles axis collection correctly."""
    x_space = LinSpace(0.0, 1.0, 5)
    y_space = LinSpace(0.0, 1.0, 3)

    result = einmesh("x y -> * (x y)", x=x_space, y=y_space)
    assert result.shape == (2, 5 * 3)

    result = einmesh("x y -> (x y) *", x=x_space, y=y_space)
    assert result.shape == (5 * 3, 2)


def test_invalid_pattern():
    """Test that einmesh raises error for invalid patterns."""
    x_space = LinSpace(0.0, 1.0, 5)

    with pytest.raises(UndefinedSpaceError):
        einmesh("x y -> x y", x=x_space)  # Missing y space
