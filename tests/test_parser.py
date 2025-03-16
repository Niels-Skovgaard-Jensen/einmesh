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
    result = einmesh("* x y z", x=x_space, y=y_space, z=z_space)

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
    result = einmesh("(x y)", x=x_space, y=y_space)

    # Basic check that we get a tensor
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert isinstance(result[0], torch.Tensor)
    assert isinstance(result[1], torch.Tensor)

    # Using einops.rearrange properly flattens the dimensions within parentheses
    # So the result should be a 1D tensor with shape (5*3,) = (15,)
    assert result[0].ndim == 1
    assert result[1].ndim == 1
    assert result[0].shape[0] == 5 * 3  # 15 elements total
    assert result[1].shape[0] == 5 * 3  # 15 elements total


def test_einmesh_output_dimension_ordering():
    """Test that einmesh respects dimension ordering in output pattern."""
    x_space = LinSpace(0.0, 1.0, 5)
    y_space = LinSpace(0.0, 1.0, 3)

    # Get original meshes
    x1_mesh, y1_mesh = einmesh("x y", x=x_space, y=y_space)
    y2_mesh, x2_mesh = einmesh("y x", x=x_space, y=y_space)

    # Ensure results are transposed
    assert torch.allclose(x1_mesh, x2_mesh.transpose(1, 0))
    assert torch.allclose(y1_mesh, y2_mesh.transpose(1, 0))


def test_star_position():
    """Test that einmesh handles star position correctly."""
    x_space = LinSpace(0.0, 1.0, 7)
    y_space = LinSpace(0.0, 1.0, 9)

    result = einmesh("* x y", x=x_space, y=y_space)
    assert isinstance(result, torch.Tensor)
    assert result.shape == (2, 7, 9)

    result = einmesh("x * y", x=x_space, y=y_space)
    assert isinstance(result, torch.Tensor)
    assert result.shape == (7, 2, 9)

    result = einmesh("x y *", x=x_space, y=y_space)
    assert isinstance(result, torch.Tensor)
    assert result.shape == (7, 9, 2)


def test_axis_collection():
    """Test that einmesh handles axis collection correctly."""
    x_space = LinSpace(0.0, 1.0, 5)
    y_space = LinSpace(0.0, 1.0, 3)

    result = einmesh("* (x y)", x=x_space, y=y_space)
    assert isinstance(result, torch.Tensor)
    assert result.shape == (2, 5 * 3)

    result = einmesh("(x y) *", x=x_space, y=y_space)
    assert isinstance(result, torch.Tensor)
    assert result.shape == (5 * 3, 2)


def test_star_in_axis_collection():
    """Test that einmesh handles star in axis collection correctly."""
    x_space = LinSpace(0.0, 1.0, 5)
    y_space = LinSpace(0.0, 1.0, 3)

    result = einmesh("(* x) y", x=x_space, y=y_space)
    assert isinstance(result, torch.Tensor)
    assert result.shape == (2 * 5, 3)


def test_invalid_pattern():
    """Test that einmesh raises error for invalid patterns."""
    x_space = LinSpace(0.0, 1.0, 5)

    with pytest.raises(UndefinedSpaceError):
        einmesh("x y", x=x_space)  # Missing y space
