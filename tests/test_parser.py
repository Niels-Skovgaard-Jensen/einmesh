import pytest
import torch

from einmesh import einmesh
from einmesh.exceptions import (
    InvalidListTypeError,
    UndefinedSpaceError,
    UnsupportedSpaceTypeError,
)
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


def test_einmesh_auto_conversion():
    """Test automatic conversion of int, float, and list kwargs."""
    x_space = LinSpace(0.0, 1.0, 3)

    # Test int -> ConstantSpace
    x_coords, y_coords = einmesh("x y", x=x_space, y=5)
    assert isinstance(y_coords, torch.Tensor)
    assert y_coords.shape == (3, 1)
    assert torch.all(y_coords == 5.0)

    # Test float -> ConstantSpace
    x_coords, z_coords = einmesh("x z", x=x_space, z=-2.5)
    assert isinstance(z_coords, torch.Tensor)
    assert z_coords.shape == (3, 1)
    assert torch.all(z_coords == -2.5)

    # Test list[int] -> ListSpace
    list_int = [1, 2, 3, 4]
    x_coords, w_coords = einmesh("x w", x=x_space, w=list_int)  # type: ignore[arg-type]
    assert isinstance(w_coords, torch.Tensor)
    assert w_coords.shape == (3, 4)
    assert torch.allclose(w_coords[0], torch.tensor(list_int, dtype=torch.float))

    # Test list[float] -> ListSpace
    list_float = [1.1, 2.2, 3.3]
    x_coords, v_coords = einmesh("x v", x=x_space, v=list_float)
    assert isinstance(v_coords, torch.Tensor)
    assert v_coords.shape == (3, 3)
    assert torch.allclose(v_coords[0], torch.tensor(list_float, dtype=torch.float))

    # Test mixed list[int | float] -> ListSpace
    list_mixed = [1, 2.5, 3]
    x_coords, u_coords = einmesh("x u", x=x_space, u=list_mixed)
    assert isinstance(u_coords, torch.Tensor)
    assert u_coords.shape == (3, 3)
    assert torch.allclose(u_coords[0], torch.tensor(list_mixed, dtype=torch.float))

    # Test combination with explicit SpaceType and stacking
    stacked_res = einmesh("* x y z", x=x_space, y=100, z=[10, 20])
    assert isinstance(stacked_res, torch.Tensor)
    assert stacked_res.shape == (3, 3, 1, 2)  # Stack, x, y (const), z (list)
    assert torch.all(stacked_res[1] == 100.0)
    assert torch.allclose(stacked_res[2, 0, 0, :], torch.tensor([10.0, 20.0]))


def test_einmesh_auto_conversion_errors():
    """Test errors raised during automatic type conversion."""
    x_space = LinSpace(0.0, 1.0, 3)

    # Test invalid type (string)
    with pytest.raises(UnsupportedSpaceTypeError) as exc_info_type:
        einmesh("x y", x=x_space, y="not a number")  # type: ignore[arg-type] # Intentionally passing invalid type
    assert "Unsupported type for space 'y': str" in str(exc_info_type.value)

    # Test list with invalid contents (string)
    with pytest.raises(InvalidListTypeError) as exc_info_list:
        einmesh("x z", x=x_space, z=[1, 2, "three"])  # type: ignore[arg-type, list-item] # Intentionally passing invalid list item type
    assert "List provided for space 'z' must contain only int or float" in str(exc_info_list.value)
    # Check that the message correctly identifies the invalid type ('str')
    assert "got types: [str]" in str(exc_info_list.value)

    x_coords_empty, w_coords_empty = einmesh("x w", x=x_space, w=[])
    assert w_coords_empty.shape == (3, 0)  # Shape should reflect the empty list dimension
