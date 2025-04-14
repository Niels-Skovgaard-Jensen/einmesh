import pytest

from einmesh import einmesh
from einmesh._exceptions import (
    InvalidListTypeError,
    UndefinedSpaceError,
    UnsupportedSpaceTypeError,
)
from einmesh.spaces import LinSpace
from tests.conftest import parametrize_backends


@parametrize_backends
def test_einmesh_basic(backend):
    """Test the basic functionality of einmesh without output pattern."""
    x_space = LinSpace(0.0, 1.0, 5)
    y_space = LinSpace(0.0, 1.0, 3)

    meshes = einmesh("x y", x=x_space, y=y_space, backend=backend)

    assert len(meshes) == 2
    assert backend.is_appropriate_type(meshes[0])
    assert backend.is_appropriate_type(meshes[1])
    assert backend.shape(meshes[0]) == (5, 3)
    assert backend.shape(meshes[1]) == (5, 3)


@parametrize_backends
def test_einmesh_star_pattern(backend):
    """Test einmesh with * pattern to stack all meshes."""
    x_space = LinSpace(0.0, 1.0, 5)
    y_space = LinSpace(0.0, 1.0, 3)
    z_space = LinSpace(0.0, 1.0, 2)

    # Using just *
    result = einmesh("* x y z", x=x_space, y=y_space, z=z_space, backend=backend)

    assert backend.is_appropriate_type(result)
    assert backend.shape(result) == (3, 5, 3, 2)  # 3 meshes stacked as first dimension

    # Check that the result contains the original meshes
    x_mesh, y_mesh, z_mesh = einmesh("x y z", x=x_space, y=y_space, z=z_space, backend=backend)
    assert backend.allclose(result[0], x_mesh)
    assert backend.allclose(result[1], y_mesh)
    assert backend.allclose(result[2], z_mesh)


@parametrize_backends
def test_einmesh_parentheses_pattern(backend):
    """Test einmesh with parentheses pattern to reshape dimensions."""
    x_space = LinSpace(0.0, 1.0, 5)
    y_space = LinSpace(0.0, 1.0, 3)

    # Using pattern with parentheses
    result = einmesh("(x y)", x=x_space, y=y_space, backend=backend)

    # Basic check that we get a tensor
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert backend.is_appropriate_type(result[0])
    assert backend.is_appropriate_type(result[1])

    # Using einops.rearrange properly flattens the dimensions within parentheses
    # So the result should be a 1D tensor with shape (5*3,) = (15,)
    assert backend.shape(result[0]) == (5 * 3,)
    assert backend.shape(result[1]) == (5 * 3,)


@parametrize_backends
def test_einmesh_output_dimension_ordering(backend):
    """Test that einmesh respects dimension ordering in output pattern."""
    x_space = LinSpace(0.0, 1.0, 5)
    y_space = LinSpace(0.0, 1.0, 3)

    # Get original meshes
    x1_mesh, y1_mesh = einmesh("x y", x=x_space, y=y_space, backend=backend)
    y2_mesh, x2_mesh = einmesh("y x", x=x_space, y=y_space, backend=backend)

    # Ensure results are transposed
    assert backend.allclose(x1_mesh, x2_mesh.transpose(1, 0))
    assert backend.allclose(y1_mesh, y2_mesh.transpose(1, 0))


@parametrize_backends
def test_star_position(backend):
    """Test that einmesh handles star position correctly."""
    x_space = LinSpace(0.0, 1.0, 7)
    y_space = LinSpace(0.0, 1.0, 9)

    result = einmesh("* x y", x=x_space, y=y_space, backend=backend)
    assert backend.is_appropriate_type(result)
    assert backend.shape(result) == (2, 7, 9)

    result = einmesh("x * y", x=x_space, y=y_space, backend=backend)
    assert backend.is_appropriate_type(result)
    assert backend.shape(result) == (7, 2, 9)

    result = einmesh("x y *", x=x_space, y=y_space, backend=backend)
    assert backend.is_appropriate_type(result)
    assert backend.shape(result) == (7, 9, 2)


@parametrize_backends
def test_axis_collection(backend):
    """Test that einmesh handles axis collection correctly."""
    x_space = LinSpace(0.0, 1.0, 5)
    y_space = LinSpace(0.0, 1.0, 3)

    result = einmesh("* (x y)", x=x_space, y=y_space, backend=backend)
    assert backend.is_appropriate_type(result)
    assert backend.shape(result) == (2, 5 * 3)

    result = einmesh("(x y) *", x=x_space, y=y_space, backend=backend)
    assert backend.is_appropriate_type(result)
    assert backend.shape(result) == (5 * 3, 2)


@parametrize_backends
def test_star_in_axis_collection(backend):
    """Test that einmesh handles star in axis collection correctly."""
    x_space = LinSpace(0.0, 1.0, 5)
    y_space = LinSpace(0.0, 1.0, 3)

    result = einmesh("(* x) y", x=x_space, y=y_space, backend=backend)
    assert backend.is_appropriate_type(result)
    assert backend.shape(result) == (2 * 5, 3)


@parametrize_backends
def test_invalid_pattern(backend):
    """Test that einmesh raises error for invalid patterns."""
    x_space = LinSpace(0.0, 1.0, 5)

    with pytest.raises(UndefinedSpaceError):
        einmesh("x y", x=x_space, backend=backend)  # Missing y space


@parametrize_backends
def test_einmesh_auto_conversion(backend):
    """Test automatic conversion of int, float, and list kwargs."""
    x_space = LinSpace(0.0, 1.0, 3)

    # Test int -> ConstantSpace
    x_coords, y_coords = einmesh("x y", x=x_space, y=5, backend=backend)
    assert backend.is_appropriate_type(y_coords)
    assert backend.shape(y_coords) == (3, 1)
    assert backend.all(y_coords == 5.0)

    # Test float -> ConstantSpace
    x_coords, z_coords = einmesh("x z", x=x_space, z=-2.5, backend=backend)
    assert backend.is_appropriate_type(z_coords)
    assert backend.shape(z_coords) == (3, 1)
    assert backend.all(z_coords == -2.5)

    # Test list[int] -> ListSpace
    list_int = [1, 2, 3, 4]
    x_coords, w_coords = einmesh("x w", x=x_space, w=list_int, backend=backend)  # type: ignore[arg-type]
    assert backend.is_appropriate_type(w_coords)
    assert backend.shape(w_coords) == (3, 4)
    assert backend.allclose(w_coords[0], backend.tensor(list_int, dtype=backend.float))

    # Test list[float] -> ListSpace
    list_float = [1.1, 2.2, 3.3]
    x_coords, v_coords = einmesh("x v", x=x_space, v=list_float, backend=backend)
    assert backend.is_appropriate_type(v_coords)
    assert backend.shape(v_coords) == (3, 3)
    assert backend.allclose(v_coords[0], backend.tensor(list_float, dtype=backend.float))

    # Test mixed list[int | float] -> ListSpace
    list_mixed = [1, 2.5, 3]
    x_coords, u_coords = einmesh("x u", x=x_space, u=list_mixed, backend=backend)
    assert backend.is_appropriate_type(u_coords)
    assert backend.shape(u_coords) == (3, 3)
    assert backend.allclose(u_coords[0], backend.tensor(list_mixed, dtype=backend.float))

    # Test combination with explicit SpaceType and stacking
    stacked_res = einmesh("* x y z", x=x_space, y=100, z=[10, 20], backend=backend)
    assert backend.is_appropriate_type(stacked_res)
    assert backend.shape(stacked_res) == (3, 3, 1, 2)  # Stack, x, y (const), z (list)
    assert backend.all(stacked_res[1] == 100.0)
    assert backend.allclose(stacked_res[2, 0, 0, :], backend.tensor([10.0, 20.0]))


@parametrize_backends
def test_einmesh_auto_conversion_errors(backend):
    """Test errors raised during automatic type conversion."""
    x_space = LinSpace(0.0, 1.0, 3)

    # Test invalid type (string)
    with pytest.raises(UnsupportedSpaceTypeError) as exc_info_type:
        einmesh("x y", x=x_space, y="not a number", backend=backend)  # type: ignore[arg-type] # Intentionally passing invalid type
    assert "Unsupported type for space 'y': str" in str(exc_info_type.value)

    # Test list with invalid contents (string)
    with pytest.raises(InvalidListTypeError) as exc_info_list:
        einmesh("x z", x=x_space, z=[1, 2, "three"], backend=backend)  # type: ignore[arg-type, list-item] # Intentionally passing invalid list item type
    assert "List provided for space 'z' must contain only int or float" in str(exc_info_list.value)
    # Check that the message correctly identifies the invalid type ('str')
    assert "got types: [str]" in str(exc_info_list.value)

    x_coords_empty, w_coords_empty = einmesh("x w", x=x_space, w=[], backend=backend)
    assert backend.shape(w_coords_empty) == (3, 0)  # Shape should reflect the empty list dimension
