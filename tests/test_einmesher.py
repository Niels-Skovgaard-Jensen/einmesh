from numpy.testing import assert_allclose

from einmesh import (
    ConstantSpace,
    LinSpace,
    ListSpace,
    NormalDistribution,
    UniformDistribution,
)
from einmesh._backends import AbstractBackend
from einmesh._einmesher import _EinMesher
from tests.conftest import parametrize_backends


@parametrize_backends
def test_einmesher_init(backend):
    """Test EinMesher initialization."""
    pattern = "x y"
    spaces = {"x": LinSpace(0, 1, 3), "y": ConstantSpace(5)}
    mesher = _EinMesher(pattern, backend=backend, **spaces)
    assert mesher.pattern == pattern
    assert mesher.spaces == spaces


@parametrize_backends
def test_einmesher_basic_mesh(backend_cls: type[AbstractBackend]):
    """Test basic mesh generation (tuple output)."""
    backend = backend_cls()
    mesher = _EinMesher(
        "x y",
        x=LinSpace(0, 1, 3),
        y=ConstantSpace(10.0),
        backend=backend,
    )
    x_coords, y_coords = mesher.sample()

    expected_x = backend.tensor([[0.0], [0.5], [1.0]])
    expected_y = backend.tensor([[10.0], [10.0], [10.0]])

    assert backend.is_appropriate_type(x_coords)
    assert backend.is_appropriate_type(y_coords)
    assert_allclose(x_coords, expected_x)
    assert_allclose(y_coords, expected_y)
    assert backend.shape(x_coords) == (3, 1)
    assert backend.shape(y_coords) == (3, 1)


@parametrize_backends
def test_einmesher_stacked_mesh(backend_cls: type[AbstractBackend]):
    """Test stacked mesh generation (single tensor output)."""
    backend = backend_cls()
    mesher = _EinMesher(
        "* x y",
        x=LinSpace(0, 1, 3),
        y=ListSpace([10.0, 20.0]),
        backend=backend,
    )
    stacked_grid = mesher.sample()

    expected_grid = backend.tensor([
        [[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]],  # x coords duplicated for y
        [[10.0, 20.0], [10.0, 20.0], [10.0, 20.0]],  # y coords broadcasted for x
    ])

    assert backend.is_appropriate_type(stacked_grid)
    assert backend.shape(stacked_grid) == (2, 3, 2)
    assert_allclose(stacked_grid, expected_grid)


@parametrize_backends
def test_einmesher_duplicate_names(backend_cls: type[AbstractBackend]):
    """Test mesh generation with duplicate names."""
    backend = backend_cls()
    mesher = _EinMesher("x x y", x=LinSpace(0, 1, 2), y=ConstantSpace(5), backend=backend)
    x0_coords, x1_coords, y_coords = mesher.sample()

    expected_x0 = backend.tensor([[[0.0], [0.0]], [[1.0], [1.0]]])  # Shape (2, 2, 1)
    expected_x1 = backend.tensor([[[0.0], [1.0]], [[0.0], [1.0]]])  # Shape (2, 2, 1)
    expected_y = backend.tensor([[[5.0], [5.0]], [[5.0], [5.0]]])  # Shape (2, 2, 1)

    assert backend.is_appropriate_type(x0_coords)
    assert backend.is_appropriate_type(x1_coords)
    assert backend.is_appropriate_type(y_coords)
    assert_allclose(x0_coords, expected_x0)
    assert_allclose(x1_coords, expected_x1)
    assert_allclose(y_coords, expected_y)
    assert backend.shape(x0_coords) == (2, 2, 1)
    assert backend.shape(x1_coords) == (2, 2, 1)
    assert backend.shape(y_coords) == (2, 2, 1)


@parametrize_backends
def test_einmesher_grouping(backend_cls: type[AbstractBackend]):
    """Test mesh generation with grouping."""
    backend = backend_cls()
    mesher = _EinMesher("x (y z)", x=LinSpace(0, 1, 2), y=LinSpace(10, 20, 3), z=ConstantSpace(5), backend=backend)
    x_coords, y_coords, z_coords = mesher.sample()

    expected_x = backend.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    expected_y = backend.tensor([[10.0, 15.0, 20.0], [10.0, 15.0, 20.0]])
    expected_z = backend.tensor([[5.0, 5.0, 5.0], [5.0, 5.0, 5.0]])

    assert backend.is_appropriate_type(x_coords)
    assert backend.is_appropriate_type(y_coords)
    assert backend.is_appropriate_type(z_coords)
    assert backend.shape(x_coords) == (2, 3)
    assert backend.shape(y_coords) == (2, 3)
    assert backend.shape(z_coords) == (2, 3)
    assert_allclose(x_coords, expected_x)
    assert_allclose(y_coords, expected_y)
    assert_allclose(z_coords, expected_z)


@parametrize_backends
def test_einmesher_stacked_grouping(backend_cls: type[AbstractBackend]):
    """Test stacked mesh generation with grouping."""
    backend = backend_cls()
    mesher = _EinMesher("* (x y)", x=LinSpace(0, 1, 2), y=ListSpace([10, 20]), backend=backend)
    stacked_grouped = mesher.sample()

    expected_stacked_grouped = backend.tensor([
        [0.0, 0.0, 1.0, 1.0],  # Flattened x_mesh
        [10.0, 20.0, 10.0, 20.0],  # Flattened y_mesh
    ])

    assert backend.is_appropriate_type(stacked_grouped)
    assert backend.shape(stacked_grouped) == (2, 4)
    assert_allclose(stacked_grouped, expected_stacked_grouped)


@parametrize_backends
def test_einmesher_random_resampling(backend_cls: type[AbstractBackend]):
    """Verify that random spaces are resampled on each mesh() call."""
    backend = backend_cls()
    mesher = _EinMesher(
        "norm uni const",
        norm=NormalDistribution(mean=0, std=1, num=5),
        uni=UniformDistribution(low=10, high=20, num=5),
        const=ConstantSpace(100),
        backend=backend,
    )

    # First call
    norm1, uni1, const1 = mesher.sample()

    # Second call
    norm2, uni2, const2 = mesher.sample()

    # Check shapes are consistent
    assert backend.shape(norm1) == backend.shape(norm2)
    assert backend.shape(uni1) == backend.shape(uni2)
    assert backend.shape(const1) == backend.shape(const2)

    # Check that random spaces generated different values
    assert backend.any(norm1 != norm2)
    assert backend.any(uni1 != uni2)

    # Check that constant space remained the same
    assert_allclose(const1, const2)


@parametrize_backends
def test_einmesher_stacked_random_resampling(backend_cls: type[AbstractBackend]):
    """Verify resampling for stacked random spaces."""
    backend = backend_cls()
    mesher = _EinMesher(
        "* norm uni",
        norm=NormalDistribution(mean=5, std=2, num=6),
        uni=UniformDistribution(low=-10, high=0, num=6),
        backend=backend,
    )

    # First call
    stacked1 = mesher.sample()

    # Second call
    stacked2 = mesher.sample()

    # Check shapes
    assert backend.shape(stacked1) == backend.shape(stacked2)
    assert backend.shape(stacked1)[0] == 2  # Should have 2 layers stacked

    # Check that the layers corresponding to random spaces are different
    assert backend.any(stacked1[0] != stacked2[0])  # norm layer
    assert backend.any(stacked1[1] != stacked2[1])  # uni layer


# Example of how to test potential errors (though most are in _einmesh)
# @parametrize_backends
# def test_einmesher_invalid_pattern(backend):
#     backend_instance = backend()
#     with pytest.raises(SomeExpectedError):
#         mesher = EinMesher("x -> y", x=LinSpace(0,1,2))
#         mesher.sample(backend=backend.framework_name)
