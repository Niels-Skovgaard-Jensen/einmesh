from typing import Any

import einops
import pytest

from einmesh._backends import AbstractBackend
from einmesh._parser import UndefinedSpaceError
from einmesh._parser import _einmesh as einmesh
from einmesh.spaces import (
    ConstantSpace,
    LinSpace,
    ListSpace,
    LogSpace,
    NormalDistribution,
    UniformDistribution,
)

# Try explicit absolute import
from tests.conftest import parametrize_backends


@parametrize_backends
def test_linear_space(backend_cls: type[AbstractBackend]):
    # Test initialization
    backend = backend_cls()
    lin_space = LinSpace(start=0.0, end=1.0, num=5)
    assert lin_space.start == 0.0
    assert lin_space.end == 1.0
    assert lin_space.num == 5

    # Test sampling
    samples = lin_space._sample(backend)
    assert backend.is_appropriate_type(samples)
    assert samples.shape == (5,)
    assert backend.allclose(samples, backend.linspace(0.0, 1.0, 5))


@parametrize_backends
def test_log_space(backend_cls: type[AbstractBackend]):
    # Test initialization
    backend = backend_cls()
    log_space = LogSpace(start=0.0, end=1.0, num=5, base=10)
    assert log_space.start == 0.0
    assert log_space.end == 1.0
    assert log_space.num == 5
    assert log_space.base == 10

    # Test sampling
    samples = log_space._sample(backend)
    assert backend.is_appropriate_type(samples)
    assert samples.shape == (5,)
    assert backend.allclose(samples, backend.logspace(0.0, 1.0, 5, base=10))


@parametrize_backends
def test_normal_distribution(backend_cls: type[AbstractBackend]):
    # Test initialization
    backend = backend_cls()
    normal_dist = NormalDistribution(mean=0.0, std=1.0, num=1000)
    assert normal_dist.mean == 0.0
    assert normal_dist.std == 1.0
    assert normal_dist.num == 1000

    # Test sampling
    samples = normal_dist._sample(backend)
    assert backend.is_appropriate_type(samples)
    assert samples.shape == (1000,)

    # Statistical tests (approximate due to random nature)
    assert abs(samples.mean().item() - normal_dist.mean) < 0.1
    assert abs(samples.std().item() - normal_dist.std) < 0.1


@parametrize_backends
def test_uniform_distribution(backend_cls: type[AbstractBackend]):
    # Test initialization
    backend = backend_cls()
    uniform_dist = UniformDistribution(low=-1.0, high=1.0, num=1000)
    assert uniform_dist.low == -1.0
    assert uniform_dist.high == 1.0
    assert uniform_dist.num == 1000

    # Test sampling
    samples = uniform_dist._sample(backend)
    assert backend.is_appropriate_type(samples)
    assert backend.shape(samples) == (1000,)

    # Check bounds - adapt comparison for backend
    assert backend.all(samples >= -1.0)
    assert backend.all(samples <= 1.0)


@parametrize_backends
def test_constant_space(backend_cls: type[AbstractBackend]):
    # Test initialization
    backend = backend_cls()
    const_space = ConstantSpace(value=5.0, num=3)
    assert const_space.value == 5.0
    assert const_space.num == 3

    # Test sampling
    samples = const_space._sample(backend)
    assert backend.is_appropriate_type(samples)
    assert backend.shape(samples) == (3,)
    assert backend.allclose(samples, backend.tensor([5.0, 5.0, 5.0]))

    # Test default num=1
    const_space_single = ConstantSpace(value=-2.0)
    assert const_space_single.num == 1
    samples_single = const_space_single._sample(backend)
    assert samples_single.shape == (1,)
    assert backend.allclose(samples_single, backend.tensor([-2.0]))


@parametrize_backends
def test_list_space(backend_cls: type[AbstractBackend]):
    # Test initialization
    backend = backend_cls()
    test_values = [1.1, 2.2, 3.3, 4.4]
    list_space = ListSpace(values=test_values)
    assert list_space.values == test_values

    # Test sampling
    samples = list_space._sample(backend)
    assert backend.is_appropriate_type(samples)
    assert samples.shape == (len(test_values),)

    # Convert test values to tensor using appropriate backend
    expected = backend.tensor(test_values)
    assert backend.allclose(samples, expected)


@parametrize_backends
def test_einmesh_integration(backend_cls: type[AbstractBackend]):
    # Test einmesh with multiple spaces
    backend = backend_cls()
    x_space = LinSpace(0.0, 1.0, 5)
    y_space = LogSpace(0.0, 1.0, 3)

    meshes = einmesh("x y", x=x_space, y=y_space, backend=backend)

    assert len(meshes) == 2
    assert backend.is_appropriate_type(meshes[0])
    assert backend.is_appropriate_type(meshes[1])
    assert meshes[0].shape == (5, 3)
    assert meshes[1].shape == (5, 3)


@parametrize_backends
def test_linsspace_integration(backend_cls: type[AbstractBackend]):
    backend = backend_cls()
    x_space = LinSpace(0.0, 1.0, 5)
    y_space = LinSpace(0.0, 1.0, 3)

    meshes = einmesh("x y", x=x_space, y=y_space, backend=backend)

    assert len(meshes) == 2


@parametrize_backends
def test_logspace_integration(backend_cls: type[AbstractBackend]):
    backend = backend_cls()
    x_space = LinSpace(0.0, 1.0, 5)
    y_space = LogSpace(0.0, 1.0, 3)

    meshes = einmesh("x y", x=x_space, y=y_space, backend=backend)

    assert len(meshes) == 2


@parametrize_backends
def test_normaldistribution_integration(backend_cls: type[AbstractBackend]):
    backend = backend_cls()
    x_space = LinSpace(0.0, 1.0, 5)
    y_space = NormalDistribution(0.0, 1.0, 3)

    meshes = einmesh("x y", x=x_space, y=y_space, backend=backend)

    assert len(meshes) == 2


@parametrize_backends
def test_uniformdistribution_integration(backend_cls: type[AbstractBackend]):
    backend = backend_cls()
    x_space = LinSpace(0.0, 1.0, 5)
    y_space = UniformDistribution(0.0, 1.0, 3)

    meshes = einmesh("x y", x=x_space, y=y_space, backend=backend)

    assert len(meshes) == 2


@parametrize_backends
def test_constantspace_integration(backend_cls: type[AbstractBackend]):
    backend = backend_cls()
    x_space = LinSpace(0.0, 1.0, 5)
    y_space = ConstantSpace(value=7.0, num=3)

    meshes = einmesh("x y", x=x_space, y=y_space, backend=backend)

    assert len(meshes) == 2
    assert meshes[0].shape == (5, 3)
    assert meshes[1].shape == (5, 3)
    # Check if y values are constant
    assert backend.all(meshes[1] == 7.0)


@parametrize_backends
def test_listspace_integration(backend_cls: type[AbstractBackend]) -> None:
    backend = backend_cls()
    x_space: LinSpace = LinSpace(start=0.0, end=1.0, num=5)
    list_values: list[float] = [10.0, 20.0, 30.0]
    y_space: ListSpace = ListSpace(values=list_values)

    meshes: Any | tuple[Any, ...] = einmesh("x y", x=x_space, y=y_space, backend=backend)

    assert len(meshes) == 2
    assert meshes[0].shape == (5, 3)
    assert meshes[1].shape == (5, 3)
    # Check if y values match the list across the x dimension
    expected_y = einops.repeat(backend.tensor(list_values), "n -> h n", h=5)
    assert backend.allclose(meshes[1], expected_y)


@parametrize_backends
def test_invalid_space(backend_cls: type[AbstractBackend]):
    backend = backend_cls()
    with pytest.raises(UndefinedSpaceError) as exc_info:
        einmesh("x y", x=LinSpace(0.0, 1.0, 5), backend=backend)  # Missing y space
    assert str(exc_info.value) == "Undefined space: y"
