import pytest
import torch

from einmesh import einmesh
from einmesh.parser import UndefinedSpaceError
from einmesh.spaces import (
    ConstantSpace,
    LinSpace,
    ListSpace,
    LogSpace,
    NormalDistribution,
    UniformDistribution,
)


def test_linear_space():
    # Test initialization
    lin_space = LinSpace(start=0.0, end=1.0, num=5)
    assert lin_space.start == 0.0
    assert lin_space.end == 1.0
    assert lin_space.num == 5

    # Test sampling
    samples = lin_space._sample()
    assert isinstance(samples, torch.Tensor)
    assert samples.shape == (5,)
    assert torch.allclose(samples, torch.linspace(0.0, 1.0, 5))


def test_log_space():
    # Test initialization
    log_space = LogSpace(start=0.0, end=1.0, num=5, base=10)
    assert log_space.start == 0.0
    assert log_space.end == 1.0
    assert log_space.num == 5
    assert log_space.base == 10

    # Test sampling
    samples = log_space._sample()
    assert isinstance(samples, torch.Tensor)
    assert samples.shape == (5,)
    assert torch.allclose(samples, torch.logspace(0.0, 1.0, 5, base=10))


def test_normal_distribution():
    # Test initialization
    normal_dist = NormalDistribution(mean=0.0, std=1.0, num=1000)
    assert normal_dist.mean == 0.0
    assert normal_dist.std == 1.0
    assert normal_dist.num == 1000

    # Test sampling
    samples = normal_dist._sample()
    assert isinstance(samples, torch.Tensor)
    assert samples.shape == (1000,)

    # Statistical tests (approximate due to random nature)
    assert abs(samples.mean().item() - normal_dist.mean) < 0.1
    assert abs(samples.std().item() - normal_dist.std) < 0.1


def test_uniform_distribution():
    # Test initialization
    uniform_dist = UniformDistribution(low=-1.0, high=1.0, num=1000)
    assert uniform_dist.low == -1.0
    assert uniform_dist.high == 1.0
    assert uniform_dist.num == 1000

    # Test sampling
    samples = uniform_dist._sample()
    assert isinstance(samples, torch.Tensor)
    assert samples.shape == torch.Size([1000])

    # Check bounds
    assert torch.all(samples >= -1.0)
    assert torch.all(samples <= 1.0)


def test_constant_space():
    # Test initialization
    const_space = ConstantSpace(value=5.0, num=3)
    assert const_space.value == 5.0
    assert const_space.num == 3

    # Test sampling
    samples = const_space._sample()
    assert isinstance(samples, torch.Tensor)
    assert samples.shape == (3,)
    assert torch.all(samples == 5.0)

    # Test default num=1
    const_space_single = ConstantSpace(value=-2.0)
    assert const_space_single.num == 1
    samples_single = const_space_single._sample()
    assert samples_single.shape == (1,)
    assert samples_single.item() == -2.0


def test_list_space():
    # Test initialization
    test_values = [1.1, 2.2, 3.3, 4.4]
    list_space = ListSpace(values=test_values)
    assert list_space.values == test_values

    # Test sampling
    samples = list_space._sample()
    assert isinstance(samples, torch.Tensor)
    assert samples.shape == (len(test_values),)
    assert torch.allclose(samples, torch.tensor(test_values))


def test_einmesh_integration():
    # Test einmesh with multiple spaces
    x_space = LinSpace(0.0, 1.0, 5)
    y_space = LogSpace(0.0, 1.0, 3)

    meshes = einmesh("x y", x=x_space, y=y_space)

    assert len(meshes) == 2
    assert isinstance(meshes[0], torch.Tensor)
    assert isinstance(meshes[1], torch.Tensor)
    assert meshes[0].shape == (5, 3)
    assert meshes[1].shape == (5, 3)


def test_linsspace_integration():
    x_space = LinSpace(0.0, 1.0, 5)
    y_space = LinSpace(0.0, 1.0, 3)

    meshes = einmesh("x y", x=x_space, y=y_space)

    assert len(meshes) == 2


def test_logspace_integration():
    x_space = LinSpace(0.0, 1.0, 5)
    y_space = LogSpace(0.0, 1.0, 3)

    meshes = einmesh("x y", x=x_space, y=y_space)

    assert len(meshes) == 2


def test_normaldistribution_integration():
    x_space = LinSpace(0.0, 1.0, 5)
    y_space = NormalDistribution(0.0, 1.0, 3)

    meshes = einmesh("x y", x=x_space, y=y_space)

    assert len(meshes) == 2


def test_uniformdistribution_integration():
    x_space = LinSpace(0.0, 1.0, 5)
    y_space = UniformDistribution(0.0, 1.0, 3)

    meshes = einmesh("x y", x=x_space, y=y_space)

    assert len(meshes) == 2


def test_constantspace_integration():
    x_space = LinSpace(0.0, 1.0, 5)
    y_space = ConstantSpace(value=7.0, num=3)

    meshes = einmesh("x y", x=x_space, y=y_space)

    assert len(meshes) == 2
    assert meshes[0].shape == (5, 3)
    assert meshes[1].shape == (5, 3)
    # Check if y values are constant
    assert torch.all(meshes[1] == 7.0)


def test_listspace_integration():
    x_space = LinSpace(0.0, 1.0, 5)
    list_values = [10.0, 20.0, 30.0]
    y_space = ListSpace(values=list_values)

    meshes = einmesh("x y", x=x_space, y=y_space)

    assert len(meshes) == 2
    assert meshes[0].shape == (5, 3)
    assert meshes[1].shape == (5, 3)
    # Check if y values match the list across the x dimension
    expected_y = torch.tensor(list_values).unsqueeze(0).expand(5, -1)
    assert torch.allclose(meshes[1], expected_y)


def test_invalid_space():
    with pytest.raises(UndefinedSpaceError) as exc_info:
        einmesh("x y", x=LinSpace(0.0, 1.0, 5))  # Missing y space
    assert str(exc_info.value) == "Undefined space: y"
