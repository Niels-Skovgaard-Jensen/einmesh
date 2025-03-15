import torch

from einmesh import einmesh
from einmesh.spaces import LinSpace


def test_linear_space():
    x_torch = torch.linspace(0, 1, 10)

    x_einmesh = einmesh("i", i=LinSpace(start=0, end=1, num=10))

    assert torch.allclose(x_torch, x_einmesh[0])


def test_linear_space_2d():
    x_torch = torch.linspace(0, 1, 10)
    y_torch = torch.linspace(0, 1, 10)

    x_torch, y_torch = torch.meshgrid(x_torch, y_torch, indexing="ij")

    x_einmesh = einmesh("i j", i=LinSpace(start=0, end=1, num=10), j=LinSpace(start=0, end=1, num=10))

    assert torch.allclose(x_torch, x_einmesh[0])
    assert torch.allclose(y_torch, x_einmesh[1])


def test_linear_space_8d():
    dims = [torch.linspace(0, 1, 10) for _ in range(8)]
    torch_meshes = torch.meshgrid(*dims, indexing="ij")

    einmesh_spaces = {f"dim{i}": LinSpace(start=0, end=1, num=10) for i in range(8)}
    einmesh_pattern = " ".join(einmesh_spaces.keys())
    einmesh_meshes = einmesh(einmesh_pattern, **einmesh_spaces)

    for torch_mesh, einmesh_mesh in zip(torch_meshes, einmesh_meshes):
        assert torch.allclose(torch_mesh, einmesh_mesh)
