from einmesh import einmesh
from einmesh.spaces import LinSpace
from tests.conftest import parametrize_backends


@parametrize_backends
def test_linear_space(backend):
    x_torch = backend.linspace(0, 1, 10)

    x_einmesh = einmesh("i", i=LinSpace(start=0, end=1, num=10), backend=backend)

    assert backend.allclose(x_torch, x_einmesh[0])


@parametrize_backends
def test_linear_space_2d(backend):
    x_torch = backend.linspace(0, 1, 10)
    y_torch = backend.linspace(0, 1, 10)

    x_torch, y_torch = backend.meshgrid(x_torch, y_torch, indexing="ij")

    x_einmesh = einmesh("i j", i=LinSpace(start=0, end=1, num=10), j=LinSpace(start=0, end=1, num=10), backend=backend)

    assert backend.allclose(x_torch, x_einmesh[0])
    assert backend.allclose(y_torch, x_einmesh[1])


@parametrize_backends
def test_linear_space_8d(backend):
    dims = [backend.linspace(0, 1, 10) for _ in range(8)]
    torch_meshes = backend.meshgrid(*dims, indexing="ij")

    einmesh_spaces = {f"dim{i}": LinSpace(start=0, end=1, num=10) for i in range(8)}
    einmesh_pattern = " ".join(einmesh_spaces.keys())
    einmesh_meshes = einmesh(einmesh_pattern, backend=backend, **einmesh_spaces)

    for torch_mesh, einmesh_mesh in zip(torch_meshes, einmesh_meshes):
        assert backend.allclose(torch_mesh, einmesh_mesh)
