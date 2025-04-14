from typing import TYPE_CHECKING

from einmesh._backends import TorchBackend
from einmesh._parser import _einmesh
from einmesh.spaces import SpaceType

if TYPE_CHECKING:
    import torch  # pyright: ignore[reportMissingImports]


def einmesh(pattern: str, **kwargs: SpaceType) -> "torch.Tensor":
    return _einmesh(pattern, backend=TorchBackend(), **kwargs)  # pyright: ignore[reportReturnType]
