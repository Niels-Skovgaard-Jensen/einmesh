from __future__ import annotations

from typing import TYPE_CHECKING

from einmesh._backends import TorchBackend
from einmesh._einmesher import _EinMesher
from einmesh._parser import _einmesh  # pyright: ignore[reportPrivateUsage]
from einmesh.spaces import SpaceType

if TYPE_CHECKING:
    import torch  # pyright: ignore[reportMissingImports]


def einmesh(pattern: str, **kwargs: SpaceType) -> torch.Tensor:
    return _einmesh(pattern, backend=TorchBackend(), **kwargs)  # pyright: ignore[reportReturnType]


# Expose a preconfigured EinMesher class bound to Torch backend
class EinMesher(_EinMesher):
    """
    EinMesher bound to the Torch backend. By default `.mesh()` will return PyTorch tensors
    without needing to pass a backend.
    """

    def __init__(self, pattern: str, **spaces: SpaceType):
        super().__init__(pattern, backend=TorchBackend(), **spaces)

    def sample(self) -> torch.Tensor | tuple[torch.Tensor, ...]:
        return super().sample()
