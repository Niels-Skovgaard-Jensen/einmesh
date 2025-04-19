from __future__ import annotations

from typing import TYPE_CHECKING, Any

from einmesh._backends import NumpyBackend
from einmesh._einmesher import _EinMesher
from einmesh._parser import _einmesh
from einmesh.spaces import SpaceType

if TYPE_CHECKING:
    import numpy as np  # pyright: ignore[reportMissingImports]


def einmesh(pattern: str, **kwargs: SpaceType) -> np.ndarray[Any, Any] | tuple[np.ndarray[Any, Any], ...]:
    return _einmesh(pattern, backend=NumpyBackend(), **kwargs)


# Expose a preconfigured EinMesher class bound to NumPy backend
class EinMesher(_EinMesher):
    """
    EinMesher bound to the NumPy backend. By default `.mesh()` will return NumPy arrays
    without needing to pass a backend.
    """

    def __init__(self, pattern: str, **spaces: SpaceType) -> None:
        super().__init__(pattern, backend=NumpyBackend(), **spaces)

    def sample(self) -> np.ndarray[Any, Any] | tuple[np.ndarray[Any, Any], ...]:
        return super().sample()
