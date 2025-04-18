from __future__ import annotations

from typing import TYPE_CHECKING, Any

from einmesh._backends import NumpyBackend
from einmesh._parser import _einmesh
from einmesh.spaces import SpaceType

if TYPE_CHECKING:
    import numpy as np  # pyright: ignore[reportMissingImports]


def einmesh(pattern: str, **kwargs: SpaceType) -> np.ndarray[Any, Any] | tuple[np.ndarray[Any, Any], ...]:
    return _einmesh(pattern, backend=NumpyBackend(), **kwargs)
