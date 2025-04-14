from typing import TYPE_CHECKING

from einmesh._backends import JaxBackend
from einmesh._parser import _einmesh
from einmesh.spaces import SpaceType

if TYPE_CHECKING:
    import jax


def einmesh(pattern: str, **kwargs: SpaceType) -> jax.Array:
    return _einmesh(pattern, backend=JaxBackend(), **kwargs)  # pyright: ignore[reportReturnType]
