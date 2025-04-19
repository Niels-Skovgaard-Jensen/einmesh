from __future__ import annotations

from typing import TYPE_CHECKING

from einmesh._backends import JaxBackend
from einmesh._einmesher import _EinMesher as _BaseEinMesher
from einmesh._parser import _einmesh  # pyright: ignore[reportPrivateUsage]
from einmesh.spaces import SpaceType

if TYPE_CHECKING:
    import jax  # pyright: ignore[reportMissingImports] need for when jax is not installed


def einmesh(pattern: str, **kwargs: SpaceType) -> jax.Array:
    return _einmesh(pattern, backend=JaxBackend(), **kwargs)  # pyright: ignore[reportReturnType]


# Expose a preconfigured EinMesher class bound to JAX backend
class EinMesher(_BaseEinMesher):
    """
    EinMesher bound to the JAX backend. By default `.mesh()` will return JAX arrays
    without needing to pass a backend.
    """

    def __init__(self, pattern: str, **spaces: SpaceType) -> None:
        super().__init__(pattern, backend=JaxBackend(), **spaces)

    def sample(self) -> jax.Array | tuple[jax.Array, ...]:
        return super().sample()
