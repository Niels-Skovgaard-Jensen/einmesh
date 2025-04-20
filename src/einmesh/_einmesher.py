from typing import Any

from ._backends import AbstractBackend
from ._parser import SpaceType, _einmesh


class _EinMesher:
    def __init__(self, pattern: str, backend: AbstractBackend | str = "numpy", **spaces: SpaceType):
        self.pattern: str = pattern
        self.backend: AbstractBackend | str = backend
        self.spaces: dict[str, SpaceType] = spaces

    def sample(self) -> Any | tuple[Any, ...]:
        return _einmesh(self.pattern, self.backend, **self.spaces)
