from typing import Any

from ._backends import AbstractBackend
from ._parser import SpaceType, _einmesh


class EinMesher:
    def __init__(self, pattern: str, **spaces: SpaceType):
        self.pattern: str = pattern
        self.spaces: dict[str, SpaceType] = spaces

    def mesh(self, backend: AbstractBackend | str = "numpy") -> Any | tuple[Any, ...]:
        return _einmesh(self.pattern, backend, **self.spaces)
