from typing import Any

from ._backends import AbstractBackend
from ._parser import KwargValueType, _einmesh


class _EinMesher:
    def __init__(
        self,
        pattern: str,
        *unamed_spaces: KwargValueType,
        backend: AbstractBackend | str = "numpy",
        **named_spaces: KwargValueType,
    ) -> None:
        self.pattern: str = pattern
        self.backend: AbstractBackend | str = backend
        self.named_spaces: dict[str, KwargValueType] = named_spaces
        self.unamed_spaces: tuple[KwargValueType, ...] = unamed_spaces

    def sample(self) -> Any | tuple[Any, ...]:
        return _einmesh(self.pattern, *self.unamed_spaces, backend=self.backend, **self.named_spaces)
