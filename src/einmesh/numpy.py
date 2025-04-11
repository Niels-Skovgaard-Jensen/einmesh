import functools

from einmesh._backends import NumpyBackend
from einmesh.parser import einmesh

einmesh = functools.partial(einmesh, backend=NumpyBackend())

__all__ = ["einmesh"]
