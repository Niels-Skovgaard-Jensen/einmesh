import functools

from einmesh._backends import TorchBackend
from einmesh.parser import einmesh

einmesh = functools.partial(einmesh, backend=TorchBackend)

__all__ = ["einmesh"]
