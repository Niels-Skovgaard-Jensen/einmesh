import functools

from einmesh._backends import JaxBackend
from einmesh.parser import einmesh

einmesh = functools.partial(einmesh, backend=JaxBackend())

__all__ = ["einmesh"]
