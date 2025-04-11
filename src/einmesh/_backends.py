import sys
from abc import ABC, abstractmethod

_loaded_backends: dict = {}
_type2backend: dict = {}
_debug_importing = False



def get_backend(tensor) -> "AbstractBackend":
    """
    Takes a correct backend (e.g. numpy backend if tensor is numpy.ndarray) for a tensor.
    If needed, imports package and creates backend
    """
    _type = type(tensor)
    _result = _type2backend.get(_type)
    if _result is not None:
        return _result

    for _framework_name, backend in list(_loaded_backends.items()):
        if backend.is_appropriate_type(tensor):
            _type2backend[_type] = backend
            return backend

    # Find backend subclasses recursively
    backend_subclasses = []
    backends = AbstractBackend.__subclasses__()
    while backends:
        backend = backends.pop()
        backends += backend.__subclasses__()
        backend_subclasses.append(backend)

    for BackendSubclass in backend_subclasses:
        if _debug_importing:
            print("Testing for subclass of ", BackendSubclass)
        if BackendSubclass.framework_name not in _loaded_backends and BackendSubclass.framework_name in sys.modules:
            # check that module was already imported. Otherwise it can't be imported
            if _debug_importing:
                print("Imported backend for ", BackendSubclass.framework_name)
            backend = BackendSubclass()
            _loaded_backends[backend.framework_name] = backend
            if backend.is_appropriate_type(tensor):
                _type2backend[_type] = backend
                return backend

    raise UnknownBackendError(type(tensor))


class AbstractBackend(ABC):
    """Base backend class, major part of methods are only for debugging purposes."""

    framework_name: str

    def is_appropriate_type(self, tensor):
        """helper method should recognize tensors it can handle"""
        raise NotImplementedError()

    def __repr__(self):
        return f"<einmesh backend for {self.framework_name}>"

    @abstractmethod
    def linspace(self, start: float, stop: float, num: int):
        raise NotImplementedError()

    @abstractmethod
    def logspace(self, start: float, stop: float, num: int, base: float = 10):
        raise NotImplementedError()

    @abstractmethod
    def arange(self, start: float, stop: float, step: float = 1):
        raise NotImplementedError()

    @abstractmethod
    def full(self, shape, fill_value):
        raise NotImplementedError()

    @abstractmethod
    def tensor(self, data, dtype=None):
        raise NotImplementedError()

    @abstractmethod
    def normal(self, mean, std, size):
        raise NotImplementedError()

    @abstractmethod
    def rand(self, size):
        raise NotImplementedError()

    @abstractmethod
    def concat(self, tensors, axis: int):
        """concatenates tensors along axis.
        Assume identical across tensors: devices, dtypes and shapes except selected axis."""
        raise NotImplementedError()

    @abstractmethod
    def allclose(self, a, b, rtol=1e-05, atol=1e-08, equal_nan=False):
        raise NotImplementedError()

    @abstractmethod
    def shape(self, tensor):
        raise NotImplementedError()

    @abstractmethod
    def meshgrid(self, *tensors, indexing="ij"):
        raise NotImplementedError()

    @abstractmethod
    def size(self, tensor):
        raise NotImplementedError()

    @abstractmethod
    def all(self, tensor):
        raise NotImplementedError()

    @abstractmethod
    def stack(self, tensors, dim: int):
        raise NotImplementedError()

    @property
    @abstractmethod
    def float(self):
        raise NotImplementedError()

    @property
    @abstractmethod
    def float32(self):
        raise NotImplementedError()

    @property
    @abstractmethod
    def float64(self):
        raise NotImplementedError()

    @property
    @abstractmethod
    def int8(self):
        raise NotImplementedError()

    @property
    @abstractmethod
    def int16(self):
        raise NotImplementedError()

    @property
    @abstractmethod
    def int32(self):
        raise NotImplementedError()

    @property
    @abstractmethod
    def int64(self):
        raise NotImplementedError()

    @property
    @abstractmethod
    def bool(self):
        raise NotImplementedError()


class NumpyBackend(AbstractBackend):
    framework_name = "numpy"

    def __init__(self):
        import numpy

        self.np = numpy

    def is_appropriate_type(self, tensor):
        return isinstance(tensor, self.np.ndarray)

    def linspace(self, start, stop, num):
        return self.np.linspace(start, stop, num)

    def concat(self, tensors, axis: int):
        return self.np.concatenate(tensors, axis=axis)

    def logspace(self, start, stop, num, base=10):
        return self.np.logspace(start, stop, num, base=base)

    def full(self, shape, fill_value):
        return self.np.full(shape, fill_value)

    def tensor(self, data, dtype=None):
        return self.np.array(data, dtype=dtype)

    def normal(self, mean, std, size):
        return self.np.random.normal(mean, std, size=size)

    def rand(self, size):
        if isinstance(size, int):
            size = (size,)
        return self.np.random.rand(*size)

    def arange(self, start, stop, step=1):
        return self.np.arange(start, stop, step)

    def allclose(self, a, b, rtol=1e-05, atol=1e-08, equal_nan=False):
        return self.np.allclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan)

    def shape(self, tensor):
        return tuple(tensor.shape)

    def meshgrid(self, *tensors, indexing="ij"):
        return self.np.meshgrid(*tensors, indexing=indexing)

    def size(self, tensor):
        return tensor.size

    def all(self, tensor):
        return tensor.all()

    def stack(self, tensors, dim: int):
        return self.np.stack(tensors, axis=dim)

    @property
    def float(self):
        return self.np.float64

    @property
    def float32(self):
        return self.np.float32

    @property
    def float64(self):
        return self.np.float64

    @property
    def int8(self):
        return self.np.int8

    @property
    def int16(self):
        return self.np.int16

    @property
    def int32(self):
        return self.np.int32

    @property
    def int64(self):
        return self.np.int64

    @property
    def bool(self):
        return self.np.bool_


class JaxBackend(NumpyBackend):
    framework_name = "jax"

    def __init__(self):
        super().__init__()
        self.onp = self.np

        import jax.numpy
        import jax.random

        self.np = jax.numpy
        self._random = jax.random
        self.key = jax.random.PRNGKey(0)

    def rand(self, size):
        new_key, subkey = self._random.split(self.key)
        self.key = new_key
        return self._random.uniform(subkey, size)

    def normal(self, mean, std, size):
        new_key, subkey = self._random.split(self.key)
        self.key = new_key
        return self._random.normal(key=subkey, shape=size) * std + mean


class TorchBackend(AbstractBackend):
    framework_name = "torch"

    def __init__(self):
        import torch

        self.torch = torch

    def is_appropriate_type(self, tensor):
        return isinstance(tensor, self.torch.Tensor)

    def arange(self, start, stop, dtype=None):
        return self.torch.arange(start, stop, dtype=dtype)

    def linspace(self, start, stop, num, dtype=None):
        return self.torch.linspace(start, stop, num, dtype=dtype)

    def logspace(self, start, stop, num, base=10):
        return self.torch.logspace(start, stop, num, base=base)

    def full(self, shape, fill_value):
        return self.torch.full(shape, fill_value)

    def tensor(self, data, dtype=None):
        return self.torch.tensor(data, dtype=dtype)

    def normal(self, mean, std, size):
        return self.torch.normal(mean, std, size=size)

    def rand(self, size):
        return self.torch.rand(size)

    def concat(self, tensors, axis: int):
        return self.torch.cat(tensors, dim=axis)

    def allclose(self, a, b, rtol=1e-05, atol=1e-08, equal_nan=False):
        return self.torch.allclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan)

    def shape(self, tensor):
        return tuple(tensor.shape)

    def meshgrid(self, *tensors, indexing="ij"):
        return self.torch.meshgrid(*tensors, indexing=indexing)

    def size(self, tensor):
        return tensor.numel()

    def all(self, tensor):
        return tensor.all()

    def stack(self, tensors, dim: int):
        return self.torch.stack(tensors, dim=dim)

    @property
    def float(self):
        return self.torch.float

    @property
    def float32(self):
        return self.torch.float32

    @property
    def float64(self):
        return self.torch.float64

    @property
    def int(self):
        return self.torch.int

    @property
    def int8(self):
        return self.torch.int8

    @property
    def int16(self):
        return self.torch.int16

    @property
    def int32(self):
        return self.torch.int32

    @property
    def int64(self):
        return self.torch.int64

    @property
    def bool(self):
        return self.torch.bool


if __name__ == "__main__":
    import jax
    import numpy as np
    import torch

    tensor = torch.randn(10, 10)
    print(get_backend(tensor))
    tensor = np.random.randn(10, 10)
    print(get_backend(tensor))
    seed = 1701
    num_steps = 100
    key, subkey = jax.random.split(jax.random.PRNGKey(seed))
    vec = jax.random.normal(subkey, (num_steps,))
    print(get_backend(vec))
