import pytest

from einmesh._backends import AbstractBackend, JaxBackend, NumpyBackend, TorchBackend, get_backend
from einmesh._einmesher import _EinMesher
from einmesh._exceptions import (
    ArrowError,
    MultipleStarError,
    UnbalancedParenthesesError,
    UnderscoreError,
    UnknownBackendError,
)
from einmesh.spaces import LinSpace
from tests.conftest import parametrize_backends

"""
Error condition tests for parser validation and backend selection.
"""

# Parser validation errors


@parametrize_backends
def test_multiple_star_error(backend_cls: type[AbstractBackend]):
    backend = backend_cls()
    mesher = _EinMesher(
        "x * y * z", x=LinSpace(0.0, 1.0, 2), y=LinSpace(0.0, 1.0, 2), z=LinSpace(0.0, 1.0, 2), backend=backend
    )
    with pytest.raises(MultipleStarError):
        mesher.sample()


@parametrize_backends
def test_unbalanced_parentheses_error(backend_cls: type[AbstractBackend]):
    backend = backend_cls()
    # Missing closing parenthesis
    mesher1 = _EinMesher("x (y z", x=LinSpace(0.0, 1.0, 2), y=LinSpace(0.0, 1.0, 2), backend=backend)
    with pytest.raises(UnbalancedParenthesesError):
        mesher1.sample()
    # Extra closing parenthesis
    mesher2 = _EinMesher("x y)", x=LinSpace(0.0, 1.0, 2), y=LinSpace(0.0, 1.0, 2), backend=backend)
    with pytest.raises(UnbalancedParenthesesError):
        mesher2.sample()


@parametrize_backends
def test_arrow_error(backend_cls: type[AbstractBackend]):
    backend = backend_cls()
    mesher = _EinMesher("x -> y", x=LinSpace(0.0, 1.0, 2), y=LinSpace(0.0, 1.0, 2), backend=backend)
    with pytest.raises(ArrowError):
        mesher.sample()


@parametrize_backends
def test_underscore_error(backend_cls: type[AbstractBackend]):
    backend = backend_cls()
    mesher = _EinMesher("x_y y", x=LinSpace(0.0, 1.0, 2), y=LinSpace(0.0, 1.0, 2), backend=backend)
    with pytest.raises(UnderscoreError):
        mesher.sample()


def test_unknown_backend_error():
    mesher = _EinMesher("x y", x=LinSpace(0.0, 1.0, 2), y=LinSpace(0.0, 1.0, 2), backend="NotDefinedBackend")
    with pytest.raises(UnknownBackendError):
        mesher.sample()


def test_get_backend_selection_torch():
    import torch

    tensor = torch.tensor([1, 2, 3])
    backend = get_backend(tensor)
    assert isinstance(backend, TorchBackend)


def test_get_backend_selection_numpy():
    import numpy as np

    array = np.array([1, 2, 3])
    backend = get_backend(array)
    assert isinstance(backend, NumpyBackend)


@pytest.mark.skipif(not JaxBackend.is_available(), reason="JAX not available")
def test_get_backend_selection_jax():
    import jax.numpy as jnp

    array = jnp.zeros((3,))
    backend = get_backend(array)
    assert isinstance(backend, JaxBackend)


# JAX backend seed initialization sanity check


def test_jax_backend_seed_variation():
    # The initial PRNGKey should differ across instances
    b1 = JaxBackend()
    b2 = JaxBackend()
    # Convert PRNGKey arrays to Python lists for comparison
    key1 = b1.key.tolist()
    key2 = b2.key.tolist()
    assert key1 != key2
