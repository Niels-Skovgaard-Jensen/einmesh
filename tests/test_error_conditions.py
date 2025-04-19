import pytest

from einmesh import EinMesher
from einmesh._backends import JaxBackend, NumpyBackend, TorchBackend, get_backend
from einmesh._exceptions import (
    ArrowError,
    MultipleStarError,
    UnbalancedParenthesesError,
    UnderscoreError,
    UnknownBackendError,
)
from einmesh.spaces import LinSpace

"""
Error condition tests for parser validation and backend selection.
"""

# Parser validation errors


def test_multiple_star_error():
    mesher = EinMesher("x * y * z", x=LinSpace(0.0, 1.0, 2), y=LinSpace(0.0, 1.0, 2), z=LinSpace(0.0, 1.0, 2))
    with pytest.raises(MultipleStarError):
        mesher.mesh(backend="numpy")


def test_unbalanced_parentheses_error():
    # Missing closing parenthesis
    mesher1 = EinMesher("x (y z", x=LinSpace(0.0, 1.0, 2), y=LinSpace(0.0, 1.0, 2))
    with pytest.raises(UnbalancedParenthesesError):
        mesher1.mesh(backend="numpy")
    # Extra closing parenthesis
    mesher2 = EinMesher("x y)", x=LinSpace(0.0, 1.0, 2), y=LinSpace(0.0, 1.0, 2))
    with pytest.raises(UnbalancedParenthesesError):
        mesher2.mesh(backend="numpy")


def test_arrow_error():
    mesher = EinMesher("x -> y", x=LinSpace(0.0, 1.0, 2), y=LinSpace(0.0, 1.0, 2))
    with pytest.raises(ArrowError):
        mesher.mesh(backend="numpy")


def test_underscore_error():
    mesher = EinMesher("x_y y", x=LinSpace(0.0, 1.0, 2), y=LinSpace(0.0, 1.0, 2))
    with pytest.raises(UnderscoreError):
        mesher.mesh(backend="numpy")


def test_unknown_backend_error():
    mesher = EinMesher("x y", x=LinSpace(0.0, 1.0, 2), y=LinSpace(0.0, 1.0, 2))
    with pytest.raises(UnknownBackendError):
        mesher.mesh(backend="invalid_backend")


# Backend selection tests


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
