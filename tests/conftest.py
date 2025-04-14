import pytest

from einmesh._backends import JaxBackend, NumpyBackend, TorchBackend

# Define the reusable decorator
parametrize_backends = pytest.mark.parametrize(
    argnames="backend",
    argvalues=[
        pytest.param(
            TorchBackend,
            marks=pytest.mark.skipif(condition=not TorchBackend.is_available(), reason="Torch not available"),
        ),
        pytest.param(
            NumpyBackend,
            marks=pytest.mark.skipif(condition=not NumpyBackend.is_available(), reason="Numpy not available"),
        ),
        pytest.param(
            JaxBackend, marks=pytest.mark.skipif(condition=not JaxBackend.is_available(), reason="Jax not available")
        ),
    ],
    ids=["torch", "numpy", "jax"],  # Optional: Add IDs for clearer test names
)


def init_backend(func, backend_class):
    def wrapper(*args, **kwargs):
        kwargs["backend"] = backend_class()
        return func(*args, **kwargs)

    return wrapper
