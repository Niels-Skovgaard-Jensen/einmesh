import pytest

from einmesh._backends import JaxBackend, NumpyBackend, TorchBackend

# Define the reusable decorator
parametrize_backends = pytest.mark.parametrize(
    "backend",
    [TorchBackend(), NumpyBackend(), JaxBackend()],
    ids=["torch", "numpy", "jax"],  # Optional: Add IDs for clearer test names
)
