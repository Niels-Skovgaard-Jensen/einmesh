from typing import TYPE_CHECKING, Any, TypeAlias

Number: TypeAlias = int | float | bool

if TYPE_CHECKING:
    import jax  # pyright: ignore[reportMissingImports]
    import numpy as np  # pyright: ignore[reportMissingImports]
    import torch  # pyright: ignore[reportMissingImports]

    Tensor: TypeAlias = np.ndarray[Any, Any] | torch.Tensor | jax.Array
else:
    Tensor: TypeAlias = Any
