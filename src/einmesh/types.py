from typing import TYPE_CHECKING, Any, TypeAlias

Number: TypeAlias = int | float | bool

if TYPE_CHECKING:
    import jax
    import numpy as np
    import torch

    Tensor: TypeAlias = np.ndarray[Any, Any] | torch.Tensor | jax.Array
else:
    Tensor: TypeAlias = Any
