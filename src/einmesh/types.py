from typing import TYPE_CHECKING, Any, TypeAlias

from numpy.typing import NDArray

Number: TypeAlias = int | float | bool

if TYPE_CHECKING:
    import jax  # pyright: ignore[reportMissingImports]
    import torch  # pyright: ignore[reportMissingImports]

    Tensor: TypeAlias = NDArray[Any] | torch.Tensor | jax.Array
else:
    Tensor = Any
