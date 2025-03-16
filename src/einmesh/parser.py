from collections import OrderedDict

import einops
import torch

from einmesh.exceptions import (
    ArrowError,
    MultipleStarError,
    UnbalancedParenthesesError,
    UndefinedSpaceError,
    UnderscoreError,
)
from einmesh.spaces import SpaceType


def einmesh(pattern: str, **kwargs: SpaceType) -> torch.Tensor | tuple["torch.Tensor", ...]:
    """
    Einmesh is a function that takes a pattern and space objects and returns tensor(s).

    The pattern can have two forms:
    - Simple form: "x y z" - creates a mesh grid from the specified spaces.
    - Extended form: "x y z -> output_pattern" - creates a mesh grid and reshapes
      according to the output pattern.

    In the output pattern:
    - "*" collects all mesh tensors into a single tensor (stacking the meshes)
    - Parentheses like "(y z)" combine dimensions by reshaping using einops.rearrange

    Examples:
        >>> x = LinSpace(0, 1, 10)
        >>> y = LinSpace(0, 1, 20)
        >>> z = LinSpace(0, 1, 30)
        >>> # Return a tuple of 3 meshes
        >>> mesh_x, mesh_y, mesh_z = einmesh("x y z", x=x, y=y, z=z)
        >>> # Stack meshes into a single tensor with shape [3, 10, 20, 30]
        >>> stacked = einmesh("x y z -> * x y z", x=x, y=y, z=z)
        >>> # Reshape combining y and z dimensions
        >>> reshaped = einmesh("x y z -> x (y z)", x=x, y=y, z=z)

    Args:
        pattern: String pattern specifying input spaces and optional output reshaping
        **kwargs: Space objects corresponding to the names in the pattern

    Returns:
        Either a single tensor or a tuple of tensors depending on the pattern
    """

    _verify_pattern(pattern)

    # get stack index
    shape_pattern = pattern.replace("(", "").replace(")", "")
    stack_idx = shape_pattern.split(" ").index("*") if "*" in shape_pattern else None

    # get sampling list

    sampling_list = shape_pattern.strip().split()

    meshes, dim_shapes = _generate_samples(sampling_list, **kwargs)

    # Store shape information for later

    # Handle star pattern for stacking meshes
    if stack_idx is not None:
        meshes = torch.stack(meshes, dim=stack_idx)

        dim_shapes["einstack"] = meshes.shape[stack_idx]

    if isinstance(meshes, torch.Tensor):
        output_pattern = pattern.replace("*", "einstack")
        input_pattern = pattern.replace("*", "einstack").replace("(", "").replace(")", "")
        meshes = einops.rearrange(meshes, f"{input_pattern} -> {output_pattern}", **dim_shapes)
    if isinstance(meshes, list):
        for i, mesh in enumerate(meshes):
            output_pattern = pattern
            input_pattern = pattern.replace("(", "").replace(")", "")
            meshes[i] = einops.rearrange(mesh, f"{input_pattern} -> {output_pattern}", **dim_shapes)

        meshes = tuple(meshes)

    return meshes


def _generate_samples(sampling_list: list[str], **kwargs: SpaceType) -> tuple[list[torch.Tensor], dict[str, int]]:
    """Generate samples from the pattern."""
    lin_samples: OrderedDict[str, torch.Tensor] = OrderedDict()
    dim_shapes: dict[str, int] = {}
    for p in sampling_list:
        if p == "*":
            continue
        if p not in kwargs:
            raise UndefinedSpaceError(p)
        samples = kwargs[p]._sample()
        lin_samples[p] = samples
        dim_shapes[p] = samples.size()[0]
    meshes = list(torch.meshgrid(*lin_samples.values(), indexing="ij"))
    return meshes, dim_shapes


def _verify_pattern(pattern: str) -> None:
    """Verify the pattern is valid."""
    if pattern.count("*") > 1:
        raise MultipleStarError()
    if pattern.count("(") != pattern.count(")"):
        raise UnbalancedParenthesesError()
    if "_" in pattern:
        raise UnderscoreError()
    if "->" in pattern:
        raise ArrowError()
