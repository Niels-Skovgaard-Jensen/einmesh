import re

import einops
import torch

from einmesh.exceptions import (
    ArrowError,
    MultipleStarError,
    UnbalancedParenthesesError,
    UndefinedSpaceError,
)
from einmesh.spaces import SpaceType


def _handle_duplicate_names(
    pattern: str,
    shape_pattern: str,
    kwargs: dict[str, SpaceType],
) -> tuple[str, dict[str, str], dict[str, SpaceType]]:
    """Handles renaming of duplicate space names in the pattern and updates kwargs."""
    seen_names: dict[str, int] = {}
    name_mapping: dict[str, str] = {}

    # First count occurrences of each name (excluding '*')
    for name in shape_pattern.split():
        if name != "*":
            seen_names[name] = seen_names.get(name, 0) + 1

    # Then rename duplicates for each unique name with counts > 1
    for name in list(seen_names.keys()):
        if seen_names[name] > 1:
            for i in range(seen_names[name]):
                new_name = f"{name}_{i}"
                # Use regex to replace only whole words to avoid partial matches
                pattern = re.sub(rf"\b{name}\b", new_name, pattern, count=1)
                name_mapping[new_name] = name

    # Update kwargs with renamed spaces
    updated_kwargs = kwargs.copy()  # Avoid modifying the original dict directly
    for new_name, orig_name in name_mapping.items():
        if orig_name in updated_kwargs:
            updated_kwargs[new_name] = updated_kwargs[orig_name]

    # Remove original names that were renamed
    orig_names_to_remove = set(name_mapping.values())
    for orig_name in orig_names_to_remove:
        if orig_name in updated_kwargs:
            updated_kwargs.pop(orig_name)

    return pattern, name_mapping, updated_kwargs


def einmesh(pattern: str, **kwargs: SpaceType) -> torch.Tensor | tuple[torch.Tensor, ...]:
    """
    Create multi-dimensional meshgrids using einops-style pattern syntax.

    einmesh allows you to easily create and manipulate multi-dimensional meshgrids by combining different sampling spaces (like LinSpace or LatinHypercube) using an intuitive pattern syntax inspired by einops.

    The pattern consists of space names that must correspond to keyword arguments. Each space name represents a dimension in the resulting meshgrid. The pattern can include:

    - Space names: e.g. "x y z" creates a meshgrid from spaces x, y, and z
    - Stacking: "*" stacks multiple meshgrids into a single tensor along that dimension
    - Grouping: "(x y)" groups dimensions x and y together

    Examples:
        >>> # Basic 2D meshgrid from linspaces
        >>> x = LinSpace(0, 1, 5)
        >>> y = LinSpace(0, 1, 3)
        >>> grid = einmesh("x y", x=x, y=y)  # shape: (5, 3)

        >>> # Stack multiple meshgrids with *
        >>> stacked = einmesh("* (x y)", x=x, y=y)  # shape: (2, 5, 3)

        >>> # Stack and reshape
        >>> x = einmesh("* (x y)",
        ...              x=LinSpace(0, 1, 5),
        ...              y=LinSpace(0, 1, 3))  # shape: (2, 5*3)

    Args:
        pattern: String pattern specifying the input spaces
        **kwargs: Space objects (LinSpace, LogSpace etc.) corresponding to names in pattern

    Returns:
        Either a single tensor or a tuple of tensors depending on the pattern.
        Returns tuple when pattern has no stacking (*).
        Returns single tensor when pattern includes stacking.

    Raises:
        UnbalancedParenthesesError: If parentheses in pattern are not balanced
        MultipleStarError: If pattern contains multiple * characters
        UndefinedSpaceError: If pattern contains space name with no corresponding kwarg
        UnderscoreError: If pattern contains invalid underscore usage
    """

    _verify_pattern(pattern)

    # get stack index
    shape_pattern = pattern.replace("(", "").replace(")", "")
    stack_idx = shape_pattern.split().index("*") if "*" in shape_pattern else None

    # Check for and handle duplicate names in pattern
    pattern, name_mapping, kwargs = _handle_duplicate_names(pattern, shape_pattern, kwargs)

    # Determine the final order of space names from the potentially modified pattern
    final_pattern_names = pattern.replace("(", "").replace(")", "").split()
    # Filter out '*' as it's not a sampling space name
    sampling_list = [name for name in final_pattern_names if name != "*"]

    # Pass the ordered sampling_list and potentially modified kwargs
    meshes, dim_shapes = _generate_samples(sampling_list, **kwargs)

    # Handle star pattern for stacking meshes
    input_sampling_list = list(sampling_list)  # Base list for input pattern
    if stack_idx is not None:
        meshes = torch.stack(meshes, dim=stack_idx)
        dim_shapes["einstack"] = meshes.shape[stack_idx]
        # Insert 'einstack' into the sampling list copy at the correct index for the input pattern
        input_sampling_list.insert(stack_idx, "einstack")

    # Define the input pattern based on the actual order of dimensions in the tensor(s)
    input_pattern = " ".join(input_sampling_list)

    if isinstance(meshes, torch.Tensor):  # Stacked case
        # Output pattern: User pattern with '*' replaced by 'einstack'
        output_pattern = pattern.replace("*", "einstack")
        meshes = einops.rearrange(meshes, f"{input_pattern} -> {output_pattern}", **dim_shapes)

    elif isinstance(meshes, list):  # Non-stacked case (must be tuple eventually)
        rearranged_meshes = []
        # Output pattern: User pattern (with renames, no '*' or 'einstack')
        output_pattern = pattern
        # Input pattern is the same for all meshes in the list
        for mesh in meshes:
            # Rearrange each mesh individually
            rearranged_mesh = einops.rearrange(mesh, f"{input_pattern} -> {output_pattern}", **dim_shapes)
            rearranged_meshes.append(rearranged_mesh)
        meshes = tuple(rearranged_meshes)  # Convert list back to tuple

    return meshes


def _generate_samples(sampling_list: list[str], **kwargs: SpaceType) -> tuple[list[torch.Tensor], dict[str, int]]:
    """Generate samples based on the provided ordered list."""
    lin_samples: list[torch.Tensor] = []
    dim_shapes: dict[str, int] = {}
    # Iterate using the provided sampling_list to ensure correct order
    for p in sampling_list:
        if p not in kwargs:
            # This check might be redundant if pattern validation is robust, but safer to keep
            raise UndefinedSpaceError(p)
        samples = kwargs[p]._sample()
        lin_samples.append(samples)
        dim_shapes[p] = samples.size()[0]
    # The order of meshes returned by torch.meshgrid(indexing='ij')
    # corresponds to the order of tensors in lin_samples.
    meshes = list(torch.meshgrid(*lin_samples, indexing="ij"))
    return meshes, dim_shapes


def _verify_pattern(pattern: str) -> None:
    """Verify the pattern is valid."""
    if pattern.count("*") > 1:
        raise MultipleStarError()
    if pattern.count("(") != pattern.count(")"):
        raise UnbalancedParenthesesError()
    # Allow underscore only if it was introduced by renaming duplicates
    # This check might need refinement if users can legimitately use underscores
    # if "_" in pattern:
    #     raise UnderscoreError()
    if "->" in pattern:
        raise ArrowError()
