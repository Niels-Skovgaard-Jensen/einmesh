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
    """
    Handles renaming of duplicate space names in the pattern and updates kwargs.

    If a space name appears multiple times in the pattern (e.g., "x x y"),
    it renames subsequent occurrences with an index suffix (e.g., "x_0 x_1 y").
    It updates the pattern string and the kwargs dictionary accordingly, adding
    entries for the new names and removing the original duplicate entries from kwargs.

    Args:
        pattern: The original einmesh pattern string.
        shape_pattern: The pattern string with parentheses removed.
        kwargs: The dictionary of space names to SpaceType objects.

    Returns:
        A tuple containing:
            - The modified pattern string with duplicates renamed.
            - A dictionary mapping new names to original names (e.g., {"x_0": "x", "x_1": "x"}).
            - The updated kwargs dictionary with renamed keys and removed originals.
    """
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
    Creates multi-dimensional meshgrids using an einops-style pattern string.

    `einmesh` simplifies the creation and manipulation of multi-dimensional
    meshgrids by specifying sampling spaces (like LinSpace, LogSpace, etc.)
    and their arrangement using an intuitive pattern inspired by `einops`.

    The pattern string defines the dimensions and structure of the output:
    - **Space Names:** Correspond to keyword arguments providing `SpaceType` objects
      (e.g., "x y z" uses spaces named `x`, `y`, `z` passed as kwargs).
    - **Repeated Names:** Handles dimensions derived from the same space type
      (e.g., "x x y" results in dimensions named `x_0`, `x_1`, `y`).
    - **Stacking (`*`):** Stacks the generated meshgrids for each space along a new
      dimension. Only one `*` is allowed. The output is a single tensor.
    - **Grouping (`()`):** Groups dimensions together in the output tensor shape,
      affecting the `einops.rearrange` operation applied internally.

    If the pattern does *not* contain `*`, the function returns a tuple of tensors,
    one for each space name in the pattern (after handling duplicates). Each tensor
    in the tuple represents the coordinates for that specific dimension across the
    entire meshgrid, ordered according to the pattern.

    If the pattern *does* contain `*`, the function returns a single tensor where
    the individual meshgrids are stacked along the dimension specified by `*`.

    Examples:
        >>> from einmesh import LinSpace, einmesh
        >>> x_space = LinSpace(0, 1, 5)
        >>> y_space = LinSpace(10, 20, 3)

        >>> # Basic 2D meshgrid (returns tuple: (x_coords, y_coords))
        >>> x_coords, y_coords = einmesh("x y", x=x_space, y=y_space)
        >>> x_coords.shape  # (5, 3) - x varies along the first dim
        torch.Size([5, 3])
        >>> y_coords.shape  # (5, 3) - y varies along the second dim
        torch.Size([5, 3])

        >>> # Stacked meshgrid (returns single tensor)
        >>> # '*' indicates stacking dimension
        >>> stacked_grid = einmesh("* x y", x=x_space, y=y_space)
        >>> stacked_grid.shape # (2, 5, 3) - Dim 0 stacks x and y grids
        torch.Size([2, 5, 3])

        >>> # Grouping affects rearrangement (here, stacks x and y coords)
        >>> stacked_grouped = einmesh("* (x y)", x=x_space, y=y_space)
        >>> stacked_grouped.shape # (2, 15) - Dim 0 stacks, dim 1 combines x & y
        torch.Size([2, 15])

        >>> # Repeated spaces
        >>> x0_coords, x1_coords = einmesh("x x", x=x_space)
        >>> x0_coords.shape
        torch.Size([5, 5])

    Args:
        pattern: The einops-style string defining meshgrid structure.
        **kwargs: Keyword arguments mapping space names in the pattern to
                  `SpaceType` objects (e.g., `x=LinSpace(0, 1, 10)`).

    Returns:
        A `torch.Tensor` if the pattern includes `*` (stacking), or a
        `tuple[torch.Tensor, ...]` if the pattern does not include `*`.

    Raises:
        UnbalancedParenthesesError: If parentheses in the pattern are not balanced.
        MultipleStarError: If the pattern contains more than one `*`.
        UndefinedSpaceError: If a name in the pattern doesn't have a corresponding
                             kwarg `SpaceType`.
        ArrowError: If the pattern contains '->', which is not supported.
        # Note: UnderscoreError is currently commented out in _verify_pattern
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
    """
    Generates 1D samples for each space and creates initial meshgrids.

    Uses `torch.meshgrid` with `indexing="ij"` based on the ordered
    `sampling_list`.

    Args:
        sampling_list: An ordered list of space names (potentially renamed
                       if duplicates existed in the original pattern).
        **kwargs: The dictionary of space names to `SpaceType` objects,
                  potentially updated with renamed keys.

    Returns:
        A tuple containing:
            - A list of `torch.Tensor` objects representing the meshgrid
              coordinates for each dimension in `sampling_list`. The order
              matches `sampling_list`.
            - A dictionary mapping space names to their corresponding dimension sizes.

    Raises:
        UndefinedSpaceError: If a name in `sampling_list` is not found in `kwargs`.
    """
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
    """
    Performs basic validation checks on the input pattern string.

    Checks for:
    - More than one stacking operator (`*`).
    - Unbalanced parentheses.
    - Presence of '->' (reserved for potential future use or indicating error).
    # - Presence of underscores ('_') - Currently commented out.

    Args:
        pattern: The input einmesh pattern string.

    Raises:
        MultipleStarError: If pattern contains more than one `*`.
        UnbalancedParenthesesError: If pattern has unbalanced parentheses.
        ArrowError: If pattern contains '->'.
        # UnderscoreError: If pattern contains '_' (currently disabled).
    """
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
