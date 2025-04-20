import re
from typing import Any

import einops

from einmesh._backends import AbstractBackend, JaxBackend, NumpyBackend, TorchBackend
from einmesh._exceptions import (
    ArrowError,
    InvalidListTypeError,
    MultipleStarError,
    UnbalancedParenthesesError,
    UndefinedSpaceError,
    UnderscoreError,
    UnknownBackendError,
    UnsupportedSpaceTypeError,
)
from einmesh.spaces import ConstantSpace, ListSpace, SpaceType


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


# Define the type alias for acceptable kwarg values
KwargValueType = SpaceType | int | float | list[int | float]


def _get_backend(backend: AbstractBackend | str) -> AbstractBackend:
    if isinstance(backend, str):
        if backend == "torch":
            return TorchBackend()
        elif backend == "jax":
            return JaxBackend()
        elif backend == "numpy":
            return NumpyBackend()
        else:
            raise UnknownBackendError(backend)
    elif isinstance(backend, AbstractBackend):
        return backend
    else:
        raise UnknownBackendError(backend)


def _einmesh(pattern: str, backend: AbstractBackend | str, **kwargs: KwargValueType) -> Any | tuple[Any, ...]:
    """
    Creates multi-dimensional meshgrids using an einops-style pattern string.

    `einmesh` simplifies the creation and manipulation of multi-dimensional
    meshgrids by specifying sampling spaces and their arrangement using an
    intuitive pattern inspired by `einops`.

    The pattern string defines the dimensions and structure of the output:
    - **Space Names:** Correspond to keyword arguments providing sampling definitions.
    - **Sampling Definitions:** Can be:
        - `SpaceType` objects (e.g., `LinSpace`, `LogSpace`).
        - Single `int` or `float` values (automatically converted to `ConstantSpace`).
        - `list` of `int` or `float` values (automatically converted to `ListSpace`).
      (e.g., `x=LinSpace(0, 1, 5)`, `y=10.0`, `z=[1, 2, 4]`)
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

        >>> # Basic 2D meshgrid (tuple output)
        >>> x_coords, y_coords = einmesh("x y", x=x_space, y=10.0)
        >>> x_coords.shape
        torch.Size([5, 1])
        >>> y_coords.shape
        torch.Size([5, 1])
        >>> y_coords # Constant value 10.0 repeated
        tensor([[10.],
                [10.],
                [10.],
                [10.],
                [10.]])

        >>> # Using a list for a dimension
        >>> x_coords, z_coords = einmesh("x z", x=x_space, z=[1, 2, 4])
        >>> z_coords.shape
        torch.Size([5, 3])
        >>> z_coords[0] # First row shows the list values
        tensor([1., 2., 4.])

        >>> # Stacked meshgrid (single tensor output)
        >>> stacked_grid = einmesh("* x y", x=x_space, y=10.0)
        >>> stacked_grid.shape
        torch.Size([2, 5, 1])

        >>> # Grouping affects rearrangement
        >>> stacked_grouped = einmesh("* (x y)", x=x_space, y=[10.0, 20.0])
        >>> stacked_grouped.shape
        torch.Size([2, 10])

        >>> # Repeated spaces (using implicit ConstantSpace)
        >>> x0_coords, x1_coords = einmesh("x x", x=5)
        >>> x0_coords.shape
        torch.Size([1, 1])
        >>> x0_coords.item()
        5.0

    Args:
        pattern: The einops-style string defining meshgrid structure.
        **kwargs: Keyword arguments mapping space names in the pattern to
                  sampling definitions (`SpaceType`, `int`, `float`, or `list`).

    Returns:
        Union[torch.Tensor, tuple[torch.Tensor, ...]]: A `torch.Tensor` if the pattern includes `*` (stacking), or a
        `tuple[torch.Tensor, ...]` if the pattern does not include `*`.

    Raises:
        UnbalancedParenthesesError: If parentheses in the pattern are not balanced.
        MultipleStarError: If the pattern contains more than one `*`.
        UndefinedSpaceError: If a name in the pattern doesn't have a corresponding
                             kwarg definition.
        ArrowError: If the pattern contains '->', which is not supported.
        TypeError: If a kwarg value is not a `SpaceType`, `int`, `float`, or a
                   valid `list` of `int`/`float`.
    """

    _verify_pattern(pattern)
    backend = _get_backend(backend)

    # Process kwargs to convert raw values to SpaceTypes
    processed_kwargs: dict[str, SpaceType] = {}
    for name, value in kwargs.items():
        if isinstance(value, (int, float)):
            processed_kwargs[name] = ConstantSpace(value=float(value))
        elif isinstance(value, list):
            # Check if list contains only numbers (int or float)
            if all(isinstance(item, (int, float)) for item in value):
                # Convert all items to float for consistency
                processed_kwargs[name] = ListSpace(values=[float(item) for item in value])
            else:
                # Explicitly annotate the list as list[str] to satisfy the type checker
                invalid_types: list[str] = [type(item).__name__ for item in value if not isinstance(item, (int, float))]
                raise InvalidListTypeError(space_name=name, invalid_types=invalid_types)
        elif isinstance(value, SpaceType):
            processed_kwargs[name] = value
        else:
            raise UnsupportedSpaceTypeError(space_name=name, invalid_type=type(value).__name__)

    # get stack index
    shape_pattern = pattern.replace("(", "").replace(")", "")
    stack_idx = shape_pattern.split().index("*") if "*" in shape_pattern else None

    # Check for and handle duplicate names in pattern using processed kwargs
    pattern, name_mapping, processed_kwargs_renamed = _handle_duplicate_names(pattern, shape_pattern, processed_kwargs)

    # Determine the final order of space names from the potentially modified pattern
    final_pattern_names = pattern.replace("(", "").replace(")", "").split()
    # Filter out '*' as it's not a sampling space name
    sampling_list = [name for name in final_pattern_names if name != "*"]

    # Pass the ordered sampling_list and processed+renamed kwargs
    meshes, dim_shapes = _generate_samples(sampling_list, backend, **processed_kwargs_renamed)

    # Handle star pattern for stacking meshes
    input_sampling_list = list(sampling_list)  # Base list for input pattern
    if stack_idx is not None:
        meshes = backend.stack(meshes, dim=stack_idx)
        dim_shapes["einstack"] = meshes.shape[stack_idx]
        # Insert 'einstack' into the sampling list copy at the correct index for the input pattern
        input_sampling_list.insert(stack_idx, "einstack")

    # Define the input pattern based on the actual order of dimensions in the tensor(s)
    input_pattern = " ".join(input_sampling_list)

    if backend.is_appropriate_type(meshes):  # Stacked case
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


def _generate_samples(sampling_list: list[str], backend: AbstractBackend, **kwargs: SpaceType):
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
    lin_samples: list[Any] = []
    dim_shapes: dict[str, int] = {}
    # Iterate using the provided sampling_list to ensure correct order
    for p in sampling_list:
        if p not in kwargs:
            # This check might be redundant if pattern validation is robust, but safer to keep
            raise UndefinedSpaceError(p)
        samples = kwargs[p]._sample(backend)
        lin_samples.append(samples)
        dim_shapes[p] = backend.shape(samples)[0]
    # The order of meshes returned by torch.meshgrid(indexing='ij')
    # corresponds to the order of tensors in lin_samples.
    meshes = list(backend.meshgrid(*lin_samples, indexing="ij"))
    return meshes, dim_shapes


def _verify_pattern(pattern: str) -> None:
    """
    Performs basic validation checks on the input pattern string.

    Checks for:
    - More than one stacking operator (`*`).
    - Unbalanced parentheses.
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
    if "_" in pattern:
        raise UnderscoreError()
    if "->" in pattern:
        raise ArrowError()
