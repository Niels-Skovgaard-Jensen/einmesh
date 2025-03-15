from collections import OrderedDict
from typing import Optional, Union

import einops
import torch

from einmesh.spaces import SpaceType


def parse_pattern(pattern: str) -> list[str]:
    """Parse a pattern string into individual dimension names."""
    return pattern.strip().split()


def split_pattern(pattern: str) -> tuple[str, Optional[str]]:
    """Split a pattern into input and output parts."""
    if "->" in pattern:
        input_pattern, output_pattern = pattern.split("->")
        return input_pattern.strip(), output_pattern.strip()
    return pattern, None


def parse_output_pattern(output_pattern: str) -> list[str]:
    """Parse the output pattern into parts, handling parentheses correctly."""
    output_parts = []
    current_part = ""
    in_parentheses = False

    for char in output_pattern:
        if char == "(":
            in_parentheses = True
            if current_part.strip():
                output_parts.append(current_part.strip())
                current_part = ""
            current_part += char
        elif char == ")":
            in_parentheses = False
            current_part += char
            output_parts.append(current_part.strip())
            current_part = ""
        elif char.isspace() and not in_parentheses:
            if current_part.strip():
                output_parts.append(current_part.strip())
                current_part = ""
        else:
            current_part += char

    if current_part.strip():
        output_parts.append(current_part.strip())

    return output_parts


def handle_star_pattern(meshes: tuple[torch.Tensor, ...], output_pattern: str) -> Optional[torch.Tensor]:
    """Handle patterns with * for stacking meshes."""
    # Special case: if output pattern is just "*", stack all meshes
    if output_pattern.strip() == "*":
        return torch.stack(meshes)

    # Handle pattern that begins with * to stack all meshes
    if "*" in output_pattern:
        tokens = output_pattern.split()
        if "*" in tokens:
            # Stack all meshes first
            stacked = torch.stack(meshes)

            # If * is the only token, return stacked tensor
            if len(tokens) == 1:
                return stacked

            # For patterns like "* x y z", interpret as
            # "put all meshes as first dimension, followed by the original dimensions"
            return stacked

    return None


def process_output_parts(
    output_parts: list[str],
    pattern_list: list[str],
    dim_to_tensor: dict[str, torch.Tensor],
    meshes: tuple[torch.Tensor, ...],
    dim_shapes: dict[str, int],
) -> Union[torch.Tensor, tuple[torch.Tensor, ...]]:
    """Process the output pattern parts and return the appropriate tensor(s)."""
    result_tensors = []
    einops_input_pattern = " ".join(pattern_list)
    einops_output_parts = []

    for part in output_parts:
        if part.startswith("(") and part.endswith(")"):
            # Handle grouped dimensions
            inner_dims = part[1:-1].strip().split()
            if not inner_dims:
                continue

            # Get the tensors for each dimension
            tensors_to_combine = [dim_to_tensor[dim] for dim in inner_dims if dim in dim_to_tensor]
            if not tensors_to_combine:
                continue

            # For parentheses, we'll create an output pattern like "(dim1 dim2)"
            einops_output_parts.append(f"({' '.join(inner_dims)})")

            # Simply use the first tensor as a placeholder - we'll use einops at the end
            result_tensors.append(tensors_to_combine[0])
        elif part in dim_to_tensor:
            # Add the tensor directly
            result_tensors.append(dim_to_tensor[part])
            einops_output_parts.append(part)

    # If we have parentheses in the pattern, we need to use einops for proper reshaping
    if any("(" in part for part in einops_output_parts):
        # We need to construct a tensor that combines all our dimensions
        # This is a simplified approach - create a sample grid with all dimensions
        # and then use einops.rearrange to reshape it
        combined_tensor = meshes[0]  # Start with the first mesh

        # Create a pattern for einops.rearrange
        einops_output_pattern = " ".join(einops_output_parts)

        # Use einops to reshape the tensor
        return einops.rearrange(combined_tensor, f"{einops_input_pattern} -> {einops_output_pattern}", **dim_shapes)

    # If no einops reshaping is needed, just return the tensors
    if len(result_tensors) == 1:
        return result_tensors[0]
    return tuple(result_tensors)


def einmesh(pattern: str, **kwargs: SpaceType) -> Union[torch.Tensor, tuple[torch.Tensor, ...]]:
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
    # Split pattern into input and output parts
    input_pattern, output_pattern = split_pattern(pattern)

    # Process input pattern
    pattern_list = parse_pattern(input_pattern)
    lin_samples: OrderedDict[str, torch.Tensor] = OrderedDict()

    for p in pattern_list:
        if p not in kwargs:
            raise UndefinedSpaceError(p)
        lin_samples[p] = kwargs[p]._sample()

    meshes = torch.meshgrid(*lin_samples.values(), indexing="ij")

    # Store shape information for later
    dim_shapes = {dim: tensor.size()[0] for dim, tensor in zip(pattern_list, lin_samples.values())}

    # If no output pattern, return the original meshes tuple
    if output_pattern is None:
        return meshes

    # Create a dictionary mapping dimension names to their corresponding mesh tensors
    dim_to_tensor: dict[str, torch.Tensor] = dict(zip(pattern_list, meshes))

    # Handle star pattern for stacking meshes
    star_result = handle_star_pattern(meshes, output_pattern)
    if star_result is not None:
        return star_result

    # Parse and process the output pattern
    output_parts = parse_output_pattern(output_pattern)
    return process_output_parts(output_parts, pattern_list, dim_to_tensor, meshes, dim_shapes)


class UndefinedSpaceError(ValueError):
    """Error raised when a required sample space is not defined."""

    def __init__(self, space_name: str) -> None:
        super().__init__(f"Undefined space: {space_name}")
