from collections import OrderedDict

import torch

from einmesh.spaces import SpaceType


class UndefinedSpaceError(ValueError):
    """Error raised when a required sample space is not defined."""

    def __init__(self, space_name: str) -> None:
        super().__init__(f"Undefined space: {space_name}")


def einmesh(pattern: str, **kwargs: SpaceType) -> tuple[torch.Tensor, ...]:
    """
    Einmesh is a function that takes a pattern and a list of tensors and returns a new tensor.
    """

    pattern_list = pattern.split(" ")
    lin_samples: OrderedDict[str, torch.Tensor] = OrderedDict()

    for p in pattern_list:
        if p not in kwargs:
            raise UndefinedSpaceError(p)
        lin_samples[p] = kwargs[p]._sample()

    meshes = torch.meshgrid(*lin_samples.values(), indexing="ij")

    return meshes
