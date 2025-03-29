import typing as t
from dataclasses import dataclass

import torch


@dataclass
class LogSpace:
    """
    Represents a sequence of points spaced logarithmically.

    This class generates a tensor of `num` points between `10**start` and `10**end`
    (or `base**start` and `base**end` if `base` is specified), spaced
    logarithmically.

    Attributes:
        start: The starting exponent of the sequence.
        end: The ending exponent of the sequence.
        num: The number of points to generate.
        base: The base of the logarithm. Defaults to 10.
    """

    start: float
    end: float
    num: int
    base: float = 10

    def _sample(self) -> torch.Tensor:
        """Generates the logarithmically spaced points."""
        return torch.logspace(self.start, self.end, self.num, base=self.base)


@dataclass
class LinSpace:
    """
    Represents a sequence of points spaced linearly.

    This class generates a tensor of `num` points evenly spaced between `start`
    and `end` (inclusive).

    Attributes:
        start: The starting value of the sequence.
        end: The ending value of the sequence.
        num: The number of points to generate.
    """

    start: float
    end: float
    num: int

    def _sample(self) -> torch.Tensor:
        """Generates the linearly spaced points."""
        return torch.linspace(self.start, self.end, self.num)


@dataclass
class NormalDistribution:
    """
    Represents a sampling from a normal (Gaussian) distribution.

    This class generates a tensor of `num` random numbers sampled from a normal
    distribution with the specified `mean` and standard deviation `std`.

    Attributes:
        mean: The mean (center) of the normal distribution.
        std: The standard deviation (spread or width) of the normal distribution.
        num: The number of samples to generate.
    """

    mean: float
    std: float
    num: int

    def _sample(self) -> torch.Tensor:
        """Generates samples from the normal distribution."""
        return torch.normal(self.mean, self.std, size=(self.num,))


@dataclass
class UniformDistribution:
    """
    Represents a sampling from a uniform distribution.

    This class generates a tensor of `num` random numbers sampled from a uniform
    distribution over the interval [`low`, `high`).

    Attributes:
        low: The lower boundary of the output interval.
        high: The upper boundary of the output interval.
        num: The number of samples to generate.
    """

    low: float
    high: float
    num: int

    def _sample(self) -> torch.Tensor:
        """Generates samples from the uniform distribution."""
        # torch.rand samples from [0, 1), so we scale and shift.
        return torch.rand(self.num) * (self.high - self.low) + self.low


SpaceType = t.Union[LogSpace, LinSpace, NormalDistribution, UniformDistribution]
