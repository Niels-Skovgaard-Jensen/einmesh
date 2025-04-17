from __future__ import annotations

import math
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from einmesh._backends import AbstractBackend
from einmesh._operators import BackendOperator, OperatorFactory


@dataclass
class SpaceType(ABC):
    """Base class for all space types."""

    operators: list[BackendOperator] = field(init=False, default_factory=list)

    def __abs__(self) -> SpaceType:
        self_copy = deepcopy(self)
        self_copy.operators.append(OperatorFactory.abs())
        return self_copy

    def __add__(self, other) -> SpaceType:
        self_copy = deepcopy(self)
        self_copy.operators.append(OperatorFactory.add(value=other))
        return self_copy

    def __radd__(self, other) -> SpaceType:
        return self.__add__(other)

    def __sub__(self, other) -> SpaceType:
        self_copy = deepcopy(self)
        self_copy.operators.append(OperatorFactory.sub(value=other))
        return self_copy

    def __rsub__(self, other) -> SpaceType:
        self_copy = deepcopy(self)
        self_copy.operators.insert(0, OperatorFactory.neg())
        self_copy.operators.insert(1, OperatorFactory.add(value=other))
        return self_copy

    def __mul__(self, other) -> SpaceType:
        self_copy = deepcopy(self)
        self_copy.operators.append(OperatorFactory.mul(value=other))
        return self_copy

    def __rmul__(self, other) -> SpaceType:
        return self.__mul__(other)

    def __mod__(self, other) -> SpaceType:
        self_copy = deepcopy(self)
        self_copy.operators.append(OperatorFactory.mod(value=other))
        return self_copy

    def __floordiv__(self, other) -> SpaceType:
        self_copy = deepcopy(self)
        self_copy.operators.append(OperatorFactory.floor_div(value=other))
        return self_copy

    def __truediv__(self, other) -> SpaceType:
        self_copy = deepcopy(self)
        self_copy.operators.append(OperatorFactory.div(value=other))
        return self_copy

    def __neg__(self) -> SpaceType:
        self_copy = deepcopy(self)
        self_copy.operators.append(OperatorFactory.neg())
        return self_copy

    def __pos__(self) -> SpaceType:
        self_copy = deepcopy(self)
        self_copy.operators.append(OperatorFactory.pos())
        return self_copy

    def __pow__(self, other) -> SpaceType:
        self_copy = deepcopy(self)
        self_copy.operators.append(OperatorFactory.pow(exponent=other))
        return self_copy

    @abstractmethod
    def _generate_samples(self, backend: AbstractBackend):
        """Create a new space type."""
        ...

    def _sample(self, backend: AbstractBackend) -> Any:
        """Sample the space type."""
        sample = self._generate_samples(backend)
        return self._apply_operators(sample, backend)

    def _apply_operators(self, sample, backend: AbstractBackend) -> Any:
        """Apply the operators to the space type."""
        for operator in self.operators:
            sample = operator(x=sample, backend=backend)
        return sample


@dataclass
class LogSpace(SpaceType):
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

    def _generate_samples(self, backend: AbstractBackend):
        """Generates the logarithmically spaced points."""
        sample = backend.logspace(self.start, self.end, self.num, base=self.base)
        return self._apply_operators(sample, backend)


@dataclass
class LgSpace(LogSpace):
    """
    Represents a sequence of points spaced in base 2.
    """

    start: float
    end: float
    num: int
    base: float = field(init=False, default=2)

    def __post_init__(self) -> None:
        object.__setattr__(self, "base", 2)


@dataclass
class LnSpace(LogSpace):
    """
    Represents a sequence of points spaced in base e.
    """

    start: float
    end: float
    num: int
    base: float = field(init=False, default=math.e)

    def __post_init__(self) -> None:
        object.__setattr__(self, "base", math.e)


@dataclass
class LinSpace(SpaceType):
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

    def _generate_samples(self, backend: AbstractBackend):
        """Generates the linearly spaced points."""
        return backend.linspace(self.start, self.end, self.num)


@dataclass
class ConstantSpace(SpaceType):
    """
    Represents a constant value repeated multiple times.

    This class generates a tensor containing the same constant value repeated
    `num` times.

    Attributes:
        value: The constant float value to be used.
        num: The number of times the constant value should be repeated. Defaults to 1.
    """

    value: float
    num: int | None = None

    def __post_init__(self) -> None:
        if self.num is None:
            object.__setattr__(self, "num", 1)

    def _generate_samples(self, backend: AbstractBackend):
        """Generates a tensor with the constant value repeated."""
        return backend.full((self.num,), self.value)


@dataclass
class ListSpace(SpaceType):
    """
    Represents a predefined list of values.

    This class generates a tensor directly from a provided list of float values.
    The number of points generated is equal to the length of the input list.

    Attributes:
        values: A list of float values to be converted into a tensor.
        num: The number of points, automatically set to length of values.
    """

    values: list[float]
    num: int = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "num", len(self.values))

    def _generate_samples(self, backend: AbstractBackend):
        """Generates a tensor from the provided list of values."""
        return backend.tensor(self.values)


@dataclass
class Distribution(SpaceType):
    """Base class for all distribution spaces."""

    ...


@dataclass
class NormalDistribution(Distribution):
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

    def _generate_samples(self, backend: AbstractBackend):
        """Generates samples from the normal distribution."""
        return backend.normal(self.mean, self.std, size=(self.num,))


@dataclass
class UniformDistribution(Distribution):
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

    def _generate_samples(self, backend: AbstractBackend):
        """Generates samples from the uniform distribution."""
        # torch.rand samples from [0, 1), so we scale and shift.
        return backend.rand(self.num) * (self.high - self.low) + self.low
