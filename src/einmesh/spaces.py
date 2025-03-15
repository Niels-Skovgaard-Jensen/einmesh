from dataclasses import dataclass

import torch


@dataclass
class LogSpace:
    start: float
    end: float
    num: int
    base: float = 10

    def _sample(self) -> torch.Tensor:
        return torch.logspace(self.start, self.end, self.num, base=self.base)


@dataclass
class LinSpace:
    start: float
    end: float
    num: int

    def _sample(self) -> torch.Tensor:
        return torch.linspace(self.start, self.end, self.num)


@dataclass
class NormalDistribution:
    mean: float
    std: float
    num: int

    def _sample(self) -> torch.Tensor:
        return torch.normal(self.mean, self.std, size=(self.num,))


@dataclass
class UniformDistribution:
    low: float
    high: float
    num: int

    def _sample(self) -> torch.Tensor:
        return torch.rand(self.num, 1) * (self.high - self.low) + self.low


@dataclass
class LatinHypercube:
    low: float
    high: float
    num: int
    connector: str = "default"

    def _sample(self) -> torch.Tensor:
        return torch.rand(self.num, 1) * (self.high - self.low) + self.low


SpaceType = LogSpace | LinSpace | NormalDistribution | UniformDistribution
