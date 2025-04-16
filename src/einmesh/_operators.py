from abc import abstractmethod

from einmesh._backends import AbstractBackend


class BackendOperator:
    """Base class for all operators acting on einmesh tensors."""

    @abstractmethod
    def __call__(self, x, backend: AbstractBackend): ...


class BackendAbs(BackendOperator):
    """Operator for absolute value."""

    def __call__(self, x, backend: AbstractBackend):
        return backend.abs(x)


class BackendAdd(BackendOperator):
    """Operator for addition."""

    def __init__(self, value: float):
        self.value = value

    def __call__(self, x, backend: AbstractBackend):
        return backend.add(x, summand=self.value)


class BackendSub(BackendOperator):
    """Operator for subtraction."""

    def __init__(self, value: float):
        self.value = value

    def __call__(self, x, backend: AbstractBackend):
        return backend.sub(x, subtrahend=self.value)


class BackendMul(BackendOperator):
    """Operator for multiplication."""

    def __init__(self, value: float):
        self.value = value

    def __call__(self, x, backend: AbstractBackend):
        return backend.mul(x, multiplier=self.value)


class BackendDiv(BackendOperator):
    """Operator for division."""

    def __init__(self, value: float):
        self.value = value

    def __call__(self, x, backend: AbstractBackend):
        return backend.truediv(x, divisor=self.value)


class BackendMod(BackendOperator):
    """Operator for modulo."""

    def __init__(self, value: float):
        self.value = value

    def __call__(self, x, backend: AbstractBackend):
        return backend.mod(x, divisor=self.value)


class BackendFloorDiv(BackendOperator):
    """Operator for floor division."""

    def __init__(self, value: float):
        self.value = value

    def __call__(self, x, backend: AbstractBackend):
        return backend.floordiv(x, divisor=self.value)


class BackendPow(BackendOperator):
    """Operator for power."""

    def __init__(self, exponent: float):
        self.exponent = exponent

    def __call__(self, x, backend: AbstractBackend):
        return backend.pow(base=x, exponent=self.exponent)


class BackendNeg(BackendOperator):
    """Operator for negation."""

    def __call__(self, x, backend: AbstractBackend):
        return backend.neg(x)


class BackendPos(BackendOperator):
    """Operator for positive value."""

    def __call__(self, x, backend: AbstractBackend):
        return backend.pos(x)


class OperatorFactory:
    """Factory for creating operators."""

    @staticmethod
    def abs() -> BackendAbs:
        return BackendAbs()

    @staticmethod
    def add(value: float) -> BackendAdd:
        return BackendAdd(value=value)

    @staticmethod
    def sub(value: float) -> BackendSub:
        return BackendSub(value=value)

    @staticmethod
    def mul(value: float) -> BackendMul:
        return BackendMul(value=value)

    @staticmethod
    def div(value: float) -> BackendDiv:
        return BackendDiv(value=value)

    @staticmethod
    def mod(value: float) -> BackendMod:
        return BackendMod(value=value)

    @staticmethod
    def floor_div(value: float) -> BackendFloorDiv:
        return BackendFloorDiv(value=value)

    @staticmethod
    def pow(exponent: float) -> BackendPow:
        return BackendPow(exponent=exponent)

    @staticmethod
    def neg() -> BackendNeg:
        return BackendNeg()

    @staticmethod
    def pos() -> BackendPos:
        return BackendPos()
