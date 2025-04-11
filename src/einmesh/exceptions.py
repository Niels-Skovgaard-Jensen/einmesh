class PatternError(ValueError):
    """Error raised when the pattern is invalid."""

    def __init__(self, message: str) -> None:
        super().__init__(f"Invalid pattern: {message}")


class UnbalancedParenthesesError(PatternError):
    """Error raised when the pattern has unbalanced parentheses."""

    def __init__(self) -> None:
        super().__init__("Unbalanced parentheses in pattern")


class MultipleStarError(PatternError):
    """Error raised when multiple '*' are found in the output pattern."""

    def __init__(self) -> None:
        super().__init__("Multiple '*' are not allowed in the output pattern")


class UnderscoreError(PatternError):
    """Error raised when an underscore is found in the pattern."""

    def __init__(self) -> None:
        super().__init__("Underscores are not allowed in pattern names")


class ArrowError(PatternError):
    """Error raised when an arrow is found in the pattern."""

    def __init__(self) -> None:
        super().__init__("Arrow '->' is not allowed in pattern")


class UndefinedSpaceError(ValueError):
    """Error raised when a required sample space is not defined."""

    def __init__(self, space_name: str) -> None:
        super().__init__(f"Undefined space: {space_name}")


class InvalidListTypeError(TypeError):
    """Error raised when a list kwarg contains non-numeric types."""

    def __init__(self, space_name: str, invalid_types: list[str]) -> None:
        types_str = ", ".join(invalid_types)
        message = (
            f"List provided for space '{space_name}' must contain only int or float values, "
            f"but got types: [{types_str}]"
        )
        super().__init__(message)


class UnsupportedSpaceTypeError(TypeError):
    """Error raised when a kwarg value has an unsupported type."""

    def __init__(self, space_name: str, invalid_type: str) -> None:
        message = (
            f"Unsupported type for space '{space_name}': {invalid_type}. "
            f"Must be a SpaceType, int, float, or list[int | float]."
        )
        super().__init__(message)

class UnknownBackendError(Exception):
    """Error raised when a tensor type is unknown to einmesh."""

    def __init__(self, tensor_type: type):
        self.message = f"Tensor type unknown to einmesh: {tensor_type}"
        super().__init__(self.message)
