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
