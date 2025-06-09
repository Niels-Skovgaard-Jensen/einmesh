# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Setup and Installation
- `make install` - Install development dependencies and pre-commit hooks using uv
- `uv sync --group dev-all` - Sync all development dependencies

### Testing and Quality Checks
- `make test` - Run pytest with coverage
- `make check` - Run all code quality tools (pre-commit, lock file check, basedpyright)
- `uv run pytest` - Run tests directly
- `uv run basedpyright` - Type checking
- `uv run ruff check` - Linting
- `uv run ruff format` - Code formatting

### Building and Publishing
- `make build` - Build wheel file
- `make docs` - Build and serve documentation with mkdocs

### Backend-Specific Testing
- `make check-dep` - Test dependencies for numpy, torch, and jax backends individually

## Architecture Overview

### Core Concept
Einmesh provides einops-style multi-dimensional meshgrid generation using string patterns. Users define coordinate spaces and combine them using patterns like `"x y"` or `"x y z *"`.

### Key Components

**Backend System** (`_backends.py`):
- Abstract backend system supporting numpy, torch, and jax
- Dynamic backend loading based on tensor types
- Each backend module (numpy.py, torch.py, jax.py) provides `einmesh` function

**Space Definitions** (`spaces.py`):
- `LinSpace`: Linearly spaced points
- `LogSpace`: Logarithmically spaced points
- `UniformDistribution`/`NormalDistribution`: Random sampling
- `ConstantSpace`, `ListSpace`, `RangeSpace`: Additional space types

**Pattern Parser** (`_parser.py`):
- Parses einops-style patterns like `"x y"`, `"x y z *"`, `"(x y) z"`
- Handles stacking (`*`), grouping (`()`), and duplicate names
- Core `_einmesh` function coordinates parsing and backend execution

**Operators** (`_operators.py`):
- Mathematical operations (sin, cos, exp) that work across backends

### Import Pattern
```python
from einmesh.numpy import einmesh  # NumPy backend
from einmesh.torch import einmesh  # PyTorch backend
from einmesh.jax import einmesh    # JAX backend
from einmesh import LinSpace, LogSpace  # Space definitions
```

### Testing Strategy
- Tests are backend-agnostic where possible
- Specific backend tests in separate test files
- Coverage includes error conditions and edge cases
