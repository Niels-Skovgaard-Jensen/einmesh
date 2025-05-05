<p align="center">
  <img src="docs/img/Einmesh Logo.svg" onerror="this.src='img/Einmesh Logo.svg'" alt="Einmesh Logo" width="600"/>
</p>

## einops-style multi dimensional meshgrids

![PyPI - Python Version](https://img.shields.io/pypi/pyversions/einmesh)
[![Release](https://img.shields.io/github/v/release/niels-skovgaard-jensen/einmesh)](https://img.shields.io/github/v/release/niels-skovgaard-jensen/einmesh)
[![Build status](https://img.shields.io/github/actions/workflow/status/niels-skovgaard-jensen/einmesh/main.yml?branch=main)](https://github.com/niels-skovgaard-jensen/einmesh/actions/workflows/main.yml?query=branch%3Amain)
[![Commit activity](https://img.shields.io/github/commit-activity/m/niels-skovgaard-jensen/einmesh)](https://img.shields.io/github/commit-activity/m/niels-skovgaard-jensen/einmesh)
[![License](https://img.shields.io/github/license/niels-skovgaard-jensen/einmesh)](https://img.shields.io/github/license/niels-skovgaard-jensen/einmesh)



# Installation
Simple installation from uv:
```
uv add einmesh
```
or pip:
```
pip install einmesh
```
# Features
- **einops-style Meshgrid Generation**: The core function `einmesh` allows creating multi-dimensional meshgrids (like `torch.meshgrid`) using a concise string pattern similar to `einops`.
- **Flexible Space Definitions**: Users define the dimensions using various "space" objects:
    - `LinSpace`: Linearly spaced points.
    - `LogSpace`: Logarithmically spaced points.
    - `UniformDistribution`: Points sampled from a uniform distribution.
    - `NormalDistribution`: Points sampled from a normal distribution.
- **Pattern Features**:
    - **Named Dimensions**: Pattern elements correspond to keyword arguments (e.g., `einmesh("x y", x=LinSpace(...), y=LogSpace(...))`).
    - **Dimension Ordering**: The order in the pattern determines the order/shape of the output tensors (using `ij` indexing convention, like NumPy).
    - **Stacking (`*`)**: A `*` in the pattern stacks the individual coordinate tensors along a new dimension, returning a single tensor.
    - **Grouping (`()`)**: Parentheses group axes for rearrangement using `einops.rearrange`.
    - **Duplicate Names**: Handles patterns like `"x x y"`, which repeats an axis and re-samples it.
- **Output**:
    - Returns a tuple of coordinate tensors if no `*` is present.
    - Returns a single stacked tensor if `*` is present.
- **Backend**: Numpy, Torch and JAX are all supported! Just import the einmesh function from the backend like
```python
from einmesh.numpy import einmesh # Creates numpy arrays
from einmesh.jax import einmesh # Creates JAX arrays
from einmesh.torch import einmesh # Creates Torch Tensors
```
- **Super-Lightweight**: Only installation dependency is einops! At runtime backends are loaded if available.
# Examples
**1. Basic 2D Linear Grid**

Create a simple 2D grid with linearly spaced points along x and y.

```python
from einmesh import LinSpace
from einmesh.numpy import einmesh

# Define the spaces
x_space = LinSpace(0, 1, 10)  # 10 points from 0 to 1
y_space = LinSpace(-1, 1, 20) # 20 points from -1 to 1

# Create the meshgrid
# Output: tuple of two tensors, each with shape (10, 20) following 'ij' indexing
x_coords, y_coords = einmesh("x y", x=x_space, y=y_space)

print(f"{x_coords.shape=}")
print(f"{y_coords.shape=}")

# Output:
# x_coords.shape=(10, 20)
# y_coords.shape=(10, 20)
```

**2. Stacked Coordinates**

Create a 3D grid and stack the coordinate tensors into a single tensor.

```python
x_space = LinSpace(0, 1, 5)
y_space = LinSpace(0, 1, 6)
z_space = LogSpace(1, 2, 7)

# Use '*' to stack the coordinates along the last dimension
# Output: single tensor with shape (5, 6, 7, 3)
coords = einmesh("x y z *", x=x_space, y=y_space, z=z_space)

print(coords.shape)
# Output: (5, 6, 7, 3)
# coords[..., 0] contains x coordinates
# coords[..., 1] contains y coordinates
# coords[..., 2] contains z coordinates
```

**3. Using Distributions**

Generate grid points by sampling from distributions.

```python
from einmesh import UniformDistribution, NormalDistribution

# Sample 10 points uniformly between -5 and 5 for x
x_dist = UniformDistribution(-5, 5, 10)
# Sample 15 points from a normal distribution (mean=0, std=1) for y
y_dist = NormalDistribution(0, 1, 15)

# Create the meshgrid
# Output: tuple of two tensors, each with shape (10, 15)
x_samples, y_samples = einmesh("x y", x=x_dist, y=y_dist)

print(x_samples.shape, y_samples.shape)
# Output: (10, 15) (10, 15)
# Note: The points along each axis will not be sorted.
```

**4. Duplicate Dimension Names**

Use the same space definition for multiple axes.

```python
space = LinSpace(0, 1, 5)

# 'x' space is used for both the first and second dimensions.
# Output shapes: (5, 5, 10) for each tensor
x0_coords, x1_coords, y_coords = einmesh("x x y", x=space, y=LinSpace(-1, 1, 10))

print(x0_coords.shape, x1_coords.shape, y_coords.shape)
# Output: (5, 5, 10) (5, 5, 10) (5, 5, 10)
```
