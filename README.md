# einmesh
## einops-style multi dimensional meshgrids

[![Release](https://img.shields.io/github/v/release/niels-skovgaard-jensen/einmesh)](https://img.shields.io/github/v/release/niels-skovgaard-jensen/einmesh)
[![Build status](https://img.shields.io/github/actions/workflow/status/niels-skovgaard-jensen/einmesh/main.yml?branch=main)](https://github.com/niels-skovgaard-jensen/einmesh/actions/workflows/main.yml?query=branch%3Amain)
[![Commit activity](https://img.shields.io/github/commit-activity/m/niels-skovgaard-jensen/einmesh)](https://img.shields.io/github/commit-activity/m/niels-skovgaard-jensen/einmesh)
[![License](https://img.shields.io/github/license/niels-skovgaard-jensen/einmesh)](https://img.shields.io/github/license/niels-skovgaard-jensen/einmesh)


# Installation
Simple installation from pip:
```
pip install einmesh
```
# API
Create high dimensional meshgrids through a single command.
```python
from einmesh import einmesh, LogSpace, LinSpace

x, y = einmesh(
    "x y",                                       # Name a set of axis einops style
    x=LinSpace(start=0, end=3, num=3),           # Provide a space to sample from
    y=LogSpace(start=0, end=1, num=3, base=10),  # -- Mix and match as you like!
)
print(f"{x=}")
print(f"{y=}")
```
Will output
```bash
x=tensor([[0.0000, 0.0000, 0.0000],
        [1.5000, 1.5000, 1.5000],
        [3.0000, 3.0000, 3.0000]])
y=tensor([[ 1.0000,  3.1623, 10.0000],
        [ 1.0000,  3.1623, 10.0000],
        [ 1.0000,  3.1623, 10.0000]])
```
Thus it is effectively equivalent to common torch code
```python
x = torch.linspace(start=0, end=1, num=3)
y = torch.logspace(start=0, end=1, num=3, base=10)

x, y = torch.meshgrid(x, y, indexing="ij")
```
Which in 2D is not too bad, but in high dimensions can get tireing!
