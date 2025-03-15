# einmesh
## einops-style multi dimensional meshgrids

[![Release](https://img.shields.io/github/v/release/niels-skovgaard-jensen/einmesh)](https://img.shields.io/github/v/release/niels-skovgaard-jensen/einmesh)
[![Build status](https://img.shields.io/github/actions/workflow/status/niels-skovgaard-jensen/einmesh/main.yml?branch=main)](https://github.com/niels-skovgaard-jensen/einmesh/actions/workflows/main.yml?query=branch%3Amain)
[![Commit activity](https://img.shields.io/github/commit-activity/m/niels-skovgaard-jensen/einmesh)](https://img.shields.io/github/commit-activity/m/niels-skovgaard-jensen/einmesh)
[![License](https://img.shields.io/github/license/niels-skovgaard-jensen/einmesh)](https://img.shields.io/github/license/niels-skovgaard-jensen/einmesh)

# Example

```python
from einmesh import einmesh
from einmesh.spaces import LinSpace, LogSpace

x = einmesh(
    "x y",                             # Name a set of axis
    x=LinSpace(start=0, end=1, num=2), # Provide a space to sample from
    y=LogSpace(start=0, end=1, num=2)  # Mix and match as you like!
)

print(x.shape)
```
