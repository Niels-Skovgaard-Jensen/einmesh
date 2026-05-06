"""Verify the per-backend einmesh modules expose an identical API.

The numpy, torch, and jax backend modules must each expose ``einmesh`` and
``EinMesher`` with the same call-time signature, and must respond identically
to the same inputs. This guards against signature drift between backends.
"""

from __future__ import annotations

import importlib
import inspect
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
from numpy.testing import assert_allclose

from einmesh import ConstantSpace, LinSpace, ListSpace
from einmesh._backends import AbstractBackend, JaxBackend, NumpyBackend, TorchBackend

_BACKEND_CLASSES: tuple[type[AbstractBackend], ...] = (NumpyBackend, TorchBackend, JaxBackend)


def _load_available_backends() -> dict[str, Any]:
    return {
        cls.framework_name: importlib.import_module(f"einmesh.{cls.framework_name}")
        for cls in _BACKEND_CLASSES
        if cls.is_available()
    }


def _to_numpy(value: object) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _signature_params(func: object, *, drop_self: bool = False) -> list[tuple[str, object, object]]:
    """Annotations differ across backends by design (np.ndarray vs torch.Tensor vs
    jax.Array), so they are excluded from the comparison.
    """
    params = list(inspect.signature(func).parameters.values())
    if drop_self and params and params[0].name == "self":
        params = params[1:]
    return [(p.name, p.kind, p.default) for p in params]


@pytest.fixture(scope="module")
def backend_modules() -> dict[str, Any]:
    modules = _load_available_backends()
    if len(modules) < 2:
        pytest.skip("Need at least two backends installed to compare APIs")
    return modules


def test_each_backend_exposes_required_names(backend_modules: dict[str, Any]) -> None:
    for name, module in backend_modules.items():
        assert hasattr(module, "einmesh"), f"{name} backend missing einmesh"
        assert hasattr(module, "EinMesher"), f"{name} backend missing EinMesher"
        assert callable(module.einmesh), f"{name}.einmesh is not callable"
        assert callable(module.EinMesher), f"{name}.EinMesher is not callable"


@pytest.mark.parametrize(
    ("label", "getter", "drop_self"),
    [
        ("einmesh", lambda m: m.einmesh, False),
        ("EinMesher", lambda m: m.EinMesher, False),
        ("EinMesher.sample", lambda m: m.EinMesher.sample, True),
    ],
)
def test_signature_is_identical(
    backend_modules: dict[str, Any],
    label: str,
    getter: Callable[[Any], Any],
    drop_self: bool,
) -> None:
    sigs = {name: _signature_params(getter(mod), drop_self=drop_self) for name, mod in backend_modules.items()}
    reference_name, reference = next(iter(sigs.items()))
    for name, params in sigs.items():
        if name == reference_name:
            continue
        assert params == reference, (
            f"{label} signature for {name} differs from {reference_name}: {params} vs {reference}"
        )


def _assert_outputs_match(outputs: dict[str, Any]) -> None:
    reference_name, reference = next(iter(outputs.items()))
    is_tuple = isinstance(reference, tuple)
    for name, out in outputs.items():
        if name == reference_name:
            continue
        assert isinstance(out, tuple) == is_tuple, (
            f"return container differs: {reference_name} tuple={is_tuple}, {name} tuple={isinstance(out, tuple)}"
        )
        if is_tuple:
            assert len(out) == len(reference), f"tuple length differs between {reference_name} and {name}"
            for ref_item, item in zip(reference, out, strict=False):
                assert_allclose(_to_numpy(item), _to_numpy(ref_item))
        else:
            assert_allclose(_to_numpy(out), _to_numpy(reference))


@pytest.mark.parametrize(
    ("args", "kwargs"),
    [
        pytest.param(("x y",), {"x": LinSpace(0, 1, 3), "y": ListSpace([10.0, 20.0])}, id="named-spaces"),
        pytest.param(("...", LinSpace(0, 1, 3), ConstantSpace(7.0)), {}, id="positional-spaces"),
        pytest.param(("... y", LinSpace(0, 1, 3)), {"y": ConstantSpace(5.0)}, id="mixed-positional-named"),
        pytest.param(("* x y",), {"x": LinSpace(0, 1, 3), "y": ListSpace([10.0, 20.0])}, id="stacked"),
        pytest.param(("x z",), {"x": LinSpace(0, 1, 4), "z": [1.0, 2.0, 4.0]}, id="implicit-constant-list"),
    ],
)
def test_einmesh_outputs_match(
    backend_modules: dict[str, Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> None:
    outputs = {name: mod.einmesh(*args, **kwargs) for name, mod in backend_modules.items()}
    _assert_outputs_match(outputs)


@pytest.mark.parametrize("pattern", ["x y", "* x y"])
def test_einmesher_sample_matches(backend_modules: dict[str, Any], pattern: str) -> None:
    outputs = {
        name: mod.EinMesher(pattern, x=LinSpace(0, 1, 3), y=ListSpace([10.0, 20.0])).sample()
        for name, mod in backend_modules.items()
    }
    _assert_outputs_match(outputs)
