from einmesh.spaces import LinSpace
from tests.conftest import parametrize_backends


@parametrize_backends
def test_addition_with_float(backend):
    backend = backend()

    linspace = LinSpace(start=0.0, end=1.0, num=5)

    added_linspace = linspace + 1.0
    added_linspace_reverse = 1.0 + linspace

    assert backend.allclose(linspace._sample(backend), backend.linspace(0.0, 1.0, 5))
    assert backend.allclose(added_linspace._sample(backend), backend.linspace(1.0, 2.0, 5))
    assert backend.allclose(added_linspace_reverse._sample(backend), backend.linspace(1.0, 2.0, 5))


@parametrize_backends
def test_subtraction_with_float(backend):
    backend = backend()

    linspace = LinSpace(start=0.0, end=1.0, num=5)

    subtracted_linspace = linspace - 1.0

    assert backend.allclose(linspace._sample(backend), backend.linspace(0.0, 1.0, 5))
    assert backend.allclose(subtracted_linspace._sample(backend), backend.linspace(0.0, 1.0, 5) - 1.0)


@parametrize_backends
def test_multiplication_with_float(backend):
    backend = backend()

    linspace = LinSpace(start=0.0, end=1.0, num=5)

    multiplied_linspace = linspace * 2.0

    assert backend.allclose(linspace._sample(backend), backend.linspace(0.0, 1.0, 5))
    assert backend.allclose(multiplied_linspace._sample(backend), backend.linspace(0.0, 2.0, 5))


@parametrize_backends
def test_truediv_with_float(backend):
    backend = backend()

    linspace = LinSpace(start=0.0, end=1.0, num=5)

    divided_linspace = linspace / 2.0

    assert backend.allclose(linspace._sample(backend), backend.linspace(0.0, 1.0, 5))
    assert backend.allclose(divided_linspace._sample(backend), backend.linspace(0.0, 0.5, 5))


@parametrize_backends
def test_mod_with_float(backend):
    backend = backend()

    linspace = LinSpace(start=0.0, end=1.0, num=5)

    mod_linspace = linspace % 2.0

    assert backend.allclose(linspace._sample(backend), backend.linspace(0.0, 1.0, 5))
    assert backend.allclose(mod_linspace._sample(backend), backend.linspace(0.0, 1.0, 5) % 2.0)


@parametrize_backends
def test_floordiv_with_float(backend):
    backend = backend()

    linspace = LinSpace(start=0.0, end=1.0, num=5)

    floordiv_linspace = linspace // 2.0

    assert backend.allclose(linspace._sample(backend), backend.linspace(0.0, 1.0, 5))
    assert backend.allclose(floordiv_linspace._sample(backend), backend.linspace(0.0, 1.0, 5) // 2.0)


@parametrize_backends
def test_pow_with_float(backend):
    backend = backend()

    linspace = LinSpace(start=0.0, end=1.0, num=5)

    pow_linspace = linspace**2.0

    assert backend.allclose(linspace._sample(backend), backend.linspace(0.0, 1.0, 5))
    assert backend.allclose(pow_linspace._sample(backend), backend.linspace(0.0, 1.0, 5) ** 2.0)


@parametrize_backends
def test_negation(backend):
    backend = backend()

    linspace = LinSpace(start=0.0, end=1.0, num=5)

    negated_linspace = -linspace

    assert backend.allclose(negated_linspace._sample(backend), backend.linspace(0.0, -1.0, 5))


@parametrize_backends
def test_pos(backend):
    backend = backend()

    linspace = LinSpace(start=0.0, end=1.0, num=5)

    pos_linspace = +linspace

    assert backend.allclose(pos_linspace._sample(backend), backend.linspace(0.0, 1.0, 5))
