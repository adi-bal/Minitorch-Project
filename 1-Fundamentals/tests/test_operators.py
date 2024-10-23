from typing import Callable, List, Tuple

import pytest
from hypothesis import given
from hypothesis.strategies import lists

from minitorch import MathTest
import minitorch
from minitorch.operators import (
    add,
    addLists,
    eq,
    exp,
    id,
    inv,
    inv_back,
    log_back,
    lt,
    max,
    mul,
    neg,
    negList,
    prod,
    relu,
    relu_back,
    sigmoid,
)

from .strategies import assert_close, small_floats

# ## Task 0.1 Basic hypothesis tests.


@pytest.mark.task0_1
@given(small_floats, small_floats)
def test_same_as_python(x: float, y: float) -> None:
    """Check that the main operators all return the same value of the python version"""
    assert_close(mul(x, y), x * y)
    assert_close(add(x, y), x + y)
    assert_close(neg(x), -x)
    assert_close(max(x, y), x if x > y else y)
    if abs(x) > 1e-5:
        assert_close(inv(x), 1.0 / x)


@pytest.mark.task0_1
@given(small_floats)
def test_relu(a: float) -> None:
    if a > 0:
        assert relu(a) == a
    if a < 0:
        assert relu(a) == 0.0


@pytest.mark.task0_1
@given(small_floats, small_floats)
def test_relu_back(a: float, b: float) -> None:
    if a > 0:
        assert relu_back(a, b) == b
    if a < 0:
        assert relu_back(a, b) == 0.0


@pytest.mark.task0_1
@given(small_floats)
def test_id(a: float) -> None:
    assert id(a) == a


@pytest.mark.task0_1
@given(small_floats)
def test_lt(a: float) -> None:
    """Check that a - 1.0 is always less than a"""
    assert lt(a - 1.0, a) == 1.0
    assert lt(a, a - 1.0) == 0.0


@pytest.mark.task0_1
@given(small_floats)
def test_max(a: float) -> None:
    assert max(a - 1.0, a) == a
    assert max(a, a - 1.0) == a
    assert max(a + 1.0, a) == a + 1.0
    assert max(a, a + 1.0) == a + 1.0


@pytest.mark.task0_1
@given(small_floats)
def test_eq(a: float) -> None:
    assert eq(a, a) == 1.0
    assert eq(a, a - 1.0) == 0.0
    assert eq(a, a + 1.0) == 0.0


@pytest.mark.task0_2
@given(small_floats)
def test_sigmoid(a: float) -> None:
    sigm_a = sigmoid(a)
    assert 0.0 <= sigm_a <= 1.0
    assert_close(sigmoid(-a), 1 - sigm_a)
    if a == 0:
        assert sigm_a == 0.5
    delta = 1e-6
    if abs(a) < 10.0:
        assert sigmoid(a) < sigmoid(a + delta)


@pytest.mark.task0_2
@given(small_floats, small_floats, small_floats)
def test_transitive(a: float, b: float, c: float) -> None:
    """Test the transitive property of less-than (a < b and b < c implies a < c)"""
    if a < b and b < c:
        assert a < c


@pytest.mark.task0_2
@given(small_floats, small_floats)
def test_symmetric(a: float, b: float) -> None:
    """Test that the multiplication function is symmetric, i.e., mul(a, b) == mul(b, a)."""
    assert mul(a, b) == mul(b, a)


@pytest.mark.task0_2
@given(small_floats, small_floats, small_floats)
def test_distribute(x: float, y: float, z: float) -> None:
    """Test the distributive property: z * (x + y) = z * x + z * y."""
    assert_close(mul(z, add(x, y)), add(mul(z, x), mul(z, y)))


@pytest.mark.task0_2
@given(small_floats, small_floats)
def test_exponential(a: float, b: float) -> None:
    expo_a = exp(a)

    # Exponential is always positive
    assert expo_a > 0

    # Exponential at 0 is 1
    if a == 0:
        assert expo_a == 1

    # Exponential is strictly increasing
    delta = 1e-6
    if abs(a) < 10.0:
        assert exp(a) < exp(a + delta)


@pytest.mark.task0_3
@given(small_floats, small_floats, small_floats, small_floats)
def test_zip_with(a: float, b: float, c: float, d: float) -> None:
    x1, x2 = addLists([a, b], [c, d])
    y1, y2 = a + c, b + d
    assert_close(x1, y1)
    assert_close(x2, y2)


@pytest.mark.task0_3
@given(
    lists(small_floats, min_size=5, max_size=5),
    lists(small_floats, min_size=5, max_size=5),
)
def test_sum_distribute(ls1: List[float], ls2: List[float]) -> None:
    # Calculate the sum of the two lists individually
    sum_ls1 = sum(ls1)
    sum_ls2 = sum(ls2)

    # Calculate the sum of element-wise addition of the two lists
    combined_sum = sum(addLists(ls1, ls2))

    # Ensure that the sum of individual sums is the same as the combined sum
    (
        assert_close(sum_ls1 + sum_ls2, combined_sum),
        "Distributive property failed: sum(ls1) + sum(ls2) != sum(ls1 + ls2)",
    )


@pytest.mark.task0_3
@given(lists(small_floats))
def test_sum(ls: List[float]) -> None:
    assert_close(sum(ls), minitorch.operators.sum(ls))


@pytest.mark.task0_3
@given(small_floats, small_floats, small_floats)
def test_prod(x: float, y: float, z: float) -> None:
    assert_close(prod([x, y, z]), x * y * z)


@pytest.mark.task0_3
@given(lists(small_floats))
def test_negList(ls: List[float]) -> None:
    check = negList(ls)
    for i, j in zip(ls, check):
        assert_close(i, -j)


# ## Generic mathematical tests

# For each unit this generic set of mathematical tests will run.


one_arg, two_arg, _ = MathTest._tests()


@given(small_floats)
@pytest.mark.parametrize("fn", one_arg)
def test_one_args(fn: Tuple[str, Callable[[float], float]], t1: float) -> None:
    name, base_fn = fn
    base_fn(t1)


@given(small_floats, small_floats)
@pytest.mark.parametrize("fn", two_arg)
def test_two_args(
    fn: Tuple[str, Callable[[float, float], float]], t1: float, t2: float
) -> None:
    name, base_fn = fn
    base_fn(t1, t2)


@given(small_floats, small_floats)
def test_backs(a: float, b: float) -> None:
    relu_back(a, b)
    inv_back(a + 2.4, b)
    log_back(abs(a) + 4, b)
