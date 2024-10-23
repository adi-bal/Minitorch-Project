from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Tuple, Protocol


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""Computes an approximation to the derivative of `f` with respect to one arg.

    Args:
    ----
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
    -------
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$

    """
    vals_plus = list(vals)
    vals_minus = list(vals)
    vals_plus[arg] += epsilon
    vals_minus[arg] -= epsilon
    return (f(*vals_plus) - f(*vals_minus)) / (2.0 * epsilon)


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        """Accumulate the derivative of the variable with respect to a given value.

        Args:
        ----
            x: The value with respect to which the derivative is accumulated.

        """
        ...

    @property
    def unique_id(self) -> int:
        """Return a unique identifier for the variable."""
        ...

    def is_leaf(self) -> bool:
        """Check if the variable is a leaf node in the computational graph.

        Returns
        -------
            bool: True if the variable has no parents, False otherwise.

        """
        ...

    def is_constant(self) -> bool:
        """Check if the variable is constant (not subject to change).

        Returns
        -------
            bool: True if the variable is constant, False otherwise.

        """
        ...

    @property
    def parents(self) -> Iterable["Variable"]:
        """Return an iterable of parent variables.

        Returns
        -------
            Iterable[Variable]: The parent variables of this variable.

        """
        ...

    def chain_rule(self, d_output: Any) -> Iterable[Tuple["Variable", Any]]:
        """Apply the chain rule to compute derivatives with respect to parents.

        Args:
        ----
            d_output: The derivative of the output with respect to this variable.

        Returns:
        -------
            Iterable[Tuple[Variable, Any]]: A list of tuples, where each tuple contains
            a parent variable and its corresponding derivative.

        """
        ...


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """Computes the topological order of the computation graph.

    Args:
    ----
        variable: The right-most variable

    Returns:
    -------
        Non-constant Variables in topological order starting from the right.

    """
    visited = set()
    result = []

    def visit(var: Variable) -> None:
        if var.unique_id in visited:
            return
        visited.add(var.unique_id)
        for parent in var.parents:
            visit(parent)
        if not var.is_constant():
            result.append(var)

    visit(variable)

    return reversed(result)


def backpropagate(variable: Variable, deriv: Any) -> None:
    """Runs backpropagation on the computation graph in order to
    compute derivatives for the leaf nodes.

    Args:
    ----
    variable: The right-most variable.
    deriv: The derivative that we want to propagate backward to the leaves.

    No return. Should write its results to the derivative values of each leaf through `accumulate_derivative`.

    """
    ordering = topological_sort(variable)
    derivative = dict()

    derivative[variable.unique_id] = deriv

    for var in ordering:
        d_var = 0
        if var.unique_id in derivative:
            d_var = derivative[var.unique_id]

        if var.is_leaf():
            var.accumulate_derivative(d_var)
        else:
            for parent, l_grad in var.chain_rule(d_var):
                if parent.unique_id not in derivative:
                    derivative[parent.unique_id] = 0
                derivative[parent.unique_id] += l_grad


@dataclass
class Context:
    """Context class is used by `Function` to store information during the forward pass."""

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        """Store the given `values` if they need to be used during backpropagation."""
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        """Return the saved tensors."""
        return self.saved_values
