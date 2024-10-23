import math
import random
from dataclasses import dataclass
from typing import List, Tuple


def make_pts(N: int) -> List[Tuple[float, float]]:
    """Generate N random points with x_1 and x_2 values in the range [0, 1).

    Args:
    ----
        N (int): The number of points to generate.

    Returns:
    -------
        List[Tuple[float, float]]: A list of tuples where each tuple contains
        two float values representing a point in 2D space.

    """
    X = []
    for i in range(N):
        x_1 = random.random()
        x_2 = random.random()
        X.append((x_1, x_2))
    return X


@dataclass
class Graph:
    """A class to represent a dataset for a binary classification problem.

    Attributes
    ----------
        N (int): The number of points in the dataset.
        X (List[Tuple[float, float]]): The list of points.
        y (List[int]): The list of labels for each point.

    """

    N: int
    X: List[Tuple[float, float]]
    y: List[int]


def simple(N: int) -> Graph:
    """Generate a dataset with points classified by a simple threshold on x_1.

    Args:
    ----
        N (int): The number of points to generate.

    Returns:
    -------
        Graph: A Graph object containing the generated points and labels.

    """
    X = make_pts(N)
    y = [1 if x_1 < 0.5 else 0 for x_1, x_2 in X]
    return Graph(N, X, y)


def diag(N: int) -> Graph:
    """Generate a dataset with points classified by a diagonal threshold on x_1 + x_2.

    Args:
    ----
        N (int): The number of points to generate.

    Returns:
    -------
        Graph: A Graph object containing the generated points and labels.

    """
    X = make_pts(N)
    y = [1 if x_1 + x_2 < 0.5 else 0 for x_1, x_2 in X]
    return Graph(N, X, y)


def split(N: int) -> Graph:
    """Generate a dataset with points classified by a split on x_1.

    Args:
    ----
        N (int): The number of points to generate.

    Returns:
    -------
        Graph: A Graph object containing the generated points and labels.

    """
    X = make_pts(N)
    y = [1 if x_1 < 0.2 or x_1 > 0.8 else 0 for x_1, x_2 in X]
    return Graph(N, X, y)


def xor(N: int) -> Graph:
    """Generate a dataset with points classified by an XOR-like function of x_1 and x_2.

    Args:
    ----
        N (int): The number of points to generate.

    Returns:
    -------
        Graph: A Graph object containing the generated points and labels.

    """
    X = make_pts(N)
    y = [
        1 if (x_1 < 0.5 and x_2 > 0.5) or (x_1 > 0.5 and x_2 < 0.5) else 0
        for x_1, x_2 in X
    ]
    return Graph(N, X, y)


def circle(N: int) -> Graph:
    """Generate a dataset with points classified by whether they are inside or outside
    a circle of radius sqrt(0.1) centered at (0.5, 0.5).

    Args:
    ----
        N (int): The number of points to generate.

    Returns:
    -------
        Graph: A Graph object containing the generated points and labels.

    """
    X = make_pts(N)
    y = [1 if (x_1 - 0.5) ** 2 + (x_2 - 0.5) ** 2 > 0.1 else 0 for x_1, x_2 in X]
    return Graph(N, X, y)


def spiral(N: int) -> Graph:
    """Generate a dataset with points arranged in two spirals.

    Args:
    ----
        N (int): The number of points to generate (even number).

    Returns:
    -------
        Graph: A Graph object containing the generated points and labels.

    """

    def x(t: float) -> float:
        return t * math.cos(t) / 20.0

    def y(t: float) -> float:
        return t * math.sin(t) / 20.0

    X = [
        (x(10.0 * (float(i) / (N // 2))) + 0.5, y(10.0 * (float(i) / (N // 2))) + 0.5)
        for i in range(N // 2)
    ]
    X += [
        (y(-10.0 * (float(i) / (N // 2))) + 0.5, x(-10.0 * (float(i) / (N // 2))) + 0.5)
        for i in range(N // 2)
    ]
    y2 = [0] * (N // 2) + [1] * (N // 2)
    return Graph(N, X, y2)


datasets = {
    "Simple": simple,
    "Diag": diag,
    "Split": split,
    "Xor": xor,
    "Circle": circle,
    "Spiral": spiral,
}
