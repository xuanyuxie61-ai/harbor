
import numpy as np
import math
from typing import List, Tuple, Iterator


def simplex_lattice_points(n: int, t: int) -> List[np.ndarray]:
    if n < 1:
        return []
    if t < 0:
        return []

    points = []
    x = np.zeros(n, dtype=int)
    x[0] = t
    more = True

    while more:
        points.append(x.copy())

        if x[0] == t and np.sum(x[1:]) == 0:


            pass


        k = -1
        for i in range(n - 1, -1, -1):
            if x[i] > 0:
                k = i
                break

        if k == -1:
            more = False
            break

        if k == n - 1:

            x[k] -= 1
            if k > 0:
                x[k - 1] += 1
        else:

            s = x[k] - 1
            x[k] = 0
            x[k + 1] = s + 1






            x[k] = 0
            x[k + 1] = s + 1
            x[k + 2:] = 0



            pass


    return _generate_simplex_lattice_direct(n, t)


def _generate_simplex_lattice_direct(n: int, t: int) -> List[np.ndarray]:
    points = []

    def helper(dim: int, remaining: int, current: List[int]):
        if dim == n - 1:
            current.append(remaining)
            points.append(np.array(current, dtype=int))
            current.pop()
            return
        for v in range(remaining + 1):
            current.append(v)
            helper(dim + 1, remaining - v, current)
            current.pop()

    helper(0, t, [])
    return points


def simplex_lattice_iterator(n: int, t: int) -> Iterator[np.ndarray]:
    def helper(dim: int, remaining: int, current: List[int]):
        if dim == n - 1:
            current.append(remaining)
            yield np.array(current, dtype=int)
            current.pop()
            return
        for v in range(remaining + 1):
            current.append(v)
            yield from helper(dim + 1, remaining - v, current)
            current.pop()

    yield from helper(0, t, [])


def portfolio_weight_grid(n_assets: int, n_grid: int) -> np.ndarray:
    points = _generate_simplex_lattice_direct(n_assets, n_grid)

    points_eq = [p for p in points if p.sum() == n_grid]
    if len(points_eq) == 0:

        weights = np.array(points, dtype=float) / n_grid

        row_sums = weights.sum(axis=1, keepdims=True)
        weights = weights / (row_sums + 1e-15)
        return weights
    weights = np.array(points_eq, dtype=float) / n_grid
    return weights


def correlation_simplex_grid(n_factors: int, n_levels: int) -> List[np.ndarray]:
    points = _generate_simplex_lattice_direct(n_factors, n_levels)
    eigenvalue_grids = []
    for p in points:
        if p.sum() > 0:
            ev = p.astype(float) / p.sum() * n_factors
            eigenvalue_grids.append(ev)
    return eigenvalue_grids


def test_lattice_enumeration():
    n, t = 3, 4
    points = _generate_simplex_lattice_direct(n, t)
    expected = int(math.comb(n + t - 1, t))
    assert len(points) == expected, f"格点数量错误: {len(points)} != {expected}"
    for p in points:
        assert p.sum() <= t, f"格点总和超限: {p.sum()} > {t}"
        assert np.all(p >= 0), "格点存在负分量"
    print(f"lattice_enumeration test passed. n={n}, t={t}, count={len(points)}")


    weights = portfolio_weight_grid(3, 4)
    assert np.allclose(weights.sum(axis=1), 1.0), "权重和不为 1"
    assert np.all(weights >= 0), "权重存在负值"
    print(f"portfolio_weight_grid test passed. shape={weights.shape}")


if __name__ == "__main__":
    test_lattice_enumeration()
