
import numpy as np
from typing import Callable, Tuple


def chebyshev_coefficients(a: float, b: float, n: int, f: Callable) -> np.ndarray:
    if n <= 0:
        raise ValueError("插值阶数 n 必须为正整数")
    if b <= a:
        raise ValueError("区间右端点 b 必须大于左端点 a")


    angles = (2.0 * np.arange(1, n + 1) - 1.0) * np.pi / (2.0 * n)
    x_nodes = np.cos(angles)

    x_physical = 0.5 * (a + b) + 0.5 * (b - a) * x_nodes
    fx = f(x_physical)

    c = np.zeros(n, dtype=float)
    for k in range(n):
        c[k] = np.sum(fx * np.cos((k) * angles))

    c *= 2.0 / n
    return c


def chebyshev_interpolant(a: float, b: float, n: int, c: np.ndarray,
                          x_eval: np.ndarray) -> np.ndarray:
    if len(c) != n:
        raise ValueError("系数向量长度必须与插值阶数一致")

    x_eval = np.asarray(x_eval, dtype=float).ravel()
    if b == a:
        return np.full_like(x_eval, c[0] * 0.5 if n > 0 else 0.0)

    y = (2.0 * x_eval - a - b) / (b - a)
    m = len(x_eval)

    d1 = np.zeros(m, dtype=float)
    d2 = np.zeros(m, dtype=float)


    for i in range(n - 1, 0, -1):
        d0 = 2.0 * y * d1 - d2 + c[i]
        d2 = d1
        d1 = d0

    value = y * d1 - d2 + 0.5 * c[0]
    return value


def lagrange_basis_1d(xd: np.ndarray, xi: np.ndarray) -> np.ndarray:
    xd = np.asarray(xd, dtype=float).ravel()
    xi = np.asarray(xi, dtype=float).ravel()
    nd = len(xd)
    ni = len(xi)

    if nd == 0:
        raise ValueError("插值节点不能为空")

    lb = np.ones((ni, nd), dtype=float)
    for j in range(nd):
        for k in range(nd):
            if k != j:
                denom = xd[j] - xd[k]
                if abs(denom) < 1e-14:
                    raise ValueError(f"插值节点重复或过于接近：xd[{j}]={xd[j]}, xd[{k}]={xd[k]}")
                lb[:, j] *= (xi - xd[k]) / denom
    return lb


def lagrange_value_1d(xd: np.ndarray, yd: np.ndarray, xi: np.ndarray) -> np.ndarray:
    yd = np.asarray(yd, dtype=float).ravel()
    lb = lagrange_basis_1d(xd, xi)
    return lb @ yd


def build_2d_chebyshev_basis(image_shape: Tuple[int, int], order: int) -> np.ndarray:










    raise NotImplementedError("Hole_1: 二维切比雪夫张量积基构造待实现")


def image_to_chebyshev_coefficients(image: np.ndarray, order: int) -> np.ndarray:
    if image.ndim != 2:
        raise ValueError("输入必须是二维图像")

    H, W = image.shape
    Psi = build_2d_chebyshev_basis((H, W), order)
    vec = image.ravel()


    A = Psi.T @ Psi
    b = Psi.T @ vec


    reg = 1e-10 * np.eye(A.shape[0])
    c = np.linalg.solve(A + reg, b)
    return c


def chebyshev_coefficients_to_image(coeffs: np.ndarray, image_shape: Tuple[int, int],
                                    order: int) -> np.ndarray:
    Psi = build_2d_chebyshev_basis(image_shape, order)
    vec = Psi @ coeffs
    return vec.reshape(image_shape)
