
import numpy as np
from typing import List, Optional


def divided_differences(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    if n == 0:
        return np.array([])
    if len(y) != n:
        raise ValueError("x 与 y 长度必须一致")
    if len(np.unique(x)) != n:
        raise ValueError("节点坐标必须互异")


    table = np.zeros((n, n))
    table[:, 0] = y
    for col in range(1, n):
        for row in range(n - col):
            denom = x[row + col] - x[row]
            if abs(denom) < 1e-15:
                raise ValueError(f"节点 {row} 与 {row+col} 过于接近")
            table[row, col] = (table[row + 1, col - 1] - table[row, col - 1]) / denom

    return table[0, :]


def newton_evaluate(x_nodes: np.ndarray, coeffs: np.ndarray,
                    x_query: float) -> float:
    n = len(coeffs)
    if n == 0:
        return 0.0
    result = coeffs[-1]
    for k in range(n - 2, -1, -1):
        result = result * (x_query - x_nodes[k]) + coeffs[k]
    return float(result)


def newton_evaluate_array(x_nodes: np.ndarray, coeffs: np.ndarray,
                          x_queries: np.ndarray) -> np.ndarray:
    return np.array([newton_evaluate(x_nodes, coeffs, xq) for xq in x_queries])


class ConcentrationInterpolator:

    def __init__(self, x_nodes: np.ndarray, C_nodes: np.ndarray,
                 max_order: int = 8):
        self.x_nodes = np.asarray(x_nodes, dtype=float)
        self.C_nodes = np.asarray(C_nodes, dtype=float)
        self.max_order = int(max_order)
        if len(self.x_nodes) != len(self.C_nodes):
            raise ValueError("节点坐标与浓度数组长度不一致")
        if self.max_order < 1:
            raise ValueError("max_order 必须 ≥ 1")

    def _select_local_window(self, x_query: float) -> tuple[np.ndarray, np.ndarray]:
        if x_query <= self.x_nodes[0]:
            idx = slice(0, min(self.max_order + 1, len(self.x_nodes)))
            return self.x_nodes[idx], self.C_nodes[idx]
        if x_query >= self.x_nodes[-1]:
            start = max(0, len(self.x_nodes) - self.max_order - 1)
            idx = slice(start, len(self.x_nodes))
            return self.x_nodes[idx], self.C_nodes[idx]


        distances = np.abs(self.x_nodes - x_query)
        center_idx = int(np.argmin(distances))


        half = self.max_order // 2
        left = max(0, center_idx - half)
        right = min(len(self.x_nodes), left + self.max_order + 1)
        left = max(0, right - self.max_order - 1)
        idx = slice(left, right)
        return self.x_nodes[idx], self.C_nodes[idx]

    def interpolate(self, x_query: float) -> float:
        x_win, c_win = self._select_local_window(x_query)
        coeffs = divided_differences(x_win, c_win)
        return newton_evaluate(x_win, coeffs, x_query)

    def interpolate_batch(self, x_queries: np.ndarray) -> np.ndarray:
        return np.array([self.interpolate(xq) for xq in x_queries])

    def reconstruct_breakthrough_curve(self, x_obs: float,
                                       C_history: np.ndarray,
                                       t_values: np.ndarray) -> np.ndarray:
        if C_history.shape[1] != len(self.x_nodes):
            raise ValueError("C_history 的空间维度必须与节点数一致")
        if len(t_values) != C_history.shape[0]:
            raise ValueError("时间数组长度必须与 C_history 的时间维度一致")

        btc = np.zeros(len(t_values))
        for n in range(len(t_values)):

            self.C_nodes = C_history[n, :]
            btc[n] = self.interpolate(x_obs)
        return btc


def cubic_spline_natural(x: np.ndarray, y: np.ndarray,
                         x_query: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    if n < 3:

        return np.interp(x_query, x, y)

    h = np.diff(x)
    if np.any(h <= 0):
        raise ValueError("x 必须严格递增")


    A = np.zeros((n, n))
    b = np.zeros(n)
    A[0, 0] = 1.0
    A[-1, -1] = 1.0
    for i in range(1, n - 1):
        A[i, i - 1] = h[i - 1] / 6.0
        A[i, i] = (h[i - 1] + h[i]) / 3.0
        A[i, i + 1] = h[i] / 6.0
        b[i] = (y[i + 1] - y[i]) / h[i] - (y[i] - y[i - 1]) / h[i - 1]

    M = np.linalg.solve(A, b)


    a_coef = y[:-1]
    b_coef = (y[1:] - y[:-1]) / h - h * (2 * M[:-1] + M[1:]) / 6.0
    c_coef = M[:-1] / 2.0
    d_coef = (M[1:] - M[:-1]) / (6.0 * h)

    x_query = np.asarray(x_query, dtype=float)
    result = np.zeros_like(x_query)
    for i in range(n - 1):
        mask = (x_query >= x[i]) & (x_query <= x[i + 1])
        if i < n - 2:
            mask = mask | ((x_query > x[i]) & (x_query <= x[i + 1]))
        dx = x_query[mask] - x[i]
        result[mask] = a_coef[i] + dx * (b_coef[i] + dx * (c_coef[i] + dx * d_coef[i]))


    result[x_query < x[0]] = y[0]
    result[x_query > x[-1]] = y[-1]
    return result


if __name__ == "__main__":
    x = np.linspace(0, 10, 11)
    y = np.sin(x)
    coeffs = divided_differences(x, y)
    val = newton_evaluate(x, coeffs, 3.5)
    assert abs(val - np.sin(3.5)) < 0.01

    interp = ConcentrationInterpolator(x, y, max_order=5)
    v2 = interp.interpolate(3.5)
    assert abs(v2 - np.sin(3.5)) < 0.01


    xq = np.linspace(0, 10, 100)
    ys = cubic_spline_natural(x, y, xq)
    assert np.all(np.isfinite(ys))
    print("interpolation_engine: 自测试通过")
