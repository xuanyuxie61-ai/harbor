"""
interpolation_engine.py
================================================================================
浓度场高精度插值与突破曲线重建模块

基于种子项目：
  - 592_interp_equal：Newton 均差插值与 Horner 求值

科学背景：
  有限元求解器仅在节点上给出浓度值 C_j = C(x_j)。然而，环境监管通常要求
  在任意监测井位置 x_obs 处评估浓度，这需要高阶插值方法。

  Newton 插值多项式：
      P_n(x) = c_0 + c_1(x-x_0) + c_2(x-x_0)(x-x_1) + ... + c_n(x-x_0)...(x-x_{n-1})
  其中系数 c_k 为 k 阶均差：
      c_k = f[x_0, x_1, ..., x_k]

  均差递推：
      f[x_i] = y_i
      f[x_i, ..., x_j] = (f[x_{i+1}, ..., x_j] - f[x_i, ..., x_{j-1}]) / (x_j - x_i)

  优势：
    - 新增节点时无需重新计算所有系数
    - 适合非等距节点（如有限元自适应加密后的节点分布）
    - 计算复杂度 O(n²) 建立，O(n) 求值

  在地下水模型中，Newton 插值用于：
    - 监测井处的浓度插值
    - 突破曲线（breakthrough curve）的连续重构
    - 自适应网格间的解传递
================================================================================
"""

import numpy as np
from typing import List, Optional


def divided_differences(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    计算 Newton 均差表的主对角线系数。

    输入：
        x : 节点坐标，必须严格递增且无重复
        y : 节点函数值
    返回：
        coeffs : 长度 len(x) 的系数数组，P(x) = Σ coeffs[k] * ω_k(x)
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    if n == 0:
        return np.array([])
    if len(y) != n:
        raise ValueError("x 与 y 长度必须一致")
    if len(np.unique(x)) != n:
        raise ValueError("节点坐标必须互异")

    # 构建均差表（上三角矩阵）
    table = np.zeros((n, n))
    table[:, 0] = y
    for col in range(1, n):
        for row in range(n - col):
            denom = x[row + col] - x[row]
            if abs(denom) < 1e-15:
                raise ValueError(f"节点 {row} 与 {row+col} 过于接近")
            table[row, col] = (table[row + 1, col - 1] - table[row, col - 1]) / denom

    return table[0, :]  # 主对角线即为 Newton 系数


def newton_evaluate(x_nodes: np.ndarray, coeffs: np.ndarray,
                    x_query: float) -> float:
    """
    使用 Horner 嵌套乘法求 Newton 插值多项式在 x_query 处的值。

    算法：
        P(x) = c_0 + (x-x_0)(c_1 + (x-x_1)(c_2 + ...))
    """
    n = len(coeffs)
    if n == 0:
        return 0.0
    result = coeffs[-1]
    for k in range(n - 2, -1, -1):
        result = result * (x_query - x_nodes[k]) + coeffs[k]
    return float(result)


def newton_evaluate_array(x_nodes: np.ndarray, coeffs: np.ndarray,
                          x_queries: np.ndarray) -> np.ndarray:
    """批量求值。"""
    return np.array([newton_evaluate(x_nodes, coeffs, xq) for xq in x_queries])


class ConcentrationInterpolator:
    """
    基于 Newton 插值的浓度场重建器。
    """

    def __init__(self, x_nodes: np.ndarray, C_nodes: np.ndarray,
                 max_order: int = 8):
        """
        参数
        ----------
        x_nodes : np.ndarray
            FEM 节点坐标
        C_nodes : np.ndarray
            节点浓度值
        max_order : int
            每个局部插值窗口的最大多项式阶数
        """
        self.x_nodes = np.asarray(x_nodes, dtype=float)
        self.C_nodes = np.asarray(C_nodes, dtype=float)
        self.max_order = int(max_order)
        if len(self.x_nodes) != len(self.C_nodes):
            raise ValueError("节点坐标与浓度数组长度不一致")
        if self.max_order < 1:
            raise ValueError("max_order 必须 ≥ 1")

    def _select_local_window(self, x_query: float) -> tuple[np.ndarray, np.ndarray]:
        """
        为查询点选择局部插值窗口：
          - 找到最近的 max_order+1 个节点
          - 确保窗口覆盖查询点
        """
        if x_query <= self.x_nodes[0]:
            idx = slice(0, min(self.max_order + 1, len(self.x_nodes)))
            return self.x_nodes[idx], self.C_nodes[idx]
        if x_query >= self.x_nodes[-1]:
            start = max(0, len(self.x_nodes) - self.max_order - 1)
            idx = slice(start, len(self.x_nodes))
            return self.x_nodes[idx], self.C_nodes[idx]

        # 找到最近的节点索引
        distances = np.abs(self.x_nodes - x_query)
        center_idx = int(np.argmin(distances))

        # 选择窗口：以 center_idx 为中心，向两侧扩展
        half = self.max_order // 2
        left = max(0, center_idx - half)
        right = min(len(self.x_nodes), left + self.max_order + 1)
        left = max(0, right - self.max_order - 1)
        idx = slice(left, right)
        return self.x_nodes[idx], self.C_nodes[idx]

    def interpolate(self, x_query: float) -> float:
        """在 x_query 处插值浓度。"""
        x_win, c_win = self._select_local_window(x_query)
        coeffs = divided_differences(x_win, c_win)
        return newton_evaluate(x_win, coeffs, x_query)

    def interpolate_batch(self, x_queries: np.ndarray) -> np.ndarray:
        """批量插值。"""
        return np.array([self.interpolate(xq) for xq in x_queries])

    def reconstruct_breakthrough_curve(self, x_obs: float,
                                       C_history: np.ndarray,
                                       t_values: np.ndarray) -> np.ndarray:
        """
        在固定空间位置 x_obs 上，重构随时间变化的突破曲线 C(x_obs, t)。

        对每一时间层，先在空间上插值到 x_obs，得到离散序列 C_obs(t_n)，
        再返回该序列（后续可进一步在时间方向插值以获得连续 BTC）。
        """
        if C_history.shape[1] != len(self.x_nodes):
            raise ValueError("C_history 的空间维度必须与节点数一致")
        if len(t_values) != C_history.shape[0]:
            raise ValueError("时间数组长度必须与 C_history 的时间维度一致")

        btc = np.zeros(len(t_values))
        for n in range(len(t_values)):
            # 更新内部浓度场
            self.C_nodes = C_history[n, :]
            btc[n] = self.interpolate(x_obs)
        return btc


def cubic_spline_natural(x: np.ndarray, y: np.ndarray,
                         x_query: np.ndarray) -> np.ndarray:
    """
    自然三次样条插值（作为 Newton 插值的补充，用于平滑突破曲线）。

    在每个区间 [x_i, x_{i+1}] 上：
        S_i(x) = a_i + b_i (x-x_i) + c_i (x-x_i)² + d_i (x-x_i)³

    连续性条件：
        S_i(x_{i+1}) = S_{i+1}(x_{i+1})
        S_i'(x_{i+1}) = S_{i+1}'(x_{i+1})
        S_i''(x_{i+1}) = S_{i+1}''(x_{i+1})
    自然边界：S''(x_0) = S''(x_n) = 0
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    if n < 3:
        # 退化为线性插值
        return np.interp(x_query, x, y)

    h = np.diff(x)
    if np.any(h <= 0):
        raise ValueError("x 必须严格递增")

    # 三对角系统求解二阶导数 M_i
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

    # 计算样条系数
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

    # 外插
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

    # 样条测试
    xq = np.linspace(0, 10, 100)
    ys = cubic_spline_natural(x, y, xq)
    assert np.all(np.isfinite(ys))
    print("interpolation_engine: 自测试通过")
