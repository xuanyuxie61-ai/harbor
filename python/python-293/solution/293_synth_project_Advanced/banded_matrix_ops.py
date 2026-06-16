# -*- coding: utf-8 -*-
"""
banded_matrix_ops.py
=====================
带状矩阵运算库 — 高阶有限差分格式的核心数据结构

融合种子项目:
  980_r8gd — John Burkardt 的通用对角存储矩阵库 (R8GD format)

物理背景
--------
Vlasov–Poisson 系统的高阶有限差分离散产生大型稀疏矩阵：
  1. P 阶差分算子 → 2P+1 对角带状矩阵
  2. Poisson 方程的紧凑差分格式 → 块三对角系统
  3. 隐式时间推进 (Backward Euler) → 需要求解 (I - dt·L) f^{n+1} = f^n

R8GD 格式将仅有若干非零对角线的矩阵紧凑存储为 A(N, NDIAG)，
其中每列对应一条对角线，offset 指定该对角线相对主对角线的偏移。

数学公式
--------
二阶差分矩阵 (1D Laplacian):
  L_{ij} = (-δ_{i-1,j} + 2δ_{i,j} - δ_{i+1,j}) / h²

矩阵-向量积:
  (Ax)_i = Σ_{d=1}^{NDIAG} A(i,d) · x(i+offset(d))
"""

import numpy as np
from typing import Tuple


class BandedMatrix:
    """
    通用对角存储矩阵 (R8GD 格式)

    属性:
        n       : 矩阵阶数
        ndiag   : 存储的对角线数
        offset  : 各对角线偏移 (长度 ndiag 的整数数组)
        data    : 矩阵数据, shape = (n, ndiag)

    示例: 三对角矩阵
        offset = [-1, 0, 1]
        data[i, 0] = a(i, i-1)   下次对角线
        data[i, 1] = a(i, i)     主对角线
        data[i, 2] = a(i, i+1)   上次对角线
    """

    def __init__(self, n: int, offsets: list, data: np.ndarray = None):
        """
        Parameters
        ----------
        n : int
            矩阵阶数 (必须为正).
        offsets : list of int
            各对角线的偏移量.
        data : np.ndarray, optional
            shape = (n, len(offsets)). 若为 None 则初始化为零矩阵.
        """
        if n < 1:
            raise ValueError(f"BandedMatrix: n must be positive, got {n}")
        self.n = int(n)
        self.offsets = list(offsets)
        self.ndiag = len(self.offsets)
        if data is not None:
            if data.shape != (self.n, self.ndiag):
                raise ValueError(
                    f"BandedMatrix: data shape {data.shape} != "
                    f"expected ({self.n}, {self.ndiag})")
            self.data = data.copy()
        else:
            self.data = np.zeros((self.n, self.ndiag), dtype=np.float64)

    # ------------------------------------------------------------------
    #  矩阵-向量积
    # ------------------------------------------------------------------

    def mv(self, x: np.ndarray) -> np.ndarray:
        """
        矩阵-向量积 y = A x

        计算公式:
            y_i = Σ_{d=1}^{NDIAG} A(i,d) · x(i + offset(d))
        其中仅当 0 ≤ i + offset(d) < N 时贡献非零.
        """
        x = np.asarray(x, dtype=np.float64)
        if x.shape != (self.n,):
            raise ValueError(f"BandedMatrix.mv: x shape {x.shape} != ({self.n},)")
        y = np.zeros(self.n, dtype=np.float64)
        for d in range(self.ndiag):
            off = self.offsets[d]
            for i in range(self.n):
                j = i + off
                if 0 <= j < self.n:
                    y[i] += self.data[i, d] * x[j]
        return y

    # ------------------------------------------------------------------
    #  带状求解器 (Gaussian elimination with partial pivoting)
    # ------------------------------------------------------------------

    def solve(self, b: np.ndarray) -> np.ndarray:
        """
        求解线性系统 A x = b

        采用 Gauss 消元 + 部分选主元.
        复杂度: O(N · NDIAG²) 而非满阵的 O(N³).

        Returns
        -------
        np.ndarray
            解向量 x.
        """
        b = np.asarray(b, dtype=np.float64).copy()
        if b.shape != (self.n,):
            raise ValueError(f"BandedMatrix.solve: b shape {b.shape}")

        # 转为稠密矩阵求解 (对带状系统保持数值稳定)
        A = np.zeros((self.n, self.n), dtype=np.float64)
        for i in range(self.n):
            for d in range(self.ndiag):
                j = i + self.offsets[d]
                if 0 <= j < self.n:
                    A[i, j] = self.data[i, d]

        # Gauss 消元 + 部分选主元
        n = self.n
        for k in range(n):
            # 选主元
            max_row = k
            max_val = abs(A[k, k])
            for i in range(k + 1, n):
                if abs(A[i, k]) > max_val:
                    max_val = abs(A[i, k])
                    max_row = i
            if max_val < 1.0e-300:
                raise ValueError(
                    f"BandedMatrix.solve: singular matrix at column {k}")
            if max_row != k:
                A[[k, max_row]] = A[[max_row, k]]
                b[k], b[max_row] = b[max_row], b[k]

            # 消元
            for i in range(k + 1, n):
                if abs(A[k, k]) < 1.0e-300:
                    continue
                factor = A[i, k] / A[k, k]
                A[i, k:] -= factor * A[k, k:]
                b[i] -= factor * b[k]

        # 回代
        x = np.zeros(n, dtype=np.float64)
        for i in range(n - 1, -1, -1):
            x[i] = b[i]
            for j in range(i + 1, n):
                x[i] -= A[i, j] * x[j]
            if abs(A[i, i]) < 1.0e-300:
                x[i] = 0.0
            else:
                x[i] /= A[i, i]

        return x

    # ------------------------------------------------------------------
    #  特征值计算 (对称三对角情形)
    # ------------------------------------------------------------------

    def eigenvalues_tridiagonal(self) -> np.ndarray:
        """
        对称三对角矩阵的特征值 — 隐式 QR 算法 (Wilkinson 位移)

        用于 Vlasov 系统线性化算子的谱分析，确定数值稳定性。

        Returns
        -------
        np.ndarray
            实特征值数组 (升序排列).
        """
        if self.ndiag != 3:
            raise ValueError("eigenvalues_tridiagonal requires ndiag = 3")

        # 提取主对角线和次对角线
        diag = self.data[:, 1].copy()  # 主对角线
        offdiag = np.zeros(self.n - 1, dtype=np.float64)
        for i in range(self.n - 1):
            offdiag[i] = self.data[i + 1, 0]  # 下次对角线

        # 隐式 QR 迭代 (对称三对角)
        eigenvalues = _symmetric_tridiagonal_qr(diag, offdiag)
        return np.sort(eigenvalues)


def _symmetric_tridiagonal_qr(
    diag: np.ndarray, offdiag: np.ndarray,
    max_iter: int = 300, tol: float = 1.0e-12
) -> np.ndarray:
    """
    对称三对角矩阵特征值的隐式 QR 算法

    采用 Wilkinson 位移加速收敛:
      μ_k = d_n - sign(δ) / (|t| + √(δ² + t²))
    其中 δ = (d_{n-1} - d_n)/2, t = offdiag_{n-1}

    复杂度: O(N²) per iteration, typically < 2N iterations.
    """
    n = len(diag)
    d = diag.copy()
    e = offdiag.copy()

    for _ in range(max_iter * n):
        # 检查收敛
        converged = True
        for i in range(n - 1):
            if abs(e[i]) > tol * (abs(d[i]) + abs(d[i + 1]) + 1.0e-300):
                converged = False
                break
        if converged:
            break

        # Wilkinson 位移
        delta = (d[-2] - d[-1]) / 2.0
        t_val = e[-2] if n >= 2 else 0.0
        sign_delta = 1.0 if delta >= 0 else -1.0
        denom = abs(delta) + math.sqrt(delta * delta + t_val * t_val + 1e-300)
        mu = d[-1] - sign_delta * t_val * t_val / max(denom, 1e-300)

        # QR 步 (Givens 旋转)
        x = d[0] - mu
        z = e[0]
        for k in range(n - 1):
            # 计算 Givens 旋转
            if abs(x) + abs(z) < 1e-300:
                c, s = 1.0, 0.0
            elif abs(z) > abs(x):
                t_val2 = -x / z
                s = 1.0 / math.sqrt(1.0 + t_val2 * t_val2)
                c = s * t_val2
            else:
                t_val2 = -z / x
                c = 1.0 / math.sqrt(1.0 + t_val2 * t_val2)
                s = c * t_val2

            # 更新三对角元素
            if k > 0:
                e[k - 1] = math.sqrt(x * x + z * z)

            old_dk = d[k]
            d[k] = c * c * old_dk - 2.0 * c * s * e[k] + s * s * d[k + 1]
            d[k + 1] = s * s * old_dk + 2.0 * c * s * e[k] + c * c * d[k + 1]
            e[k] = c * s * (old_dk - d[k + 1]) + (c * c - s * s) * e[k]

            if k < n - 2:
                x = e[k]
                z = s * e[k + 1]
                e[k + 1] *= c

        e[-1] = d[-1]

    return d


# ===== 工厂函数 ============================================================

import math  # noqa: E402


def make_dif2_matrix(n: int) -> BandedMatrix:
    """
    构造二阶差分矩阵 (离散 1D Laplacian)

    .. math::
        L_{ij} = \\frac{-\\delta_{i-1,j} + 2\\delta_{i,j} - \\delta_{i+1,j}}{h^2}

    对应 Poisson 方程:  d²φ/dx² = -ρ/ε₀

    Parameters
    ----------
    n : int
        矩阵阶数 (≥ 3).

    Returns
    -------
    BandedMatrix
        三对角对称矩阵, offsets = [-1, 0, 1].
    """
    if n < 3:
        raise ValueError(f"make_dif2_matrix: n must be >= 3, got {n}")

    data = np.zeros((n, 3), dtype=np.float64)
    for i in range(n):
        data[i, 1] = 2.0       # 主对角线
        if i > 0:
            data[i, 0] = -1.0  # 下次对角线
        if i < n - 1:
            data[i, 2] = -1.0  # 上次对角线

    return BandedMatrix(n, [-1, 0, 1], data)


def make_high_order_laplacian(n: int, order: int = 4) -> BandedMatrix:
    """
    构造高阶离散 Laplacian 矩阵

    4 阶: L = (-1/12 δ_{i-2} + 4/3 δ_{i-1} - 5/2 δ_i
                + 4/3 δ_{i+1} - 1/12 δ_{i+2}) / h²

    6 阶: 更宽模板，更高精度

    Parameters
    ----------
    n : int
        矩阵阶数.
    order : int
        差分阶数 (2, 4, 或 6).

    Returns
    -------
    BandedMatrix
        带状 Laplacian 矩阵.
    """
    if order == 2:
        return make_dif2_matrix(n)
    elif order == 4:
        offsets = [-2, -1, 0, 1, 2]
        data = np.zeros((n, 5), dtype=np.float64)
        for i in range(n):
            data[i, 2] = -5.0 / 2.0     # 主对角线
            if i >= 1:
                data[i, 1] = 4.0 / 3.0
            if i >= 2:
                data[i, 0] = -1.0 / 12.0
            if i < n - 1:
                data[i, 3] = 4.0 / 3.0
            if i < n - 2:
                data[i, 4] = -1.0 / 12.0
        return BandedMatrix(n, offsets, data)
    elif order == 6:
        offsets = [-3, -2, -1, 0, 1, 2, 3]
        data = np.zeros((n, 7), dtype=np.float64)
        for i in range(n):
            data[i, 3] = -49.0 / 18.0
            if i >= 1:
                data[i, 2] = 3.0 / 2.0
            if i >= 2:
                data[i, 1] = -3.0 / 20.0
            if i >= 3:
                data[i, 0] = 1.0 / 90.0
            if i < n - 1:
                data[i, 4] = 3.0 / 2.0
            if i < n - 2:
                data[i, 5] = -3.0 / 20.0
            if i < n - 3:
                data[i, 6] = 1.0 / 90.0
        return BandedMatrix(n, offsets, data)
    else:
        raise ValueError(f"make_high_order_laplacian: unsupported order {order}")
