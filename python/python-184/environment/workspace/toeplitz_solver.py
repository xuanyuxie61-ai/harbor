"""
Toeplitz System Solver via Levinson-Durbin Recursion
=====================================================
源自种子项目 1000_r8to (Toeplitz Matrix Operations)。

在 time series 分析中，自协方差矩阵 R 满足 Toeplitz 结构：
    R[i,j] = r_{|i-j|},    r_k = E[x_t x_{t-k}]
Yule-Walker 方程 R a = -r 的求解是 AR 模型参数估计的核心。
Levinson-Durbin 递推将复杂度从 O(n^3) 降到 O(n^2)。

数学推导：
------------
设 n 阶对称正定 Toeplitz 矩阵 T_n，其第一列为 (t_0, t_1, ..., t_{n-1})^T。
定义前向预测误差滤波器 a^{(n)} 满足
    T_n a^{(n)} = -[t_1, t_2, ..., t_n]^T
Levinson 递推的核心是反射系数 (reflection coefficient) k_n：

    k_n = -(t_n + sum_{i=1}^{n-1} a_i^{(n-1)} t_{n-i}) / E_{n-1}

其中 E_{n-1} 为第 n-1 阶前向预测误差功率：
    E_{n-1} = t_0 + sum_{i=1}^{n-1} a_i^{(n-1)} t_i

更新公式：
    a_i^{(n)} = a_i^{(n-1)} + k_n * a_{n-i}^{(n-1)},   i=1,...,n-1
    a_n^{(n)} = k_n
    E_n = E_{n-1} * (1 - k_n^2)

边界条件：E_0 = t_0, 且 |k_n| < 1 保证正定性。
"""

import numpy as np


class ToeplitzSolver:
    """
    求解对称正定 Toeplitz 线性系统 T x = b，
    以及 Levinson-Durbin 递推用于 AR 系数估计。
    """

    def __init__(self, eps: float = 1e-14):
        self.eps = eps

    def _check_toeplitz(self, t: np.ndarray) -> None:
        if t.ndim != 1:
            raise ValueError("Toeplitz first column must be 1-D array.")
        if len(t) < 1:
            raise ValueError("Toeplitz first column must have length >= 1.")
        if t[0] <= self.eps:
            raise ValueError("Leading diagonal element t_0 must be positive.")

    def solve_yule_walker(self, autocorr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
         Levinson-Durbin 递推求解 Yule-Walker 方程。

        Parameters
        ----------
        autocorr : np.ndarray, shape (p+1,)
            自相关序列 [r_0, r_1, ..., r_p]，要求 r_0 > 0 且对称正定。

        Returns
        -------
        ar_coefs : np.ndarray, shape (p,)
            AR 系数 [a_1, a_2, ..., a_p]，满足
                x_t = -sum_{i=1}^p a_i x_{t-i} + e_t
               
        reflection_coefs : np.ndarray, shape (p,)
            反射系数 k_1, ..., k_p，用于格型滤波器与 Schur-Cohn 稳定性检验。
        """
        self._check_toeplitz(autocorr)
        p = len(autocorr) - 1
        if p == 0:
            return np.array([]), np.array([])

        a = np.zeros(p)
        k = np.zeros(p)
        E = autocorr[0]

        for n in range(1, p + 1):
            # TODO: Hole_1 - 实现 Levinson-Durbin 递推核心步骤
            # 要求：
            #   1. 计算反射系数 k_n（基于自相关和前阶AR系数）
            #   2. 边界处理：|k_n| 接近 1 时的数值稳定性处理
            #   3. 更新AR系数 a^{(n)}（利用前阶系数和反射系数）
            #   4. 更新预测误差功率 E_n
            # 提示：
            #   - 反射系数公式：k_n = -(t_n + Σ_{i=1}^{n-1} a_i^{(n-1)} t_{n-i}) / E_{n-1}
            #   - AR系数更新：a_i^{(n)} = a_i^{(n-1)} + k_n * a_{n-i}^{(n-1)}
            #   - 误差功率更新：E_n = E_{n-1} * (1 - k_n^2)
            raise NotImplementedError("Hole_1: 请实现 Levinson-Durbin 递推核心")

        return a, k

    def solve_toeplitz(self, t: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        一般对称 Toeplitz 系统 T x = b 的 Levinson 递推求解。

        Parameters
        ----------
        t : np.ndarray, shape (n,)
            Toeplitz 矩阵第一列 [t_0, t_1, ..., t_{n-1}]。
        b : np.ndarray, shape (n,)
            右端向量。

        Returns
        -------
        x : np.ndarray, shape (n,)
            解向量。
        """
        self._check_toeplitz(t)
        n = len(t)
        if b.shape != (n,):
            raise ValueError("b must have same length as t.")

        x = np.zeros(n)
        if n == 1:
            return np.array([b[0] / t[0]])

        # 初始化一阶解
        x[0] = b[0] / t[0]
        E = t[0]

        # 辅助向量 y 满足 T_{m} y^{(m)} = -[t_1, ..., t_m]^T
        y = np.zeros(n - 1)
        y[0] = -t[1] / t[0]

        for m in range(1, n):
            # 计算前向误差
            delta = np.dot(t[1:m + 1][::-1], x[:m]) if m > 0 else 0.0
            if m == 0:
                delta = 0.0
            alpha = (b[m] - delta) / E if abs(E) > self.eps else 0.0

            # 更新 x
            x_prev = x[:m].copy()
            x[:m] = x_prev + alpha * y[:m]
            x[m] = alpha

            if m == n - 1:
                break

            # 更新辅助向量 y
            gamma = np.dot(t[1:m + 1][::-1], y[:m]) if m > 0 else 0.0
            beta = -(t[m + 1] + gamma) / E if abs(E) > self.eps else 0.0
            y_prev = y[:m].copy()
            y[:m] = y_prev + beta * y_prev[::-1]
            y[m] = beta

            # 更新能量
            E = E * (1.0 - beta ** 2)
            if E < self.eps:
                E = self.eps

        return x

    def schur_cohn_stability(self, reflection_coefs: np.ndarray) -> bool:
        """
        Schur-Cohn 稳定性判据：若所有 |k_i| < 1，则 AR 多项式
            A(z) = 1 + a_1 z^{-1} + ... + a_p z^{-p}
        的所有根位于单位圆内，系统稳定（因果且平稳）。
        """
        if len(reflection_coefs) == 0:
            return True
        return bool(np.all(np.abs(reflection_coefs) < 1.0))

    def autocorr_to_toeplitz(self, autocorr: np.ndarray) -> np.ndarray:
        """将自相关序列展开为完整 Toeplitz 矩阵（用于验证）。"""
        p = len(autocorr)
        i = np.arange(p)
        j = np.arange(p)
        return autocorr[np.abs(i[:, None] - j)]
