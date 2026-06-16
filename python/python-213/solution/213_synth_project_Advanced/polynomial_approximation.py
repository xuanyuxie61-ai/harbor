"""
polynomial_approximation.py - 多项式逼近与 Vandermonde 系统模块
===============================================================

融合种子项目:
  - 1382_vandermonde_approx_1d: Vandermonde 矩阵与多项式拟合

核心数学: 中心路径的多项式逼近

Interior Point Method 沿中心路径追踪最优解.
中心路径参数化:
  C = { z(mu) : F(z, mu) = 0, mu > 0 }

其中 F 是 KKT 映射的障碍扰动版本.

当 mu 连续变化时, z(mu) 形成光滑曲线.
多项式逼近用于:
  1. 预测器-校正方法中的预测步
  2. 障碍参数 mu 的最优选择
  3. 外推加速收敛

Vandermonde 逼近:
  给定数据点 {(t_i, y_i)}, 求次数 <= m 的多项式 p:
    p(t) = c_0 + c_1 t + c_2 t^2 + ... + c_m t^m

  使得 sum_i (p(t_i) - y_i)^2 最小.

  等价于最小二乘问题:
    min ||V c - y||_2^2
  其中 V_{ij} = t_i^j 是 Vandermonde 矩阵.

  条件数: kappa(V) ~ O(h^{-(m+1)}) 对等距节点
  这导致高阶 Vandermonde 系统高度病态.

  改善条件数的方法:
    1. 缩放: t_hat = (t - mean) / std
    2. Chebyshev 节点 (代替等距节点)
    3. QR 分解 (代替正规方程)

  本模块使用缩放 + QR 分解.
"""

import numpy as np
from typing import Tuple, Optional


class VandermondeApproximation:
    """
    Vandermonde 矩阵多项式逼近.

    用于中心路径跟踪中的预测步:
      已知 mu_1, mu_2, ..., mu_k 处的解 z(mu_i),
      用多项式外推预测 z(mu_{k+1}).

    Parameters
    ----------
    degree : int
        多项式次数
    """

    def __init__(self, degree: int):
        self.degree = degree
        self._coeffs: Optional[np.ndarray] = None
        self._scale_mean: float = 0.0
        self._scale_std: float = 1.0

    def build_vandermonde_matrix(self, x: np.ndarray, m: int) -> np.ndarray:
        """
        构建 Vandermonde 矩阵 (融合 1382_vandermonde_approx_1d).

        V = [ 1  x_1  x_1^2  ...  x_1^m ]
            [ 1  x_2  x_2^2  ...  x_2^m ]
            [ ...                         ]
            [ 1  x_n  x_n^2  ...  x_n^m ]

        V_{ij} = x_i^j,  j = 0, ..., m

        条件数问题:
          当 m 较大时, Vandermonde 矩阵高度病态.
          kappa(V) 随 m 指数增长.

          解决方案: 列缩放 (归一化 x)
          x_hat = (x - mean(x)) / std(x)

        Parameters
        ----------
        x : ndarray, shape (n,)
            数据点
        m : int
            多项式次数

        Returns
        -------
        V : ndarray, shape (n, m+1)
            Vandermonde 矩阵
        """
        n = len(x)
        V = np.zeros((n, m + 1))
        V[:, 0] = 1.0
        for j in range(1, m + 1):
            V[:, j] = V[:, j - 1] * x
        return V

    def fit(self, t_data: np.ndarray, y_data: np.ndarray
             ) -> np.ndarray:
        """
        最小二乘拟合 Vandermonde 多项式.

        求解:
          min_c ||V c - y||_2

        使用 QR 分解避免正规方程的条件数平方问题:
          V = Q R  =>  R c = Q^T y

        与正规方程 V^T V c = V^T y 相比,
        QR 分解的条件数为 kappa(V) 而非 kappa(V)^2.

        Parameters
        ----------
        t_data : ndarray, shape (n,)
            参数值
        y_data : ndarray, shape (n,) or (n, k)
            函数值

        Returns
        -------
        coeffs : ndarray, shape (degree+1,) or (degree+1, k)
            多项式系数
        """
        m = self.degree
        n = len(t_data)
        assert n >= m + 1, f"数据点 ({n}) 不足以拟合 {m} 次多项式"

        # 列缩放改善条件数
        self._scale_mean = np.mean(t_data)
        self._scale_std = np.std(t_data)
        if self._scale_std < 1.0e-15:
            self._scale_std = 1.0

        t_scaled = (t_data - self._scale_mean) / self._scale_std

        # 构建 Vandermonde 矩阵
        V = self.build_vandermonde_matrix(t_scaled, m)

        # QR 分解求解
        if y_data.ndim == 1:
            Q, R = np.linalg.qr(V)
            self._coeffs = np.linalg.solve(R, Q.T @ y_data)
        else:
            Q, R = np.linalg.qr(V)
            Qt_y = Q.T @ y_data
            self._coeffs = np.zeros((m + 1, y_data.shape[1]))
            for k in range(y_data.shape[1]):
                self._coeffs[:, k] = np.linalg.solve(R, Qt_y[:, k])

        return self._coeffs

    def evaluate(self, t: np.ndarray) -> np.ndarray:
        """
        在指定点求值多项式.

        p(t) = c_0 + c_1 t_hat + c_2 t_hat^2 + ... + c_m t_hat^m
        其中 t_hat = (t - mean) / std

        Parameters
        ----------
        t : ndarray
            求值点

        Returns
        -------
        ndarray
            多项式值
        """
        if self._coeffs is None:
            raise ValueError("必须先调用 fit()")

        t = np.atleast_1d(t)
        t_scaled = (t - self._scale_mean) / self._scale_std

        V = self.build_vandermonde_matrix(t_scaled, self.degree)
        return V @ self._coeffs

    def predict_center_path(self, mu_values: np.ndarray,
                             solution_values: np.ndarray,
                             mu_target: float) -> np.ndarray:
        """
        使用 Vandermonde 外推预测中心路径上的解.

        给定 mu_1 > mu_2 > ... > mu_k 处的解 z(mu_i),
        外推预测 z(mu_target).

        这在 predictor-corrector 内点法中用作预测步:
          z_pred = p(mu_target)
          然后在 z_pred 附近进行 Newton 校正.

        Parameters
        ----------
        mu_values : ndarray
            已知的 mu 值
        solution_values : ndarray
            对应的解
        mu_target : float
            目标 mu 值

        Returns
        -------
        ndarray
            预测的解
        """
        self.fit(mu_values, solution_values)
        return self.evaluate(np.array([mu_target]))


class PolynomialContinuation:
    """
    障碍参数的多项式连续策略.

    内点法的核心策略: 如何减小 mu.

    简单策略: mu_{k+1} = sigma * mu_k (几何衰减)
    高级策略: 基于中心路径曲率自适应选择 sigma.

    中心路径的曲率:
      kappa(mu) = ||z''(mu)|| / (1 + ||z'(mu)||^2)^{3/2}

    曲率大 => 路径弯曲 => 需要小的 mu 减小步
    曲率小 => 路径平坦 => 可以大的 mu 减小步

    公式:
      sigma_k = 1 - delta / (1 + kappa(mu_k) * mu_k)
      mu_{k+1} = sigma_k * mu_k
"""

    def __init__(self, initial_mu: float, target_mu: float = 1.0e-10,
                 reduction_factor: float = 0.2):
        self.mu = initial_mu
        self.mu_target = target_mu
        self.reduction_factor = reduction_factor
        self.history = [(0, initial_mu)]
        self.iteration = 0

    def update(self, curvature: float = 0.0) -> float:
        """
        更新障碍参数.

        自适应策略:
          sigma = 1 - delta / (1 + curvature * mu)

        Parameters
        ----------
        curvature : float
            中心路径曲率估计

        Returns
        -------
        float
            新的 mu 值
        """
        self.iteration += 1

        # 自适应 sigma
        sigma = 1.0 - self.reduction_factor / (1.0 + curvature * self.mu)
        sigma = np.clip(sigma, 0.01, 0.99)

        self.mu *= sigma
        self.mu = max(self.mu, self.mu_target)
        self.history.append((self.iteration, self.mu))

        return self.mu

    def converged(self) -> bool:
        """检查是否收敛."""
        return self.mu <= self.mu_target

    def get_mu(self) -> float:
        """获取当前 mu."""
        return self.mu
