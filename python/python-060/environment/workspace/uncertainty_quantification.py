# -*- coding: utf-8 -*-
r"""
uncertainty_quantification.py
不确定性量化模块：基于稀疏网格随机配点与 Cholesky 采样的
平流层臭氧浓度预测不确定性分析。

核心公式：

  设模型输出 Y = \mathcal{M}(\boldsymbol{\xi})，其中 \boldsymbol{\xi} 为
  服从标准正态分布的随机参数向量，协方差矩阵为 \Sigma。

  通过 Cholesky 分解 \Sigma = L L^T，将相关正态变量转化为独立标准正态：

      \boldsymbol{\xi} = L \mathbf{z}, \quad \mathbf{z} \sim \mathcal{N}(0, I)

  统计矩通过稀疏网格数值积分估计：

      \mathbb{E}[Y] \approx \sum_{i=1}^{N_q} w_i \, \mathcal{M}(L \mathbf{z}_i)

      \mathrm{Var}(Y) \approx \sum_{i=1}^{N_q} w_i \,
          \left[\mathcal{M}(L \mathbf{z}_i) - \mathbb{E}[Y]\right]^2

融合来源：
  - 026_asa007: Cholesky 分解
  - 1103_sparse_grid_cc: 稀疏网格求积
  - 1006_random_data: 随机采样
"""

import numpy as np
from sparse_quadrature import sparse_grid_cc
from linear_solvers import cholesky_solve_dense
from utils import clip_positive


def build_covariance_matrix(n_param, correlation_length=0.5):
    r"""
    构造参数不确定性协方差矩阵（高斯型相关结构）：

        \Sigma_{ij} = \sigma^2 \exp\!
            \left(-\frac{(i-j)^2}{2\ell^2}\right)

    Parameters
    ----------
    n_param : int
        参数个数。
    correlation_length : float
        相关长度 ℓ。

    Returns
    -------
    Sigma : ndarray, shape (n_param, n_param)
    """
    i = np.arange(n_param)
    j = np.arange(n_param)
    I, J = np.meshgrid(i, j)
    Sigma = np.exp(-0.5 * ((I - J) / correlation_length) ** 2)
    return Sigma


def transform_to_physical(z, mu, L):
    r"""
    将标准正态样本变换为相关正态分布：

        \boldsymbol{\xi} = \boldsymbol{\mu} + L \mathbf{z}

    Parameters
    ----------
    z : ndarray
        标准正态样本。
    mu : ndarray
        均值向量。
    L : ndarray
        Cholesky 因子（下三角）。

    Returns
    -------
    xi : ndarray
    """
    z = np.asarray(z, dtype=float)
    mu = np.asarray(mu, dtype=float)
    return mu + L.dot(z)


class UQAnalyzer:
    r"""
    不确定性量化分析器。

    流程：
      1. 定义模型参数的先验分布（均值 μ，协方差 Σ）；
      2. Cholesky 分解 Σ = L L^T；
      3. 在标准正态空间构造稀疏网格；
      4. 对每个配点运行模型；
      5. 数值积分估计均值、方差及灵敏度指标。
    """

    def __init__(self, n_param=4, level_max=2):
        r"""
        Parameters
        ----------
        n_param : int
            随机参数维数。
        level_max : int
            稀疏网格层级。
        """
        self.n_param = n_param
        self.level_max = level_max
        self.Sigma = build_covariance_matrix(n_param)
        # Cholesky 分解
        self.L = np.linalg.cholesky(self.Sigma)
        self.mu = np.zeros(n_param)
        # 预生成稀疏网格（映射到 [0,1] 后通过逆 CDF 转标准正态）
        self.grid_points, self.grid_weights = sparse_grid_cc(n_param, level_max)
        # 映射到标准正态空间（逆误差函数变换）
        from scipy.special import erfinv
        self.grid_points_z = np.sqrt(2.0) * erfinv(2.0 * self.grid_points - 1.0)
        # 过滤权重过小或 NaN 的节点
        valid = np.isfinite(self.grid_points_z).all(axis=1) & (np.abs(self.grid_weights) > 1e-16)
        self.grid_points_z = self.grid_points_z[valid]
        self.grid_weights = self.grid_weights[valid]

    def sample_parameters(self):
        r"""
        返回所有稀疏网格配点对应的物理参数样本。

        Returns
        -------
        samples : list of ndarray
        """
        samples = []
        for z in self.grid_points_z:
            xi = transform_to_physical(z, self.mu, self.L)
            samples.append(xi)
        return samples

    def estimate_moments(self, model_outputs):
        r"""
        基于模型输出估计统计矩。

        Parameters
        ----------
        model_outputs : ndarray, shape (n_q, ...)
            每个配点对应的模型输出。

        Returns
        -------
        mean : ndarray
        variance : ndarray
        std : ndarray
        """
        w = self.grid_weights
        y = np.asarray(model_outputs)
        # 权重归一化
        w_sum = np.sum(w)
        if abs(w_sum) < 1e-15:
            w_sum = 1.0
        mean = np.tensordot(w, y, axes=([0], [0])) / w_sum
        diff = y - mean
        var = np.tensordot(w, diff ** 2, axes=([0], [0])) / w_sum
        var = np.maximum(var, 0.0)
        return mean, var, np.sqrt(var)

    def sobol_first_order(self, model_outputs, param_idx):
        r"""
        一阶 Sobol 灵敏度指标近似（基于条件方差）。

            S_i = \frac{\mathrm{Var}_{\xi_i}(\mathbb{E}[Y|\xi_i])}{\mathrm{Var}(Y)}

        这里使用稀疏网格配点的简化估计。

        Parameters
        ----------
        model_outputs : ndarray, shape (n_q,)
        param_idx : int

        Returns
        -------
        S_i : float
        """
        y = np.asarray(model_outputs, dtype=float)
        w = self.grid_weights
        _, var_y, _ = self.estimate_moments(y)
        if var_y < 1e-20:
            return 0.0
        # 按参数值分组近似条件期望
        z_vals = self.grid_points_z[:, param_idx]
        # 使用核平滑近似
        h = 0.5
        n_q = len(y)
        cond_var = 0.0
        w_sum = np.sum(w)
        for j in range(n_q):
            kernel = np.exp(-0.5 * ((z_vals - z_vals[j]) / h) ** 2)
            kernel_w = kernel * w
            kw_sum = np.sum(kernel_w)
            if kw_sum > 1e-15:
                e_y = np.sum(kernel_w * y) / kw_sum
                cond_var += w[j] * (e_y ** 2)
        cond_var = cond_var / w_sum
        S_i = np.clip(cond_var / var_y, 0.0, 1.0)
        return float(S_i)


def monte_carlo_uncertainty(model_func, n_samples=500, n_param=4):
    r"""
    蒙特卡洛不确定性传播。

    Parameters
    ----------
    model_func : callable
        model_func(xi) -> output
    n_samples : int
    n_param : int

    Returns
    -------
    mean, std : float
    """
    Sigma = build_covariance_matrix(n_param)
    L = np.linalg.cholesky(Sigma)
    outputs = []
    for _ in range(n_samples):
        z = np.random.randn(n_param)
        xi = L.dot(z)
        out = model_func(xi)
        outputs.append(out)
    outputs = np.array(outputs)
    return float(np.mean(outputs)), float(np.std(outputs))
