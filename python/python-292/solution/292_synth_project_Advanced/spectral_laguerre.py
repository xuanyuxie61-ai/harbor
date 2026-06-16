"""
spectral_laguerre.py — Laguerre 谱展开速度空间求解器
=======================================================

融合自:
  - 641_laguerre_polynomial: Laguerre 多项式、 associated Laguerre、
    Gauss-Laguerre 求积规则

物理背景:
  动力学等离子体中电子速度分布函数 f(v) 用 Laguerre 多项式展开:
    f(v) = sum_{n=0}^{N} c_n * L_n(v^2 / v_th^2) * exp(-v^2 / (2*v_th^2))

  其中 L_n 为 Laguerre 多项式，满足三项递推:
    (n+1)*L_{n+1}(x) = (2n+1-x)*L_n(x) - n*L_{n-1}(x)

  矩计算:
    密度:  n_e = integral f(v) dv = c_0 * sqrt(pi/2) * v_th
    热能:  (3/2)*n_e*T_e = integral (m*v^2/2)*f(v) dv
    热通量: q_e = integral (m*v^2/2)*v*f(v) dv

  Gauss-Laguerre 求积:
    integral_0^inf w(x)*f(x) dx = sum_i w_i * f(x_i)
  其中节点 x_i 为 Laguerre 多项式的零点，权重 w_i 由
  Jacobi 矩阵特征值问题确定。
"""

import numpy as np


class LaguerreSpectralSolver:
    """Laguerre 谱展开速度空间求解器。"""

    def __init__(self, n_modes=12, v_max=6.0):
        self.n_modes = n_modes
        self.v_max = v_max
        self.v_th = 1.0  # 热速度

        self.quad_nodes = None   # 求积节点
        self.quad_weights = None # 求积权重
        self.n_quad = 2 * n_modes + 4
        self.coeffs = None       # 展开系数
        self.f_reconstructed = None
        self.truncation_error = None

    def build_quadrature(self):
        """
        构建 Gauss-Laguerre 求积规则。

        Jacobi 矩阵 (对称三对角):
          J_ii = 2*i + 1 + alpha  (对角元)
          J_{i,i+1} = sqrt((i+1)*(i+1+alpha))  (次对角元)

        节点 = J 的特征值
        权重 = 2 * (特征向量的第一分量)^2 * Gamma(alpha+1)
        """
        alpha = 0.0  # 标准 Laguerre
        n = self.n_quad
        diag = np.zeros(n)
        offdiag = np.zeros(n - 1)
        for i in range(n):
            diag[i] = 2 * i + 1 + alpha
        for i in range(n - 1):
            offdiag[i] = np.sqrt((i + 1) * (i + 1 + alpha))

        # 对角化 Jacobi 矩阵 (QL 算法简化版)
        J = np.diag(diag) + np.diag(offdiag, 1) + np.diag(offdiag, -1)
        eigvals, eigvecs = np.linalg.eigh(J)

        self.quad_nodes = eigvals
        # 权重: w_i = Gamma(alpha+1) * (v_0^{(i)})^2
        from math import gamma as gamma_func
        self.quad_weights = gamma_func(alpha + 1) * eigvecs[0, :] ** 2

        # 转换到速度空间: v = sqrt(2*x)*v_th
        self.quad_nodes_v = np.sqrt(2 * np.maximum(self.quad_nodes, 0)) * self.v_th

    def _laguerre_values(self, x, n_max=None):
        """
        计算 Laguerre 多项式 L_n(x) 的值 (三项递推)。

        L_0(x) = 1
        L_1(x) = 1 - x
        (n+1)*L_{n+1}(x) = (2n+1-x)*L_n(x) - n*L_{n-1}(x)
        """
        if n_max is None:
            n_max = self.n_modes
        L = np.zeros((len(x), n_max + 1))
        L[:, 0] = 1.0
        if n_max >= 1:
            L[:, 1] = 1.0 - x
        for n in range(1, n_max):
            L[:, n + 1] = ((2 * n + 1 - x) * L[:, n] - n * L[:, n - 1]) / (n + 1)
        return L

    def compute_expansion_coefficients(self, f_data):
        """
        计算 Laguerre 展开系数。

        c_n = integral f(v) * L_n(v^2/v_th^2) * w(v) dv / h_n
        其中 h_n = integral L_n^2 * w dv = 1 (归一化)

        使用 Gauss-Laguerre 求积近似。
        """
        # 将 f_data 插值到求积节点
        if len(f_data) != self.n_quad:
            v_data = np.linspace(-self.v_max, self.v_max, len(f_data))
            f_interp = np.interp(self.quad_nodes_v[:len(f_data)],
                                 np.abs(v_data), f_data)
            if len(f_interp) < self.n_quad:
                f_interp = np.pad(f_interp, (0, self.n_quad - len(f_interp)),
                                  mode='constant')
        else:
            f_interp = f_data

        x_quad = self.quad_nodes[:len(f_interp)]
        w_quad = self.quad_weights[:len(f_interp)]
        L_vals = self._laguerre_values(x_quad)

        self.coeffs = np.zeros(self.n_modes + 1)
        for n in range(self.n_modes + 1):
            self.coeffs[n] = np.sum(w_quad * f_interp[:len(x_quad)] * L_vals[:len(x_quad), n])

    def reconstruct_distribution(self):
        """重构分布函数。"""
        x = np.linspace(0, self.v_max ** 2 / self.v_th ** 2, 100)
        L_vals = self._laguerre_values(x)
        self.f_reconstructed = np.zeros(len(x))
        for n in range(self.n_modes + 1):
            self.f_reconstructed += self.coeffs[n] * L_vals[:, n]
        weight = np.exp(-x / 2)
        self.f_reconstructed *= weight

        # 截断误差: 最后一个系数的相对大小
        if np.max(np.abs(self.coeffs)) > 1e-30:
            self.truncation_error = abs(self.coeffs[-1]) / np.max(np.abs(self.coeffs))
        else:
            self.truncation_error = 0.0

    def compute_energy_moment(self):
        """
        计算电子热能矩 <v^2/2>。

        <v^2/2> = integral (v^2/2) * f(v) dv / integral f(v) dv
        使用 Laguerre 展开的解析矩公式。
        """
        if self.coeffs is None:
            return 0.0
        # <v^2/2> = (3/2) * v_th^2 * c_0 / c_0 = (3/2) * v_th^2 (Maxwellian)
        # 微扰修正: + v_th^2 * c_1 / c_0
        v_th2 = self.v_th ** 2
        energy = 1.5 * v_th2
        if abs(self.coeffs[0]) > 1e-30:
            energy += v_th2 * self.coeffs[1] / self.coeffs[0]
        return float(energy)

    def compute_heat_flux(self):
        """
        计算电子热通量 q_e。

        q_e = integral (m*v^2/2)*v*f(v) dv
        对于 Maxwellian 背景，q_e 正比于 c_1 的梯度。
        """
        if self.coeffs is None:
            return 0.0
        # 热通量正比于 c_1 (各向异性一阶矩)
        q_e = self.v_th ** 3 * self.coeffs[1] * 0.5
        return float(q_e)
