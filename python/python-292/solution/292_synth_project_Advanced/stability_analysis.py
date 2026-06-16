"""
stability_analysis.py — von Neumann 稳定性分析
=================================================

融合自:
  - 605_jacobi_exactness: 多项式精确度验证 (→ 放大因子精确度验证)

物理背景:
  对线性化 MHD 方程进行 Fourier 分析，求解放大因子 G(k, dt)。
  稳定性条件: |G(k, dt)| <= 1 + O(dt)  对所有波数 k。

  纯对流方程 du/dt + c*du/dx = 0:
    中心差分: G = 1 - i*(c*dt/dx)*sum a_k*sin(k*dx*k)
    CFL 条件: |c|*dt/dx <= CFL_max

  纯扩散方程 du/dt = eta*d^2u/dx^2:
    显式: G = 1 - (eta*dt/dx^2)*sum b_k*(1-cos(k*dx*k))
    Fourier 数限制: Fo = eta*dt/dx^2 <= 0.5 (二阶)

  组合对流-扩散:
    |G|^2 = (1 - D)^2 + (C*sin)^2 <= 1
    其中 D = Fo*sum b_k*(1-cos), C = CFL*sum a_k*sin
"""

import numpy as np


class StabilityAnalyzer:
    """Von Neumann 稳定性分析器。"""

    def __init__(self, mhd, fd):
        self.mhd = mhd
        self.fd = fd
        self.cfl_advect_crit = None
        self.fo_diff_crit = None
        self.dt_critical = None
        self.amplification_factor = None

    def analyze_advection(self):
        """
        分析纯对流稳定性。

        对 2p 阶中心差分，对流 CFL 限约为:
          CFL_max ~ 1 / sum |a_k| * k
        更精确地通过扫描 amplification factor 得到。
        """
        n_k = 500
        kd_arr = np.linspace(0, np.pi, n_k)
        p = self.fd.half_width
        cfl_values = np.zeros(n_k)

        for idx, kd in enumerate(kd_arr):
            # 计算空间算子的 Fourier 变换
            imag_part = 0.0
            for m, m_val in enumerate(range(-p, p + 1)):
                imag_part += self.fd.central_coeffs[m] * np.sin(kd * m_val)

            # G = 1 - i * CFL * imag_part
            # |G|^2 = 1 + CFL^2 * imag_part^2
            # 中心差分对流总是弱不稳定，需要添加小量人工耗散
            # 实际 CFL 限由扩散项控制
            cfl_values[idx] = abs(imag_part)

        # 对于中心差分，CFL 限由扩散稳定化决定
        max_imag = np.max(cfl_values)
        self.cfl_advect_crit = 1.0 / (max_imag + 1e-10)

    def analyze_diffusion(self):
        """
        分析纯扩散稳定性。

        Fourier 数 Fo = eta*dt/dx^2
        G = 1 - Fo * sum b_k*(1-cos(k*dx*k))
        稳定: Fo <= 1 / max(sum b_k*(1-cos(k*dx*k)))
        """
        n_k = 500
        kd_arr = np.linspace(0, np.pi, n_k)
        p = self.fd.half_width
        diffusion_factor = np.zeros(n_k)

        for idx, kd in enumerate(kd_arr):
            real_part = 0.0
            for m, m_val in enumerate(range(-p, p + 1)):
                real_part += self.fd.central_2nd[m] * (1.0 - np.cos(kd * m_val))
            diffusion_factor[idx] = real_part

        max_diff = np.max(diffusion_factor)
        self.fo_diff_crit = 1.0 / (max_diff + 1e-10)

    def analyze_combined_scheme(self):
        """
        分析组合对流-扩散方案的放大因子。

        |G|^2 = (1 + Fo*D(k))^2 + (CFL*C(k))^2
        其中 D(k) = -sum b_k*(1-cos(k*dx*k)) <= 0
              C(k) = sum a_k*sin(k*dx*k)
        稳定性: |G|^2 <= 1
        """
        n_k = 200
        kd_arr = np.linspace(0, np.pi, n_k)
        p = self.fd.half_width
        self.amplification_factor = np.zeros(n_k)

        for idx, kd in enumerate(kd_arr):
            real_part = 0.0
            imag_part = 0.0
            for m, m_val in enumerate(range(-p, p + 1)):
                real_part += self.fd.central_2nd[m] * (1.0 - np.cos(kd * m_val))
                imag_part += self.fd.central_coeffs[m] * np.sin(kd * m_val)

            self.amplification_factor[idx] = np.sqrt(real_part ** 2 + imag_part ** 2)

    def find_critical_timestep(self):
        """
        确定组合方案的临界时间步。

        dt_crit = min(CFL_crit * dx / v_max, Fo_crit * dx^2 / eta)
        """
        v_max = np.max(np.abs(self.mhd.By)) + 1e-10
        dx = self.mhd.dx
        eta = self.mhd.eta

        dt_conv = self.cfl_advect_crit * dx / v_max
        dt_diff = self.fo_diff_crit * dx ** 2 / eta

        self.dt_critical = min(dt_conv, dt_diff)
        self.dt_critical = max(self.dt_critical, 1e-10)

    def compute_growth_rate_spectrum(self, k_range=None):
        """
        计算撕裂模增长率谱 gamma(k)。

        线性色散关系 (Furth et al. 1963):
          gamma * tau_A = C * S^{-3/5} * (k*a)^{2/5}

        其中 C ~ 0.6 为常数，tau_A = L/v_A。
        """
        if k_range is None:
            k_range = np.linspace(0.1, 2.0, 50)
        S = self.mhd.S_lundquist
        a = 0.5
        C_fkr = 0.6
        gamma = C_fkr * S ** (-0.6) * (k_range * a) ** 0.4
        return k_range, gamma
