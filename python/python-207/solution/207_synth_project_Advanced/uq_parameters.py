"""
uq_parameters.py — 随机抛物型 PDE 不确定性量化问题的物理/数学参数定义

科学背景
========
求解随机热传导方程 (Stochastic Heat Equation, SHE):

    ∂u/∂t = ∂/∂x [ κ(x,ω) · ∂u/∂x ] + f(x,t),
        x ∈ (0, L),  t ∈ (0, T]

其中 κ(x,ω) = κ₀ + σ_κ · Y(x,ω) 为随机扩散系数场,
Y(x,ω) 为具有平方指数核的平稳高斯随机场:

    C(r) = σ² · exp( -r² / (2·ℓ²) )

边界条件: Dirichlet,  u(0,t) = u_L,  u(L,t) = u_R
初始条件:  u(x,0) = u₀(x) + 扰动

本模块定义所有计算所需的数值参数, 包括空间/时间离散化,
随机场参数, 蒙特卡洛样本量, 置信水平等.

核心公式
========
1. 网格 Peclet 数:  Pe_h = |c|·h / (2·κ₀)  (对流-扩散情形)
2. 网格 Fourier 数:  Fo = κ₀·Δt / h²        (稳定性约束)
3. 相关长度与网格比: ρ = ℓ / h               (分辨率指标)
4. 有效维度:  N_eff = L / ℓ                    (独立涨落单元数)
5. KL 截断能量:  Σ_{k>K} λ_k / Σ_{k} λ_k < ε
"""

import numpy as np


class UQProblemParameters:
    """封装 SHE-UQ 问题的全部参数."""

    def __init__(self):
        # ============================================================
        # 空间域 [0, L] 的离散化
        # ============================================================
        self.L = 1.0               # 空间域长度 (m)
        self.nx = 51               # 空间节点数 (含边界)
        self.x = np.linspace(0.0, self.L, self.nx)
        self.dx = self.x[1] - self.x[0]

        # ============================================================
        # 时间域 [0, T] 的离散化
        # ============================================================
        self.T = 0.5               # 终止时间 (s)
        self.nt = 101              # 时间步数 (含 t=0)
        self.t = np.linspace(0.0, self.T, self.nt)
        self.dt = self.t[1] - self.t[0]

        # ============================================================
        # 确定性物理参数
        # ============================================================
        self.kappa_0 = 1.0e-2      # 基准扩散系数 (m²/s)
        self.u_left = 100.0        # 左边界 Dirichlet 温度 (K)
        self.u_right = 25.0        # 右边界 Dirichlet 温度 (K)
        self.u_init = 50.0         # 均匀初始温度 (K)

        # ============================================================
        # 随机场参数 (平方指数核)
        # ============================================================
        self.sigma_kappa = 3.0e-3  # κ 的标准差 (m²/s)
        self.correlation_length = 0.1   # 相关长度 ℓ (m)

        # ============================================================
        # 随机源项参数 (可选, 加法噪声)
        # ============================================================
        self.sigma_source = 1.0    # 源项标准差 (K/s)
        self.include_source_noise = True

        # ============================================================
        # 蒙特卡洛 / 采样参数
        # ============================================================
        self.n_mc = 400            # Monte Carlo 样本量
        self.n_kl_modes = 15       # Karhunen-Loève 截断阶数
        self.quadrature_order = 20  # Gauss-Chebyshev 求积阶数

        # ============================================================
        # 置信 / 预测区间参数
        # ============================================================
        self.alpha_ci = 0.05       # 点态置信水平 (95% CI)
        self.alpha_pi = 0.05       # 点态预测水平 (95% PI)
        self.alpha_sim = 0.05      # 同时带覆盖率 (SCB)
        self.coverage_target = 1.0 - self.alpha_sim
        self.bootstrap_n = 500     # Bootstrap 重采样次数
        self.fixed_point_tol = 1.0e-8   # 不动点迭代容差
        self.fixed_point_max_iter = 200

        # ============================================================
        # 数值稳定性参数
        # ============================================================
        self.cholesky_jitter = 1.0e-12   # Cholesky 正则化
        self.newton_tol = 1.0e-10        # Newton/Regula 容差
        self.max_bisection_iter = 300    # 对分法最大迭代
        self.eps_machine = np.finfo(float).eps

        # ============================================================
        # 高斯求积节点 / 权重 (Gauss-Chebyshev 第一类)
        # 节点: x_k = cos((2k-1)π/(2n)),  k=1,...,n
        # 权重: w_k = π/n
        # ============================================================
        self._build_quadrature()

    def _build_quadrature(self):
        """构造 Gauss-Chebyshev 第一类求积规则."""
        n = self.quadrature_order
        k = np.arange(1, n + 1)
        self.gc_nodes = np.cos((2.0 * k - 1.0) * np.pi / (2.0 * n))
        self.gc_weights = np.full(n, np.pi / n)

    # ---------------------------------------------------------------
    # 派生诊断量
    # ---------------------------------------------------------------
    @property
    def fourier_number(self):
        """网格 Fourier 数: Fo = κ₀·Δt / Δx²."""
        return self.kappa_0 * self.dt / (self.dx ** 2)

    @property
    def correlation_ratio(self):
        """相关长度 / 网格间距: ρ = ℓ / Δx."""
        return self.correlation_length / self.dx

    @property
    def effective_dimension(self):
        """有效独立单元数: N_eff = L / ℓ."""
        return self.L / self.correlation_length

    @property
    def kl_energy_retained(self):
        """KL 截断保留能量的近似 (基于平方指数核特征值衰减)."""
        # 对于平方指数核, 第 k 个特征值近似衰减为
        #   λ_k ~ C · exp(-c·k^(2/d))  (d=1 时: λ_k ~ exp(-c·k²))
        # 此处用经验常数估算
        c_decay = 0.5 * (self.L / self.correlation_length) ** 2
        all_energy = np.sum(np.exp(-c_decay * np.arange(1, 200) ** 2))
        kept_energy = np.sum(np.exp(-c_decay * np.arange(1, self.n_kl_modes + 1) ** 2))
        return kept_energy / all_energy if all_energy > 0 else 0.0

    def validate(self):
        """验证参数一致性并报告诊断信息."""
        info = {}

        # Fourier 数检查 (隐式格式理论上无条件稳定,
        # 但 Fo >> 1 会增大时间截断误差)
        info['fourier_number'] = self.fourier_number
        if self.fourier_number > 5.0:
            info['fourier_warning'] = (
                f"Fourier 数 Fo={self.fourier_number:.2f} >> 1, "
                "时间截断误差可能较大"
            )
        else:
            info['fourier_warning'] = None

        # 相关长度 vs 网格间距
        info['correlation_ratio'] = self.correlation_ratio
        if self.correlation_ratio < 2.0:
            info['mesh_warning'] = (
                f"相关长度/网格比 ρ={self.correlation_ratio:.2f} < 2, "
                "网格不足以分辨随机场涨落"
            )
        else:
            info['mesh_warning'] = None

        # KL 截断能量
        info['kl_energy_retained'] = self.kl_energy_retained
        if info['kl_energy_retained'] < 0.95:
            info['kl_warning'] = (
                f"KL 截断仅保留 {info['kl_energy_retained']*100:.1f}% 能量, "
                f"建议增加 n_kl_modes (当前={self.n_kl_modes})"
            )
        else:
            info['kl_warning'] = None

        # Monte Carlo 样本量 vs 有效维度
        info['mc_efficiency'] = self.n_mc / max(1, int(self.effective_dimension))
        if info['mc_efficiency'] < 10:
            info['mc_warning'] = (
                f"MC 样本量/有效维度比 = {info['mc_efficiency']:.1f}, "
                "建议增加蒙特卡洛样本量"
            )
        else:
            info['mc_warning'] = None

        return info

    def summary(self):
        """返回参数的文本摘要."""
        val = self.validate()
        lines = [
            "=" * 70,
            "SHE-UQ 问题参数摘要",
            "=" * 70,
            f"  空间域:        [0, {self.L}] m,  nx={self.nx},  dx={self.dx:.6f}",
            f"  时间域:        [0, {self.T}] s,  nt={self.nt},  dt={self.dt:.6f}",
            f"  基准扩散系数:  κ₀ = {self.kappa_0:.4e} m²/s",
            f"  随机扰动强度:  σ_κ = {self.sigma_kappa:.4e} m²/s",
            f"  相关长度:      ℓ = {self.correlation_length:.4f} m",
            f"  Fourier 数:    Fo = {val['fourier_number']:.4f}",
            f"  相关比:        ρ = {val['correlation_ratio']:.2f}",
            f"  有效维度:      N_eff = {self.effective_dimension:.2f}",
            f"  KL 截断阶数:  K = {self.n_kl_modes}",
            f"  KL 保留能量:  {val['kl_energy_retained']*100:.2f}%",
            f"  MC 样本量:    N_mc = {self.n_mc}",
            f"  置信水平:      1-α = {self.coverage_target:.4f}",
            f"  求积阶数:      n_q = {self.quadrature_order}",
            "=" * 70,
        ]
        for key in ['fourier_warning', 'mesh_warning', 'kl_warning', 'mc_warning']:
            if val.get(key):
                lines.append(f"  ⚠ {val[key]}")
        return "\n".join(lines)
