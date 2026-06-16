"""
stability_eigenvalue.py - 阿尔芬波稳定性本征值分析模块

本模块对阿尔芬波-高能粒子耦合系统进行本征值分析，
判断系统的稳定性。核心算法改编自 companion_matrix (203) 项目。

物理背景:
  将阿尔芬波方程线性化后写为:
    ∂y/∂t = L y
  其中 y 为状态向量 (ψ, φ)，L 为线性算子。

  空间离散后:
    dy/dt = A y
  其中 A 为 N×N 矩阵。

  稳定性由 A 的本征值决定:
    y(t) = Σ cₖ exp(λₖ t) vₖ
  其中 λₖ = γₖ + iωₖ 为本征值。

  稳定性判据:
    γₖ < 0 ∀k  → 稳定 (所有模衰减)
    γₖ > 0 ∃k  → 不稳定 (至少一个模增长)

  对于阿尔芬波-EP 系统:
    γ > 0 表示 EP 驱动的不稳定性 (如 TAE 不稳定性)
    γ < 0 表示背景阻尼 (连续阻尼、辐射阻尼等)

数值方法:
  1. 直接本征值求解: 对离散算子矩阵 A 求全部/部分本征值
  2. 伴随矩阵法: 将色散多项式转化为伴随矩阵本征值问题
  3. 能量原理: δW = δW_MHD + δW_EP 的正定性判断

作者: DA 博士级合成项目 PROJECT_290
"""

import numpy as np
from scipy import linalg


class StabilityAnalyzer:
    """
    阿尔芬波系统稳定性分析器。

    提供多种稳定性分析方法:
      1. 矩阵本征值分析
      2. 能量原理 (δW) 分析
      3. Nyquist 稳定性判据
    """

    def __init__(self, plasma_params, geometry):
        self.params = plasma_params
        self.geom = geometry
        self.n_r = plasma_params.n_r
        self.n_theta = plasma_params.n_theta

    def build_linear_operator(self, fd_order=4):
        """
        构建线性化阿尔芬波算子的矩阵表示。

        对于线性化简化 MHD:
          ∂ψ/∂t = -∇_∥ φ + η ∇²ψ
          ∂φ/∂t = v_A² ∇_∥ ψ

        写成块矩阵:
          d/dt [ψ]   [η∇²    -∇_∥  ] [ψ]
               [φ] = [v_A²∇_∥   0   ] [φ]

        参数:
          fd_order: int, 有限差分阶数

        返回:
          A: ndarray, shape (2N, 2N), 线性算子矩阵
        """
        from high_order_operators import fd_coefficients_second_derivative, fd_coefficients_first_derivative

        n_r = self.n_r
        n_theta = self.n_theta
        N = n_r * n_theta

        dr = self.params.a_minor / max(n_r - 1, 1)
        dtheta = 2.0 * np.pi / n_theta
        v_A = self.params.v_alfven
        eta = self.params.eta_resistivity

        q_vals = self.params.q_profile(
            np.linspace(0.01, 1.0, n_r)
        )

        # 构建子矩阵
        # 1. η ∇² (径向 + 极向)
        d0, coeffs = fd_coefficients_second_derivative(fd_order)

        # 径向 Laplacian (块三对角)
        L_radial = np.zeros((N, N))
        for j in range(n_r):
            r = j * dr
            for k in range(n_theta):
                idx = j * n_theta + k
                L_radial[idx, idx] = d0 / dr ** 2
                # 加上 (1/r) ∂/∂r 项
                if r > 1e-10:
                    fd1 = fd_coefficients_first_derivative(fd_order)
                    for m_idx, c_m in enumerate(fd1):
                        m = m_idx + 1
                        j_plus = min(j + m, n_r - 1)
                        j_minus = max(j - m, 0)
                        idx_plus = j_plus * n_theta + k
                        idx_minus = j_minus * n_theta + k
                        L_radial[idx, idx_plus] += c_m / (r * dr)
                        L_radial[idx, idx_minus] -= c_m / (r * dr)

                for m_idx, c_m in enumerate(coeffs):
                    m = m_idx + 1
                    j_plus = min(j + m, n_r - 1)
                    j_minus = max(j - m, 0)
                    idx_plus = j_plus * n_theta + k
                    idx_minus = j_minus * n_theta + k
                    L_radial[idx, idx_plus] += c_m / dr ** 2
                    L_radial[idx, idx_minus] += c_m / dr ** 2

        # 极向 Laplacian (块对角)
        L_poloidal = np.zeros((N, N))
        for j in range(n_r):
            r = j * dr
            if r > 1e-10:
                for k in range(n_theta):
                    idx = j * n_theta + k
                    fd1_2, coeffs_2 = fd_coefficients_second_derivative(fd_order)
                    L_poloidal[idx, idx] += d0 / (r * dtheta) ** 2
                    for m_idx, c_m in enumerate(coeffs_2):
                        m = m_idx + 1
                        k_plus = (k + m) % n_theta
                        k_minus = (k - m) % n_theta
                        L_poloidal[idx, j * n_theta + k_plus] += c_m / (r * dtheta) ** 2
                        L_poloidal[idx, j * n_theta + k_minus] += c_m / (r * dtheta) ** 2

        L_total = L_radial + L_poloidal

        # 2. ∇_∥ 算子
        G_par = np.zeros((N, N))
        fd1 = fd_coefficients_first_derivative(fd_order)
        for j in range(n_r):
            q = q_vals[j]
            prefactor = 1.0 / (q * self.params.R0 * dtheta) if abs(q) > 1e-10 else 0.0
            for k in range(n_theta):
                idx = j * n_theta + k
                for m_idx, c_m in enumerate(fd1):
                    m = m_idx + 1
                    k_plus = (k + m) % n_theta
                    k_minus = (k - m) % n_theta
                    G_par[idx, j * n_theta + k_plus] += prefactor * c_m
                    G_par[idx, j * n_theta + k_minus] -= prefactor * c_m

        # 组装块矩阵
        A = np.zeros((2 * N, 2 * N))
        A[:N, :N] = eta * L_total     # η ∇²
        A[:N, N:] = -G_par            # -∇_∥
        A[N:, :N] = v_A ** 2 * G_par  # v_A² ∇_∥
        # A[N:, N:] = 0

        return A

    def eigenvalue_analysis(self, A, n_modes=None):
        """
        对线性算子进行本征值分析。

        计算 A 的全部本征值 (对小系统) 或部分本征值。

        参数:
          A: ndarray, 线性算子矩阵
          n_modes: int or None, 返回的模数

        返回:
          result: dict, 包含本征值、增长率、频率、稳定性判断
        """
        N = A.shape[0] // 2

        # 计算全部本征值
        eigenvalues = linalg.eigvals(A)

        # 按实部排序 (增长率从大到小)
        idx_sort = np.argsort(-eigenvalues.real)
        eigenvalues = eigenvalues[idx_sort]

        # 提取增长率和频率
        growth_rates = eigenvalues.real
        frequencies = eigenvalues.imag / (2.0 * np.pi)

        # 稳定性判断
        max_growth = np.max(growth_rates)
        is_stable = max_growth < 1e-10

        if n_modes is not None:
            eigenvalues = eigenvalues[:n_modes]
            growth_rates = growth_rates[:n_modes]
            frequencies = frequencies[:n_modes]

        return {
            'eigenvalues': eigenvalues,
            'growth_rates': growth_rates,
            'frequencies_hz': frequencies,
            'max_growth_rate': max_growth,
            'is_stable': is_stable,
            'n_unstable': int(np.sum(growth_rates > 1e-10)),
            'dominant_mode_idx': int(np.argmax(growth_rates)),
        }

    def energy_principle_analysis(self, state):
        """
        能量原理稳定性分析 (δW 方法)。

        理想 MHD 能量原理:
          δW = δW_F + δW_S + δW_V

        其中:
          δW_F = (1/2) ∫ [|Q_⊥|²/μ₀ + γp|∇·ξ|² + |J×ξ + ∇(ξ·∇p)|²/...] dV
          δW_S = (1/2) ∫ |Q_∥|²/μ₀ dV  (稳定贡献)
          δW_V = (1/2) ∫_{vacuum} |Q̂|²/μ₀ dV

        简化版:
          δW ≈ (1/2) ∫ [v_A² |∇_⊥ ψ|² - |∇_∥ ψ|² v_A² + EP修正] dV

        δW > 0 → 稳定
        δW < 0 → 不稳定

        参数:
          state: AlfvenWaveState

        返回:
          result: dict, 包含 δW 各分量及总体稳定性
        """
        from high_order_operators import apply_first_derivative, apply_second_derivative
        from boundary_handler import periodic_index

        n_r = self.n_r
        n_theta = self.n_theta
        dr = self.params.a_minor / max(n_r - 1, 1)
        dtheta = 2.0 * np.pi / n_theta
        v_A = self.params.v_alfven

        psi = state.psi
        phi = state.phi

        # δW_MHD: 磁能 + 动能
        dw_mag = 0.0
        dw_kin = 0.0
        dw_parallel = 0.0

        q_vals = self.params.q_profile(np.linspace(0.01, 1.0, n_r))

        for j in range(1, n_r - 1):
            r = j * dr
            q = q_vals[j]
            for k in range(n_theta):
                # 径向导数
                dpsi_dr = (psi[j + 1, k] - psi[j - 1, k]) / (2.0 * dr)
                dphi_dr = (phi[j + 1, k] - phi[j - 1, k]) / (2.0 * dr)

                # 极向导数
                k_plus = periodic_index(k + 1, n_theta)
                k_minus = periodic_index(k - 1, n_theta)
                dpsi_dth = (psi[j, k_plus] - psi[j, k_minus]) / (2.0 * dtheta)
                dphi_dth = (phi[j, k_plus] - phi[j, k_minus]) / (2.0 * dtheta)

                # 沿场线导数
                if abs(q) > 1e-10:
                    grad_par_psi = (1.0 / (q * self.params.R0)) * dpsi_dth
                else:
                    grad_par_psi = 0.0

                # 体积元 r dr dθ
                dV = r * dr * dtheta

                # 磁能 (梯度能)
                dw_mag += 0.5 * (dpsi_dr ** 2 + dpsi_dth ** 2 / (r ** 2 + 1e-30)) * dV
                # 动能
                dw_kin += 0.5 * (dphi_dr ** 2 + dphi_dth ** 2 / (r ** 2 + 1e-30)) * dV
                # 平行项 (稳定/不稳定取决于符号)
                dw_parallel += -0.5 * v_A ** 2 * grad_par_psi ** 2 * dV

        # 高能粒子贡献 (简化: 如果 β_fast 足够大则提供稳定化)
        dw_ep = self.params.beta_fast * dw_mag * 0.1

        dw_total = dw_mag * v_A ** 2 + dw_kin + dw_parallel + dw_ep

        is_stable = dw_total > 0

        return {
            'delta_W_total': dw_total,
            'delta_W_magnetic': dw_mag * v_A ** 2,
            'delta_W_kinetic': dw_kin,
            'delta_W_parallel': dw_parallel,
            'delta_W_EP': dw_ep,
            'is_stable': is_stable,
            'stability_margin': dw_total / (abs(dw_mag * v_A ** 2) + abs(dw_kin) + 1e-30),
        }

    def nyquist_stability_check(self, omega_array, D_array):
        """
        Nyquist 稳定性判据。

        绘制色散函数 D(ω) 的 Nyquist 图，
        通过计算包围原点的圈数判断不稳定模的数量。

        参数:
          omega_array: ndarray, 频率扫描值
          D_array: ndarray, 对应的色散函数值 (复数)

        返回:
          n_encirclements: int, 逆时针包围原点的圈数
          is_stable: bool, Nyquist 稳定性判断
        """
        # 计算环绕数 (幅角原理)
        phases = np.angle(D_array)
        phase_diffs = np.diff(phases)

        # 处理 ±π 跳变
        phase_diffs = np.where(phase_diffs > np.pi, phase_diffs - 2 * np.pi, phase_diffs)
        phase_diffs = np.where(phase_diffs < -np.pi, phase_diffs + 2 * np.pi, phase_diffs)

        total_phase = np.sum(phase_diffs)
        n_encirclements = int(round(total_phase / (2.0 * np.pi)))

        is_stable = n_encirclements <= 0

        return {
            'n_encirclements': n_encirclements,
            'total_phase_change': total_phase,
            'is_stable': is_stable,
        }
