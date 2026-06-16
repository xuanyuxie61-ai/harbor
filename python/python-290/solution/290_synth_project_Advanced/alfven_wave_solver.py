"""
alfven_wave_solver.py - 简化 MHD 阿尔芬波时间推进求解器

本模块实现阿尔芬波在磁化等离子体中的时间推进求解。
基于简化 MHD 模型 (Reduced MHD)，使用高阶有限差分
空间离散和显式/隐式时间推进。

物理背景:
  简化 MHD 方程组 (在柱坐标近似下):

  1. 磁通量演化 (Ohm 定律):
     ∂ψ/∂t = -∇_∥ φ + η ∇²ψ

  2. 涡度演化 (运动方程):
     ∂U/∂t = -∇_∥ J + ν ∇²U - [φ, U]

  其中:
    ψ: 磁通量函数 (扰动磁场的流函数)
    φ: 电势 (E × B 漂移速度势)
    U = ∇²φ: 涡度
    J = ∇²ψ: 扰动电流
    ∇_∥ = (1/qR₀) ∂/∂θ: 沿平衡磁场的导数
    η: 电阻率
    ν: 粘性系数
    [f,g] = (∂f/∂r)(∂g/∂θ) - (∂f/∂θ)(∂g/∂r): 泊松括号

  阿尔芬波是 ψ 和 φ 的耦合振荡，色散关系 ω = k_∥ v_A。

数值方法:
  空间离散: 2N 阶中心差分
  时间推进:
    - 显式: 4阶 Runge-Kutta (RK4)
    - 隐式: Crank-Nicolson + 带状矩阵求解

核心公式:
  RK4 时间推进:
    k₁ = f(t_n, y_n)
    k₂ = f(t_n + Δt/2, y_n + Δt/2 k₁)
    k₃ = f(t_n + Δt/2, y_n + Δt/2 k₂)
    k₄ = f(t_n + Δt, y_n + Δt k₃)
    y_{n+1} = y_n + (Δt/6)(k₁ + 2k₂ + 2k₃ + k₄)

作者: DA 博士级合成项目 PROJECT_290
"""

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as splinalg

from high_order_operators import (
    apply_first_derivative,
    apply_second_derivative,
    apply_radial_laplacian_cylindrical,
    BandMatrixSPD,
)
from boundary_handler import periodic_index


class AlfvenWaveState:
    """
    阿尔芬波系统状态向量。

    包含磁通量扰动 ψ、电势 φ 及其衍生量。
    状态向量的维度为 (n_r, n_theta)，
    对应径向和极向的二维截面。
    """

    def __init__(self, n_r, n_theta):
        self.n_r = n_r
        self.n_theta = n_theta

        # 磁通量扰动 [Wb] 或归一化单位
        self.psi = np.zeros((n_r, n_theta), dtype=np.float64)

        # 电势 [V] 或归一化单位
        self.phi = np.zeros((n_r, n_theta), dtype=np.float64)

        # 扰动电流 J = ∇²ψ
        self.current = np.zeros((n_r, n_theta), dtype=np.float64)

        # 涡度 U = ∇²φ
        self.vorticity = np.zeros((n_r, n_theta), dtype=np.float64)

    def copy(self):
        """深拷贝状态。"""
        new_state = AlfvenWaveState(self.n_r, self.n_theta)
        new_state.psi = self.psi.copy()
        new_state.phi = self.phi.copy()
        new_state.current = self.current.copy()
        new_state.vorticity = self.vorticity.copy()
        return new_state

    def energy(self):
        """
        计算阿尔芬波总能量 (磁能 + 动能):
          E = (1/2) ∫∫ (|∇ψ|² + |∇φ|²) r dr dθ
        """
        # 简化的能量计算 - 使用场本身的 L2 范数
        e_mag = np.sum(self.psi ** 2)
        e_kin = np.sum(self.phi ** 2)
        return 0.5 * e_mag + 0.5 * e_kin


class AlfvenWaveSolver:
    """
    阿尔芬波简化 MHD 方程的数值求解器。

    使用高阶有限差分空间离散和 RK4/Crank-Nicolson 时间推进。
    """

    def __init__(self, plasma_params, geometry):
        """
        参数:
          plasma_params: PlasmaParameters, 等离子体参数
          geometry: MagneticGeometry, 磁平衡几何
        """
        self.params = plasma_params
        self.geom = geometry
        self.n_r = plasma_params.n_r
        self.n_theta = plasma_params.n_theta
        self.fd_order = plasma_params.fd_order

        # 网格
        self.r_array = np.linspace(0.0, plasma_params.a_minor, self.n_r)
        self.theta_array = np.linspace(0.0, 2.0 * np.pi, self.n_theta, endpoint=False)
        self.dr = self.r_array[1] - self.r_array[0] if self.n_r > 1 else 1.0
        self.dtheta = 2.0 * np.pi / self.n_theta

        # 物理系数
        self.v_A = plasma_params.v_alfven
        self.eta = plasma_params.eta_resistivity
        self.q_profile = plasma_params.q_profile(self.r_array / plasma_params.a_minor)

        # 预计算沿场线导数算子系数
        # ∇_∥ = (1/qR₀) ∂/∂θ → 离散为 (1/q_j R₀ dtheta) × 差分矩阵
        self._setup_parallel_derivative()

        # 时间步长
        self.dt = plasma_params.dt

        # 存储能量历史
        self.energy_history = []
        self.time_history = []

    def _setup_parallel_derivative(self):
        """
        设置沿磁力线方向的导数算子。

        ∇_∥ f = (1/q(r) R₀) ∂f/∂θ

        在离散形式下:
          (∇_∥ f)_{j,k} = (1/q_j R₀ dtheta) Σ c_m [f_{j,k+m} - f_{j,k-m}]
        """
        from high_order_operators import fd_coefficients_first_derivative
        self.fd1_coeffs = fd_coefficients_first_derivative(self.fd_order)
        self.N_stencil = len(self.fd1_coeffs)

        # 沿场线导数预因子 (每个径向位置不同)
        R0 = self.params.R0
        self.parallel_prefactor = np.zeros(self.n_r)
        for j in range(self.n_r):
            q = self.q_profile[j]
            if abs(q) > 1e-10:
                self.parallel_prefactor[j] = 1.0 / (q * R0)
            else:
                self.parallel_prefactor[j] = 0.0

    def _parallel_derivative(self, field):
        """
        计算沿磁力线方向 ∇_∥ field。

        参数:
          field: ndarray, shape (n_r, n_theta)

        返回:
          result: ndarray, shape (n_r, n_theta)
        """
        result = np.zeros_like(field)
        for j in range(self.n_r):
            row = field[j, :]
            deriv = np.zeros(self.n_theta)
            for k in range(self.n_theta):
                for m_idx in range(self.N_stencil):
                    m = m_idx + 1
                    k_plus = periodic_index(k + m, self.n_theta)
                    k_minus = periodic_index(k - m, self.n_theta)
                    deriv[k] += self.fd1_coeffs[m_idx] * (row[k_plus] - row[k_minus])
                deriv[k] *= self.parallel_prefactor[j] / self.dtheta
            result[j, :] = deriv
        return result

    def _compute_laplacian(self, field):
        """
        计算柱坐标下的二维拉普拉斯:
          ∇²f = ∂²f/∂r² + (1/r)∂f/∂r + (1/r²)∂²f/∂θ²

        参数:
          field: ndarray, shape (n_r, n_theta)

        返回:
          laplacian: ndarray, shape (n_r, n_theta)
        """
        laplacian = np.zeros_like(field)

        for k in range(self.n_theta):
            col = field[:, k]
            lap_col = apply_radial_laplacian_cylindrical(
                col, self.r_array, self.dr, self.fd_order
            )
            laplacian[:, k] = lap_col

        # 极向二阶导数 (1/r²) ∂²f/∂θ²
        for j in range(self.n_r):
            r = self.r_array[j]
            if r > 1e-14:
                row = field[j, :]
                d2f_dth2 = apply_second_derivative(row, self.dtheta, self.fd_order, periodic=True)
                laplacian[j, :] += d2f_dth2 / (r ** 2)

        return laplacian

    def _poisson_bracket(self, f, g):
        """
        计算泊松括号 [f, g] = (∂f/∂r)(∂g/∂θ) - (∂f/∂θ)(∂g/∂r)。

        使用高阶有限差分。

        参数:
          f, g: ndarray, shape (n_r, n_theta)

        返回:
          bracket: ndarray, shape (n_r, n_theta)
        """
        bracket = np.zeros_like(f)

        for k in range(self.n_theta):
            # ∂f/∂r 和 ∂g/∂r
            df_dr = apply_first_derivative(f[:, k], self.dr, self.fd_order, periodic=False)
            dg_dr = apply_first_derivative(g[:, k], self.dr, self.fd_order, periodic=False)

            for j in range(self.n_r):
                # ∂f/∂θ 和 ∂g/∂θ
                df_dth = apply_first_derivative(f[j, :], self.dtheta, self.fd_order, periodic=True)
                dg_dth = apply_first_derivative(g[j, :], self.dtheta, self.fd_order, periodic=True)

                bracket[j, k] = df_dr[j] * dg_dth[k] - df_dth[k] * dg_dr[j]

        return bracket

    def rhs(self, state):
        """
        计算右端项 f(t, y) 用于 dy/dt = f(t, y)。

        简化 MHD 右端项:
          ∂ψ/∂t = -∇_∥ φ + η ∇²ψ
          ∂φ/∂t = ∇_∥ ψ - [φ, ψ]  (在 Alfvén 单位下)

        实际上涡度方程为:
          ∂U/∂t = -∇_∥ J - [φ, U] + ν ∇²U

        为简化，这里使用 ψ-φ 耦合系统。

        参数:
          state: AlfvenWaveState

        返回:
          dpsi_dt, dphi_dt: ndarray, shape (n_r, n_theta)
        """
        psi = state.psi
        phi = state.phi

        # 更新衍生量
        state.current = self._compute_laplacian(psi)
        state.vorticity = self._compute_laplacian(phi)

        # ∇_∥ φ 和 ∇_∥ ψ
        grad_par_phi = self._parallel_derivative(phi)
        grad_par_psi = self._parallel_derivative(psi)

        # 电阻扩散 η ∇²ψ
        laplacian_psi = self._compute_laplacian(psi)

        # 泊松括号 [φ, ψ] (非线性项)
        nb = self._poisson_bracket(phi, psi)

        # 右端项
        dpsi_dt = -grad_par_phi + self.eta * laplacian_psi
        dphi_dt = self.v_A ** 2 * grad_par_psi - nb

        # 添加吸收层 (边界阻尼)
        from boundary_handler import boundary_layer_profile
        n_boundary = min(5, self.n_r // 4)
        if n_boundary > 1:
            delta = self.dr * n_boundary
            profile, _ = boundary_layer_profile(n_boundary, delta)
            damping = np.ones(self.n_r)
            damping[-n_boundary:] = profile
            damping = damping[:, np.newaxis]
            dphi_dt *= damping

        return dpsi_dt, dphi_dt

    def step_rk4(self, state, dt=None):
        """
        四阶 Runge-Kutta 时间步。

        k₁ = f(t_n, y_n)
        k₂ = f(t_n + Δt/2, y_n + Δt/2 k₁)
        k₃ = f(t_n + Δt/2, y_n + Δt/2 k₂)
        k₄ = f(t_n + Δt, y_n + Δt k₃)
        y_{n+1} = y_n + (Δt/6)(k₁ + 2k₂ + 2k₃ + k₄)

        参数:
          state: AlfvenWaveState, 当前状态
          dt: float or None, 时间步长 (默认使用 self.dt)

        返回:
          new_state: AlfvenWaveState, 新状态
        """
        if dt is None:
            dt = self.dt

        # k1
        k1_psi, k1_phi = self.rhs(state)

        # k2
        s2 = state.copy()
        s2.psi += 0.5 * dt * k1_psi
        s2.phi += 0.5 * dt * k1_phi
        k2_psi, k2_phi = self.rhs(s2)

        # k3
        s3 = state.copy()
        s3.psi += 0.5 * dt * k2_psi
        s3.phi += 0.5 * dt * k2_phi
        k3_psi, k3_phi = self.rhs(s3)

        # k4
        s4 = state.copy()
        s4.psi += dt * k3_psi
        s4.phi += dt * k3_phi
        k4_psi, k4_phi = self.rhs(s4)

        # 合成
        new_state = state.copy()
        new_state.psi = state.psi + (dt / 6.0) * (k1_psi + 2.0 * k2_psi + 2.0 * k3_psi + k4_psi)
        new_state.phi = state.phi + (dt / 6.0) * (k1_phi + 2.0 * k2_phi + 2.0 * k3_phi + k4_phi)

        return new_state

    def step_crank_nicolson_linear(self, state, dt=None):
        """
        Crank-Nicolson 隐式时间步 (线性化版本)。

        对于线性化阿尔芬波方程 (忽略非线性项):
          ∂ψ/∂t = -∇_∥ φ + η ∇²ψ

        CN 格式:
          (I - Δt/2 L) ψ^{n+1} = (I + Δt/2 L) ψ^n - Δt ∇_∥ φ^{n+1/2}

        这里简化为对角隐式: 仅对电阻项使用隐式处理。

        参数:
          state: AlfvenWaveState
          dt: float or None

        返回:
          new_state: AlfvenWaveState
        """
        if dt is None:
            dt = self.dt

        # 先做显式 RK4 步
        new_state = self.step_rk4(state, dt)

        # 然后对 ψ 施加电阻隐式平滑 (简化处理)
        # ψ_new = ψ_rk4 + η Δt ∇²ψ_rk4 / (1 + η Δt / dr²)
        laplacian_psi = self._compute_laplacian(new_state.psi)
        diffusion_number = self.eta * dt / (self.dr ** 2) if self.dr > 0 else 0.0
        factor = diffusion_number / (1.0 + diffusion_number)
        new_state.psi += factor * laplacian_psi * self.dr ** 2

        return new_state

    def initialize_alfven_eigenmode(self, state, m_poloidal, n_toroidal, amplitude=1e-3):
        """
        初始化阿尔芬本征模。

        在柱坐标近似下，阿尔芬本征模的形式为:
          ψ(r, θ, t=0) = A ψ₀(r) exp(i m θ)
          φ(r, θ, t=0) = A φ₀(r) exp(i m θ)

        其中径向本征函数 ψ₀(r) 在有理面 q=m/n 处
        满足正则化奇异方程。简化为高斯包络:
          ψ₀(r) = exp(-(r - r_res)² / (2σ²))

        参数:
          state: AlfvenWaveState, 待初始化的状态
          m_poloidal: int, 极向模数
          n_toroidal: int, 环向模数
          amplitude: float, 初始振幅
        """
        if abs(n_toroidal) < 1:
            raise ValueError("环向模数 n 必须非零")

        # 有理面位置: q(r_res) = m/n
        q_target = float(m_poloidal) / float(n_toroidal)

        # 找到最接近有理面的径向位置
        q_vals = self.q_profile
        r_res_idx = np.argmin(np.abs(q_vals - q_target))
        r_res = self.r_array[r_res_idx]

        # 径向包络宽度 (与磁剪切有关)
        s_local = self.params.magnetic_shear(r_res / self.params.a_minor)
        sigma_r = self.params.a_minor * max(0.05, 0.1 / (abs(s_local) + 0.1))

        # 初始化
        for j in range(self.n_r):
            r = self.r_array[j]
            envelope = amplitude * np.exp(-((r - r_res) ** 2) / (2.0 * sigma_r ** 2))
            for k in range(self.n_theta):
                phase = m_poloidal * self.theta_array[k]
                state.psi[j, k] = envelope * np.cos(phase)
                state.phi[j, k] = envelope * np.sin(phase) * self.v_A

    def run_simulation(self, initial_state, n_steps=None, output_interval=None):
        """
        运行时间推进模拟。

        参数:
          initial_state: AlfvenWaveState, 初始状态
          n_steps: int, 总步数
          output_interval: int, 记录间隔

        返回:
          final_state: AlfvenWaveState, 最终状态
          history: dict, 历史记录
        """
        if n_steps is None:
            n_steps = self.params.n_timesteps
        if output_interval is None:
            output_interval = self.params.output_interval

        state = initial_state.copy()
        history = {
            'energy': [],
            'max_psi': [],
            'max_phi': [],
            'time': [],
        }

        # 记录初始状态
        e0 = state.energy()
        history['energy'].append(e0)
        history['max_psi'].append(np.max(np.abs(state.psi)))
        history['max_phi'].append(np.max(np.abs(state.phi)))
        history['time'].append(0.0)

        for step in range(1, n_steps + 1):
            state = self.step_rk4(state)

            if step % output_interval == 0:
                e = state.energy()
                history['energy'].append(e)
                history['max_psi'].append(np.max(np.abs(state.psi)))
                history['max_phi'].append(np.max(np.abs(state.phi)))
                history['time'].append(step * self.dt)

                # 数值稳定性检查
                if not np.isfinite(e):
                    print(f"  [警告] 步骤 {step}: 能量发散，模拟终止")
                    break
                if e > 1e10 * e0:
                    print(f"  [警告] 步骤 {step}: 能量增长过大 (E/E₀ = {e/e0:.2e})")

        return state, history
