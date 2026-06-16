"""
stability_analysis.py
=====================
恒星振荡模式的稳定性分析与分岔检测.

融合种子项目:
  - border-collision-bifurcation (1085): 时间延迟系统 → 模式切换分岔
  - Jacobi (603): 迭代收敛 → 稳定域判定
  - biochemical_linear_ode (090): ODE 稳定性 → 线性稳定性矩阵

科学背景
--------
恒星振荡的稳定性由复频率 ω = ω_R + i γ 决定:
  - γ > 0: 不稳定模式 (增长)
  - γ < 0: 稳定模式 (衰减)
  - γ = 0: 中性模式

不稳定性机制:
  1. κ-机制 (不透明度机制): 在 He II 电离区, dκ/dT < 0 驱动脉动
  2. ε-机制 (核反应机制): 温度敏感产能 ε ∝ T^ν
  3. 对流不稳定性: Schwarzschild/Ledoux 判据

  判据 (非绝热分析):
    ∫₀ᴿ κ_T (δT/T)(δρ/ρ) dr > 0  →  驱动

  其中 κ_T = (∂lnκ/∂lnT)_ρ 为不透明度的温度灵敏度.

分岔分析:
  当恒星演化穿越不稳定带时, 振荡模式的稳定性发生变化.
  这对应于动力系统理论中的分岔:

  Hopf 分岔: 实部穿越零点, 出现极限环脉动
  Border-collision: 分段线性系统中模式突然切换

本模块实现:
  1. 非绝热稳定性矩阵构造
  2. 复频率求解 (QR 算法)
  3. 分岔检测 (border-collision)
  4. 非线性脉动振幅方程
"""

import numpy as np
from typing import Tuple, Dict, Optional


class NonAdiabaticStabilityMatrix:
    """
    非绝热稳定性矩阵.

    在准绝热近似 (QAA) 下, 增长率 γ 可以估算为:

    γ = -1/(2ω² E_k) ∫₀ᴹ [(Γ₃-1)(ρ δε - δL_r'/dm)] dm

    其中 E_k 是模式动能:
      E_k = 1/2 ∫ |ξ|² dm

    简化实现: 构造 2N 维状态空间矩阵, 通过其特征值判定稳定性.

    状态向量:
      Y = [y₁, y₂, ..., y_N, ẏ₁, ẏ₂, ..., ẏ_N]

    其中 y_i 是第 i 个网格点上的径向位移.

    运动方程:
      M ÿ + C ẏ + K y = 0

    →  Ẏ = A Y,  其中 A = [[0, I], [-M⁻¹K, -M⁻¹C]]

    参数
    ----
    n_modes : int
        径向网格点数 (自由度)
    stellar_model : object
        恒星结构模型
    """

    def __init__(self, n_modes: int, stellar_model):
        if n_modes < 4:
            raise ValueError(f"n_modes={n_modes} 至少需要 4")
        self.n = n_modes
        self.model = stellar_model

        # 构造质量、阻尼、刚度矩阵
        self.M = self._build_mass_matrix()
        self.C = self._build_damping_matrix()
        self.K = self._build_stiffness_matrix()

    def _build_mass_matrix(self) -> np.ndarray:
        """
        构造质量矩阵 M (对角).

        M_{ii} = Δm_i = 4π r_i² ρ_i Δr

        Parameters
        ----------
        (uses stellar_model)

        Returns
        -------
        M : ndarray, shape (n, n)
        """
        r = self.model.r[:self.n]
        rho = self.model.rho[:self.n]
        dr = self.model.dr

        M_diag = 4.0 * PI * r**2 * rho * dr
        M_diag[0] = max(M_diag[0], 1e-30)  # 中心保护

        return np.diag(M_diag)

    def _build_damping_matrix(self) -> np.ndarray:
        """
        构造阻尼矩阵 C (来自非绝热效应).

        C_{ii} = -2 γ_i M_{ii}

        其中 γ_i 是局部增长率, 由非绝热分析给出:

        γ ∝ (Γ₃-1) ρ κ_T / (c_s T) × (dL_r/dr)

        对于 κ-机制, 在电离区:
          κ_T = ∂lnκ/∂lnT < 0  →  正阻尼 (稳定)
          κ_ρ = ∂lnκ/∂lnρ > 0  →  驱动 (不稳定)

        Returns
        -------
        C : ndarray, shape (n, n)
        """
        n = self.n
        r = self.model.r[:n]
        T = self.model.T[:n]
        rho = self.model.rho[:n]
        Gamma1 = self.model.Gamma1[:n]
        Gamma3 = 1.0 + (Gamma1 - 1.0)  # 简化: Γ₃ - 1 = Γ₁ - 1

        # 不透明度温度灵敏度 (Kramers 不透明度: κ ∝ ρ T^{-3.5})
        kappa_T = -3.5  # dlnκ/dlnT (Kramers)
        kappa_rho = 1.0  # dlnκ/dlnρ

        # 局部增长率 (简化模型)
        gamma_local = np.zeros(n)
        for i in range(1, n - 1):
            if T[i] > 0 and rho[i] > 0:
                # 热驱动力 (简化)
                # γ ∝ -(Γ₃-1) κ_T ε_nuc / (c_v T)
                eps_nuc = 1e-3 * (T[i] / 1e7)**4  # 简化产能率
                c_v = 1.5 * K_BOLTZMANN / (self.model.mu * M_H)
                gamma_local[i] = (Gamma3[i] - 1.0) * kappa_T * eps_nuc / (
                    c_v * max(T[i], 1e10)
                )

        # 阻尼矩阵: 对角
        M_diag = np.diag(self.M)
        C_diag = -2.0 * gamma_local * M_diag
        return np.diag(C_diag)

    def _build_stiffness_matrix(self) -> np.ndarray:
        """
        构造刚度矩阵 K (弹性恢复力).

        K 来自压强梯度和重力的扰动:
          K_{ij} = ∫ (Γ₁ P / ρ) ∇φ_i · ∇φ_j dr
                 + 重力项 (A 矩阵贡献)

        使用三对角差分近似:
          K = -D(Γ₁P/ρ D) + 重力修正

        Returns
        -------
        K : ndarray, shape (n, n)
        """
        n = self.n
        dr = self.model.dr
        r = self.model.r[:n]
        Gamma1 = self.model.Gamma1[:n]
        P = self.model.P[:n]
        rho = self.model.rho[:n]
        N2 = self.model.N2[:n]
        g = self.model.g[:n]

        # 声速²
        c_s2 = Gamma1 * P / np.maximum(rho, 1e-100)

        # 三对角刚度矩阵
        K = np.zeros((n, n))
        for i in range(1, n - 1):
            r_i = max(r[i], 1e-10)
            # 主对角
            K[i, i] = (c_s2[i + 1] + c_s2[i]) / (2.0 * dr**2) \
                      + (c_s2[i] + c_s2[i - 1]) / (2.0 * dr**2) \
                      - N2[i]  # 浮力项
            # 次对角
            K[i, i - 1] = -(c_s2[i] + c_s2[i - 1]) / (2.0 * dr**2)
            K[i, i + 1] = -(c_s2[i + 1] + c_s2[i]) / (2.0 * dr**2)

            # 球面几何修正
            K[i, i] += c_s2[i] * 2.0 / (r_i * dr)

        # 边界
        K[0, 0] = K[1, 1]
        K[-1, -1] = K[-2, -2]

        # 乘以质量元
        dm = np.diag(self.M)
        for i in range(n):
            if dm[i] > 1e-30:
                K[i, :] *= dm[i]

        return K

    def compute_complex_frequencies(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        求解复频率 (特征值问题).

        Ẏ = A Y,  A = [[0, I], [-M⁻¹K, -M⁻¹C]]

        特征值 λ = -iω + γ:
          ω = -Im(λ): 振荡频率
          γ = Re(λ): 增长率

        Returns
        -------
        omega : ndarray
            复频率 ω = ω_R + iγ
        eigenvalues : ndarray
            原始特征值
        """
        n = self.n

        # 求逆
        M_inv = np.diag(1.0 / np.maximum(np.diag(self.M), 1e-30))

        # 构造 2n × 2n 状态空间矩阵
        A = np.zeros((2 * n, 2 * n))
        A[:n, n:] = np.eye(n)
        A[n:, :n] = -M_inv @ self.K
        A[n:, n:] = -M_inv @ self.C

        # QR 算法求解特征值
        eigenvalues = np.linalg.eigvals(A)

        # 转换为复频率: λ = γ - iω → ω_complex = ω_R + iγ
        omega = -np.imag(eigenvalues) + 1j * np.real(eigenvalues)

        # 过滤物理模式 (正频率)
        return omega, eigenvalues

    def growth_rate_spectrum(self) -> Dict[str, np.ndarray]:
        """
        获取增长率谱.

        Returns
        -------
        result : dict
            omega_real: 实频率 [rad/s]
            gamma: 增长率 [s⁻¹]
            is_stable: 稳定性标志
        """
        omega, _ = self.compute_complex_frequencies()

        # 提取正频率模式
        pos_mask = np.real(omega) > 0
        omega_pos = omega[pos_mask]

        # 去重 (共轭对)
        unique_omega = []
        seen = set()
        for w in omega_pos:
            w_rounded = round(np.real(w), 6)
            if w_rounded not in seen:
                seen.add(w_rounded)
                unique_omega.append(w)
        unique_omega = np.array(unique_omega)

        return {
            "omega_real": np.real(unique_omega),
            "gamma": np.imag(unique_omega),
            "is_stable": np.imag(unique_omega) < 0,
        }


class BorderCollisionBifurcationDetector:
    """
    Border-collision 分岔检测器.

    融合 border-collision-bifurcation (1085) 项目:
    当参数 μ 变化时, 分段线性系统可能发生 border-collision 分岔,
    即不动点穿过分段边界时发生突然的模式切换.

    在恒星振荡中, 这对应于:
    1. 演化过程中恒星穿越 HR 图的不稳定带
    2. 对流边界移动导致模式特征函数突然改变
    3. 化学成分跃变导致 g 模和 p 模的耦合/反交叉

    检测条件 (Núñez et al. 2019):
    对于分段线性映射 x_{n+1} = f(x_n, μ),
    当 μ 穿过 μ_c 时, 若:
    - 左侧不动点 x_L(μ_c) 与右侧不动点 x_R(μ_c) 碰撞于边界
    - 两侧斜率乘积 s_L × s_R ≠ 1  (非光滑)
    则发生 border-collision 分岔

    参数
    ----
    n_steps : int
        参数扫描步数
    """

    def __init__(self, n_steps: int = 200):
        self.n_steps = n_steps

    def detect_anticrossing(
        self,
        freq_p: np.ndarray,
        freq_g: np.ndarray,
        coupling_strength: float = 0.1,
    ) -> Dict[str, any]:
        """
        检测 p 模和 g 模的反交叉 (avoided crossing).

        当 p 模频率 ν_p 和 g 模频率 ν_g 接近时,
        耦合导致频率排斥:

        ν_± = (ν_p + ν_g)/2 ± √((ν_p - ν_g)/2)² + q²

        其中 q 为耦合强度.

        在反交叉点:
        - 模式特征函数混合
        - 频率间距最小 = 2q
        - 类似于量子力学中的避免交叉

        Parameters
        ----------
        freq_p : ndarray
            p 模频率序列
        freq_g : ndarray
            g 模频率序列
        coupling_strength : float
            耦合强度 q

        Returns
        -------
        result : dict
            avoided_crossings: 反交叉事件列表
            min_spacing: 最小频率间距
            critical_params: 临界参数值
        """
        q = coupling_strength
        crossings = []

        for fp in freq_p:
            for fg in freq_g:
                delta = fp - fg
                # 耦合后的频率
                nu_plus = (fp + fg) / 2.0 + np.sqrt(delta**2 / 4.0 + q**2)
                nu_minus = (fp + fg) / 2.0 - np.sqrt(delta**2 / 4.0 + q**2)

                # 反交叉条件: 未耦合间距 < 某个阈值
                if abs(delta) < 5.0 * q:
                    crossings.append({
                        "freq_p": fp,
                        "freq_g": fg,
                        "delta": delta,
                        "nu_plus": nu_plus,
                        "nu_minus": nu_minus,
                        "min_spacing": 2.0 * q,
                    })

        return {
            "avoided_crossings": crossings,
            "n_crossings": len(crossings),
            "min_spacing": 2.0 * q,
            "coupling_strength": q,
        }

    def iterate_piecewise_map(
        self,
        mu: float,
        tau: float,
        x0: float,
        n_iter: int = 1000,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        迭代分段线性映射 (融合 1085 项目的核心算法).

        x_{n+1} = f(x_n, μ) = -x_n + μ + b·sgn(x_{n-τ})

        这是驱动时间延迟系统的简化模型.

        在恒星振荡中的对应:
        - x: 径向位移
        - μ: 控制参数 (光度/温度)
        - τ: 热时标延迟 (脉冲星/δ Scuti 星中的 κ-机制)

        Parameters
        ----------
        mu : float
            控制参数
        tau : float
            延迟时间 (无量纲)
        x0 : float
            初始条件
        n_iter : int
            迭代次数

        Returns
        -------
        X : ndarray
            轨道序列
        T : ndarray
            时间序列
        """
        # 历史缓冲区 (延迟)
        history_len = max(int(tau * 100), 10)
        history = [x0] * history_len
        b = 0.5  # 反馈强度

        X = [x0]
        T = [0.0]
        x = x0

        for step in range(n_iter):
            # 反馈项 (历史延迟)
            hist_idx = max(0, len(history) - history_len)
            x_delayed = history[hist_idx]

            # 分段线性映射
            sign_drive = 1.0 if (step % 2 == 0) else -1.0
            feedback = np.sign(x) * (-1) ** (step + 1)
            drive = feedback + b * sign_drive

            # 时间延迟项
            feedback_delay = -np.sign(x_delayed) * (-1) ** (step + 1)

            # 更新
            dt = 0.01
            x_new = x + dt * (drive + feedback_delay * mu)

            # 数值保护
            x_new = np.clip(x_new, -100.0, 100.0)

            history.append(x_new)
            X.append(x_new)
            T.append((step + 1) * dt)
            x = x_new

        return np.array(X), np.array(T)

    def scan_bifurcation(
        self,
        mu_range: Tuple[float, float],
        tau: float = 0.95,
        x0: float = 0.01,
    ) -> Dict[str, np.ndarray]:
        """
        扫描分岔图.

        Parameters
        ----------
        mu_range : tuple
            参数范围 (mu_min, mu_max)
        tau : float
            延迟参数
        x0 : float
            初始条件

        Returns
        -------
        result : dict
            mu_values: 参数值
            attractors: 每个参数下的吸引子点
        """
        mu_vals = np.linspace(mu_range[0], mu_range[1], self.n_steps)
        attractors = []

        for mu in mu_vals:
            # 预热 (丢弃瞬态)
            X_burn, _ = self.iterate_piecewise_map(mu, tau, x0, n_iter=500)
            # 收集吸引子
            X_attr, _ = self.iterate_piecewise_map(
                mu, tau, X_burn[-1], n_iter=200
            )
            # 取最后 50 个点
            attractors.append(X_attr[-50:])

        return {
            "mu_values": mu_vals,
            "attractors": attractors,
        }


class PulsationAmplitudeEquation:
    """
    脉动振幅方程 (非线性稳定性).

    对于接近分岔点的模式, 振幅演化满足 Landau 方程:

    dA/dt = σ A - l |A|² A

    其中:
      σ = γ + iδω  (线性增长率 + 频率修正)
      l = l_R + il_I  (非线性饱和系数)

    平衡振幅:
      |A|²_eq = γ / l_R  (超临界 Hopf 分岔)

    多模耦合 (Goupil & Buchler 1994):
    dA_j/dt = σ_j A_j - l_j |A_j|² A_j - Σ_{k≠j} q_{jk} |A_k|² A_j

    其中 q_{jk} 是模间耦合系数.

    参数
    ----
    n_modes : int
        耦合模式数量
    """

    def __init__(self, n_modes: int = 3):
        self.n = n_modes

    def evolve_amplitude(
        self,
        sigma: np.ndarray,
        l_coeff: np.ndarray,
        q_coupling: Optional[np.ndarray] = None,
        A0: Optional[np.ndarray] = None,
        dt: float = 0.01,
        n_steps: int = 10000,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        时间积分振幅方程.

        dA_j/dt = σ_j A_j - l_j |A_j|² A_j - Σ_k q_{jk} |A_k|² A_j

        使用 RK4 方法.

        Parameters
        ----------
        sigma : ndarray, shape (n,)
            线性增长率 (复数)
        l_coeff : ndarray, shape (n,)
            非线性饱和系数 (复数)
        q_coupling : ndarray, shape (n, n), optional
            模间耦合系数
        A0 : ndarray, shape (n,), optional
            初始振幅
        dt : float
            时间步长
        n_steps : int
            积分步数

        Returns
        -------
        A : ndarray, shape (n_steps, n)
            振幅时间序列
        T : ndarray, shape (n_steps,)
            时间
        """
        n = self.n

        if A0 is None:
            A0 = 0.01 * np.ones(n, dtype=complex)

        if q_coupling is None:
            q_coupling = 0.1 * np.ones((n, n))
            np.fill_diagonal(q_coupling, 0.0)

        T = np.arange(n_steps) * dt
        A = np.zeros((n_steps, n), dtype=complex)
        A[0] = A0

        def rhs(A_curr):
            dA = np.zeros(n, dtype=complex)
            for j in range(n):
                dA[j] = sigma[j] * A_curr[j]
                dA[j] -= l_coeff[j] * np.abs(A_curr[j])**2 * A_curr[j]
                for k in range(n):
                    if k != j:
                        dA[j] -= q_coupling[j, k] * np.abs(A_curr[k])**2 * A_curr[j]
            return dA

        # RK4 积分
        for step in range(n_steps - 1):
            A_curr = A[step]

            k1 = rhs(A_curr)
            k2 = rhs(A_curr + dt * k1 / 2.0)
            k3 = rhs(A_curr + dt * k2 / 2.0)
            k4 = rhs(A_curr + dt * k3)

            A_new = A_curr + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)

            # 数值保护: 限制振幅
            amp = np.abs(A_new)
            max_amp = 1.0  # 物理上限
            for j in range(n):
                if amp[j] > max_amp:
                    A_new[j] *= max_amp / amp[j]

            A[step + 1] = A_new

        return A, T


# ============================================================
# 物理常数 (局部定义, 避免循环导入)
# ============================================================
PI = np.pi
G_CGS = 6.67430e-8
K_BOLTZMANN = 1.380649e-16
M_H = 1.6735575e-24
