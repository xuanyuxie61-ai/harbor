"""
磁电耦合熵迹分析模块
=====================
对应种子项目: 1147_dharmavira_Hilbert-Smith (熵迹模拟 → 序参量涨落熵)

物理背景:
    多铁性材料中, 极化和磁化的涨落携带系统的信息熵.
    定义磁电耦合熵迹 Θ(t) 来表征系统趋向平衡的过程:

    Θ(t) = ∫ dω · [Σᵢ ψᵢ(ω)]² · exp(-t·ω²)

    其中 ψᵢ(ω) 为序参量涨落的频率分量:
        ψᵢ(ω) = wᵢ · cos(φᵢ·ω) · exp(-α·ω²)

    φᵢ 为各自由度的相位 (与畴壁振动模式相关),
    wᵢ 为权重 (与模式振幅相关).

    物理含义:
        Θ(t) 大 → 系统有序 (低频模式主导)
        Θ(t) 小 → 系统无序 (高频模式主导)

    在相变点附近, Θ(t) 的行为变化可检测临界现象.

方法论 (对应种子项目 1147):
    1. 生成 N 个符号点, 每个具有随机相位 φᵢ
    2. 预计算 ψᵢ(γ) = wᵢ·cos(φᵢ·γ)·exp(-α·γ²)
    3. 对每个 t, 数值积分 Θ(t) = trapz((Σψᵢ)²·exp(-tγ²), γ)
    4. 分析 Θ(t) 的标度行为

核心公式:
    von Neumann 熵:
        S = -Tr(ρ·ln ρ)
    对于高斯分布:
        S = ½·ln(2πe·σ²)

    互信息:
        I(P;M) = S(P) + S(M) - S(P,M)
    表征 P 和 M 之间的磁电关联强度.
"""

import numpy as np


class MagnetoelectricEntropyTrace:
    """
    磁电耦合熵迹分析器.

    计算序参量涨落的熵随平滑参数 t 的变化,
    用于检测相变和磁电耦合强度.
    """

    def __init__(self, n_modes=256, alpha_gaussian=0.5, seed=285):
        """
        参数:
            n_modes: 模式数 N
            alpha_gaussian: 高斯包络参数 α
            seed: 随机种子
        """
        self.n_modes = n_modes
        self.alpha = alpha_gaussian
        self.rng = np.random.RandomState(seed)

        # 生成模式参数
        self.phases = None
        self.weights = None
        self._init_modes()

    def _init_modes(self):
        """
        初始化涨落模式.

        相位 φᵢ: 均匀分布于 [-π, π]
        权重 wᵢ: 指数衰减分布 (低频模式权重更大)
        """
        # 相位: 均匀分布
        self.phases = self.rng.uniform(-np.pi, np.pi, self.n_modes)

        # 权重: 指数衰减
        indices = np.arange(self.n_modes)
        self.weights = np.exp(-indices / (self.n_modes / 4))
        self.weights /= np.sum(self.weights)  # 归一化

    # ============================================================
    # 熵迹计算 (对应种子项目 1147)
    # ============================================================

    def compute_entropy_trace(self, t_values, gamma_max=10.0,
                              n_gamma=512):
        """
        计算熵迹 Θ(t).

        Θ(t) = ∫₀^{γ_max} dγ · [Σᵢ ψᵢ(γ)]² · exp(-t·γ²)

        其中 ψᵢ(γ) = wᵢ · cos(φᵢ·γ) · exp(-α·γ²)

        对应种子项目 1147 的核心计算:
        1. 预计算 ψᵢ(γ) 矩阵
        2. 对每个 t, 计算 Θ(t)

        参数:
            t_values: 平滑参数数组
            gamma_max: γ 积分上限
            n_gamma: γ 离散点数

        返回:
            t_values: 平滑参数
            theta: 熵迹值数组
        """
        gamma = np.linspace(0, gamma_max, n_gamma)
        d_gamma = gamma[1] - gamma[0]

        # 预计算 ψᵢ(γ) = wᵢ · cos(φᵢ·γ) · exp(-α·γ²)
        psi_matrix = np.zeros((self.n_modes, n_gamma))
        gaussian_env = np.exp(-self.alpha * gamma ** 2)

        for i in range(self.n_modes):
            psi_matrix[i] = (self.weights[i] *
                             np.cos(self.phases[i] * gamma) *
                             gaussian_env)

        # 总和: Σᵢ ψᵢ(γ)
        psi_sum = np.sum(psi_matrix, axis=0)
        psi_sum_sq = psi_sum ** 2

        # 对每个 t 计算 Θ(t)
        theta = np.zeros(len(t_values))
        for k, t in enumerate(t_values):
            # Θ(t) = ∫ (Σψᵢ)² · exp(-t·γ²) dγ
            integrand = psi_sum_sq * np.exp(-t * gamma ** 2)
            theta[k] = np.trapz(integrand, gamma)

        return t_values, theta

    # ============================================================
    # 更新模式参数 (从模拟数据)
    # ============================================================

    def update_from_simulation(self, P_field_history, M_field_history):
        """
        从模拟轨迹更新模式参数.

        从 P(t) 和 M(t) 的时间序列中提取频率分量.

        步骤:
            1. 对 P(t) 和 M(t) 做 FFT
            2. 提取主要频率和相位
            3. 更新 self.phases 和 self.weights

        参数:
            P_field_history: P 时间序列, shape (n_steps, 3) 或 (n_steps,)
            M_field_history: M 时间序列, shape (n_steps, 3) 或 (n_steps,)
        """
        # 计算总序参量
        if P_field_history.ndim == 2:
            P_total = np.linalg.norm(P_field_history, axis=1)
        else:
            P_total = P_field_history

        if M_field_history.ndim == 2:
            M_total = np.linalg.norm(M_field_history, axis=1)
        else:
            M_total = M_field_history

        # FFT
        n_steps = len(P_total)
        n_freq = min(n_steps // 2, self.n_modes)

        P_fft = np.fft.rfft(P_total - np.mean(P_total))
        M_fft = np.fft.rfft(M_total - np.mean(M_total))

        # 功率谱
        P_power = np.abs(P_fft[:n_freq])
        M_power = np.abs(M_fft[:n_freq])
        ME_power = 0.5 * (P_power + M_power)

        # 相位
        ME_phase = np.angle(P_fft[:n_freq] * np.conj(M_fft[:n_freq]))

        # 更新
        if n_freq > 0:
            self.phases[:n_freq] = ME_phase
            total_power = np.sum(ME_power)
            if total_power > 1e-30:
                self.weights[:n_freq] = ME_power / total_power

    # ============================================================
    # 信息论分析
    # ============================================================

    def mutual_information(self, P_samples, M_samples, n_bins=20):
        """
        计算极化与磁化之间的互信息.

        I(P;M) = H(P) + H(M) - H(P,M)

        其中 H 为 Shannon 熵:
            H(X) = -Σ p(x)·ln(p(x))

        参数:
            P_samples: 极化样本, shape (n,)
            M_samples: 磁化样本, shape (n,)
            n_bins: 直方图箱数

        返回:
            I_PM: 互信息 (nats)
            H_P: P 的熵
            H_M: M 的熵
            H_PM: 联合熵
        """
        # 2D 直方图
        hist_2d, x_edges, y_edges = np.histogram2d(
            P_samples, M_samples, bins=n_bins
        )

        # 归一化为概率
        p_xy = hist_2d / np.sum(hist_2d)
        p_xy = p_xy[p_xy > 0]
        H_PM = -np.sum(p_xy * np.log(p_xy + 1e-30))

        # 边缘分布
        p_x = np.sum(hist_2d, axis=1)
        p_x = p_x[p_x > 0]
        p_x = p_x / np.sum(p_x)
        H_P = -np.sum(p_x * np.log(p_x + 1e-30))

        p_y = np.sum(hist_2d, axis=0)
        p_y = p_y[p_y > 0]
        p_y = p_y / np.sum(p_y)
        H_M = -np.sum(p_y * np.log(p_y + 1e-30))

        I_PM = H_P + H_M - H_PM

        return max(I_PM, 0.0), H_P, H_M, H_PM

    def von_neumann_entropy(self, rho):
        """
        计算 von Neumann 熵.

        S = -Tr(ρ·ln ρ) = -Σ λᵢ·ln(λᵢ)

        其中 λᵢ 为密度矩阵 ρ 的本征值.

        参数:
            rho: 密度矩阵, shape (d, d)

        返回:
            S: von Neumann 熵 (nats)
        """
        eigenvalues = np.linalg.eigvalsh(rho)
        eigenvalues = eigenvalues[eigenvalues > 1e-30]
        S = -np.sum(eigenvalues * np.log(eigenvalues))
        return max(S, 0.0)

    # ============================================================
    # 标度分析
    # ============================================================

    def scaling_exponent(self, t_values, theta_values, t_range=None):
        """
        分析 Θ(t) 的标度行为.

        Θ(t) ~ t^{-β}  (幂律标度)

        通过 ln Θ vs ln t 的线性回归得到 β.

        物理含义:
            β 与系统的关联长度指数 ν 相关:
            β = d·ν/2 (d 为有效维数)

        参数:
            t_values: 平滑参数
            theta_values: 熵迹值
            t_range: 拟合范围 (t_min, t_max)

        返回:
            beta: 标度指数
            r_squared: 拟合优度
        """
        from scipy.stats import linregress

        if t_range is not None:
            mask = (t_values >= t_range[0]) & (t_values <= t_range[1])
            t_fit = t_values[mask]
            theta_fit = theta_values[mask]
        else:
            t_fit = t_values
            theta_fit = theta_values

        # 过滤正值
        valid = (t_fit > 0) & (theta_fit > 0)
        if np.sum(valid) < 3:
            return 0.0, 0.0

        ln_t = np.log(t_fit[valid])
        ln_theta = np.log(theta_fit[valid])

        result = linregress(ln_t, ln_theta)
        beta = -result.slope  # 负号因为 Θ ~ t^{-β}
        r_squared = result.rvalue ** 2

        return beta, r_squared
