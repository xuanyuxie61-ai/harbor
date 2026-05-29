"""
stochastic_perturbation.py
==========================
随机扰动与热噪声模块

融合种子项目：
- 029_asa053: Wishart 分布 / 正态随机数生成
- 060_axon_ode: Hodgkin-Huxley 模型中的门控变量随机动力学思想

核心内容：
1. Box-Muller / Marsaglia 方法生成标准正态随机数
2. 热涨落噪声的相场模型（Fluctuation-Dissipation Theorem）
3. Wishart 分布协方差矩阵采样
4. 随机相位场方程（Stochastic Allen-Cahn）
5. 噪声强度标度与界面宽度关系

随机 Allen-Cahn 方程：
    τ ∂φ/∂t = ε²∇²φ - W'(φ) + ξ(x,t)

其中热噪声 ξ(x,t) 满足：
    <ξ(x,t) ξ(x',t')> = 2k_B T M_φ δ(x-x') δ(t-t')

离散化后，每时间步在每个网格点添加独立高斯噪声：
    ξ_{i,j}^n ~ N(0, σ²)
    σ² = 2k_B T M_φ / (Δx Δy Δt)
"""

import numpy as np


class RandomNumberGenerator:
    """
    标准随机数生成器。
    基于种子项目 029_asa053 的 rnorm 方法。
    """

    @staticmethod
    def rnorm_marsaglia():
        """
        Marsaglia polar method 生成两个独立标准正态随机数。

        算法：
            1. 在单位圆 [-1,1]² 内均匀采样 (U1, U2)
            2. 计算 S = U1² + U2²
            3. 若 S < 1，返回 Z1 = U1 * sqrt(-2 ln S / S), Z2 = U2 * sqrt(-2 ln S / S)

        Returns
        -------
        tuple
            (z1, z2) 两个标准正态随机数。
        """
        while True:
            u1 = np.random.uniform(-1.0, 1.0)
            u2 = np.random.uniform(-1.0, 1.0)
            s = u1 ** 2 + u2 ** 2
            if 0 < s <= 1.0:
                break

        factor = np.sqrt(-2.0 * np.log(s) / s)
        return u1 * factor, u2 * factor

    @staticmethod
    def standard_normal_array(shape):
        """
        生成指定形状的标准正态随机数组。

        Parameters
        ----------
        shape : tuple
            输出数组形状。

        Returns
        -------
        ndarray
            标准正态随机数组。
        """
        return np.random.standard_normal(shape)


class WishartSampler:
    """
    Wishart 分布采样器。
    基于种子项目 029_asa053 的 wshrt 方法。

    Wishart 分布 W_p(n, Σ) 是协方差矩阵的分布。
    若 X_1, ..., X_n ~ N_p(0, Σ)，则 S = Σ X_i X_i^T ~ W_p(n, Σ)。
    """

    @staticmethod
    def wishart_variate(Sigma, n, np_size):
        """
        生成 Wishart 分布随机矩阵。

        简化实现：通过 Cholesky 分解和正态随机数生成。

        Parameters
        ----------
        Sigma : ndarray
            协方差矩阵。
        n : int
            自由度。
        np_size : int
            矩阵维度。

        Returns
        -------
        ndarray
            Wishart 随机矩阵。
        """
        # Cholesky 分解
        L = np.linalg.cholesky(Sigma)

        # 生成 n 个 N(0, I) 向量
        X = np.random.standard_normal((np_size, n))

        # 变换为 N(0, Sigma)
        Y = L @ X

        # Wishart 矩阵
        W = Y @ Y.T

        return W


class ThermalNoise:
    """
    热噪声生成器，用于随机相场模型。
    """

    def __init__(self, nx, ny, dx, dy, dt, kbt=0.01, mobility=1.0):
        """
        初始化热噪声参数。

        Parameters
        ----------
        nx, ny : int
            网格点数。
        dx, dy : float
            空间步长。
        dt : float
            时间步长。
        kbt : float
            热能量 k_B T（无量纲）。
        mobility : float
            界面迁移率 M_φ。
        """
        self.nx = nx
        self.ny = ny
        self.dx = dx
        self.dy = dy
        self.dt = dt
        self.kbt = kbt
        self.mobility = mobility

        # 计算噪声标准差
        # σ² = 2 k_B T M_φ / (dx dy dt)
        self.noise_std = np.sqrt(
            2.0 * kbt * mobility / (dx * dy * dt)
        )

    def generate_white_noise(self):
        """
        生成空间-时间白噪声场。

        Returns
        -------
        ndarray
            噪声场，形状 (nx, ny)。
        """
        return self.noise_std * np.random.standard_normal((self.nx, self.ny))

    def generate_colored_noise(self, correlation_length=2.0):
        """
        生成空间相关噪声（有色噪声）。

        通过白噪声与高斯核卷积实现空间相关性：
            ξ_c(x) = ∫ G(x-x') ξ_w(x') dx'
        其中 G(r) = exp(-r²/(2l_c²)) / (√(2π) l_c)。

        Parameters
        ----------
        correlation_length : float
            相关长度 l_c（以网格点数为单位）。

        Returns
        -------
        ndarray
            有色噪声场。
        """
        white = self.generate_white_noise()

        # 构建高斯核
        size = int(3 * correlation_length) + 1
        x = np.arange(-size, size + 1)
        y = np.arange(-size, size + 1)
        X, Y = np.meshgrid(x, y, indexing='ij')
        kernel = np.exp(-(X ** 2 + Y ** 2) / (2.0 * correlation_length ** 2))
        kernel = kernel / np.sum(kernel)

        # 2D 卷积
        from scipy.signal import convolve2d
        colored = convolve2d(white, kernel, mode='same', boundary='fill')

        return colored

    def apply_to_phase_field(self, phi_rhs, noise_type='white', **kwargs):
        """
        将热噪声添加到相场方程右端项。

        Parameters
        ----------
        phi_rhs : ndarray
            相场方程右端项。
        noise_type : str
            'white' 或 'colored'。
        **kwargs
            传递给噪声生成器的额外参数。

        Returns
        -------
        ndarray
            添加噪声后的右端项。
        """
        if noise_type == 'white':
            noise = self.generate_white_noise()
        elif noise_type == 'colored':
            noise = self.generate_colored_noise(**kwargs)
        else:
            raise ValueError(f"不支持的噪声类型: {noise_type}")

        return phi_rhs + noise


class StochasticAllenCahn:
    """
    随机 Allen-Cahn 方程求解器。

    方程：
        τ ∂φ/∂t = ε²∇²φ - W'(φ) + √(2k_B T M_φ) η(x,t)

    其中 η(x,t) 为空间-时间白噪声。
    """

    def __init__(self, phase_field_model, thermal_noise):
        """
        初始化随机相场求解器。

        Parameters
        ----------
        phase_field_model : PhaseFieldModel
            相场模型实例。
        thermal_noise : ThermalNoise
            热噪声生成器。
        """
        self.pf = phase_field_model
        self.noise = thermal_noise

    def rhs_with_noise(self, phi, T, C, velocity_x=None, velocity_y=None,
                       noise_type='white'):
        """
        计算带噪声的相场方程右端项。

        Parameters
        ----------
        phi : ndarray
            序参量场。
        T : ndarray
            温度场。
        C : ndarray
            浓度场。
        velocity_x, velocity_y : ndarray, optional
            速度场。
        noise_type : str
            噪声类型。

        Returns
        -------
        ndarray
            带噪声的右端项。
        """
        rhs = self.pf.phase_field_rhs(phi, T, C, velocity_x, velocity_y)
        rhs = self.noise.apply_to_phase_field(rhs, noise_type=noise_type)
        return rhs


class HodgkinHuxleyInspiredGating:
    """
    基于 Hodgkin-Huxley 思想的相变门控动力学。

    将 HH 模型的门控变量思想应用于相变过程：
    定义 "相变激活度" 变量，描述原子在晶格位置上的跃迁概率。

    类比 HH 模型的门控变量 n, m, h：
        α_n(V) = 0.01(10-V)/(exp((10-V)/10)-1)
        β_n(V) = 0.125 exp(-V/80)

    相变版本（以过冷度 ΔT 为驱动力）：
        α(ΔT) = α_0 exp(-Q_a/(k_B T)) * f(ΔT)
        β(ΔT) = β_0 exp(-Q_d/(k_B T)) * g(ΔT)
    """

    def __init__(self, alpha0=1.0, beta0=1.0, Qa=1.0, Qd=1.0, kbt=0.1):
        """
        初始化门控动力学参数。

        Parameters
        ----------
        alpha0, beta0 : float
            前置因子。
        Qa, Qd : float
            激活能和去激活能。
        kbt : float
            热能量。
        """
        self.alpha0 = alpha0
        self.beta0 = beta0
        self.Qa = Qa
        self.Qd = Qd
        self.kbt = kbt

    def activation_rate(self, undercooling):
        """
        计算激活速率（固相→液相转变）。

        Parameters
        ----------
        undercooling : float
            过冷度 ΔT = T_M - T。

        Returns
        -------
        float
            激活速率。
        """
        thermal_factor = np.exp(-self.Qa / max(self.kbt, 1e-14))
        driving_force = max(undercooling, 0.0)
        return self.alpha0 * thermal_factor * driving_force

    def deactivation_rate(self, undercooling):
        """
        计算去激活速率（液相→固相转变）。

        Parameters
        ----------
        undercooling : float
            过冷度。

        Returns
        -------
        float
            去激活速率。
        """
        thermal_factor = np.exp(-self.Qd / max(self.kbt, 1e-14))
        driving_force = max(undercooling, 0.0)
        return self.beta0 * thermal_factor * (1.0 + driving_force)

    def gating_variable_derivative(self, g, undercooling):
        """
        计算门控变量的时间导数：
            dg/dt = α(1 - g) - βg

        Parameters
        ----------
        g : float
            当前门控变量值（范围 [0,1]）。
        undercooling : float
            过冷度。

        Returns
        -------
        float
            dg/dt。
        """
        alpha = self.activation_rate(undercooling)
        beta = self.deactivation_rate(undercooling)
        return alpha * (1.0 - g) - beta * g
