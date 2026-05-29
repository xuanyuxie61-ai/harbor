"""
================================================================================
气候随机微分方程数值求解模块
================================================================================

融合来源：
  - 833_ode_trapezoidal:    隐式梯形法ODE求解
  - 1063_sde:               Euler-Maruyama / Milstein SDE求解
  - 856_peaks_movie:        周期性气候模态（peaks函数作为温度场基模态）

科学应用：
  模拟气候状态的随机演化，包括热扩散、牛顿冷却、外部强迫和内部变率。

核心公式：
  1. 能量平衡模型（EBM）:
     C_p ∂T/∂t = D ∇²T - λT + F_ext(t) + σ(T)·η(t)
     其中:
       C_p = 体积热容 (W·yr·m^{-2}·K^{-1})
       D   = 热扩散系数
       λ   = 气候反馈参数 (W·m^{-2}·K^{-1})
       F_ext = 外部辐射强迫
       η(t)  = 白噪声过程

  2. 空间离散后的矩阵形式:
     dT/dt = A·T + b(t) + Σ(T)·dW
     A = D·L - λ·I  (L 是球面拉普拉斯矩阵的离散近似)

  3. 梯形隐式法（ODE部分）:
     T_{n+1} = T_n + (h/2)·[f(T_n) + f(T_{n+1})]
     通过不动点迭代求解:
     z_{k+1} = T_n + (h/2)·[f(T_n) + f(z_k)]

  4. Milstein方法（SDE部分）:
     T_{n+1} = T_n + f(T_n)·Δt + g(T_n)·ΔW
               + 0.5·g(T_n)·g'(T_n)·(ΔW² - Δt)
     其中 g(T) = σ(T) 是扩散系数。

  5. Peaks气候模态（源自 856_peaks_movie）:
     将经典的 MATLAB peaks 函数改造为球面温度场基模态:
     P(λ,φ) = 3(1-φ')² exp(-φ'²-(λ'+1)²)
            - 10(φ'/5 - φ'³ - λ'⁵) exp(-φ'²-λ'²)
            - 1/3 exp(-(φ'+1)² - λ'²)
     其中 (λ', φ') 是经度/纬度的归一化坐标。
================================================================================
"""

import numpy as np


class ClimateForcing:
    """
    外部气候强迫生成器：包含周期性太阳活动、火山喷发和温室气体强迫。
    """

    def __init__(self, amplitude: float = 0.15, volcanic_freq: float = 0.03):
        self.amplitude = amplitude
        self.volcanic_freq = volcanic_freq
        self.cycle_periods = [11.0, 88.0, 210.0]  # 太阳周期、Gleissberg周期、de Vries周期

    def compute(self, t: float, n_grid: int):
        """
        计算时刻 t 的外部强迫场。
        F(t) = A·[sin(2πt/11) + 0.3·sin(2πt/88) + 0.1·sin(2πt/210)]
               - V·exp(-((t-t_v)/τ_v)²)
        """
        solar = 0.0
        for period in self.cycle_periods:
            solar += np.sin(2.0 * np.pi * t / period) / len(self.cycle_periods)

        # 随机火山喷发事件（泊松过程近似）
        volcanic = 0.0
        np.random.seed(int(t * 1000) % 2**31)
        if np.random.rand() < self.volcanic_freq:
            volcanic = -0.5 * np.exp(-0.01 * (t % 50.0) ** 2)

        forcing_scalar = self.amplitude * solar + volcanic
        return forcing_scalar * np.ones(n_grid, dtype=np.float64)


class StochasticClimateModel:
    """
    随机气候动力学模型。
    """

    def __init__(
        self,
        n_grid: int,
        diffusion_coeff: float = 1.2e-6,
        damping_coeff: float = 0.02,
        forcing_amplitude: float = 0.15,
        noise_intensity: float = 0.08,
        dt_years: float = 1.0,
    ):
        self.n_grid = n_grid
        self.D = diffusion_coeff
        self.lambda_ = damping_coeff
        self.noise_intensity = noise_intensity
        self.dt = dt_years
        self.forcing = ClimateForcing(amplitude=forcing_amplitude)

        # 构建简化拉普拉斯矩阵（基于最近邻扩散）
        self._build_laplacian()

    def _build_laplacian(self):
        """
        构建简化的球面拉普拉斯矩阵 L。
        使用环形拓扑近似：L_{ii} = -2, L_{i,i±1} = 1。
        对于球面，考虑周期性边界条件。
        """
        n = self.n_grid
        self.L = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            self.L[i, i] = -2.0
            left = (i - 1) % n
            right = (i + 1) % n
            self.L[i, left] = 1.0
            self.L[i, right] = 1.0
        # 归一化
        self.L = self.L / max(1.0, n / 100.0)

    def _drift(self, T: np.ndarray, t: float) -> np.ndarray:
        """
        漂移项 f(T) = D·L·T - λ·T + F_ext(t)。
        """
        forcing = self.forcing.compute(t, self.n_grid)
        return self.D * (self.L @ T) - self.lambda_ * T + forcing

    def _diffusion(self, T: np.ndarray) -> np.ndarray:
        """
        扩散系数 g(T) = σ · (1 + 0.1·|T|)。
        温度异常越大，内部变率越强。
        """
        return self.noise_intensity * (1.0 + 0.1 * np.abs(T))

    def _diffusion_derivative(self, T: np.ndarray) -> np.ndarray:
        """
        g'(T) = 0.1·σ·sign(T)  (当 T≠0 时)。
        """
        return self.noise_intensity * 0.1 * np.sign(T)

    def trapezoidal_step(self, T: np.ndarray, t: float) -> np.ndarray:
        """
        纯ODE梯形隐式步进（无随机项）。
        融合 833_ode_trapezoidal 的迭代求解思想。

        算法:
            z_0 = T_n
            z_{k+1} = T_n + (h/2)·[f(T_n) + f(z_k)]
            迭代至收敛
        """
        h = self.dt
        f_n = self._drift(T, t)
        z = T.copy()

        for _ in range(20):  # 最多20次迭代
            z_new = T + 0.5 * h * (f_n + self._drift(z, t + h))
            if np.linalg.norm(z_new - z) < 1e-12:
                z = z_new
                break
            z = z_new

        return z

    def euler_maruyama_step(self, T: np.ndarray, t: float) -> np.ndarray:
        """
        Euler-Maruyama 方法求解 SDE。
        融合 1063_sde/em 的算法。

        T_{n+1} = T_n + f(T_n)·Δt + g(T_n)·ΔW
        """
        dW = np.sqrt(self.dt) * np.random.randn(self.n_grid)
        drift = self._drift(T, t)
        diff = self._diffusion(T)
        return T + drift * self.dt + diff * dW

    def milstein_step(self, T: np.ndarray, t: float = 0.0) -> np.ndarray:
        """
        Milstein 方法求解 SDE（强阶 1.0）。
        融合 1063_sde/milstrong 的算法。

        T_{n+1} = T_n + f(T_n)·Δt + g(T_n)·ΔW
                  + 0.5·g(T_n)·g'(T_n)·(ΔW² - Δt)
        """
        dW = np.sqrt(self.dt) * np.random.randn(self.n_grid)
        drift = self._drift(T, t)
        diff = self._diffusion(T)
        diff_deriv = self._diffusion_derivative(T)

        # Milstein 修正项
        correction = 0.5 * diff * diff_deriv * (dW ** 2 - self.dt)

        return T + drift * self.dt + diff * dW + correction

    def initial_state(self) -> np.ndarray:
        """
        生成初始温度场：基态 + peaks模态扰动。
        融合 856_peaks_movie 的 peaks 函数作为气候模态。
        """
        n = self.n_grid
        # 基态温度场（随纬度变化的平均温度）
        lat = np.linspace(-90, 90, n)
        base_temp = 15.0 - 30.0 * np.sin(np.radians(lat)) ** 2

        # Peaks 模态扰动（经向波状结构）
        lon = np.linspace(-180, 180, n)
        x = lon / 60.0
        y = lat / 30.0

        peaks = (
            3.0 * (1.0 - x) ** 2 * np.exp(-x ** 2 - (y + 1.0) ** 2)
            - 10.0 * (x / 5.0 - x ** 3 - y ** 5) * np.exp(-x ** 2 - y ** 2)
            - (1.0 / 3.0) * np.exp(-(x + 1.0) ** 2 - y ** 2)
        )
        # 缩放peaks幅度
        peaks = 0.3 * peaks

        return base_temp + peaks

    def strong_convergence_test(self, T0: np.ndarray, t_final: float = 1.0):
        """
        Milstein 方法的强收敛性检验。
        参考 1063_sde/milstrong 的收敛分析。
        """
        n_ref = 2 ** 11
        dt_ref = t_final / n_ref
        m = 200  # 路径数

        # 生成参考布朗增量
        dW_ref = np.sqrt(dt_ref) * np.random.randn(m, n_ref)

        r_values = [1, 16, 32, 64, 128]
        errors = []

        for r in r_values[1:]:
            dt = r * dt_ref
            L = n_ref // r
            x_mil = np.zeros(m)

            for p in range(m):
                x_temp = T0[0]  # 单点测试
                for j in range(L):
                    winc = np.sum(dW_ref[p, r * j:r * (j + 1)])
                    # 简化的Milstein（标量版本）
                    drift = -self.lambda_ * x_temp
                    diff = self.noise_intensity * (1.0 + 0.1 * abs(x_temp))
                    diff_deriv = self.noise_intensity * 0.1 * np.sign(x_temp)
                    x_temp = (
                        x_temp
                        + drift * dt
                        + diff * winc
                        + 0.5 * diff * diff_deriv * (winc ** 2 - dt)
                    )
                x_mil[p] = x_temp

            # 参考解（最小步长）
            x_ref = np.zeros(m)
            for p in range(m):
                x_temp = T0[0]
                for j in range(n_ref):
                    winc = dW_ref[p, j]
                    drift = -self.lambda_ * x_temp
                    diff = self.noise_intensity * (1.0 + 0.1 * abs(x_temp))
                    diff_deriv = self.noise_intensity * 0.1 * np.sign(x_temp)
                    x_temp = (
                        x_temp
                        + drift * dt_ref
                        + diff * winc
                        + 0.5 * diff * diff_deriv * (winc ** 2 - dt_ref)
                    )
                x_ref[p] = x_temp

            err = np.mean(np.abs(x_mil - x_ref))
            errors.append(err)

        dtvals = np.array([r * dt_ref for r in r_values[1:]])
        A = np.vstack([np.ones(len(dtvals)), np.log(dtvals)]).T
        rhs = np.log(errors)
        sol = np.linalg.lstsq(A, rhs, rcond=None)[0]
        q = sol[1]
        return q, errors, dtvals
