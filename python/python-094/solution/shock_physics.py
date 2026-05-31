"""
shock_physics.py
================
非线性声学冲击波传播的物理模型与核心公式。

融合种子项目：无直接融合，提供物理基础。
注入的物理模型：
  - 非线性Burgers方程 (viscous)
  - KZK方程 (Khokhlov-Zabolotskaya-Kuznetsov)
  - 冲击波形成距离 (Fubini-Ghiron解)
  - 频谱能量级联与熵产生率
  - 状态方程 (Tait方程) 用于水中声波
"""

import numpy as np
from numpy.polynomial.legendre import leggauss


class NonlinearAcousticsPhysics:
    """
    封装非线性声学冲击波传播的全部物理模型。
    """

    def __init__(self, medium='water', f0=1e6, p0=1e5, geometry='planar'):
        """
        Parameters
        ----------
        medium : str
            传播介质，'water' 或 'air'。
        f0 : float
            声源中心频率 (Hz)。
        p0 : float
            声源峰值压力 (Pa)。
        geometry : str
            波束几何：'planar', 'spherical', 'cylindrical'。
        """
        self.medium = medium
        self.f0 = float(f0)
        self.p0 = float(p0)
        self.geometry = geometry

        # 介质物理参数
        if medium == 'water':
            self.c0 = 1500.0          # 声速 m/s
            self.rho0 = 1000.0        # 密度 kg/m^3
            self.nu = 1.0e-6          # 运动粘度 m^2/s
            self.beta = 3.5           # 非线性系数 (B/A + 1)
            self.gamma = 7.0          # Tait方程指数
            self.B_tait = 3.046e8     # Tait方程常数 Pa
        elif medium == 'air':
            self.c0 = 343.0
            self.rho0 = 1.21
            self.nu = 1.5e-5
            self.beta = 1.2
            self.gamma = 1.4
            self.B_tait = 1.013e5
        else:
            raise ValueError(f"Unknown medium: {medium}")

        # 派生物理量
        self.omega0 = 2.0 * np.pi * self.f0
        self.k0 = self.omega0 / self.c0
        self.lambda_wave = self.c0 / self.f0
        self.wavelength = self.lambda_wave
        self.acoustic_impedance = self.rho0 * self.c0

        # 小信号Mach数 (声源处)
        self.M0 = self.p0 / (self.rho0 * self.c0 ** 2)
        if self.M0 <= 0.0 or self.M0 > 0.5:
            raise ValueError(f"Mach number {self.M0} out of valid range (0, 0.5]")

        # 热粘滞吸收系数 (经典吸收) — 必须在 Goldberg 数之前计算
        self.classical_absorption = self._classical_absorption()

        # 冲击波形成距离 (Fubini解)
        self.shock_formation_distance = self._compute_shock_formation_distance()

        # Gol'dberg数 (衡量衍射/非线性竞争)
        self.Goldberg_number = self._compute_goldberg_number()

    # =====================================================================
    # 核心物理公式
    # =====================================================================

    def _compute_shock_formation_distance(self):
        r"""
        计算冲击波形成距离 x_s。

        对于平面波，Fubini解给出：

        .. math::
            x_s = \frac{1}{\beta \, k_0 \, M_0}

        其中 :math:`\beta = 1 + B/(2A)` 为非线性系数，
        :math:`k_0 = \omega_0 / c_0` 为波数，
        :math:`M_0 = p_0 / (\rho_0 c_0^2)` 为声Mach数。

        Returns
        -------
        float
            冲击波形成距离 (m)。
        """
        x_s = 1.0 / (self.beta * self.k0 * self.M0)
        if x_s <= 0.0 or not np.isfinite(x_s):
            raise ValueError("Shock formation distance computed as non-positive or non-finite.")
        return x_s

    def _compute_goldberg_number(self):
        r"""
        计算 Gol'dberg 数 N_G，衡量非线性与吸收之间的竞争。

        .. math::
            N_G = \frac{1}{\alpha \, x_s}

        其中 :math:`\alpha` 为热粘滞吸收系数。

        Returns
        -------
        float
            Gol'dberg数。
        """
        alpha = self.classical_absorption
        if alpha <= 0.0:
            return np.inf
        Ng = 1.0 / (alpha * self.shock_formation_distance)
        return Ng

    def _classical_absorption(self):
        r"""
        经典热粘滞吸收系数 (Stokes-Kirchhoff)。

        .. math::
            \alpha_{cl} = \frac{\omega^2}{2 \rho_0 c_0^3}
            \left( \frac{4}{3} \mu + \kappa \left( \frac{1}{C_v} - \frac{1}{C_p} \right) \right)

        这里采用简化形式 :math:`\alpha_{cl} \approx \nu \, \omega^2 / c_0^2`。

        Returns
        -------
        float
            吸收系数 (Np/m)。
        """
        alpha = self.nu * self.omega0 ** 2 / (self.c0 ** 2)
        if alpha < 0.0:
            raise ValueError("Classical absorption coefficient negative.")
        return alpha

    def tait_equation(self, p):
        r"""
        Tait 状态方程，描述介质在高压下的非线性压缩性。

        .. math::
            p = B \left[ \left( \frac{\rho}{\rho_0} \right)^{\gamma} - 1 \right]

        Parameters
        ----------
        p : float or np.ndarray
            压力 (Pa)。

        Returns
        -------
        rho : float or np.ndarray
            对应密度 (kg/m^3)。
        """
        p = np.asarray(p)
        # 确保 p > -B_tait，避免非物理值
        p_safe = np.where(p <= -self.B_tait, -0.999 * self.B_tait, p)
        rho = self.rho0 * ((p_safe / self.B_tait) + 1.0) ** (1.0 / self.gamma)
        return rho

    def nonlinear_wave_speed(self, p):
        r"""
        非线性声速 :math:`c(p) = c_0 + \beta \, p / (\rho_0 c_0)`。

        由 :math:`c^2 = dp/d\rho` 及 Tait 方程导出。

        Parameters
        ----------
        p : float or np.ndarray
            压力。

        Returns
        -------
        float or np.ndarray
            局部声速。
        """
        return self.c0 + self.beta * p / (self.rho0 * self.c0)

    def burgers_rhs(self, u, x, nu_eff):
        r"""
        一维粘性Burgers方程的右端项：

        .. math::
            \frac{\partial u}{\partial t} = - u \frac{\partial u}{\partial x}
            + \nu_{\mathrm{eff}} \frac{\partial^2 u}{\partial x^2}

        其中 :math:`\nu_{\mathrm{eff}}` 为有效粘性（包含热粘滞与非线性耗散修正）。

        Parameters
        ----------
        u : np.ndarray
            质点速度场 (m/s)。
        x : np.ndarray
            空间坐标 (m)，等距或不等距。
        nu_eff : float
            有效粘性系数 (m^2/s)。

        Returns
        -------
        np.ndarray
            时间导数 du/dt。
        """
        if u.size < 3:
            raise ValueError("Velocity array too small for spatial derivative.")
        if x.size != u.size:
            raise ValueError("x and u must have same size.")

        # 一阶对流项：upwind-biased 差分，处理冲击波陡峭前沿
        dx = np.diff(x)
        if np.any(dx <= 0.0):
            raise ValueError("x coordinates must be strictly increasing.")

        # 使用保守形式计算：-u * du/dx
        du_dx = np.zeros_like(u)
        # 内部：中心差分，配合 minmod 限制器思想（简单实现）
        du_dx[1:-1] = (u[2:] - u[:-2]) / (x[2:] - x[:-2])
        # 边界：单侧差分
        du_dx[0] = (u[1] - u[0]) / (x[1] - x[0])
        du_dx[-1] = (u[-1] - u[-2]) / (x[-1] - x[-2])

        # 二阶扩散项
        d2u_dx2 = np.zeros_like(u)
        d2u_dx2[1:-1] = 2.0 * (
            (u[2:] - u[1:-1]) / (x[2:] - x[1:-1]) -
            (u[1:-1] - u[:-2]) / (x[1:-1] - x[:-2])
        ) / (x[2:] - x[:-2])
        d2u_dx2[0] = d2u_dx2[1]
        d2u_dx2[-1] = d2u_dx2[-2]

        rhs = -u * du_dx + nu_eff * d2u_dx2
        return rhs

    def kzk_rhs(self, p, r_grid, z, diffraction=True, absorption=True):
        r"""
        KZK 方程右端项（轴对称简化形式）。

        KZK方程描述弱非线性、弱衍射声束：

        .. math::
            \frac{\partial p}{\partial z} =
            \frac{c_0}{2} \int_{0}^{r} \frac{\partial^2 p}{\partial z'^2} \, dr'
            + \frac{\delta}{2 c_0^3} \frac{\partial^2 p}{\partial \tau^2}
            + \frac{\beta \, p}{\rho_0 c_0^3} \frac{\partial p}{\partial \tau}

        这里采用频域/伪时间形式处理。

        Parameters
        ----------
        p : np.ndarray, shape (Nr,)
            轴对称压力分布 (Pa)。
        r_grid : np.ndarray, shape (Nr,)
            径向坐标 (m)，必须 r_grid[0] = 0。
        z : float
            当前传播距离 (m)。
        diffraction : bool
            是否包含衍射项。
        absorption : bool
            是否包含吸收项。

        Returns
        -------
        np.ndarray
            dz/dz 的压力变化率。
        """
        p = np.asarray(p, dtype=float)
        r_grid = np.asarray(r_grid, dtype=float)
        if p.size != r_grid.size:
            raise ValueError("p and r_grid must have same size.")
        if r_grid[0] != 0.0:
            raise ValueError("r_grid must start at 0 for axisymmetric geometry.")
        if np.any(np.diff(r_grid) <= 0.0):
            raise ValueError("r_grid must be strictly increasing.")

        Nr = p.size
        dp_dz = np.zeros(Nr, dtype=float)

        # 衍射项: (1/2r) * d/dr(r * dp/dr)，轴对称Laplacian
        if diffraction:
            dp_dr = np.zeros(Nr, dtype=float)
            dp_dr[1:-1] = (p[2:] - p[:-2]) / (r_grid[2:] - r_grid[:-2])
            # 轴心处使用 L'Hopital: 2 * d2p/dr2
            if Nr >= 3:
                dp_dr[0] = 0.0  # 轴对称边界
                dp_dr[-1] = (p[-1] - p[-2]) / (r_grid[-1] - r_grid[-2])

            d2p_dr2 = np.zeros(Nr, dtype=float)
            d2p_dr2[1:-1] = 2.0 * (
                (p[2:] - p[1:-1]) / (r_grid[2:] - r_grid[1:-1]) -
                (p[1:-1] - p[:-2]) / (r_grid[1:-1] - r_grid[:-2])
            ) / (r_grid[2:] - r_grid[:-2])
            # 轴心
            if Nr >= 3:
                d2p_dr2[0] = 2.0 * (p[1] - p[0]) / (r_grid[1] - r_grid[0]) ** 2
                d2p_dr2[-1] = d2p_dr2[-2]

            # 1/r * d/dr(r * dp/dr) = d2p/dr2 + (1/r) * dp/dr
            laplacian_r = d2p_dr2.copy()
            if Nr > 1:
                # 避免 r=0 除零
                r_safe = r_grid.copy()
                r_safe[0] = r_safe[1]  # 临时值，下一行用mask处理
                with np.errstate(divide='ignore', invalid='ignore'):
                    laplacian_r += dp_dr / r_safe
                laplacian_r[0] = 2.0 * d2p_dr2[0]  # L'Hopital极限

            # 衍射修正系数 (1 / (2 * k0))
            diffraction_coeff = 1.0 / (2.0 * self.k0)
            dp_dz += diffraction_coeff * laplacian_r

        # 吸收项 (简化热粘滞)
        if absorption:
            alpha = self.classical_absorption
            # 简化为线性吸收: -alpha * p
            # 实际应为频散算子，这里用伪时间近似
            dp_dz -= alpha * p

        # 非线性项作为伪源（在z步进中单独处理，这里返回线性部分）
        return dp_dz

    def entropy_production_rate(self, u, x):
        r"""
        冲击波面的熵产生率（基于Rankine-Hugoniot跳跃条件）。

        .. math::
            \dot{s} = \frac{1}{T_0} \int_{\Sigma}
            [\![ \rho_0 c_0^2 u^2 / 2 + \beta \rho_0 c_0^2 u^3 / 3 \!]\!] \, dx

        这里简化为基于速度梯度的数值估计。

        Parameters
        ----------
        u : np.ndarray
            质点速度场。
        x : np.ndarray
            空间坐标。

        Returns
        -------
        float
            熵产生率估计 (J/(K·m^2·s))。
        """
        if u.size < 3 or x.size < 3:
            return 0.0
        T0 = 300.0  # 参考温度 K
        du_dx = np.zeros_like(u)
        du_dx[1:-1] = (u[2:] - u[:-2]) / (x[2:] - x[:-2])
        # 熵产生正比于 (du/dx)^2 在冲击面附近
        entropy_rate = np.sum(du_dx ** 2) * np.mean(np.diff(x)) * self.rho0 * self.nu / T0
        return float(entropy_rate)

    def shock_mach_number(self, u_post, u_pre=0.0):
        r"""
        基于Rankine-Hugoniot条件的冲击波Mach数。

        .. math::
            M_s = \frac{U_s}{c_0} = 1 + \frac{\beta}{2} \frac{u_{post} - u_{pre}}{c_0}

        Parameters
        ----------
        u_post : float
            冲击波后质点速度。
        u_pre : float
            冲击波前质点速度（默认为0）。

        Returns
        -------
        float
            冲击波Mach数。
        """
        delta_u = u_post - u_pre
        Ms = 1.0 + (self.beta / 2.0) * (delta_u / self.c0)
        return Ms

    def spectral_cascade_energy(self, u_hat, k_vec):
        r"""
        计算频谱能量级联分布。

        .. math::
            E(k) = \frac{1}{2} |\hat{u}(k)|^2

        Parameters
        ----------
        u_hat : np.ndarray
            速度的傅里叶系数。
        k_vec : np.ndarray
            波数向量。

        Returns
        -------
        np.ndarray
            能量谱 E(k)。
        """
        u_hat = np.asarray(u_hat)
        k_vec = np.asarray(k_vec)
        if u_hat.shape != k_vec.shape:
            raise ValueError("u_hat and k_vec must have same shape.")
        E = 0.5 * np.abs(u_hat) ** 2
        return E

    def validate_physical_state(self, u, p):
        """
        验证物理状态的边界条件与数值稳定性。

        Parameters
        ----------
        u : np.ndarray
            速度场。
        p : np.ndarray
            压力场。

        Returns
        -------
        bool
            是否通过验证。
        """
        if np.any(~np.isfinite(u)) or np.any(~np.isfinite(p)):
            return False
        if np.any(np.abs(u) > 10.0 * self.c0):
            return False  # 速度不应超过10倍声速（极端物理限制）
        if np.any(p < -self.B_tait * 0.999):
            return False  # 压力不能低于Tait方程的奇点
        return True
