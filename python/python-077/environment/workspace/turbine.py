"""
turbine.py
风力机气动-机械耦合模型

融合源项目：
- 925_pwl_approx_1d: 分段线性逼近（风速-功率曲线分段线性化）
- 468_geometry: 圆面积计算（转子扫掠面积）
"""

import numpy as np
from typing import Callable, Optional, Tuple
from numerical_utils import pwl_approx_1d


class WindTurbine:
    """
    水平轴风力机（HAWT）模型。

    物理模型：
    -----------
    风力机从风能中提取的机械功率：

        P_mech = 0.5 · ρ · A · C_p(λ, β) · u³

    其中：
        - ρ = 1.225 kg/m³（标准空气密度）
        - A = π·R² = π·(D/2)²  转子扫掠面积 [m²]
        - C_p(λ, β)  功率系数，与叶尖速比 λ 和桨距角 β 相关
        - u  来流风速 [m/s]

    叶尖速比：
        λ = (ω·R) / u

    其中 ω 为转子角速度 [rad/s]。

    推力系数：
        C_T = C_p / η

    其中 η 为整体效率（约 0.4~0.5 的贝兹极限为 C_p,max ≈ 16/27 ≈ 0.593）。

    分段线性功率曲线：
        - u < u_cut_in:     P = 0
        - u_cut_in ≤ u < u_rated:  P = PWL(u)（分段线性插值）
        - u_rated ≤ u < u_cut_out: P = P_rated
        - u ≥ u_cut_out:    P = 0
    """

    def __init__(self, D: float = 126.0, hub_height: float = 90.0,
                 rated_power: float = 5.0, u_cut_in: float = 3.0,
                 u_rated: float = 12.0, u_cut_out: float = 25.0,
                 cp_max: float = 0.45, ct_at_rated: float = 0.8):
        """
        Parameters
        ----------
        D : float
            转子直径 [m]。
        hub_height : float
            轮毂高度 [m]。
        rated_power : float
            额定功率 [MW]。
        u_cut_in : float
            切入风速 [m/s]。
        u_rated : float
            额定风速 [m/s]。
        u_cut_out : float
            切出风速 [m/s]。
        cp_max : float
            最大功率系数。
        ct_at_rated : float
            额定工况推力系数。
        """
        if D <= 0:
            raise ValueError("转子直径必须为正")
        if hub_height <= 0:
            raise ValueError("轮毂高度必须为正")
        if not (0 < u_cut_in < u_rated < u_cut_out):
            raise ValueError("风速参数必须满足 u_cut_in < u_rated < u_cut_out")
        if cp_max <= 0 or cp_max > 16.0 / 27.0:
            raise ValueError(f"功率系数必须在 (0, 16/27] 范围内，给定 {cp_max}")
        if ct_at_rated <= 0 or ct_at_rated >= 2.0:
            raise ValueError("推力系数必须在 (0, 2) 范围内")

        self.D = D
        self.R = D / 2.0
        self.hub_height = hub_height
        self.rated_power = rated_power
        self.u_cut_in = u_cut_in
        self.u_rated = u_rated
        self.u_cut_out = u_cut_out
        self.cp_max = cp_max
        self.ct_at_rated = ct_at_rated
        self.rho = 1.225  # kg/m³

        # 扫掠面积（源自 468_geometry 的 circle_area_2d）
        self.swept_area = np.pi * self.R ** 2

        # 构建分段线性功率曲线
        self._build_pwl_power_curve()

    def _build_pwl_power_curve(self, n_control: int = 10):
        """
        构建风速-功率分段线性逼近曲线。

        在 [u_cut_in, u_rated] 区间使用分段线性函数逼近理论功率曲线：

            P_theory(u) = 0.5·ρ·A·C_p(u)·u³

        其中 C_p(u) 简化为随 u 变化的函数：
            C_p(u) = C_p,max · sin²( (π/2)·(u - u_cut_in)/(u_rated - u_cut_in) )
        """
        # 在切入-额定区间取数据点
        nd = 50
        xd = np.linspace(self.u_cut_in, self.u_rated, nd)

        # 理论功率曲线（平滑过渡）
        ratio = (xd - self.u_cut_in) / (self.u_rated - self.u_cut_in)
        cp_curve = self.cp_max * np.sin(np.pi / 2.0 * ratio) ** 2
        yd = 0.5 * self.rho * self.swept_area * cp_curve * xd**3 / 1e6  # MW

        # 限制不超过额定功率
        yd = np.minimum(yd, self.rated_power)

        # 分段线性逼近（源自 925_pwl_approx_1d）
        nc = n_control
        xc = np.linspace(self.u_cut_in, self.u_rated, nc)
        self.pwl_yc = pwl_approx_1d(nd, xd, yd, nc, xc)
        self.pwl_xc = xc

        # 构建可调用功率曲线
        self._power_curve_func = self._make_pwl_power_func()

    def _make_pwl_power_func(self) -> Callable[[float], float]:
        """构造分段线性功率曲线函数。"""
        xc = self.pwl_xc
        yc = self.pwl_yc

        def power_func(u: float) -> float:
            if u < self.u_cut_in or u >= self.u_cut_out:
                return 0.0
            if u >= self.u_rated:
                return self.rated_power

            # 分段线性插值
            if u <= xc[0]:
                return max(0.0, yc[0])
            if u >= xc[-1]:
                return min(self.rated_power, yc[-1])

            j = np.searchsorted(xc, u, side='right') - 1
            j = max(0, min(j, len(xc) - 2))
            dx = xc[j + 1] - xc[j]
            if dx < 1e-14:
                return float(yc[j])
            t = (u - xc[j]) / dx
            val = yc[j] * (1.0 - t) + yc[j + 1] * t
            return float(max(0.0, min(self.rated_power, val)))

        return power_func

    def power(self, u: float) -> float:
        """
        计算给定风速下的输出功率 [MW]。

        Parameters
        ----------
        u : float
            来流风速 [m/s]。

        Returns
        -------
        float
            输出功率 [MW]。
        """
        if u < 0:
            return 0.0
        return self._power_curve_func(u)

    def thrust_coefficient(self, u: float) -> float:
        """
        计算给定风速下的推力系数 C_T。

        简化模型：
            - u < u_cut_in:      C_T = 0
            - u_cut_in ≤ u < u_rated: C_T 线性增加到额定值
            - u_rated ≤ u < u_cut_out: C_T 随 u^{-2} 衰减（恒功率区）
            - u ≥ u_cut_out:     C_T = 0
        """
        if u < self.u_cut_in or u >= self.u_cut_out:
            return 0.0
        if u < self.u_rated:
            ratio = (u - self.u_cut_in) / (self.u_rated - self.u_cut_in)
            return self.ct_at_rated * ratio
        else:
            # 恒功率区：P = 0.5·ρ·A·C_p·u³ = const，故 C_p ∝ u^{-3}
            # 近似 C_T ∝ u^{-2}
            return self.ct_at_rated * (self.u_rated / u) ** 2

    def axial_induction_factor(self, u: float) -> float:
        """
        计算轴向诱导因子 a。

        根据动量理论：
            C_T = 4·a·(1 - a)        (a ≤ 0.5, 标准动量理论)

        对于高推力状态（a > 0.5，湍流尾流区），需采用 Glauert 经验修正：
            a = 0.5 · (1 + √(1 - C_T / 0.96))   (C_T > 1.0 时)

        解得：
            a = (1 - √(1 - C_T)) / 2            (标准区)
            a = 0.5 · (1 + √(1 - C_T / 0.96))   (湍流尾流区，Glauert 修正)
        """
        # TODO: Hole 2 — 请完成轴向诱导因子的计算，需同时支持标准动量理论与 Glauert 经验修正
        raise NotImplementedError("Hole 2: 待修复")

    def tip_speed_ratio(self, u: float, omega_rpm: float = 15.0) -> float:
        """
        计算叶尖速比 λ。

            λ = (ω·R) / u = (2π·n/60)·R / u

        Parameters
        ----------
        u : float
            风速 [m/s]。
        omega_rpm : float
            转子转速 [rpm]。

        Returns
        -------
        float
            叶尖速比。
        """
        if u < 1e-6:
            return 0.0
        omega_rad = 2.0 * np.pi * omega_rpm / 60.0
        return (omega_rad * self.R) / u

    def capacity_factor(self, wind_field) -> float:
        """
        计算容量因子。

            CF = E_annual / (P_rated · 8760)

        其中 E_annual 为年发电量。

        Parameters
        ----------
        wind_field : WindField
            风资源场对象。

        Returns
        -------
        float
            容量因子 [0, 1]。
        """
        from numerical_utils import weibull_pdf
        # 数值积分计算年发电量
        n = 500
        u = np.linspace(0.0, self.u_cut_out * 1.5, n)
        du = u[1] - u[0]
        pdf = weibull_pdf(u, wind_field.A, wind_field.k)
        powers = np.array([self.power(float(ui)) for ui in u])
        aep = 8760.0 * np.trapezoid(powers * pdf, u)
        max_annual = self.rated_power * 8760.0
        if max_annual < 1e-10:
            return 0.0
        return float(aep / max_annual)


class TurbineFarm:
    """
    风电场风机集合管理。
    """

    def __init__(self, turbine: WindTurbine, positions: Optional[np.ndarray] = None):
        """
        Parameters
        ----------
        turbine : WindTurbine
            风机原型。
        positions : Optional[np.ndarray]
            风机坐标数组，形状 (n, 2)。
        """
        self.turbine = turbine
        self.positions = positions if positions is not None else np.zeros((0, 2))

    def add_turbine(self, x: float, y: float):
        """添加风机。"""
        self.positions = np.vstack([self.positions, [x, y]])

    def n_turbines(self) -> int:
        return len(self.positions)

    def pairwise_distances(self) -> np.ndarray:
        """
        计算所有风机间的 pairwise 距离矩阵。

        Returns
        -------
        np.ndarray
            n×n 距离矩阵 [m]。
        """
        n = self.n_turbines()
        if n == 0:
            return np.zeros((0, 0))
        dist = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                d = np.linalg.norm(self.positions[i] - self.positions[j])
                dist[i, j] = d
                dist[j, i] = d
        return dist

    def min_spacing(self) -> float:
        """最小风机间距 [m]。"""
        dist = self.pairwise_distances()
        n = self.n_turbines()
        if n <= 1:
            return float('inf')
        # 忽略对角线
        np.fill_diagonal(dist, float('inf'))
        return float(np.min(dist))
