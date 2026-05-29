"""
wake_model.py
风力机尾流模型与叠加算法

融合源项目：
- 468_geometry: 圆面积、三角形面积、多边形面积（转子扫掠面积与尾流锥体截面计算）
- 1147_square_integrals: 单位正方形积分（尾流亏损在扫掠面上的积分平均）
"""

import numpy as np
from typing import List, Tuple, Optional


class WakeModel:
    """
    Jensen/Park 尾流模型实现。

    物理模型：
    -----------
    对于位于下游距离 x 处的尾流截面，风速亏损为：

        Δu = u_0 · (1 - √(1 - C_T)) / (1 + k·x/D)²

    其中：
        - u_0 : 来流风速 [m/s]
        - C_T : 推力系数
        - k   : 尾流扩展系数（陆上 k ≈ 0.075，海上 k ≈ 0.04~0.05）
        - D   : 转子直径 [m]
        - x   : 下游距离 [m]

    尾流半径随下游距离线性扩展：

        R_w(x) = R + k·x = D/2 + k·x

    尾流区域内的局部风速：

        u(x, r) = u_0 · [1 - (1 - √(1 - C_T)) / (1 + k·x/D)²]      (r ≤ R_w)
        u(x, r) = u_0                                               (r > R_w)

    尾流叠加（能量亏损线性叠加）：

        1 - u/u_0 = √( Σ_i (1 - u_i/u_0)² )

    参考：
        N. O. Jensen, "A note on wind generator interaction,"
        Risø National Laboratory, 1983.
    """

    def __init__(self, k_wake: float = 0.05, a: float = 0.293, D: float = 126.0):
        """
        Parameters
        ----------
        k_wake : float
            尾流扩展系数，海上风电场推荐 0.04~0.05。
        a : float
            轴向诱导因子，典型值 0.1~0.5（额定工况约 0.25~0.33）。
        D : float
            转子直径 [m]。
        """
        # TODO: Hole 1-A — 请完成基于轴向诱导因子 a 的初始化与参数校验
        # 要求：
        #   1. 校验 k_wake > 0, 0 < a < 0.5, D > 0
        #   2. 保存 self.k_wake, self.a, self.D, self.R = D/2
        raise NotImplementedError("Hole 1-A: 待修复")

    def wake_radius(self, x: float) -> float:
        """
        计算下游距离 x 处的尾流半径。

            R_w(x) = R + k·x
        """
        if x < 0:
            return self.R
        return self.R + self.k_wake * x

    def wake_deficit(self, x: float) -> float:
        """
        计算下游距离 x 处的中心线风速亏损比 (1 - u/u_0)。

            δ(x) = (1 - √(1 - C_T)) / (1 + k·x/D)²
        """
        if x <= 0:
            return 0.0
        denom = 1.0 + self.k_wake * x / self.D
        # TODO: Hole 1-B — 请完成 Jensen 尾流亏损公式（基于轴向诱导因子 a 的版本）
        # 物理关系：δ(x) = 2·a / (1 + k·x/D)²
        raise NotImplementedError("Hole 1-B: 待修复")

    def velocity_deficit_ratio(self, x: float, r: float) -> float:
        """
        计算位置 (x, r) 处的风速亏损比，其中 r 为到尾流中心线的径向距离。

        采用顶帽模型（top-hat）：
            亏损在尾流截面内均匀分布。
        """
        if x <= 0:
            return 0.0
        Rw = self.wake_radius(x)
        if r > Rw:
            return 0.0
        return self.wake_deficit(x)

    def local_velocity(self, u0: float, x: float, r: float) -> float:
        """
        计算尾流区域内的局部风速。

            u = u_0 · (1 - δ(x, r))
        """
        deficit = self.velocity_deficit_ratio(x, r)
        return u0 * (1.0 - deficit)

    @staticmethod
    def combine_deficits(deficits: List[float]) -> float:
        """
        能量亏损线性叠加（RSS 叠加）。

        对于 n 个尾流源，叠加后的总亏损：

            δ_total = √( Σ_{i=1}^{n} δ_i² )

        Parameters
        ----------
        deficits : List[float]
            各尾流源的风速亏损比列表。

        Returns
        -------
        float
            叠加后的总亏损比。
        """
        s = 0.0
        for d in deficits:
            dd = max(0.0, min(d, 1.0))
            s += dd ** 2
        return np.sqrt(s)

    def swept_area_average_deficit(self, x: float, y_offset: float = 0.0) -> float:
        """
        计算下游风机转子扫掠面积上的平均风速亏损。

        将下游风机转子圆盘（面积 A = π·R²）与尾流截面圆盘（半径 R_w）
        求交，在交叠区域上积分平均亏损：

            δ_avg = (1/A) ∫∫_A δ(x, r) dA

        采用圆-圆交叠面积解析公式：

            A_overlap = R²·arccos((d² + R² - R_w²)/(2·d·R))
                      + R_w²·arccos((d² + R_w² - R²)/(2·d·R_w))
                      - 0.5·√((-d + R + R_w)(d + R - R_w)(d - R + R_w)(d + R + R_w))

        其中 d = |y_offset| 为两圆心间距。

        Parameters
        ----------
        x : float
            下游距离。
        y_offset : float
            横向偏移。

        Returns
        -------
        float
            扫掠面积上的平均风速亏损比。
        """
        if x <= 0:
            return 0.0
        Rw = self.wake_radius(x)
        d = abs(y_offset)
        R = self.R

        # 无交叠
        if d >= R + Rw:
            return 0.0
        # 完全覆盖
        if d <= abs(Rw - R) and Rw >= R:
            return self.wake_deficit(x)

        # 圆-圆交叠面积（源自 468_geometry 的 circle_area_2d 思想扩展）
        # Heron 公式计算四边形面积项
        a1 = (-d + R + Rw)
        a2 = (d + R - Rw)
        a3 = (d - R + Rw)
        a4 = (d + R + Rw)
        if a1 <= 0 or a2 <= 0 or a3 <= 0 or a4 <= 0:
            return 0.0

        term_sqrt = 0.5 * np.sqrt(a1 * a2 * a3 * a4)

        # 避免 arccos 定义域问题
        cos1 = np.clip((d**2 + R**2 - Rw**2) / (2.0 * d * R + 1e-14), -1.0, 1.0)
        cos2 = np.clip((d**2 + Rw**2 - R**2) / (2.0 * d * Rw + 1e-14), -1.0, 1.0)

        A_overlap = R**2 * np.arccos(cos1) + Rw**2 * np.arccos(cos2) - term_sqrt
        A_swept = np.pi * R**2

        if A_swept < 1e-14:
            return 0.0

        # 平均亏损 = 亏损值 × (交叠面积 / 扫掠面积)
        return self.wake_deficit(x) * (A_overlap / A_swept)


class WakeFarm:
    """
    风电场级尾流叠加计算引擎。
    """

    def __init__(self, wake_model: WakeModel):
        self.wm = wake_model

    def compute_effective_velocity(self, turbines: List[Tuple[float, float]],
                                    i: int, u0: float, wind_dir: float) -> float:
        """
        计算第 i 台风机的有效来流风速。

        考虑所有上游风机对该风机的尾流影响。

        Parameters
        ----------
        turbines : List[Tuple[float, float]]
            所有风机的 (x, y) 坐标 [m]。
        i : int
            目标风机索引。
        u0 : float
            环境风速 [m/s]。
        wind_dir : float
            来流风向，角度 [度]，0 表示正东，90 表示正北。

        Returns
        -------
        float
            有效来流风速 [m/s]。
        """
        if i < 0 or i >= len(turbines):
            raise ValueError("风机索引越界")
        if u0 <= 0:
            return 0.0

        theta = np.radians(wind_dir)
        # 风向单位向量：风从该方向吹来
        wx = np.cos(theta)
        wy = np.sin(theta)

        deficits = []
        xi, yi = turbines[i]

        for j, (xj, yj) in enumerate(turbines):
            if j == i:
                continue
            # 计算 j 到 i 的向量
            dx = xi - xj
            dy = yi - yj
            # 投影到风向方向，判断 j 是否在 i 的上游
            proj = dx * wx + dy * wy
            if proj <= 1e-6:
                continue  # j 不在上游

            # 横向偏移（垂直于风向的距离）
            # 叉积大小 = |dx × dy| = |dx·wy - dy·wx|
            cross = abs(dx * wy - dy * wx)

            deficit = self.wm.swept_area_average_deficit(proj, cross)
            if deficit > 1e-6:
                deficits.append(deficit)

        if not deficits:
            return u0

        total_deficit = WakeModel.combine_deficits(deficits)
        total_deficit = min(total_deficit, 0.99)  # 上限保护
        return u0 * (1.0 - total_deficit)

    def compute_farm_power(self, turbines: List[Tuple[float, float]],
                           u0: float, wind_dir: float,
                           power_curve: callable) -> Tuple[float, List[float]]:
        """
        计算整个风电场的总功率及各风机功率。

        Parameters
        ----------
        turbines : List[Tuple[float, float]]
            风机坐标列表。
        u0 : float
            环境风速 [m/s]。
        wind_dir : float
            风向 [度]。
        power_curve : callable
            功率曲线函数 P(u) [MW]。

        Returns
        -------
        total_power : float
            总功率 [MW]。
        powers : List[float]
            各风机功率列表 [MW]。
        """
        powers = []
        for i in range(len(turbines)):
            u_eff = self.compute_effective_velocity(turbines, i, u0, wind_dir)
            p = power_curve(u_eff)
            powers.append(p)
        return sum(powers), powers
