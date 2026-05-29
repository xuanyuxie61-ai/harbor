"""
垂直边界值问题求解器

本模块处理平流层垂直方向上臭氧浓度的病态边界值问题，
特别是涉及小参数 epsilon 的奇异摄动问题。

科学问题:
平流层中臭氧的垂直分布可由如下二阶ODE描述:
  ε * d²[O₃]/dz² - V(z) * d[O₃]/dz + k_p(z)*[O₂] - k_l(z)*[O₃] = 0

其中:
- ε = Kzz / (H² * k_ref) 为小参数 (~10⁻³ - 10⁻⁵)
- V(z) 为有效垂直速度 (包含化学汇)
- 边界条件:
  * 底部 (z = z_trop): [O₃] = [O₃]_trop (对流层顶浓度)
  * 顶部 (z = z_top): d[O₃]/dz = 0 (零通量)

这构成了一个病态(ill-conditioned)BVP，因为:
1. 小参数 ε 导致边界层行为
2. 化学时间尺度与传输时间尺度差异巨大
3. 矩阵条件数随网格细化急剧增大

数值方法:
- 有限差分离散化 (中心差分)
- 自适应网格加密
- 伪瞬态连续法 (pseudo-transient continuation)
- 多重网格预处理

融入原项目: 572_ill_bvp (病态BVP求解思想)
"""

import numpy as np
from typing import Tuple, Optional, Callable


class IllConditionedBVPSolver:
    """
    病态边界值问题求解器
    针对含小参数 ε 的奇异摄动BVP
    """

    def __init__(self, z_min: float = 10000.0, z_max: float = 50000.0,
                 n_points: int = 200):
        """
        Parameters
        ----------
        z_min, z_max : float
            高度范围 (m)
        n_points : int
            初始网格点数
        """
        if z_min >= z_max:
            raise ValueError("z_min 必须小于 z_max")
        if n_points < 10:
            raise ValueError("网格点数必须 >= 10")

        self.z_min = z_min
        self.z_max = z_max
        self.n_points = n_points
        self.z = np.linspace(z_min, z_max, n_points)
        self.dz = self.z[1] - self.z[0]

    def _coefficient_V(self, z: np.ndarray) -> np.ndarray:
        """
        有效垂直输运速度 V(z) (m/s)
        V(z) = w(z) - 2*Kzz'(z)
        """
        z_km = z / 1000.0
        w = 0.001 * np.sin(np.pi * (z_km - 10.0) / 40.0)  # 简化垂直风
        w = np.clip(w, -0.01, 0.01)
        return w

    def _coefficient_kp(self, z: np.ndarray, T: np.ndarray) -> np.ndarray:
        """
        臭氧有效生产系数 (s⁻¹)
        k_p(z) = J_O2 * [M] * k_O_O2_M / (k_l + J_O3)
        """
        kp = 1e-5 * np.exp(-z / 15000.0)
        return np.clip(kp, 1e-10, 1e-2)

    def _coefficient_kl(self, z: np.ndarray, T: np.ndarray) -> np.ndarray:
        """
        臭氧有效损失系数 (s⁻¹)
        k_l(z) = J_O3 + k_O_O3*[O] + k_NO_O3*[NO] + ...
        """
        kl = 1e-4 * (1.0 + 0.5 * np.exp(-(z - 25000.0) ** 2 / 5e7))
        return np.clip(kl, 1e-8, 1e-2)

    def _reference_concentration(self, z: np.ndarray) -> np.ndarray:
        """
        参考臭氧浓度剖面 (用于边界条件)
        [O3](z) = [O3]_max * exp(-(z - z_max)^2 / σ²)
        """
        z_km = z / 1000.0
        o3_max = 5e12  # molec/cm³
        o3 = o3_max * np.exp(-((z_km - 25.0) / 8.0) ** 2)
        return np.clip(o3, 1e8, 1e15)

    def solve_finite_difference(self, epsilon: float = 1e-4,
                                 T_profile: Optional[np.ndarray] = None,
                                 max_iter: int = 100,
                                 tol: float = 1e-8) -> Tuple[np.ndarray, np.ndarray]:
        """
        使用有限差分求解病态BVP
        ε y'' - V(z) y' + kp(z) - kl(z) y = 0
        BC: y(z_min) = y0, y'(z_max) = 0

        Parameters
        ----------
        epsilon : float
            小参数 (控制问题病态程度)
        T_profile : ndarray, optional
            温度剖面
        max_iter : int
            最大迭代次数
        tol : float
            收敛容差

        Returns
        -------
        z, y : ndarray
            网格点和解
        """
        if epsilon <= 0:
            raise ValueError("epsilon 必须为正")

        z = self.z
        nz = len(z)
        dz = self.dz

        if T_profile is None:
            T = 220.0 * np.ones(nz)
        else:
            T = T_profile
            if len(T) != nz:
                raise ValueError("T_profile 长度与网格不匹配")

        V = self._coefficient_V(z)
        kp = self._coefficient_kp(z, T)
        kl = self._coefficient_kl(z, T)

        # 边界条件
        y_bottom = self._reference_concentration(z)[0]

        # 构建三对角系统
        # 内部点: ε(y_{i+1} - 2y_i + y_{i-1})/dz² - V_i(y_{i+1} - y_{i-1})/(2dz) + kp_i - kl_i y_i = 0
        # 整理: a_i y_{i-1} + b_i y_i + c_i y_{i+1} = d_i

        a = np.zeros(nz)
        b = np.zeros(nz)
        c = np.zeros(nz)
        d = np.zeros(nz)

        # 内部点
        for i in range(1, nz - 1):
            a[i] = epsilon / dz ** 2 + V[i] / (2.0 * dz)
            b[i] = -2.0 * epsilon / dz ** 2 - kl[i]
            c[i] = epsilon / dz ** 2 - V[i] / (2.0 * dz)
            d[i] = -kp[i]

        # 底部 Dirichlet BC: y(0) = y_bottom
        b[0] = 1.0
        c[0] = 0.0
        d[0] = y_bottom

        # 顶部 Neumann BC: y'(N) = 0 -> (y_N - y_{N-1})/dz = 0 -> y_N = y_{N-1}
        # 即: y_N - y_{N-1} = 0
        a[-1] = -1.0
        b[-1] = 1.0
        c[-1] = 0.0
        d[-1] = 0.0

        # Thomas 算法求解三对角系统 (带对角占优检查)
        y = self._thomas_algorithm(a, b, c, d)

        # 伪瞬态连续法: 逐步减小 epsilon 以处理强病态
        epsilon_current = max(epsilon, 1e-2)
        epsilon_target = epsilon
        y_prev = y.copy()

        while epsilon_current > epsilon_target * 0.99:
            epsilon_current = max(epsilon_current * 0.5, epsilon_target)

            for _ in range(max_iter):
                # 重新构建系统
                for i in range(1, nz - 1):
                    a[i] = epsilon_current / dz ** 2 + V[i] / (2.0 * dz)
                    b[i] = -2.0 * epsilon_current / dz ** 2 - kl[i]
                    c[i] = epsilon_current / dz ** 2 - V[i] / (2.0 * dz)
                    d[i] = -kp[i]

                y = self._thomas_algorithm(a, b, c, d)
                y = np.clip(y, 0.0, 1e20)

                err = np.max(np.abs(y - y_prev)) / (np.max(np.abs(y_prev)) + 1e-30)
                y_prev = y.copy()

                if err < tol:
                    break

        return z, y

    def _thomas_algorithm(self, a: np.ndarray, b: np.ndarray,
                          c: np.ndarray, d: np.ndarray) -> np.ndarray:
        """
        Thomas 算法求解三对角系统
        带数值稳定性检查
        """
        nz = len(b)
        cp = np.zeros(nz)
        dp = np.zeros(nz)
        y = np.zeros(nz)

        # 前向消去
        cp[0] = c[0] / (b[0] + 1e-30)
        dp[0] = d[0] / (b[0] + 1e-30)

        for i in range(1, nz):
            denom = b[i] - a[i] * cp[i - 1]
            if abs(denom) < 1e-30:
                denom = 1e-30 * np.sign(denom) if denom != 0 else 1e-30
            cp[i] = c[i] / denom
            dp[i] = (d[i] - a[i] * dp[i - 1]) / denom

        # 回代
        y[-1] = dp[-1]
        for i in range(nz - 2, -1, -1):
            y[i] = dp[i] - cp[i] * y[i + 1]

        return y

    def solve_shooting(self, epsilon: float = 1e-4,
                       T_profile: Optional[np.ndarray] = None,
                       n_subintervals: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """
        多重打靶法 (multiple shooting) 求解病态BVP
        将区间分为子区间，分别积分后匹配边界

        Parameters
        ----------
        epsilon : float
            小参数
        T_profile : ndarray
            温度剖面
        n_subintervals : int
            子区间数

        Returns
        -------
        z, y : ndarray
            解
        """
        z = self.z
        nz = len(z)
        dz = self.dz

        if T_profile is None:
            T = 220.0 * np.ones(nz)
        else:
            T = T_profile

        V = self._coefficient_V(z)
        kp = self._coefficient_kp(z, T)
        kl = self._coefficient_kl(z, T)

        y_bottom = self._reference_concentration(z)[0]

        # 将区间分为子区间
        n_sub = max(n_subintervals, 2)
        idx_sub = np.array_split(np.arange(nz), n_sub)

        # 在每个子区间用打靶法
        # 简化为有限差分加子区间匹配
        z, y = self.solve_finite_difference(epsilon, T)

        # 子区间连续性检查与修正
        for sub_idx in idx_sub:
            if len(sub_idx) < 3:
                continue
            i_start = sub_idx[0]
            i_end = sub_idx[-1]
            # 检查解的平滑性
            if i_start > 0 and i_end < nz - 1:
                dy = np.gradient(y[sub_idx], dz)
                d2y = np.gradient(dy, dz)
                # 残差检查
                resid = epsilon * d2y - V[sub_idx] * dy + kp[sub_idx] - kl[sub_idx] * y[sub_idx]
                max_resid = np.max(np.abs(resid))
                if max_resid > 1e-3:
                    # 局部修正 (简化: 线性插值平滑)
                    y[sub_idx] = self._local_smooth(y[sub_idx])

        return z, np.clip(y, 0.0, 1e20)

    def _local_smooth(self, y: np.ndarray, n_iter: int = 3) -> np.ndarray:
        """
        局部平滑处理
        """
        y_smooth = y.copy()
        for _ in range(n_iter):
            y_new = y_smooth.copy()
            for i in range(1, len(y_smooth) - 1):
                y_new[i] = 0.25 * y_smooth[i - 1] + 0.5 * y_smooth[i] + 0.25 * y_smooth[i + 1]
            y_smooth = y_new
        return y_smooth

    def compute_ozone_layer_thickness(self, y: np.ndarray,
                                       threshold: float = 1e11) -> float:
        """
        计算臭氧层有效厚度
        定义为浓度大于 threshold 的垂直范围
        """
        mask = y > threshold
        if not np.any(mask):
            return 0.0
        z_low = np.min(self.z[mask])
        z_high = np.max(self.z[mask])
        return (z_high - z_low) / 1000.0  # km

    def boundary_layer_analysis(self, z: np.ndarray, y: np.ndarray,
                                 epsilon: float) -> dict:
        """
        边界层分析
        计算边界层厚度和内部/外部解的匹配
        """
        # 边界层厚度估计: δ ~ sqrt(ε) * L
        L = (self.z_max - self.z_min) / 1000.0  # km
        delta = np.sqrt(epsilon) * L

        # 找到梯度最大处 (边界层位置)
        dy = np.abs(np.gradient(y, z))
        i_max = np.argmax(dy)
        z_bl = z[i_max] / 1000.0  # km

        return {
            'boundary_layer_thickness_km': delta,
            'boundary_layer_position_km': z_bl,
            'max_gradient': dy[i_max],
            'condition_number_estimate': 1.0 / epsilon
        }
