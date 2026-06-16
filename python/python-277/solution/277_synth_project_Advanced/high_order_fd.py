# -*- coding: utf-8 -*-
"""
high_order_fd.py
================
高阶有限差分算子与离散化模块

本模块融合以下种子项目算法:
- 491_grid_display: 网格生成与显示 → 结构化/非结构化网格生成
- 1107_Gelens-Lab_cellcyclemodules: 延迟微分方程 → 时间离散化格式

核心数值方法:
-------------
位错运动的控制方程为:
B ∂u/∂t = σ_applied - σ_back(u) - σ_PN(u)

其中 σ_back 是弹性back-stress，涉及空间高阶导数:
σ_back = -μ b²/(4π(1-ν)) * ∂²u/∂x² + α * ∂⁴u/∂x⁴ - ...

高阶有限差分:
- 2阶中心差分: f'(x) ≈ (f_{i+1} - f_{i-1})/(2h)
- 4阶中心差分: f'(x) ≈ (-f_{i+2} + 8f_{i+1} - 8f_{i-1} + f_{i-2})/(12h)
- 6阶中心差分: 使用更宽的模板
- 8阶中心差分: 最高精度

数值色散与耗散分析:
- 修正波数: k*h vs 数值波数
- 群速度误差
"""

import math
from physical_constants import PI, DEFAULT_MATERIAL


# ============================================================================
# 网格生成 — 来自 491_grid_display
# 用于位错计算域的离散化
# ============================================================================

class StructuredGrid1D:
    """
    一维结构化网格

    支持均匀和非均匀网格，用于位错位移场的空间离散化。

    网格属性:
    - N: 网格点数
    - x_min, x_max: 计算域
    - dx: 网格间距 (均匀) 或最小间距 (非均匀)
    - 网格类型: 'uniform', 'stretched', 'clustered'
    """

    def __init__(self, x_min, x_max, n_points, grid_type='uniform',
                 cluster_center=None, cluster_ratio=3.0):
        """
        初始化一维网格

        Args:
            x_min: 左边界 (m)
            x_max: 右边界 (m)
            n_points: 网格点数
            grid_type: 网格类型
            cluster_center: 聚集中心位置 (m)
            cluster_ratio: 聚集比例 (中心处加密倍数)
        """
        self.x_min = x_min
        self.x_max = x_max
        self.n_points = n_points
        self.grid_type = grid_type

        if grid_type == 'uniform':
            self.points = self._uniform_grid()
        elif grid_type == 'stretched':
            self.points = self._stretched_grid(ratio=1.5)
        elif grid_type == 'clustered':
            center = cluster_center if cluster_center is not None else (x_min + x_max) / 2.0
            self.points = self._clustered_grid(center, cluster_ratio)
        else:
            raise ValueError(f"不支持的网格类型: {grid_type}")

        self.dx_min = min(self.points[i+1] - self.points[i]
                          for i in range(len(self.points) - 1))
        self.dx_max = max(self.points[i+1] - self.points[i]
                          for i in range(len(self.points) - 1))
        self.dx_avg = (x_max - x_min) / (n_points - 1)

    def _uniform_grid(self):
        """均匀网格: x_i = x_min + i * dx"""
        dx = (self.x_max - self.x_min) / (self.n_points - 1)
        return [self.x_min + i * dx for i in range(self.n_points)]

    def _stretched_grid(self, ratio=1.5):
        """
        拉伸网格 (几何递推)

        dx_{i+1} = ratio * dx_i

        用于边界层附近的位错运动
        """
        L = self.x_max - self.x_min
        # dx_0 * (1 + r + r² + ... + r^{N-2}) = L
        # dx_0 = L * (r-1) / (r^{N-1} - 1)
        if abs(ratio - 1.0) < 1e-12:
            return self._uniform_grid()

        dx0 = L * (ratio - 1.0) / (ratio**(self.n_points - 1) - 1.0)
        points = [self.x_min]
        dx = dx0
        for i in range(1, self.n_points):
            points.append(points[-1] + dx)
            dx *= ratio
        return points

    def _clustered_grid(self, center, cluster_ratio):
        """
        聚集网格 (双曲正切)

        x(ξ) = center + L/2 * tanh(β(ξ - 0.5)) / tanh(β/2)

        其中 β 控制聚集强度

        用于位错核心附近的自适应加密
        """
        L = self.x_max - self.x_min
        beta = math.log(cluster_ratio) * 2.0  # 聚集参数

        points = []
        for i in range(self.n_points):
            xi = i / (self.n_points - 1.0)  # ξ ∈ [0, 1]
            # 双曲正切映射
            t = math.tanh(beta * (xi - 0.5)) / math.tanh(beta / 2.0)
            x = center + (L / 2.0) * t
            points.append(x)

        # 确保边界精确
        points[0] = self.x_min
        points[-1] = self.x_max
        return points

    def get_spacing(self):
        """获取所有网格间距"""
        return [self.points[i+1] - self.points[i]
                for i in range(self.n_points - 1)]

    def get_quality_metrics(self):
        """计算网格质量指标"""
        spacings = self.get_spacing()
        ratios = [spacings[i+1] / max(spacings[i], 1e-30)
                  for i in range(len(spacings) - 1)]

        return {
            'min_dx': min(spacings),
            'max_dx': max(spacings),
            'avg_dx': sum(spacings) / len(spacings),
            'max_ratio': max(ratios) if ratios else 1.0,
            'uniformity': min(spacings) / max(spacings) if max(spacings) > 0 else 1.0,
        }


# ============================================================================
# 高阶有限差分算子
# ============================================================================

class HighOrderFDOperators:
    """
    高阶有限差分算子

    提供从2阶到8阶精度的中心差分模板:

    1阶导数:
    2阶: f'(x) ≈ (-f_{i+1} + f_{i-1}) / (2h)
    4阶: f'(x) ≈ (f_{i-2} - 8f_{i-1} + 8f_{i+1} - f_{i+2}) / (12h)
    6阶: f'(x) ≈ (-f_{i-3} + 9f_{i-2} - 45f_{i-1} + 45f_{i+1} - 9f_{i+2} + f_{i+3}) / (60h)
    8阶: f'(x) ≈ (f_{i-4} - 32f_{i-3} + 168f_{i-2} - 672f_{i-1}
                    + 672f_{i+1} - 168f_{i+2} + 32f_{i+3} - f_{i+4}) / (840h)

    2阶导数:
    2阶: f''(x) ≈ (f_{i-1} - 2f_i + f_{i+1}) / h²
    4阶: f''(x) ≈ (-f_{i-2} + 16f_{i-1} - 30f_i + 16f_{i+1} - f_{i+2}) / (12h²)

    4阶导数 (用于线张力项):
    2阶: f''''(x) ≈ (f_{i-2} - 4f_{i-1} + 6f_i - 4f_{i+1} + f_{i+2}) / h⁴
    """

    # 1阶导数模板
    DERIV1_STENCILS = {
        2: {
            'coeffs': [-1.0, 0.0, 1.0],
            'denom': 2.0,
            'radius': 1,
        },
        4: {
            'coeffs': [1.0, -8.0, 0.0, 8.0, -1.0],
            'denom': 12.0,
            'radius': 2,
        },
        6: {
            'coeffs': [-1.0, 9.0, -45.0, 0.0, 45.0, -9.0, 1.0],
            'denom': 60.0,
            'radius': 3,
        },
        8: {
            'coeffs': [3.0, -32.0, 168.0, -672.0, 0.0, 672.0, -168.0, 32.0, -3.0],
            'denom': 840.0,
            'radius': 4,
        },
    }

    # 2阶导数模板
    DERIV2_STENCILS = {
        2: {
            'coeffs': [1.0, -2.0, 1.0],
            'denom': 1.0,
            'radius': 1,
        },
        4: {
            'coeffs': [-1.0, 16.0, -30.0, 16.0, -1.0],
            'denom': 12.0,
            'radius': 2,
        },
        6: {
            'coeffs': [1.0, -12.0, 90.0, -160.0, 90.0, -12.0, 1.0],
            'denom': 90.0,
            'radius': 3,
        },
    }

    # 4阶导数模板
    DERIV4_STENCILS = {
        2: {
            'coeffs': [1.0, -4.0, 6.0, -4.0, 1.0],
            'denom': 1.0,
            'radius': 2,
        },
    }

    def __init__(self, order=4):
        """
        初始化有限差分算子

        Args:
            order: 精度阶数 (2, 4, 6, 或 8)
        """
        if order not in [2, 4, 6, 8]:
            raise ValueError(f"不支持的精度阶数: {order}, 可选: 2, 4, 6, 8")
        self.order = order
        self.radius = order // 2  # 模板半径

    def apply_deriv1(self, f, dx):
        """
        应用1阶导数算子

        ∂f/∂x ≈ Σ_k c_k f_{i+k} / (denom * dx)

        Args:
            f: 函数值列表
            dx: 网格间距

        Returns:
            list: df/dx 在各网格点的值
        """
        n = len(f)
        stencil = self.DERIV1_STENCILS[self.order]
        coeffs = stencil['coeffs']
        denom = stencil['denom']
        r = stencil['radius']

        result = [0.0] * n

        # 内部点
        for i in range(r, n - r):
            s = 0.0
            for k, c in enumerate(coeffs):
                s += c * f[i - r + k]
            result[i] = s / (denom * dx)

        # 边界: 使用前向/后向差分 (降低精度)
        for i in range(r):
            if i == 0:
                result[i] = (f[1] - f[0]) / dx if n > 1 else 0.0
            else:
                result[i] = (f[i+1] - f[i-1]) / (2.0 * dx) if i+1 < n else 0.0

        for i in range(n - r, n):
            if i == n - 1:
                result[i] = (f[-1] - f[-2]) / dx if n > 1 else 0.0
            else:
                result[i] = (f[i+1] - f[i-1]) / (2.0 * dx) if i-1 >= 0 else 0.0

        return result

    def apply_deriv2(self, f, dx):
        """
        应用2阶导数算子

        ∂²f/∂x² ≈ Σ_k c_k f_{i+k} / (denom * dx²)

        Args:
            f: 函数值列表
            dx: 网格间距

        Returns:
            list: d²f/dx² 在各网格点的值
        """
        n = len(f)
        # 使用与1阶相同的精度阶数
        order2 = min(self.order, 6)  # 2阶导数最多6阶
        if order2 not in self.DERIV2_STENCILS:
            order2 = 4
        stencil = self.DERIV2_STENCILS[order2]
        coeffs = stencil['coeffs']
        denom = stencil['denom']
        r = stencil['radius']

        result = [0.0] * n

        for i in range(r, n - r):
            s = 0.0
            for k, c in enumerate(coeffs):
                s += c * f[i - r + k]
            result[i] = s / (denom * dx * dx)

        # 边界处理
        for i in range(min(r, n)):
            if i == 0 and n > 2:
                result[i] = (f[2] - 2*f[1] + f[0]) / (dx * dx)
            elif i > 0 and i < n-1:
                result[i] = (f[i+1] - 2*f[i] + f[i-1]) / (dx * dx)
            else:
                result[i] = 0.0
        if n > 2:
            result[-1] = (f[-1] - 2*f[-2] + f[-3]) / (dx * dx)

        return result

    def apply_deriv4(self, f, dx):
        """
        应用4阶导数算子

        ∂⁴f/∂x⁴ 用于位错线张力的梯度过正化项

        Args:
            f: 函数值列表
            dx: 网格间距

        Returns:
            list: ∂⁴f/∂x⁴ 在各网格点的值
        """
        n = len(f)
        stencil = self.DERIV4_STENCILS[2]
        coeffs = stencil['coeffs']
        denom = stencil['denom']
        r = stencil['radius']

        result = [0.0] * n
        dx4 = dx**4

        for i in range(r, n - r):
            s = 0.0
            for k, c in enumerate(coeffs):
                s += c * f[i - r + k]
            result[i] = s / (denom * dx4)

        # 边界
        for i in range(r):
            result[i] = 0.0
        for i in range(n - r, n):
            result[i] = 0.0

        return result

    def modified_wavenumber(self, k, dx):
        """
        计算修正波数 (数值色散分析)

        对于精确1阶导数: ∂/∂x → ik
        对于有限差分: ∂/∂x → ik'(k)

        4阶中心差分的修正波数:
        k'h = (1/(12h)) * (-e^{-2ikh} + 8e^{-ikh} - 8e^{ikh} + e^{2ikh})
            = (1/6h) * (8 sin(kh) - sin(2kh))

        理想: k'h = kh
        误差: ε = k'h/(kh) - 1

        Args:
            k: 物理波数 (1/m)
            dx: 网格间距

        Returns:
            dict: 修正波数信息
        """
        kh = k * dx

        if self.order == 2:
            k_prime_h = math.sin(kh)
        elif self.order == 4:
            k_prime_h = (8.0 * math.sin(kh) - math.sin(2.0 * kh)) / 6.0
        elif self.order == 6:
            k_prime_h = (45.0 * math.sin(kh) - 9.0 * math.sin(2.0*kh) + math.sin(3.0*kh)) / 60.0
        elif self.order == 8:
            k_prime_h = (672.0*math.sin(kh) - 168.0*math.sin(2.0*kh) + 32.0*math.sin(3.0*kh) - 3.0*math.sin(4.0*kh)) / 840.0
        else:
            k_prime_h = math.sin(kh)

        # 误差
        if abs(kh) > 1e-15:
            dispersion_error = k_prime_h / kh - 1.0
        else:
            dispersion_error = 0.0

        return {
            'k_phys_h': kh,
            'k_prime_h': k_prime_h,
            'dispersion_error': dispersion_error,
            'resolution_elements_per_wavelength': 2.0 * PI / kh if abs(kh) > 1e-15 else float('inf'),
        }


# ============================================================================
# 延迟微分方程时间积分 — 来自 1107_Gelens-Lab_cellcyclemodules
# ============================================================================

class DelayIntegrator:
    """
    延迟微分方程(DDE)积分器

    位错运动可能包含延迟效应:
    B du/dt = σ(t) - σ_back(u(t), u(t-τ)) - σ_PN(u(t))

    其中 τ 是声子传播延迟:
    τ = L / c_s (L: 特征长度, c_s: 剪切波速)

    积分方法: Runge-Kutta 4阶 (RK4) 配合历史插值

    延迟项的历史通过线性插值获得:
    u(t-τ) ≈ u(t_n) + (t_n - (t-τ)) / (t_n - t_{n-1}) * (u(t_n) - u(t_{n-1}))
    """

    def __init__(self, delay_time=0.0, n_history=100):
        """
        初始化DDE积分器

        Args:
            delay_time: 延迟时间 τ (s)
            n_history: 存储的历史点数
        """
        self.delay = delay_time
        self.n_history = n_history
        self.time_history = []
        self.state_history = []

    def interpolate_history(self, t_delayed):
        """
        从历史数据插值延迟状态

        线性插值:
        u(t-τ) = u(t_k) + (t-τ - t_k)/(t_{k+1} - t_k) * (u(t_{k+1}) - u(t_k))

        Args:
            t_delayed: 延迟时间 (s)

        Returns:
            float: 插值的历史状态
        """
        if not self.time_history:
            return 0.0

        if t_delayed <= self.time_history[0]:
            return self.state_history[0]

        if t_delayed >= self.time_history[-1]:
            return self.state_history[-1]

        # 二分查找
        lo, hi = 0, len(self.time_history) - 1
        while lo < hi - 1:
            mid = (lo + hi) // 2
            if self.time_history[mid] <= t_delayed:
                lo = mid
            else:
                hi = mid

        # 线性插值
        t0, t1 = self.time_history[lo], self.time_history[hi]
        u0, u1 = self.state_history[lo], self.state_history[hi]
        dt = t1 - t0
        if abs(dt) < 1e-30:
            return u0
        alpha = (t_delayed - t0) / dt
        return u0 + alpha * (u1 - u0)

    def update_history(self, t, state):
        """更新历史缓存"""
        self.time_history.append(t)
        self.state_history.append(state)

        # 保持历史长度限制
        if len(self.time_history) > self.n_history:
            self.time_history.pop(0)
            self.state_history.pop(0)

    def rk4_step(self, f, u, t, dt, state_size):
        """
        Runge-Kutta 4阶单步积分 (含延迟)

        k1 = f(u(t), u(t-τ), t)
        k2 = f(u(t) + h/2 k1, u(t+dt/2-τ), t + dt/2)
        k3 = f(u(t) + h/2 k2, u(t+dt/2-τ), t + dt/2)
        k4 = f(u(t) + h k3, u(t+dt-τ), t + dt)

        u(t+dt) = u(t) + (dt/6)(k1 + 2k2 + 2k3 + k4)

        Args:
            f: 右端函数 f(u_current, u_delayed, t)
            u: 当前状态
            t: 当前时间
            dt: 时间步长
            state_size: 状态维度

        Returns:
            float: 更新后的状态
        """
        if isinstance(u, (list, tuple)):
            u = list(u)
        else:
            u = [u]

        t_delayed = t - self.delay

        # k1
        u_del = self.interpolate_history(t_delayed)
        if state_size == 1:
            k1 = f(u[0], u_del, t)
        else:
            k1 = f(u, u_del, t)

        # k2
        u_mid = [u[i] + 0.5 * dt * (k1[i] if state_size > 1 else k1) for i in range(state_size)]
        u_del_mid = self.interpolate_history(t_delayed + 0.5 * dt)
        if state_size == 1:
            k2 = f(u_mid[0], u_del_mid, t + 0.5 * dt)
        else:
            k2 = f(u_mid, u_del_mid, t + 0.5 * dt)

        # k3
        u_mid2 = [u[i] + 0.5 * dt * (k2[i] if state_size > 1 else k2) for i in range(state_size)]
        if state_size == 1:
            k3 = f(u_mid2[0], u_del_mid, t + 0.5 * dt)
        else:
            k3 = f(u_mid2, u_del_mid, t + 0.5 * dt)

        # k4
        u_full = [u[i] + dt * (k3[i] if state_size > 1 else k3) for i in range(state_size)]
        u_del_full = self.interpolate_history(t_delayed + dt)
        if state_size == 1:
            k4 = f(u_full[0], u_del_full, t + dt)
        else:
            k4 = f(u_full, u_del_full, t + dt)

        # 合成
        if state_size == 1:
            u_new = u[0] + (dt / 6.0) * (k1 + 2.0*k2 + 2.0*k3 + k4)
            return u_new
        else:
            u_new = [u[i] + (dt / 6.0) * (k1[i] + 2.0*k2[i] + 2.0*k3[i] + k4[i])
                     for i in range(state_size)]
            return u_new


# ============================================================================
# 边界条件处理
# ============================================================================

class BoundaryCondition:
    """
    边界条件处理器

    位错问题中的典型边界条件:
    1. 周期性: u(0) = u(L), u'(0) = u'(L)
    2. 固定: u(0) = 0, u(L) = nb (n个Burgers矢量)
    3. 自由: u'(0) = 0, u'(L) = 0 (镜像力)
    4. 吸收: u到达边界时被吸收
    5. 反射: 镜像力使位错反弹
    """

    PERIODIC = 'periodic'
    FIXED = 'fixed'
    FREE = 'free'
    ABSORBING = 'absorbing'
    MIRROR = 'mirror'

    def __init__(self, bc_type='fixed', n_images=5):
        """
        初始化边界条件

        Args:
            bc_type: 边界条件类型
            n_images: 镜像法中的镜像数量
        """
        self.bc_type = bc_type
        self.n_images = n_images

    def apply(self, u, dx, mu, nu, b, L):
        """
        应用边界条件

        Args:
            u: 位移场列表
            dx: 网格间距
            mu: 剪切模量
            nu: 泊松比
            b: Burgers矢量
            L: 域半宽度

        Returns:
            list: 施加边界条件后的位移场
        """
        n = len(u)
        u_bc = list(u)

        if self.bc_type == self.PERIODIC:
            # 周期性边界: u[0] = u[-1], ghost points
            u_bc[0] = u_bc[-2]
            u_bc[-1] = u_bc[1]

        elif self.bc_type == self.FIXED:
            # 固定边界: u(0) = 0, u(L) = b
            u_bc[0] = 0.0
            u_bc[-1] = b

        elif self.bc_type == self.FREE:
            # 自由表面: 镜像力
            # F_image = -μb²/(4πh) 作用于边界附近
            h_left = u_bc[1] - u_bc[0] if n > 1 else dx
            h_right = u_bc[-1] - u_bc[-2] if n > 1 else dx

            if abs(h_left) > 1e-30:
                K = 1.0 / (1.0 - nu)
                sigma_image = -mu * b**2 / (4.0 * PI * abs(h_left)) * K
                u_bc[0] += sigma_image * dx**2 / mu

            if abs(h_right) > 1e-30:
                K = 1.0 / (1.0 - nu)
                sigma_image = -mu * b**2 / (4.0 * PI * abs(h_right)) * K
                u_bc[-1] += sigma_image * dx**2 / mu

        elif self.bc_type == self.MIRROR:
            # 镜像边界: 对称性条件
            if n > 1:
                u_bc[0] = u_bc[1]
                u_bc[-1] = u_bc[-2]

        return u_bc

    def mirror_force(self, position, boundary_pos, mu, b, nu):
        """
        计算镜像力

        F_image = -μ b² / (4π h) * Σ_{n=0}^{N} 1/(h + 2nL)

        其中 h 是位错到边界的距离

        Args:
            position: 位错位置 (m)
            boundary_pos: 边界位置 (m)
            mu: 剪切模量
            b: Burgers矢量
            nu: 泊松比

        Returns:
            float: 镜像力 (Pa·m, 每单位位错长度)
        """
        h = abs(position - boundary_pos)
        if h < b:
            h = b  # 截断防止奇异

        K = 1.0 / (1.0 - nu) if True else 1.0
        F = -mu * b**2 / (4.0 * PI * h) * K

        # 高阶镜像修正 (多次反射)
        L_total = 1e-6  # 总长度尺度
        for n in range(1, self.n_images + 1):
            h_image = h + 2.0 * n * L_total
            F += -mu * b**2 / (4.0 * PI * h_image) * K * 0.5**n  # 衰减因子

        return F


if __name__ == '__main__':
    print("=" * 70)
    print("高阶有限差分算子验证")
    print("=" * 70)

    # 测试网格生成
    print("\n网格生成:")
    grid_u = StructuredGrid1D(-1e-8, 1e-8, 21, 'uniform')
    grid_c = StructuredGrid1D(-1e-8, 1e-8, 21, 'clustered', cluster_center=0.0)
    q_u = grid_u.get_quality_metrics()
    q_c = grid_c.get_quality_metrics()
    print(f"  均匀网格: dx_min={q_u['min_dx']:.3e}, uniformity={q_u['uniformity']:.4f}")
    print(f"  聚集网格: dx_min={q_c['min_dx']:.3e}, uniformity={q_c['uniformity']:.4f}")

    # 测试有限差分精度
    print("\n有限差分精度 (f(x) = sin(kx), k=1e8):")
    n_test = 201
    L = 1e-8
    dx = 2*L / (n_test - 1)
    x = [-L + i*dx for i in range(n_test)]
    k_wave = 1e8

    f_exact = [math.sin(k_wave * xi) for xi in x]
    df_exact = [k_wave * math.cos(k_wave * xi) for xi in x]

    for order in [2, 4, 6, 8]:
        fd = HighOrderFDOperators(order=order)
        df_fd = fd.apply_deriv1(f_exact, dx)
        max_err = max(abs(df_fd[i] - df_exact[i]) for i in range(order//2 + 1, n_test - order//2 - 1))
        print(f"  {order}阶精度: max_error = {max_err:.4e}")

    # 测试修正波数
    print("\n数值色散分析 (kh = π/4):")
    fd4 = HighOrderFDOperators(order=4)
    info = fd4.modified_wavenumber(PI / (4 * dx), dx)
    print(f"  4阶: k'h = {info['k_prime_h']:.6f}, 误差 = {info['dispersion_error']:.4e}")

    # 测试延迟积分器
    print("\n延迟微分方程积分器:")
    dde = DelayIntegrator(delay_time=1e-13, n_history=50)
    # 简单测试: du/dt = -u(t) + 0.5*u(t-τ)
    u_val = 1.0
    t_val = 0.0
    dt = 1e-14
    for step in range(10):
        u_val = dde.rk4_step(
            lambda u, u_del, t: -u + 0.5 * u_del,
            u_val, t_val, dt, 1
        )
        t_val += dt
        dde.update_history(t_val, u_val)
    print(f"  10步后: u = {u_val:.6f} (t = {t_val:.3e})")
