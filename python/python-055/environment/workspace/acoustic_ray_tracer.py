"""
acoustic_ray_tracer.py
基于种子项目 1432_zero_rc（Brent 反通信求根）与
021_asa_geometry_2011（线参数化与三角形距离），
构建分层海洋声速剖面下的声线追踪模块。

科学背景：在海洋声纳探测中，声速随深度变化 c(z) 导致声线弯曲。
根据 Snell 定律（折射定律）：

    sin(θ(z)) / c(z) = p = 常数  （射线参数，ray parameter）

由此可得射线曲率与声速梯度关系：
    dθ/ds = -(1/c) · (dc/dz) · sin(θ)

其中 s 为弧长参数。对于分层介质（声速仅随深度变化），
射线轨迹满足常微分方程组：
    dx/ds = cos(θ)
    dz/ds = sin(θ)
    dθ/ds = -(1/c(z)) · (dc/dz) · sin(θ)

本模块采用数值积分结合 Brent 求根算法，精确计算声线
与海底地形面片的交点，进而得到双程传播时间（TTW）
与对应的水深反演值。
"""

import numpy as np
from seafloor_geometry import point_to_triangle_distance


class SoundSpeedProfile:
    """
    海洋声速剖面模型。

    采用 Munk 标准声速剖面的变体：
        c(z) = c₀ + g·z + Δc · exp( -(z - z₁)² / (2σ²) )

    其中：
        c₀ — 海面参考声速 (m/s)
        g  — 线性梯度 (s⁻¹)
        Δc — 声速异常幅值 (m/s)
        z₁ — 声道轴深度 (m)
        σ  — 声道高斯宽度 (m)
    """

    def __init__(
        self,
        c0: float = 1500.0,
        g: float = 0.015,
        delta_c: float = 30.0,
        z1: float = 1000.0,
        sigma: float = 500.0
    ):
        self.c0 = float(c0)
        self.g = float(g)
        self.delta_c = float(delta_c)
        self.z1 = float(z1)
        self.sigma = float(sigma)

    def evaluate(self, z: np.ndarray) -> np.ndarray:
        """
        计算给定深度处的声速。

        参数:
            z: 深度数组（米，正值向下）
        返回:
            声速数组 (m/s)
        """
        z = np.asarray(z, dtype=np.float64)
        if np.any(z < 0.0):
            # 对负深度做边界保护
            z = np.where(z < 0.0, 0.0, z)
        c = self.c0 + self.g * z + self.delta_c * np.exp(-((z - self.z1) ** 2) / (2.0 * self.sigma ** 2))
        return c

    def gradient(self, z: np.ndarray) -> np.ndarray:
        """
        计算声速梯度 dc/dz。

        公式:
            dc/dz = g - Δc · (z - z₁) / σ² · exp(-(z-z₁)²/(2σ²))
        """
        z = np.asarray(z, dtype=np.float64)
        z = np.where(z < 0.0, 0.0, z)
        dz = z - self.z1
        exp_term = np.exp(-(dz ** 2) / (2.0 * self.sigma ** 2))
        grad = self.g - self.delta_c * dz / (self.sigma ** 2) * exp_term
        return grad


class BrentZeroRC:
    """
    Brent 反通信求根算法（源自 zero_rc.m）。

    算法说明：
        在区间 [a, b] 上寻找函数 f 的零点，要求 f(a)·f(b) < 0。
        结合二分法、弦截法与逆二次插值，具有超线性收敛速度。

    核心迭代公式（逆二次插值步）：
        设 a, b, c 为三个历史点，对应函数值 fa, fb, fc，
        逆二次插值给出下一个试探点：
            p = s·(2·m·q·(q-r) - (b-a)·(r-1))
            q = (q-1)·(r-1)·(s-1)
        其中 s = fb/fa, q = fa/fc, r = fb/fc, m = (c-b)/2
    """

    def __init__(self, a: float, b: float, tol: float = 1e-10):
        self.a = float(a)
        self.b = float(b)
        self.tol = float(tol)
        self._status = 0
        self._arg = 0.0
        # 内部持久状态
        self._c = 0.0
        self._d = 0.0
        self._e = 0.0
        self._fa = 0.0
        self._fb = 0.0
        self._fc = 0.0
        self._sa = 0.0
        self._sb = 0.0

    def start(self) -> float:
        """启动求根过程，返回第一个求值点。"""
        self._status = 1
        self._sa = self.a
        self._sb = self.b
        self._e = self._sb - self._sa
        self._d = self._e
        self._arg = self._sa
        return self._arg

    def iterate(self, value: float) -> tuple:
        """
        接收函数值并返回下一步状态。

        参数:
            value: 在 _arg 处的函数值
        返回:
            (status, arg)
            status > 0: 继续，需要在 arg 处求值
            status = 0: 收敛，arg 为近似根
            status < 0: 失败（无变号区间）
        """
        if self._status == 1:
            self._fa = value
            self._status = 2
            self._arg = self._sb
            return self._status, self._arg

        if self._status == 2:
            self._fb = value
            if self._fa * self._fb > 0.0:
                self._status = -1
                return self._status, self._arg
            self._c = self._sa
            self._fc = self._fa
        else:
            self._fb = value
            if (self._fb > 0.0 and self._fc > 0.0) or (self._fb <= 0.0 and self._fc <= 0.0):
                self._c = self._sa
                self._fc = self._fa
                self._e = self._sb - self._sa
                self._d = self._e

        # 主迭代逻辑
        if abs(self._fc) < abs(self._fb):
            self._sa = self._sb
            self._sb = self._c
            self._c = self._sa
            self._fa = self._fb
            self._fb = self._fc
            self._fc = self._fa

        tol = 2.0 * np.finfo(float).eps * abs(self._sb) + self.tol
        m = 0.5 * (self._c - self._sb)

        if abs(m) <= tol or self._fb == 0.0:
            self._status = 0
            self._arg = self._sb
            return self._status, self._arg

        if abs(self._e) < tol or abs(self._fa) <= abs(self._fb):
            self._e = m
            self._d = self._e
        else:
            s = self._fb / self._fa
            if self._sa == self._c:
                p = 2.0 * m * s
                q = 1.0 - s
            else:
                q = self._fa / self._fc
                r = self._fb / self._fc
                p = s * (2.0 * m * q * (q - r) - (self._sb - self._sa) * (r - 1.0))
                q = (q - 1.0) * (r - 1.0) * (s - 1.0)

            if p > 0.0:
                q = -q
            else:
                p = -p

            s = self._e
            self._e = self._d

            if 2.0 * p < 3.0 * m * q - abs(tol * q) and p < abs(0.5 * s * q):
                self._d = p / q
            else:
                self._e = m
                self._d = self._e

        self._sa = self._sb
        self._fa = self._fb

        if abs(self._d) > tol:
            self._sb = self._sb + self._d
        elif m > 0.0:
            self._sb = self._sb + tol
        else:
            self._sb = self._sb - tol

        self._arg = self._sb
        self._status += 1
        return self._status, self._arg

    def solve(self, func) -> float:
        """
        便捷接口：直接求根。

        参数:
            func: 目标函数 f(x)
        返回:
            近似根
        """
        arg = self.start()
        max_iter = 100
        for _ in range(max_iter):
            val = func(arg)
            status, arg = self.iterate(val)
            if status <= 0:
                break
        return arg


class AcousticRayTracer:
    """
    分层海洋声速剖面下的声线追踪器。
    """

    def __init__(self, ssp: SoundSpeedProfile):
        self.ssp = ssp

    def trace_ray(
        self,
        x0: float,
        z0: float,
        theta0_deg: float,
        z_bottom_func,
        dt: float = 0.01,
        max_steps: int = 50000
    ) -> dict:
        """
        追踪单条声线直到与海底相交。

        数值积分采用四阶 Runge-Kutta 方法求解射线方程：
            dx/dt = c(z)·cos(θ)
            dz/dt = c(z)·sin(θ)
            dθ/dt = -dc/dz · sin(θ)

        这里 t 为时间参数（不是弧长），因此积分结果直接给出传播时间。

        参数:
            x0, z0: 初始位置 (m)
            theta0_deg: 初始掠射角（度，0 为水平，90 为垂直向下）
            z_bottom_func: 海底深度函数 z = f(x)，必须接受标量 x 返回 z
            dt: 时间步长 (s)
            max_steps: 最大积分步数
        返回:
            字典，包含射线轨迹、交点、传播时间等
        """
        theta = np.radians(float(theta0_deg))
        x = float(x0)
        z = float(z0)

        traj_x = [x]
        traj_z = [z]
        travel_time = 0.0

        # 射线参数 p = sin(theta)/c(z) 应保持不变（检验数值误差）
        c0 = float(self.ssp.evaluate(np.array([z]))[0])
        ray_param = np.sin(theta) / c0

        for step in range(max_steps):
            # RK4 单步
            k1_x, k1_z, k1_th = self._ray_derivatives(x, z, theta)
            k2_x, k2_z, k2_th = self._ray_derivatives(
                x + 0.5 * dt * k1_x, z + 0.5 * dt * k1_z, theta + 0.5 * dt * k1_th
            )
            k3_x, k3_z, k3_th = self._ray_derivatives(
                x + 0.5 * dt * k2_x, z + 0.5 * dt * k2_z, theta + 0.5 * dt * k2_th
            )
            k4_x, k4_z, k4_th = self._ray_derivatives(
                x + dt * k3_x, z + dt * k3_z, theta + dt * k3_th
            )

            dx = dt / 6.0 * (k1_x + 2.0 * k2_x + 2.0 * k3_x + k4_x)
            dz = dt / 6.0 * (k1_z + 2.0 * k2_z + 2.0 * k3_z + k4_z)
            dth = dt / 6.0 * (k1_th + 2.0 * k2_th + 2.0 * k3_th + k4_th)

            x_new = x + dx
            z_new = z + dz
            theta_new = theta + dth

            # 深度非负边界保护
            if z_new < 0.0:
                z_new = 0.0
                theta_new = -theta_new  # 海面反射

            travel_time += dt

            # 检测与海底相交
            z_bot = float(z_bottom_func(x_new))
            if z_new >= z_bot:
                # 使用 Brent 求根精确计算交点
                try:
                    x_hit = self._find_intersection_brent(x, z, x_new, z_new, z_bottom_func)
                    z_hit = float(z_bottom_func(x_hit))
                    # 插值计算击中时间
                    ratio = (x_hit - x) / (x_new - x + 1e-15)
                    t_hit = travel_time - dt + ratio * dt
                    traj_x.append(x_hit)
                    traj_z.append(z_hit)
                    return {
                        'hit': True,
                        'x_hit': x_hit,
                        'z_hit': z_hit,
                        'travel_time': t_hit,
                        'ray_param': ray_param,
                        'trajectory_x': np.array(traj_x),
                        'trajectory_z': np.array(traj_z),
                        'n_steps': step + 1,
                    }
                except Exception:
                    # Brent 失败时回退到线性插值
                    ratio = (z_bot - z) / (z_new - z + 1e-15)
                    x_hit = x + ratio * (x_new - x)
                    z_hit = z_bot
                    t_hit = travel_time - dt + ratio * dt
                    traj_x.append(x_hit)
                    traj_z.append(z_hit)
                    return {
                        'hit': True,
                        'x_hit': x_hit,
                        'z_hit': z_hit,
                        'travel_time': t_hit,
                        'ray_param': ray_param,
                        'trajectory_x': np.array(traj_x),
                        'trajectory_z': np.array(traj_z),
                        'n_steps': step + 1,
                    }

            x, z, theta = x_new, z_new, theta_new
            traj_x.append(x)
            traj_z.append(z)

        return {
            'hit': False,
            'x_hit': None,
            'z_hit': None,
            'travel_time': travel_time,
            'ray_param': ray_param,
            'trajectory_x': np.array(traj_x),
            'trajectory_z': np.array(traj_z),
            'n_steps': max_steps,
        }

    def _ray_derivatives(self, x: float, z: float, theta: float) -> tuple:
        """
        计算射线方程的右端项。

        方程组（时间参数化）:
            dx/dt = c(z) · cos(θ)
            dz/dt = c(z) · sin(θ)
            dθ/dt = - (dc/dz) · sin(θ)
        """
        # TODO: 请补全射线方程的右端项计算
        # 科学背景：根据 Snell 定律，分层海洋中声线满足 ODE 组：
        #   dx/dt = c(z) * cos(θ)
        #   dz/dt = c(z) * sin(θ)
        #   dθ/dt = -(dc/dz) * sin(θ)
        # 需要调用 self.ssp.evaluate(z) 获取声速 c，
        # 调用 self.ssp.gradient(z) 获取声速梯度 dc/dz。
        # 返回元组 (dxdt, dzdt, dthdt)。
        raise NotImplementedError("Hole_1: 请实现 _ray_derivatives 方法体")

    def _find_intersection_brent(self, x1, z1, x2, z2, z_bottom_func):
        """
        在 [x1, x2] 区间内用 Brent 求根找到射线与海底的精确交点。

        定义函数 f(x) = z_ray(x) - z_bottom(x)，其中 z_ray(x)
        用线性插值近似：z_ray(x) ≈ z1 + (z2-z1)*(x-x1)/(x2-x1)。
        """
        def f(x):
            z_ray = z1 + (z2 - z1) * (x - x1) / (x2 - x1 + 1e-15)
            z_bot = float(z_bottom_func(x))
            return z_ray - z_bot

        # 确保区间端点变号
        f1 = f(x1)
        f2 = f(x2)
        if f1 * f2 > 0:
            # 不变号，回退到中点
            return (x1 + x2) / 2.0

        solver = BrentZeroRC(x1, x2, tol=1e-8)
        arg = solver.start()
        for _ in range(80):
            val = f(arg)
            status, arg = solver.iterate(val)
            if status <= 0:
                break
        return arg

    def compute_ttw_depth(
        self,
        ttw: float,
        theta0_deg: float,
        z_bottom_func,
        x0: float = 0.0,
        z0: float = 0.0
    ) -> float:
        """
        由双程传播时间反演估算水深。

n        假设声线对称（发射与接收路径相同），单程时间为 ttw/2。
        通过追踪声线找到与海底的交点深度。

        参数:
            ttw: 双程传播时间 (s)
            theta0_deg: 发射掠射角 (度)
            z_bottom_func: 海底深度函数
            x0, z0: 发射点坐标
        返回:
            估算水深 (m)
        """
        result = self.trace_ray(x0, z0, theta0_deg, z_bottom_func)
        if not result['hit']:
            return -1.0
        one_way = result['travel_time']
        # 若单程时间与期望单程时间接近，直接返回
        expected_one_way = ttw / 2.0
        if abs(one_way - expected_one_way) < 0.1:
            return result['z_hit']
        # 否则进行角度修正（简化处理：垂直近似）
        c_avg = float(self.ssp.evaluate(np.array([result['z_hit'] / 2.0]))[0])
        depth_est = expected_one_way * c_avg * np.cos(np.radians(theta0_deg))
        return float(depth_est)
