"""
time_integrator.py
==================
时间积分器模块

融合种子项目：
- 831_ode_trapezoidal: 梯形法隐式 ODE 求解器
- 1138_spring_double_ode: 双弹簧耦合 ODE 系统

核心内容：
1. 显式 Euler 方法
2. 改进的 Euler 方法（Heun 方法）
3. 梯形法（隐式，Picard 迭代）
4. Runge-Kutta 4 阶方法
5. 显式-隐式混合方法（用于 stiff 系统）
6. 自适应时间步长控制

梯形法（隐式）：
    y_{n+1} = y_n + (h/2) [f(t_n, y_n) + f(t_{n+1}, y_{n+1})]

Picard 迭代求解：
    y_{n+1}^{(k+1)} = y_n + (h/2) [f(t_n, y_n) + f(t_{n+1}, y_{n+1}^{(k)})]

Runge-Kutta 4 阶：
    k1 = h f(t_n, y_n)
    k2 = h f(t_n + h/2, y_n + k1/2)
    k3 = h f(t_n + h/2, y_n + k2/2)
    k4 = h f(t_n + h, y_n + k3)
    y_{n+1} = y_n + (k1 + 2k2 + 2k3 + k4) / 6
"""

import numpy as np


class ODESolver:
    """
    常微分方程数值求解器集合。
    """

    @staticmethod
    def explicit_euler(f, t0, y0, t_end, h):
        """
        显式 Euler 方法。

        y_{n+1} = y_n + h f(t_n, y_n)

        Parameters
        ----------
        f : callable
            右端函数 f(t, y)。
        t0 : float
            初始时间。
        y0 : ndarray
            初始值。
        t_end : float
            终止时间。
        h : float
            时间步长。

        Returns
        -------
        tuple
            (t_array, y_array) 时间序列和数值解序列。
        """
        n_steps = int(np.ceil((t_end - t0) / h))
        t = np.linspace(t0, t_end, n_steps + 1)
        y = np.zeros((n_steps + 1,) + np.shape(y0))
        y[0] = y0

        for i in range(n_steps):
            y[i + 1] = y[i] + h * f(t[i], y[i])

        return t, y

    @staticmethod
    def trapezoidal_implicit(f, t0, y0, t_end, h, max_iter=10, tol=1e-10):
        """
        梯形法隐式求解。

        基于种子项目 831_ode_trapezoidal。

        y_{n+1} = y_n + (h/2) [f(t_n, y_n) + f(t_{n+1}, y_{n+1})]

        采用 Picard 迭代求解隐式方程。

        Parameters
        ----------
        f : callable
            右端函数 f(t, y)。
        t0 : float
            初始时间。
        y0 : ndarray
            初始值。
        t_end : float
            终止时间。
        h : float
            时间步长。
        max_iter : int
            Picard 最大迭代次数。
        tol : float
            收敛容差。

        Returns
        -------
        tuple
            (t_array, y_array)
        """
        n_steps = int(np.ceil((t_end - t0) / h))
        t = np.linspace(t0, t_end, n_steps + 1)
        y = np.zeros((n_steps + 1,) + np.shape(y0))
        y[0] = y0

        for i in range(n_steps):
            f_n = f(t[i], y[i])
            z = y[i].copy()  # 初始猜测

            for _ in range(max_iter):
                z_new = y[i] + 0.5 * h * (f_n + f(t[i + 1], z))
                if np.max(np.abs(z_new - z)) < tol:
                    z = z_new
                    break
                z = z_new

            y[i + 1] = z

        return t, y

    @staticmethod
    def runge_kutta4(f, t0, y0, t_end, h):
        """
        经典 4 阶 Runge-Kutta 方法。

        Parameters
        ----------
        f : callable
            右端函数 f(t, y)。
        t0 : float
            初始时间。
        y0 : ndarray
            初始值。
        t_end : float
            终止时间。
        h : float
            时间步长。

        Returns
        -------
        tuple
            (t_array, y_array)
        """
        n_steps = int(np.ceil((t_end - t0) / h))
        t = np.linspace(t0, t_end, n_steps + 1)
        y = np.zeros((n_steps + 1,) + np.shape(y0))
        y[0] = y0

        for i in range(n_steps):
            k1 = h * f(t[i], y[i])
            k2 = h * f(t[i] + 0.5 * h, y[i] + 0.5 * k1)
            k3 = h * f(t[i] + 0.5 * h, y[i] + 0.5 * k2)
            k4 = h * f(t[i] + h, y[i] + k3)
            y[i + 1] = y[i] + (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0

        return t, y

    @staticmethod
    def adaptive_rk45(f, t0, y0, t_end, h0, tol=1e-6, safety=0.9):
        """
        自适应步长 Runge-Kutta-Fehlberg (RK45) 方法。

        同时计算 4 阶和 5 阶近似，利用两者之差估计局部截断误差，
        并自适应调整步长。

        Parameters
        ----------
        f : callable
            右端函数 f(t, y)。
        t0 : float
            初始时间。
        y0 : ndarray
            初始值。
        t_end : float
            终止时间。
        h0 : float
            初始时间步长。
        tol : float
            误差容限。
        safety : float
            安全因子。

        Returns
        -------
        tuple
            (t_list, y_list)
        """
        # RKF45 系数（Cash-Karp）
        a = [0.0, 0.2, 0.3, 0.6, 1.0, 0.875]
        b = [
            [],
            [0.2],
            [3.0/40.0, 9.0/40.0],
            [3.0/10.0, -9.0/10.0, 6.0/5.0],
            [-11.0/54.0, 5.0/2.0, -70.0/27.0, 35.0/27.0],
            [1631.0/55296.0, 175.0/512.0, 575.0/13824.0, 44275.0/110592.0, 253.0/4096.0]
        ]
        c = [37.0/378.0, 0.0, 250.0/621.0, 125.0/594.0, 0.0, 512.0/1771.0]
        c_star = [2825.0/27648.0, 0.0, 18575.0/48384.0, 13525.0/55296.0, 277.0/14336.0, 0.25]

        t_list = [t0]
        y_list = [y0.copy()]
        t = t0
        y = y0.copy()
        h = h0

        while t < t_end:
            if t + h > t_end:
                h = t_end - t

            k = []
            for i in range(6):
                ti = t + a[i] * h
                yi = y.copy()
                for j in range(i):
                    yi += h * b[i][j] * k[j]
                k.append(f(ti, yi))

            y4 = y.copy()
            y5 = y.copy()
            for i in range(6):
                y4 += h * c[i] * k[i]
                y5 += h * c_star[i] * k[i]

            error = np.max(np.abs(y5 - y4))
            if error < 1e-14:
                error = tol * 0.1

            if error <= tol:
                t += h
                y = y5.copy()
                t_list.append(t)
                y_list.append(y.copy())

            # 调整步长
            h = safety * h * (tol / error) ** 0.2
            h = max(h, h0 * 0.01)
            h = min(h, h0 * 10.0)

        return np.array(t_list), np.array(y_list)


class CoupledOscillator:
    """
    耦合振荡器系统。
    基于种子项目 1138_spring_double_ode 的双弹簧系统思想。

    将双弹簧系统转化为两相界面的耦合振动模型：
        固相-液相界面可视为具有等效弹簧常数的弹性边界。

    系统方程：
        m1 d²u1/dt² = -k1 u1 + k2 (u2 - u1)
        m2 d²u2/dt² = -k2 (u2 - u1)

    转化为一阶系统：
        du1/dt = v1
        dv1/dt = (-k1 u1 + k2 (u2 - u1)) / m1
        du2/dt = v2
        dv2/dt = -k2 (u2 - u1) / m2
    """

    def __init__(self, m1=3.0, m2=5.0, k1=1.0, k2=10.0):
        """
        初始化双弹簧振荡器参数。

        Parameters
        ----------
        m1, m2 : float
            两个质量块的质量。
        k1, k2 : float
            两个弹簧的弹性系数。
        """
        self.m1 = m1
        self.m2 = m2
        self.k1 = k1
        self.k2 = k2

    def rhs(self, t, y):
        """
        计算耦合振荡器的一阶 ODE 右端项。

        Parameters
        ----------
        t : float
            时间。
        y : ndarray, shape (4,)
            状态向量 [u1, v1, u2, v2]。

        Returns
        -------
        ndarray
            时间导数 [du1/dt, dv1/dt, du2/dt, dv2/dt]。
        """
        u1, v1, u2, v2 = y

        du1dt = v1
        dv1dt = (-self.k1 * u1 + self.k2 * (u2 - u1)) / self.m1
        du2dt = v2
        dv2dt = -self.k2 * (u2 - u1) / self.m2

        return np.array([du1dt, dv1dt, du2dt, dv2dt])

    def solve(self, t0, y0, t_end, h, method='rk4'):
        """
        求解耦合振荡器系统。

        Parameters
        ----------
        t0 : float
            初始时间。
        y0 : ndarray
            初始状态 [u1, v1, u2, v2]。
        t_end : float
            终止时间。
        h : float
            时间步长。
        method : str
            'euler', 'trapezoidal', 'rk4'。

        Returns
        -------
        tuple
            (t, y) 解的时间序列。
        """
        solver = ODESolver()
        if method == 'euler':
            return solver.explicit_euler(self.rhs, t0, y0, t_end, h)
        elif method == 'trapezoidal':
            return solver.trapezoidal_implicit(self.rhs, t0, y0, t_end, h)
        elif method == 'rk4':
            return solver.runge_kutta4(self.rhs, t0, y0, t_end, h)
        else:
            raise ValueError(f"不支持的方法: {method}")


class PhaseFieldTimeStepper:
    """
    相场方程专用时间推进器。
    采用显式-隐式混合策略处理 stiff 非线性系统。
    """

    def __init__(self, dt, dx, dy, epsilon, tau, diffusion_coeff=1.0):
        """
        初始化时间推进器。

        Parameters
        ----------
        dt : float
            时间步长。
        dx, dy : float
            空间步长。
        epsilon : float
            界面宽度。
        tau : float
            弛豫时间。
        diffusion_coeff : float
            扩散系数。
        """
        self.dt = dt
        self.dx = dx
        self.dy = dy
        self.epsilon = epsilon
        self.tau = tau
        self.diffusion_coeff = diffusion_coeff

        # 稳定性检查
        dt_diff_limit = 0.25 * min(dx ** 2, dy ** 2) / max(diffusion_coeff, 1e-14)
        if dt > dt_diff_limit:
            # 自适应调整
            self.dt = 0.5 * dt_diff_limit

    def explicit_step(self, phi, rhs_func):
        """
        显式 Euler 单步推进：
            φ^{n+1} = φ^n + Δt * rhs(φ^n)

        Parameters
        ----------
        phi : ndarray
            当前场。
        rhs_func : callable
            右端项函数 rhs_func(phi) 返回 ∂φ/∂t。

        Returns
        -------
        ndarray
            新时刻场。
        """
        return phi + self.dt * rhs_func(phi)

    def semi_implicit_step(self, phi, rhs_nonlinear):
        """
        半隐式单步：隐式处理扩散项，显式处理非线性项。

        (I - Δt D ∇²) φ^{n+1} = φ^n + Δt * rhs_nonlinear(φ^n)

        对于均匀网格和常系数，可用 FFT 快速求解。
        这里采用简化的近似：对高频分量施加限制。

        Parameters
        ----------
        phi : ndarray
            当前场。
        rhs_nonlinear : ndarray
            非线性右端项。

        Returns
        -------
        ndarray
            新时刻场。
        """
        # 简化的半隐式：对显式结果进行光滑化
        phi_new = phi + self.dt * rhs_nonlinear

        # 对高频噪声进行抑制（数值扩散）
        lap_phi = np.zeros_like(phi)
        lap_phi[1:-1, 1:-1] = (
            (phi_new[2:, 1:-1] - 2.0 * phi_new[1:-1, 1:-1] + phi_new[:-2, 1:-1]) / (self.dx ** 2) +
            (phi_new[1:-1, 2:] - 2.0 * phi_new[1:-1, 1:-1] + phi_new[1:-1, :-2]) / (self.dy ** 2)
        )

        # 添加隐式扩散修正
        phi_new += 0.1 * self.dt * self.diffusion_coeff * lap_phi

        return phi_new

    def runge_kutta_step(self, phi, rhs_func):
        """
        4 阶 Runge-Kutta 单步推进。

        Parameters
        ----------
        phi : ndarray
            当前场。
        rhs_func : callable
            右端项函数。

        Returns
        -------
        ndarray
            新时刻场。
        """
        k1 = self.dt * rhs_func(phi)
        k2 = self.dt * rhs_func(phi + 0.5 * k1)
        k3 = self.dt * rhs_func(phi + 0.5 * k2)
        k4 = self.dt * rhs_func(phi + k3)

        return phi + (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
