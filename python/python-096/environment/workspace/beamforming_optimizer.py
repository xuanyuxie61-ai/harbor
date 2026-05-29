"""
beamforming_optimizer.py
========================
波束赋形自适应优化与动力学演化模块

核心算法来源：
  - 619_kepler_perturbed_ode：受摄开普勒问题
  - 645_langford_ode：Langford 混沌系统
  - 1029_rk12：一阶/二阶显式 Runge-Kutta 及其误差估计

在电磁学波束赋形中的角色：
  1. 受摄开普勒 ODE 的摄动项建模阵列单元间互耦引起的等效势场扰动
  2. Langford ODE 的环面分岔结构用于描述自适应波束收敛到稳定态的相变过程
  3. RK12 提供带误差估计的相位权重自适应迭代演化求解器
"""

import numpy as np
from typing import Callable, Tuple, Optional


class RK12Solver:
    """
    一阶/二阶显式 Runge-Kutta 求解器（带误差估计）。

    来源：1029_rk12

    数学模型（显式 RK2 / Heun 方法）：
      k_1 = dt * f(t_n, y_n)
      k_2 = dt * f(t_n + dt, y_n + k_1)
      y_{n+1} = y_n + (k_1 + k_2) / 2

      局部截断误差估计：
        e_{n+1} = y_{n+1}^{(RK2)} - y_{n+1}^{(RK1)}
                = (k_2 - k_1) / 2

      其中 RK1 解为 y_{n+1}^{(RK1)} = y_n + k_1（显式欧拉）。
    """

    def __init__(self, yprime: Callable[[float, np.ndarray], np.ndarray]):
        self.yprime = yprime

    def solve(self, tspan: Tuple[float, float], y0: np.ndarray,
              n_steps: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        求解 ODE。

        参数：
            tspan: (t0, t1)
            y0:    初始条件，形状 (m,)
            n_steps: 步数

        返回：
            t: (n_steps+1,) 时间数组
            y: (n_steps+1, m) 解
            e: (n_steps+1, m) 误差估计
        """
        y0 = np.asarray(y0, dtype=float).flatten()
        m = y0.size
        t = np.zeros(n_steps + 1, dtype=float)
        y = np.zeros((n_steps + 1, m), dtype=float)
        e = np.zeros((n_steps + 1, m), dtype=float)

        dt = (tspan[1] - tspan[0]) / n_steps
        t[0] = tspan[0]
        y[0, :] = y0
        e[0, :] = 0.0

        for i in range(n_steps):
            k1 = dt * self.yprime(t[i], y[i, :])
            yt = y[i, :] + k1
            k2 = dt * self.yprime(t[i] + dt, yt)
            t[i + 1] = t[i] + dt
            y[i + 1, :] = y[i, :] + 0.5 * (k1 + k2)
            e[i + 1, :] = 0.5 * (k2 - k1)

        return t, y, e

    def adaptive_solve(self, tspan: Tuple[float, float], y0: np.ndarray,
                       tol: float = 1e-6, max_steps: int = 10000) -> Tuple[np.ndarray, np.ndarray]:
        """
        简单自适应步长求解器（基于误差估计的步长减半/加倍策略）。
        """
        y0 = np.asarray(y0, dtype=float).flatten()
        t0, t1 = tspan
        dt = (t1 - t0) / 100.0
        t_list = [t0]
        y_list = [y0.copy()]
        t_curr = t0
        y_curr = y0.copy()
        steps = 0

        while t_curr < t1 and steps < max_steps:
            dt = min(dt, t1 - t_curr)
            k1 = dt * self.yprime(t_curr, y_curr)
            yt = y_curr + k1
            k2 = dt * self.yprime(t_curr + dt, yt)
            y_next = y_curr + 0.5 * (k1 + k2)
            err_est = np.max(np.abs(0.5 * (k2 - k1)))

            if err_est < tol or dt < 1e-12:
                t_curr += dt
                t_list.append(t_curr)
                y_list.append(y_next.copy())
                y_curr = y_next
                steps += 1
                if err_est < tol * 0.1:
                    dt *= 2.0
            else:
                dt *= 0.5
                if dt < 1e-14:
                    break

        return np.array(t_list), np.vstack(y_list)


class KeplerPerturbedArrayCoupling:
    """
    受摄开普勒动力学用于建模天线单元间等效互耦势场。

    来源：619_kepler_perturbed_ode

    物理映射：
      在天线阵列中，相邻单元的电磁互耦可等效为粒子间的引力/斥力势场。
      设单元位置为 \mathbf{q} = (q_1, q_2)，动量/状态为 \mathbf{p} = (p_1, p_2)，
      则互耦势能在近场区域包含 1/r^3 项（开普勒）以及高阶摄动 1/r^5 项：

        H = \frac{1}{2}(p_1^2 + p_2^2) - \frac{1}{r} - \frac{\delta}{3 r^3}

      其中 r = \sqrt{q_1^2 + q_2^2}，\delta 为互耦强度参数。

      Hamilton 方程给出：
        \dot{q}_1 = p_1,   \dot{q}_2 = p_2
        \dot{p}_1 = -q_1 / r^3 - \delta q_1 / r^5
        \dot{p}_2 = -q_2 / r^3 - \delta q_2 / r^5
    """

    def __init__(self, delta: float = 0.015, e: float = 0.6):
        self.delta = delta
        self.e = e
        # 初始条件由偏心率 e 决定
        p0 = 1.0 - e
        p1 = 0.0
        q0 = 0.0
        q1 = np.sqrt((1.0 + e) / max(1.0 - e, 1e-12))
        self.y0 = np.array([p0, p1, q0, q1], dtype=float)

    def derivative(self, t: float, y: np.ndarray) -> np.ndarray:
        """ODE 右端项。"""
        # TODO: Hole 2 - 请根据受摄开普勒 Hamiltonian 方程实现 ODE 右端项
        # 提示：Hamiltonian H = 0.5(p1^2 + p2^2) - 1/r - delta/(3*r^3)
        # 需要计算 dq/dt 和 dp/dt
        raise NotImplementedError("Hole 2: 受摄开普勒 ODE 右端项 derivative 待实现")

    def conserved_quantity(self, y: np.ndarray) -> float:
        """
        计算守恒量（摄动 Hamiltonian）。

        H = 0.5(p_1^2 + p_2^2) - 1/r - delta/(3 r^3)
        """
        q1, q2, p1, p2 = y[0], y[1], y[2], y[3]
        r = np.sqrt(q1 ** 2 + q2 ** 2)
        r = max(r, 1e-12)
        return 0.5 * (p1 ** 2 + p2 ** 2) - 1.0 / r - self.delta / (3.0 * r ** 3)


class LangfordBeamPhaseDynamics:
    """
    Langford 动力学系统用于描述阵列相位权重的非线性演化。

    来源：645_langford_ode

    物理映射：
      将 (x, y, z) 映射为阵列两个正交极化通道的相位误差
      与幅度控制状态变量。Langford 系统的环面分岔特性
      对应自适应波束赋形算法收敛时的极限环振荡或稳定不动点：

        \dot{x} = (z - b) x - d y
        \dot{y} = d x + (z - b) y
        \dot{z} = c + a z - z^3/3 - (x^2 + y^2)(1 + e z) + f z x^3

      参数 a=0.95, b=0.7, c=0.6, d=3.5, e=0.25, f=0.1 时系统呈现
      从稳定焦点经 Hopf 分岔到极限环的相变。
    """

    def __init__(self, a: float = 0.95, b: float = 0.7, c: float = 0.6,
                 d: float = 3.5, e: float = 0.25, f: float = 0.1):
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.e = e
        self.f = f
        self.y0 = np.array([0.1, 1.0, 0.0], dtype=float)

    def derivative(self, t: float, xyz: np.ndarray) -> np.ndarray:
        x, y, z = xyz[0], xyz[1], xyz[2]
        dxdt = (z - self.b) * x - self.d * y
        dydt = self.d * x + (z - self.b) * y
        dzdt = (self.c + self.a * z - z ** 3 / 3.0
                - (x ** 2 + y ** 2) * (1.0 + self.e * z)
                + self.f * z * x ** 3)
        return np.array([dxdt, dydt, dzdt])

    def lyapunov_exponent_estimate(self, tspan: Tuple[float, float] = (0.0, 100.0),
                                   n_steps: int = 5000) -> float:
        """
        通过有限时间演化估计最大 Lyapunov 指数。

        数学定义：
          \lambda_{max} = \lim_{t \to \infty} \frac{1}{t} \ln \frac{\|\delta \mathbf{x}(t)\|}{\|\delta \mathbf{x}(0)\|}
        """
        solver = RK12Solver(self.derivative)
        t, y, _ = solver.solve(tspan, self.y0, n_steps)
        # 简单估计：用轨迹相邻点的平均发散率
        if y.shape[0] < 10:
            return 0.0
        diffs = np.linalg.norm(y[1:, :] - y[:-1, :], axis=1)
        dt = (tspan[1] - tspan[0]) / n_steps
        rates = np.log(diffs[1:] / np.maximum(diffs[:-1], 1e-18)) / dt
        return float(np.mean(rates))


class AdaptiveBeamformerODE:
    """
    基于 ODE 的自适应波束赋形相位权重演化。

    科学模型：
      设阵列有 M 个单元，相位权重向量 \mathbf{w}(t) \in \mathbb{C}^M。
      目标函数（波束方向图在期望方向的功率最大化+旁瓣抑制）：

        J(\mathbf{w}) = -\mathbf{w}^H \mathbf{R}_s \mathbf{w}
                        + \lambda (|\mathbf{w}^H \mathbf{a}(\theta_0)| - 1)^2
                        + \mu \sum_{k} |\mathbf{w}^H \mathbf{a}(\theta_k)|^2

      其中 \mathbf{R}_s 为信号协方差矩阵，\mathbf{a}(\theta) 为方向向量。

      梯度流方程：
        \dot{\mathbf{w}} = -\nabla_{\mathbf{w}} J + \text{noise}

      本实现将相位置于实数向量中演化。
    """

    def __init__(self, n_elements: int, steering_angle: float = 0.0,
                 lambda_reg: float = 0.5, mu_reg: float = 0.1,
                 element_spacing: float = 0.5):
        self.n_elements = n_elements
        self.steering_angle = steering_angle
        self.lambda_reg = lambda_reg
        self.mu_reg = mu_reg
        self.element_spacing = element_spacing  # 波长单位
        self.k0 = 2.0 * np.pi  # 自由空间波数（归一化到波长）

    def _array_response(self, theta: float) -> np.ndarray:
        """均匀线阵方向向量。"""
        n = np.arange(self.n_elements)
        return np.exp(1j * self.k0 * self.element_spacing * n * np.sin(theta))

    def gradient_flow_derivative(self, t: float, phase_vec: np.ndarray) -> np.ndarray:
        """
        相位权重的梯度流演化方程。

        参数：
            phase_vec: (n_elements,) 实数相位（弧度）
        """
        w = np.exp(1j * phase_vec)
        a0 = self._array_response(self.steering_angle)
        # 主瓣约束梯度
        g_main = -2.0 * self.lambda_reg * (np.abs(np.vdot(a0, w)) - 1.0) * np.angle(np.vdot(a0, w)) * np.ones(self.n_elements)
        # 旁瓣抑制梯度（在几个离散角度上）
        g_sidelobe = np.zeros(self.n_elements, dtype=float)
        sidelobe_angles = np.linspace(-np.pi / 2, np.pi / 2, 21)
        for th in sidelobe_angles:
            if abs(th - self.steering_angle) < 0.1:
                continue
            a = self._array_response(th)
            p = np.abs(np.vdot(a, w)) ** 2
            g_sidelobe += 2.0 * self.mu_reg * p * np.imag(np.conj(w) * a * np.vdot(a, w))
        # 信号协方差（简化单位矩阵）
        g_signal = -2.0 * np.imag(np.conj(w) * w)
        return -(g_main + g_sidelobe + g_signal)

    def evolve(self, tspan: Tuple[float, float] = (0.0, 10.0),
               n_steps: int = 500, initial_phase: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        演化相位权重。

        返回：
            t: 时间数组
            phase_history: (n_steps+1, n_elements) 相位演化历史
        """
        if initial_phase is None:
            initial_phase = np.zeros(self.n_elements, dtype=float)
        solver = RK12Solver(self.gradient_flow_derivative)
        t, y, _ = solver.solve(tspan, initial_phase, n_steps)
        return t, y
