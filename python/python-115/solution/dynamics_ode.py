"""
dynamics_ode.py
反应坐标动力学与常微分方程积分模块

核心功能：
- Runge-Kutta-Fehlberg (RKF45) 自适应步长 ODE 积分器
- 糖酵解振荡动力学系统（测试与基准）
- 反应坐标演化方程（Langevin 方程确定性部分）
- 误差控制与步长自适应

科学背景：
在酶催化过渡态搜索中，反应坐标 ξ(t) 的演化遵循 Langevin 方程：
    dξ/dt = - (1/γ) ∂G/∂ξ + √(2k_B T/γ) η(t)

其中 G(ξ) 为反应坐标的自由能，γ 为摩擦系数，η(t) 为高斯白噪声。
其确定性部分（平均场）为：
    dξ/dt = - (1/γ) ∂G/∂ξ

该方程可用 RKF45 积分：
    k_1 = h f(t_n, y_n)
    k_2 = h f(t_n + h/4, y_n + k_1/4)
    k_3 = h f(t_n + 3h/8, y_n + 3k_1/32 + 9k_2/32)
    ...
    y_{n+1} = y_n + (16k_1/135 + 6656k_3/12825 + 28561k_4/56430 - 9k_5/50 + 2k_6/55)
    z_{n+1} = y_n + (25k_1/216 + 1408k_3/2565 + 2197k_4/4104 - k_5/5)

误差估计：
    err = |y_{n+1} - z_{n+1}| ≈ O(h^6)

步长调整：
    h_new = h * min(5, 0.9 * (tol/err)^{1/5})

---
糖酵解模型（Selkov, 1968）作为非线性动力学测试：
    du/dt = -u + a v + u² v
    dv/dt = b - a v - u² v

该系统的极限环行为可类比酶催化中底物-产物浓度的振荡动力学。
"""

import numpy as np


class RKF45Integrator:
    """
    Runge-Kutta-Fehlberg 自适应步长积分器
    """

    def __init__(self, f, neqn, relerr=1e-6, abserr=1e-8):
        """
        参数：
            f: 右端函数 f(t, y) -> yp
            neqn: 方程个数
            relerr: 相对误差容限
            abserr: 绝对误差容限
        """
        self.f = f
        self.neqn = neqn
        self.relerr = max(relerr, 2.0 * np.finfo(float).eps + 1e-12)
        self.abserr = max(abserr, 0.0)
        self.nfe = 0
        self.maxnfe = 3000
        self.remin = 1e-12

    def _fehl_step(self, t, y, yp, h):
        """
        单步 Fehlberg 4-5 阶推进

        返回：
            f1, f2, f3, f4, f5: 中间导数值
            s: 5 阶解估计
        """
        neqn = self.neqn
        ch = h / 4.0
        f5_temp = y + ch * yp
        f1 = self.f(t + ch, f5_temp)
        self.nfe += 1

        ch = 3.0 * h / 32.0
        f5_temp = y + ch * (yp + 3.0 * f1)
        f2 = self.f(t + 3.0 * h / 8.0, f5_temp)
        self.nfe += 1

        ch = h / 2197.0
        f5_temp = y + ch * (1932.0 * yp + 7296.0 * f2 - 7200.0 * f1)
        f3 = self.f(t + 12.0 * h / 13.0, f5_temp)
        self.nfe += 1

        ch = h / 4104.0
        f5_temp = y + ch * ((8341.0 * yp - 845.0 * f3) + (29440.0 * f2 - 32832.0 * f1))
        f4 = self.f(t + h, f5_temp)
        self.nfe += 1

        ch = h / 20520.0
        f5_temp = y + ch * ((-6080.0 * yp + 9295.0 * f3 - 5643.0 * f4) + (41040.0 * f1 - 28352.0 * f2))
        f5_val = self.f(t + h / 2.0, f5_temp)
        self.nfe += 1

        ch = h / 7618050.0
        s = y + ch * ((902880.0 * yp + 3855735.0 * f3 - 1371249.0 * f4) + (3953664.0 * f2 + 277020.0 * f5_val))

        return f1, f2, f3, f4, f5_val, s

    def integrate(self, t0, y0, tout):
        """
        从 t0 积分到 tout

        参数：
            t0: 初始时间
            y0: 初始条件 (neqn,)
            tout: 目标时间
        返回：
            t, y: 最终时间和状态
        """
        y = np.asarray(y0, dtype=float).copy()
        t = float(t0)
        dt = tout - t

        # 初始导数
        yp = self.f(t, y)
        self.nfe = 1

        # 估计初始步长
        h = abs(dt)
        toln = 0.0
        for k in range(self.neqn):
            tol = self.relerr * abs(y[k]) + self.abserr
            if tol > 0:
                toln = tol
                ypk = abs(yp[k])
                if tol < ypk * h ** 5:
                    h = (tol / ypk) ** 0.2

        if toln <= 0:
            h = 0.0
        h = max(h, 26.0 * np.finfo(float).eps * max(abs(t), abs(dt)))
        h = h if dt >= 0 else -h

        while True:
            hfaild = 0
            hmin = 26.0 * np.finfo(float).eps * abs(t)
            dt_rem = tout - t

            if 2.0 * abs(h) > abs(dt_rem):
                if abs(dt_rem) <= abs(h):
                    h = dt_rem
                else:
                    h = 0.5 * dt_rem

            while True:
                if self.nfe > self.maxnfe:
                    raise RuntimeError("RKF45: 函数评估次数超过上限")

                f1, f2, f3, f4, f5, s = self._fehl_step(t, y, yp, h)

                # 误差估计
                eeoet = 0.0
                for k in range(self.neqn):
                    et = abs(y[k]) + abs(s[k]) + 2.0 / self.relerr * self.abserr
                    if et <= 0:
                        raise RuntimeError("RKF45: 解为零，无法进行相对误差测试")
                    ee = abs((-2090.0 * yp[k] + (21970.0 * f3[k] - 15048.0 * f4[k])) +
                             (22528.0 * f2[k] - 27360.0 * f5[k]))
                    eeoet = max(eeoet, ee / et)

                esttol = abs(h) * eeoet * 2.0 / self.relerr / 752400.0

                if esttol <= 1.0:
                    break

                # 步长失败，减小步长
                hfaild = 1
                if esttol < 59049.0:
                    s_scale = 0.9 / esttol ** 0.2
                else:
                    s_scale = 0.1
                h = s_scale * h
                if abs(h) < hmin:
                    raise RuntimeError("RKF45: 步长小于最小允许值")

            # 成功步
            t = t + h
            y = s.copy()
            yp = self.f(t, y)
            self.nfe += 1

            # 调整下一步步长
            if 0.0001889568 < esttol:
                s_scale = 0.9 / esttol ** 0.2
            else:
                s_scale = 5.0
            if hfaild:
                s_scale = min(s_scale, 1.0)

            if h >= 0:
                h = max(s_scale * abs(h), hmin)
            else:
                h = -max(s_scale * abs(h), hmin)

            if abs(tout - t) < 1e-12:
                break

        return t, y


class GlycolysisModel:
    """
    糖酵解振荡动力学模型（Selkov, 1968）

    方程：
        du/dt = -u + a v + u² v
        dv/dt =  b - a v - u² v

    参数：
        a = 0.08, b = 0.6（典型值）

    平衡解：
        u* = b,    v* = b / (a + b²)
    """

    def __init__(self, a=0.08, b=0.6):
        self.a = a
        self.b = b

    def derivatives(self, t, y):
        """计算右端项"""
        u, v = y[0], y[1]
        dudt = -u + self.a * v + u ** 2 * v
        dvdt = self.b - self.a * v - u ** 2 * v
        return np.array([dudt, dvdt])

    def equilibrium(self):
        """返回平衡解"""
        u_eq = self.b
        v_eq = self.b / (self.a + self.b ** 2)
        return np.array([u_eq, v_eq])

    def jacobian(self, y):
        """
        Jacobian 矩阵：
            J = [[-1 + 2uv,    a + u²],
                 [-2uv,       -a - u²]]
        """
        u, v = y[0], y[1]
        return np.array([
            [-1.0 + 2.0 * u * v, self.a + u ** 2],
            [-2.0 * u * v, -self.a - u ** 2]
        ])


class ReactionCoordinateDynamics:
    """
    反应坐标确定性动力学

    方程：
        dξ/dt = - (1/γ) dG/dξ

    其中 G(ξ) 为用 RBF 插值得到的自由能曲线。
    """

    def __init__(self, free_energy_func, gamma=1.0):
        """
        参数：
            free_energy_func: 自由能函数 G(ξ)
            gamma: 摩擦系数
        """
        self.G = free_energy_func
        self.gamma = gamma

    def derivatives(self, t, y):
        """计算反应坐标演化"""
        xi = y[0]
        # 数值梯度
        h = 1e-6
        dG = (self.G(xi + h) - self.G(xi - h)) / (2.0 * h)
        dxdt = -dG / self.gamma
        return np.array([dxdt])


def integrate_glycolysis(t0=0.0, y0=None, tstop=50.0):
    """
    积分糖酵解模型并返回轨迹
    """
    model = GlycolysisModel()
    if y0 is None:
        y0 = np.array([0.9, 0.7])

    integrator = RKF45Integrator(model.derivatives, 2, relerr=1e-8, abserr=1e-10)

    # 多步积分，记录轨迹
    times = [t0]
    states = [y0.copy()]
    t = t0
    y = y0.copy()

    n_steps = 200
    dt = (tstop - t0) / n_steps
    for _ in range(n_steps):
        t_next = min(t + dt, tstop)
        t, y = integrator.integrate(t, y, t_next)
        times.append(t)
        states.append(y.copy())

    return np.array(times), np.array(states)
