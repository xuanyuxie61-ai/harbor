r"""
resistivity_evolution.py
=======================
反常电阻率的反应-扩散方程求解模块。
在太阳耀斑磁重联中，经典 Spitzer 电阻率不足以解释快速重联，
需要引入反常电阻率模型，描述由微观不稳定性（如双流不稳定性、
低混杂漂移不稳定性）引起的有效电阻率增强。

核心物理模型
------------
反常电阻率演化方程（反应-扩散型）:

    partial eta/partial t = D_eta nabla^2 eta + R(eta, J, |E|)

其中:
    D_eta: 电阻率扩散系数 [m^2/s]
    R: 反应项，描述微观不稳定性导致的电阻率增长/饱和

反应项的具体形式（基于阈值模型）:
    R(eta, J) = alpha * max(0, |J| - J_c) * (eta_max - eta) / eta_max
              - beta * eta

其中:
    J_c: 临界电流密度阈值 [A/m^2]
    eta_max: 最大反常电阻率 [Ohm m]
    alpha: 增长率系数 [s^-1]
    beta: 衰减率系数 [s^-1]

物理意义:
    当电流密度 |J| 超过临界值 J_c 时，微观不稳定性被激发，
    电阻率向 eta_max 增长；否则电阻率指数衰减到经典值。

一维波动方程类比（源自 artery_pde）:
    原项目中的动脉 PDE:
        d^2u/dt^2 + beta * du/dt + alpha * u = gamma * x * dp * f(t)

    可改写为反应-扩散方程的阻尼波动形式:
        partial^2 eta/partial t^2 + nu partial eta/partial t = c^2 nabla^2 eta + S

数值方法:
    使用有限差分法（Crank-Nicolson）求解扩散项，
    显式欧拉处理反应项。

融入原项目:
- 020_artery_pde: 双曲型波动方程的参数结构、阻尼项、强迫项
"""

import numpy as np
from typing import Callable, Tuple, Optional


class AnomalousResistivity:
    """
    反常电阻率演化模型。
    """

    def __init__(self,
                 D_eta: float = 1.0e3,        # 扩散系数 [m^2/s]
                 eta_classical: float = 1e-6, # 经典 Spitzer 电阻率 [Ohm m]
                 eta_max: float = 1e-3,       # 最大反常电阻率 [Ohm m]
                 J_critical: float = 1.0e-4,  # 临界电流密度 [A/m^2]
                 alpha_grow: float = 1.0e3,   # 增长率 [s^-1]
                 beta_decay: float = 1.0e2,   # 衰减率 [s^-1]
                 y_max: float = 3.0e5,        # 空间域半宽 [m]
                 ny: int = 128):
        if D_eta <= 0:
            raise ValueError("D_eta 必须为正")
        if eta_max < eta_classical:
            raise ValueError("eta_max 必须大于 eta_classical")
        if J_critical < 0:
            raise ValueError("J_critical 必须非负")

        self.D_eta = D_eta
        self.eta_cl = eta_classical
        self.eta_max = eta_max
        self.J_c = J_critical
        self.alpha_g = alpha_grow
        self.beta_d = beta_decay
        self.y_max = y_max
        self.ny = ny
        self.y = np.linspace(-y_max, y_max, ny)
        self.dy = self.y[1] - self.y[0]

    def reaction_term(self, eta: np.ndarray, J: np.ndarray) -> np.ndarray:
        """
        计算反应项 R(eta, J)。
        公式:
            R = alpha * max(0, |J| - J_c) * (eta_max - eta) / eta_max
                - beta * (eta - eta_cl)
        """
        eta = np.asarray(eta, dtype=float)
        J = np.asarray(J, dtype=float)
        if len(eta) != len(J):
            raise ValueError("eta 与 J 长度不匹配")

        J_abs = np.abs(J)
        threshold = np.maximum(0.0, J_abs - self.J_c)
        growth = self.alpha_g * threshold * (self.eta_max - eta) / self.eta_max
        decay = -self.beta_d * (eta - self.eta_cl)
        R = growth + decay
        # 数值截断，保证 eta 在物理范围内
        return R

    def _build_diffusion_matrix(self, dt: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        构造 Crank-Nicolson 扩散矩阵。
        (I - 0.5*dt*D*L) eta^{n+1} = (I + 0.5*dt*D*L) eta^n + dt*R^n
        """
        n = self.ny
        r = 0.5 * dt * self.D_eta / (self.dy ** 2)
        main = np.full(n, 1.0 + 2.0 * r)
        off = np.full(n - 1, -r)
        A = np.diag(main) + np.diag(off, k=1) + np.diag(off, k=-1)
        B = np.diag(main - 2.0 * r) + np.diag(-off, k=1) + np.diag(-off, k=-1)
        # Neumann 边界
        A[0, 1] = -2.0 * r
        A[-1, -2] = -2.0 * r
        B[0, 1] = 2.0 * r
        B[-1, -2] = 2.0 * r
        return A, B

    def step(self,
             eta: np.ndarray,
             J: np.ndarray,
             dt: float) -> np.ndarray:
        """
        单步时间推进（Crank-Nicolson + 显式反应项）。
        """
        eta = np.asarray(eta, dtype=float)
        if dt <= 0:
            raise ValueError("dt 必须为正")
        if dt > 0.5 * self.dy ** 2 / self.D_eta:
            # CFL 条件警告
            pass

        A, B = self._build_diffusion_matrix(dt)
        R = self.reaction_term(eta, J)
        rhs = B @ eta + dt * R
        eta_new = np.linalg.solve(A, rhs)
        # 物理截断
        eta_new = np.clip(eta_new, self.eta_cl, self.eta_max)
        return eta_new

    def evolve(self,
               eta0: np.ndarray,
               J_history: Callable[[int], np.ndarray],
               dt: float,
               n_steps: int) -> np.ndarray:
        """
        多步演化。
        J_history(k) 返回第 k 步的电流密度分布。
        """
        eta = np.copy(eta0)
        for k in range(n_steps):
            J = J_history(k)
            eta = self.step(eta, J, dt)
        return eta

    def equilibrium_eta(self, J: np.ndarray) -> np.ndarray:
        """
        计算反应项为零时的稳态电阻率分布。
        由 R=0 得:
            eta_eq = [alpha*threshold*eta_max + beta*eta_cl] / [alpha*threshold + beta]
        """
        J = np.asarray(J, dtype=float)
        threshold = np.maximum(0.0, np.abs(J) - self.J_c)
        num = self.alpha_g * threshold * self.eta_max + self.beta_d * self.eta_cl
        den = self.alpha_g * threshold + self.beta_d
        den_safe = np.where(den < 1e-30, 1e-30, den)
        return num / den_safe


class WaveDampingModel:
    """
    阻尼波动方程模型，改编自 artery_pde 的结构。
    用于描述电阻率在 Alfven 波驱动下的振荡响应。
    """

    def __init__(self,
                 alpha: float = 1.0,
                 beta_damp: float = 2.0,
                 gamma_force: float = 0.5,
                 nx: int = 51,
                 L: float = 5.0e-2):
        self.alpha = alpha
        self.beta = beta_damp
        self.gamma = gamma_force
        self.nx = nx
        self.L = L
        self.x = np.linspace(0.0, L, nx)
        self.dx = self.x[1] - self.x[0]

    def rhs(self, t: float, w: np.ndarray) -> np.ndarray:
        """
        状态向量 w = [u; v]，其中 u=eta, v=d eta/dt。
        du/dt = v
        dv/dt = -alpha * u - beta * v + gamma * x * dp * f(t)
        """
        nx = self.nx
        if len(w) != 2 * nx:
            raise ValueError(f"w 长度应为 {2*nx}")
        u = w[:nx]
        v = w[nx:]
        # 强迫项（模拟 Alfven 波驱动的周期性扰动）
        dp = 0.25 * 133.32
        a = 10.0 * 133.32
        b = 133.32
        omega = 2.0 * np.pi / 0.8
        forcing = self.gamma * self.x * dp * (a + b * np.cos(omega * t))
        dudt = v
        dvdt = -self.alpha * u - self.beta * v + forcing
        return np.concatenate([dudt, dvdt])

    def simulate(self, t_span: Tuple[float, float], nt: int = 200) -> Tuple[np.ndarray, np.ndarray]:
        """
        使用 RK4 积分。
        """
        nx = self.nx
        w = np.zeros(2 * nx)
        t0, tf = t_span
        dt = (tf - t0) / nt
        times = np.linspace(t0, tf, nt + 1)
        states = np.zeros((nt + 1, 2 * nx))
        states[0] = w
        for i in range(nt):
            t = times[i]
            k1 = self.rhs(t, w)
            k2 = self.rhs(t + 0.5 * dt, w + 0.5 * dt * k1)
            k3 = self.rhs(t + 0.5 * dt, w + 0.5 * dt * k2)
            k4 = self.rhs(t + dt, w + dt * k3)
            w = w + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
            states[i + 1] = w
        return times, states


def demo_resistivity():
    """
    演示反常电阻率的演化。
    """
    print("\n[Resistivity] 演示: 反常电阻率演化")
    model = AnomalousResistivity(ny=64)
    # 初始为经典电阻率
    eta0 = np.full(model.ny, model.eta_cl)
    # 模拟电流片中心电流密度高的场景
    J = model.J_c * 5.0 * np.exp(-model.y ** 2 / (2.0 * (model.y_max / 8.0) ** 2))
    dt = 1.0e-4
    n_steps = 1000
    eta_final = model.evolve(eta0, lambda k: J, dt, n_steps)
    eta_eq = model.equilibrium_eta(J)
    print(f"  初始电阻率: {eta0[model.ny//2]:.3e} Ohm m")
    print(f"  最终电阻率: {eta_final[model.ny//2]:.3e} Ohm m")
    print(f"  稳态电阻率: {eta_eq[model.ny//2]:.3e} Ohm m")

    print("\n[Resistivity] 演示: 阻尼波动方程")
    wave = WaveDampingModel(nx=51)
    t, states = wave.simulate((0.0, 2.0), nt=400)
    u_final = states[-1, :wave.nx]
    print(f"  最终 eta 幅度范围: [{np.min(u_final):.3e}, {np.max(u_final):.3e}]")


if __name__ == "__main__":
    demo_resistivity()
