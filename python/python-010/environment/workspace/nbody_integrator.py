"""
nbody_integrator.py
===================
N 体粒子运动积分器

采用 Leapfrog（蛙跳）积分方案，并结合 RK12 自适应步长控制（融入 rk12 核心思想），
用于追踪暗物质粒子在膨胀宇宙中的轨道演化。

核心物理公式
------------
粒子运动方程（共动坐标）:
    d²x/dt² + 2 H(a) dx/dt = - (1/a²) ∇Φ

    其中 x 为共动坐标，t 为宇宙时间，H(a) 为 Hubble 参数，
    Φ 为引力势，右边为物理引力加速度除以 a²。

Leapfrog 积分（kick-drift-kick 形式）:
    v(t + Δt/2) = v(t) + (Δt/2) a(t)
    x(t + Δt) = x(t) + Δt v(t + Δt/2)
    v(t + Δt) = v(t + Δt/2) + (Δt/2) a(t + Δt)

能量守恒误差:
    对于哈密顿系统，Leapfrog 为二阶辛积分器，能量误差在长时间演化中有界:
        |ΔE/E| ~ O(Δt²)

时间步长限制（Courant 条件）:
    Δt < η √(ε / |g|)
    其中 η ~ 0.1-0.5 为精度参数，ε 为力 soften 长度。

RK12 自适应误差控制（融入 rk12）:
    用一阶 Euler 与二阶 Heun 方法估计局部截断误差 e:
        e = |y_{Heun} - y_{Euler}|
    若 e > tol，则减小步长并重新积分。
"""

import numpy as np
from typing import Tuple, Callable
from cosmology import Cosmology


class NBodyIntegrator:
    """
    N 体粒子积分器，支持 Leapfrog 与 RK12 自适应步长。
    """

    def __init__(
        self,
        cosmology: Cosmology,
        softening: float = 0.5,
        eta: float = 0.2,
        use_adaptive_step: bool = False,
        tol: float = 1e-4,
    ):
        """
        Parameters
        ----------
        cosmology : Cosmology
            宇宙学模型
        softening : float
            力软化长度（Mpc/h）
        eta : float
            时间步长参数
        use_adaptive_step : bool
            是否使用 RK12 自适应步长
        tol : float
            RK12 局部误差容差
        """
        self.cosmo = cosmology
        self.softening = max(softening, 1e-6)
        self.eta = eta
        self.use_adaptive_step = use_adaptive_step
        self.tol = tol

    def compute_timestep(self, acc: np.ndarray) -> float:
        """
        基于局部加速度计算最大允许时间步长:
            Δt = η √(ε / max|a|)

        边界处理:
            若加速度过小，使用最大步长上限 1.0 Gyr。
        """
        acc_mag = np.linalg.norm(acc, axis=1)
        max_acc = acc_mag.max()
        if max_acc < 1e-15:
            return 1.0
        dt = self.eta * np.sqrt(self.softening / max_acc)
        # 限制在合理范围
        dt = np.clip(dt, 1e-5, 1.0)
        return dt

    def drift_step(
        self, pos: np.ndarray, vel: np.ndarray, dt: float, L: float
    ) -> np.ndarray:
        """
        漂移步: x_new = x + v dt

        周期性边界处理:
            x_new = x_new mod L
        """
        pos_new = pos + vel * dt
        pos_new = pos_new % L
        return pos_new

    def kick_step(self, vel: np.ndarray, acc: np.ndarray, dt: float) -> np.ndarray:
        """
        踢步: v_new = v + a dt
        """
        return vel + acc * dt

    def leapfrog_step(
        self,
        pos: np.ndarray,
        vel: np.ndarray,
        acc: np.ndarray,
        dt: float,
        L: float,
        compute_acc: Callable[[np.ndarray], np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        单步 Leapfrog（kick-drift-kick）。

        Parameters
        ----------
        pos, vel, acc : np.ndarray
            当前位置、速度、加速度
        dt : float
            时间步长
        L : float
            盒子边长
        compute_acc : callable
            计算加速度的函数 acc = compute_acc(pos)

        Returns
        -------
        pos_new, vel_new, acc_new : np.ndarray
            更新后的位置、速度、加速度
        """
        # TODO: 实现单步 Leapfrog 积分（需考虑膨胀宇宙中的 Hubble 阻尼项）
        raise NotImplementedError("请实现 leapfrog_step 方法")

    def rk12_step(
        self,
        pos: np.ndarray,
        vel: np.ndarray,
        dt: float,
        L: float,
        compute_acc: Callable[[np.ndarray], np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """
        RK12 自适应步长单步（融入 rk12 核心算法）。

        将二阶系统化为一阶:
            y = [x, v]
            dy/dt = [v, a(x)]

        分别用 Euler（一阶）和 Heun（二阶）积分，
        误差估计驱动步长调整。

        Returns
        -------
        pos_new, vel_new, acc_new, error : np.ndarray/array/float
            更新后的状态与估计误差
        """
        n_part = pos.shape[0]

        # Euler 步
        acc0 = compute_acc(pos)
        pos_euler = (pos + vel * dt) % L
        vel_euler = vel + acc0 * dt

        # Heun 步（预测-校正）
        pos_pred = (pos + vel * dt) % L
        acc_pred = compute_acc(pos_pred)
        vel_pred = vel + acc_pred * dt
        pos_heun = (pos + 0.5 * (vel + vel_pred) * dt) % L
        vel_heun = vel + 0.5 * (acc0 + acc_pred) * dt

        # 误差估计
        err_pos = np.linalg.norm(pos_heun - pos_euler, axis=1).max()
        err_vel = np.linalg.norm(vel_heun - vel_euler, axis=1).max()
        error = max(err_pos, err_vel)

        return pos_heun, vel_heun, acc_pred, error

    def evolve(
        self,
        pos0: np.ndarray,
        vel0: np.ndarray,
        t_span: Tuple[float, float],
        L: float,
        compute_acc: Callable[[np.ndarray], np.ndarray],
        n_steps: int = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        从 t_span[0] 到 t_span[1] 演化 N 体系统。

        Parameters
        ----------
        pos0, vel0 : np.ndarray
            初始位置与速度
        t_span : (t0, t1)
            时间区间（Gyr）
        L : float
            盒子边长
        compute_acc : callable
            加速度计算函数
        n_steps : int, optional
            固定步数（若不使用自适应步长）

        Returns
        -------
        t_arr : np.ndarray
            时间序列
        pos_arr : np.ndarray
            位置历史
        vel_arr : np.ndarray
            速度历史
        acc_arr : np.ndarray
            加速度历史
        """
        t0, t1 = t_span
        pos = pos0.copy()
        vel = vel0.copy()
        acc = compute_acc(pos)

        if self.use_adaptive_step:
            # 自适应步长
            t = t0
            t_list = [t]
            pos_list = [pos.copy()]
            vel_list = [vel.copy()]
            acc_list = [acc.copy()]
            max_steps = 10000
            step_count = 0
            while t < t1 and step_count < max_steps:
                dt = self.compute_timestep(acc)
                dt = min(dt, t1 - t)
                pos_new, vel_new, acc_new, err = self.rk12_step(
                    pos, vel, dt, L, compute_acc
                )
                # 误差控制
                if err > self.tol and dt > 1e-5:
                    dt = dt * 0.5
                    continue
                pos, vel, acc = pos_new, vel_new, acc_new
                t += dt
                # 若误差远小于容差，可增大步长
                if err < self.tol * 0.1:
                    dt = min(dt * 2.0, 1.0)
                t_list.append(t)
                pos_list.append(pos.copy())
                vel_list.append(vel.copy())
                acc_list.append(acc.copy())
                step_count += 1
            return (
                np.array(t_list),
                np.array(pos_list),
                np.array(vel_list),
                np.array(acc_list),
            )
        else:
            # 固定步长 Leapfrog
            if n_steps is None:
                n_steps = 100
            dt = (t1 - t0) / n_steps
            t_arr = np.zeros(n_steps + 1)
            pos_arr = np.zeros((n_steps + 1, pos0.shape[0], 3))
            vel_arr = np.zeros((n_steps + 1, pos0.shape[0], 3))
            acc_arr = np.zeros((n_steps + 1, pos0.shape[0], 3))
            t_arr[0] = t0
            pos_arr[0] = pos
            vel_arr[0] = vel
            acc_arr[0] = acc
            for i in range(n_steps):
                pos, vel, acc = self.leapfrog_step(pos, vel, acc, dt, L, compute_acc)
                t_arr[i + 1] = t0 + (i + 1) * dt
                pos_arr[i + 1] = pos
                vel_arr[i + 1] = vel
                acc_arr[i + 1] = acc
            return t_arr, pos_arr, vel_arr, acc_arr


def total_energy(
    pos: np.ndarray,
    vel: np.ndarray,
    mass: np.ndarray,
    phi: np.ndarray,
    pm_solver,
) -> Tuple[float, float, float]:
    """
    计算系统的总能量（动能 + 势能）。

    动能:
        K = (1/2) Σ_p m_p v_p²
    势能（网格近似）:
        U = (1/2) Σ_p m_p Φ(x_p)

    Returns
    -------
    E, K, U : float
        总能量、动能、势能
    """
    K = 0.5 * np.sum(mass[:, None] * vel ** 2)
    # 将势能从网格插值到粒子位置（简化用最近邻）
    N = pm_solver.N
    L = pm_solver.L
    idx = ((pos / L) * N).astype(int) % N
    phi_p = phi[idx[:, 0], idx[:, 1], idx[:, 2]]
    U = 0.5 * np.sum(mass * phi_p)
    return K + U, K, U


if __name__ == "__main__":
    from pm_solver import PMSolver
    from cosmology import Cosmology

    cosmo = Cosmology()
    N = 16
    L = 100.0
    solver = PMSolver(N, L)
    n_part = N ** 3
    pos = np.random.rand(n_part, 3) * L
    vel = np.random.randn(n_part, 3) * 1e-3
    mass = np.ones(n_part) * 1e10
    rho_mean = n_part * 1e10 / (L ** 3)

    def get_acc(p):
        return solver.compute_gravity(p, mass, rho_mean)

    integrator = NBodyIntegrator(cosmo, use_adaptive_step=False)
    t_arr, pos_arr, vel_arr, acc_arr = integrator.evolve(
        pos, vel, (0.0, 1.0), L, get_acc, n_steps=50
    )
    print(f"演化完成: {len(t_arr)} 步")
    print(f"最终位置均值: {pos_arr[-1].mean(axis=0)}")
