# -*- coding: utf-8 -*-
"""
ray_tracer.py

基于 stetter_ode 的激光射线追踪模块。

原项目 1162_stetter_ode 研究的是具有周期性分段线性系数的常微分方程。
在激光-等离子体相互作用中，该思想被提升到追踪激光射线在
非均匀折射率介质中的哈密顿轨迹。

核心物理模型：几何光学近似下的射线方程
    哈密顿量:
        H(r, k) = c * |k| * η(r)
    其中 η(r) = sqrt(1 - ω_p(r)^2 / ω_0^2) 为局域折射率。

    射线方程 (Hamilton 方程):
        dr/dt = ∂H/∂k = c * η(r) * k / |k|
        dk/dt = -∂H/∂r = - c * |k| * ∇η(r)

    等价地，用光程长度 s 作为自变量:
        dr/ds = k / |k|
        dk/ds = (ω_0 / c) * ∇η(r)

数值方法：
    采用四阶 Runge-Kutta (RK4) 积分，带自适应步长控制。
    当射线到达临界密度面 (η -> 0) 时自动终止（反射或吸收判断）。

边界处理：
    - 密度查询超出网格范围时采用常值外推或截断。
    - η < η_min (默认 1e-4) 时判定为到达截止面。
    - 步长受限于 Courant 条件: Δs <= C * min(dx, dy) / η_max。
"""

import numpy as np
from physics_constants import plasma_frequency, C_LIGHT


class RayTracer:
    """
    激光射线追踪器。

    Attributes
    ----------
    omega0 : float
        激光角频率 [rad/s]。
    eta_min : float
        截止折射率阈值。
    max_steps : int
        单条射线的最大积分步数。
    """

    def __init__(self, omega0, eta_min=1e-4, max_steps=50000):
        self.omega0 = float(omega0)
        if self.omega0 <= 0:
            raise ValueError("激光角频率必须为正。")
        self.eta_min = float(eta_min)
        self.max_steps = int(max_steps)

    def _eta_and_gradient(self, x, y, density_interp_func):
        """
        计算某点的折射率及其梯度。

        使用中心差分近似梯度:
            ∇η ≈ (η(x+δ, y) - η(x-δ, y), η(x, y+δ) - η(x, y-δ)) / (2δ)

        Parameters
        ----------
        x, y : float
            空间坐标 [m]。
        density_interp_func : callable
            密度插值函数 ne(x, y) -> float。

        Returns
        -------
        eta : float
            折射率。
        grad_eta : ndarray, shape (2,)
            折射率梯度 [m^{-1}]。
        """
        ne = density_interp_func(x, y)
        omega_p = plasma_frequency(ne)
        ratio = (omega_p / self.omega0) ** 2
        ratio = np.clip(ratio, 0.0, 1.0)
        eta = np.sqrt(1.0 - ratio)

        # 差分步长
        delta = max(1e-8 * max(abs(x), abs(y), 1.0), 1e-12)

        ne_px = density_interp_func(x + delta, y)
        ne_mx = density_interp_func(x - delta, y)
        ne_py = density_interp_func(x, y + delta)
        ne_my = density_interp_func(x, y - delta)

        omega_p_px = plasma_frequency(ne_px)
        omega_p_mx = plasma_frequency(ne_mx)
        omega_p_py = plasma_frequency(ne_py)
        omega_p_my = plasma_frequency(ne_my)

        ratio_px = np.clip((omega_p_px / self.omega0) ** 2, 0.0, 1.0)
        ratio_mx = np.clip((omega_p_mx / self.omega0) ** 2, 0.0, 1.0)
        ratio_py = np.clip((omega_p_py / self.omega0) ** 2, 0.0, 1.0)
        ratio_my = np.clip((omega_p_my / self.omega0) ** 2, 0.0, 1.0)

        eta_px = np.sqrt(1.0 - ratio_px)
        eta_mx = np.sqrt(1.0 - ratio_mx)
        eta_py = np.sqrt(1.0 - ratio_py)
        eta_my = np.sqrt(1.0 - ratio_my)

        grad_eta = np.array([
            (eta_px - eta_mx) / (2.0 * delta),
            (eta_py - eta_my) / (2.0 * delta)
        ], dtype=float)

        return eta, grad_eta

    def trace_ray(self, r0, k0, density_interp_func, domain_bounds,
                  ds_init=1e-7, ds_min=1e-12, ds_max=1e-5, courant_factor=0.5):
        """
        追踪单条激光射线。

        Parameters
        ----------
        r0 : ndarray, shape (2,)
            初始位置 [m]。
        k0 : ndarray, shape (2,)
            初始波矢量 [rad/m]。
        density_interp_func : callable
            密度插值函数 ne(x, y)。
        domain_bounds : tuple
            ((xmin, xmax), (ymin, ymax)) 计算域边界。
        ds_init : float, optional
            初始步长 [m]。
        ds_min : float, optional
            最小步长 [m]。
        ds_max : float, optional
            最大步长 [m]。
        courant_factor : float, optional
            Courant 数。

        Returns
        -------
        trajectory : ndarray, shape (N, 2)
            射线位置序列。
        k_trajectory : ndarray, shape (N, 2)
            射线波矢量序列。
        s_vals : ndarray, shape (N,)
            光程长度序列。
        status : str
            终止状态: 'cutoff', 'domain_exit', 'max_steps', 'ok'。
        """
        r0 = np.asarray(r0, dtype=float)
        k0 = np.asarray(k0, dtype=float)
        if len(r0) != 2 or len(k0) != 2:
            raise ValueError("r0 和 k0 必须是二维向量。")

        (xmin, xmax), (ymin, ymax) = domain_bounds
        dx_dom = xmax - xmin
        dy_dom = ymax - ymin

        # 初始折射率
        ne0 = density_interp_func(r0[0], r0[1])
        omega_p0 = plasma_frequency(ne0)
        ratio0 = np.clip((omega_p0 / self.omega0) ** 2, 0.0, 1.0)
        eta0 = np.sqrt(1.0 - ratio0)
        if eta0 < self.eta_min:
            return r0.reshape(1, -1), k0.reshape(1, -1), np.array([0.0]), 'cutoff'

        # RK4 积分
        r = r0.copy()
        k = k0.copy()
        s = 0.0
        ds = ds_init

        traj = [r.copy()]
        k_traj = [k.copy()]
        s_list = [0.0]

        status = 'ok'

        for step in range(self.max_steps):
            # 计算 RHS
            eta_r, grad_eta = self._eta_and_gradient(r[0], r[1], density_interp_func)

            if eta_r < self.eta_min:
                status = 'cutoff'
                break

            k_norm = np.linalg.norm(k)
            if k_norm < 1e-20:
                status = 'stagnation'
                break

            # RHS for dr/ds = k / |k|
            # RHS for dk/ds = (ω_0 / c) * ∇η
            def rhs(r_in, k_in):
                eta_tmp, grad_eta_tmp = self._eta_and_gradient(r_in[0], r_in[1], density_interp_func)
                if eta_tmp < self.eta_min:
                    return np.zeros(2), np.zeros(2)
                drds = k_in / np.linalg.norm(k_in)
                dkds = (self.omega0 / C_LIGHT) * grad_eta_tmp
                return drds, dkds

            # RK4 step
            dr1, dk1 = rhs(r, k)
            dr2, dk2 = rhs(r + 0.5 * ds * dr1, k + 0.5 * ds * dk1)
            dr3, dk3 = rhs(r + 0.5 * ds * dr2, k + 0.5 * ds * dk2)
            dr4, dk4 = rhs(r + ds * dr3, k + ds * dk3)

            r_new = r + (ds / 6.0) * (dr1 + 2 * dr2 + 2 * dr3 + dr4)
            k_new = k + (ds / 6.0) * (dk1 + 2 * dk2 + 2 * dk3 + dk4)

            # 检查是否出域
            if not (xmin <= r_new[0] <= xmax and ymin <= r_new[1] <= ymax):
                # 线性插值找到边界交点
                alpha = 1.0
                for dim, (lo, hi) in enumerate(zip([xmin, ymin], [xmax, ymax])):
                    if r_new[dim] < lo:
                        a = (lo - r[dim]) / (r_new[dim] - r[dim]) if r_new[dim] != r[dim] else 0.0
                        alpha = min(alpha, a)
                    elif r_new[dim] > hi:
                        a = (hi - r[dim]) / (r_new[dim] - r[dim]) if r_new[dim] != r[dim] else 0.0
                        alpha = min(alpha, a)
                r_new = r + alpha * (r_new - r)
                k_new = k + alpha * (k_new - k)
                s += alpha * ds
                traj.append(r_new.copy())
                k_traj.append(k_new.copy())
                s_list.append(s)
                status = 'domain_exit'
                break

            r = r_new
            k = k_new
            s += ds

            traj.append(r.copy())
            k_traj.append(k.copy())
            s_list.append(s)

            # 自适应步长: 基于 Courant 条件
            eta_next = eta_r
            ds_courant = courant_factor * min(dx_dom, dy_dom) * eta_next
            ds = np.clip(ds_courant, ds_min, ds_max)

            # 额外限制: 密度梯度大时减小步长
            grad_norm = np.linalg.norm(grad_eta)
            if grad_norm > 1e3:
                ds = max(ds * 0.5, ds_min)

        else:
            status = 'max_steps'

        trajectory = np.array(traj)
        k_trajectory = np.array(k_traj)
        s_vals = np.array(s_list)

        return trajectory, k_trajectory, s_vals, status

    def trace_beam(self, positions, directions, density_interp_func, domain_bounds):
        """
        追踪一束激光射线（多条射线）。

        Parameters
        ----------
        positions : ndarray, shape (N_rays, 2)
            各射线初始位置。
        directions : ndarray, shape (N_rays, 2)
            各射线初始方向（将被归一化）。
        density_interp_func : callable
            密度插值函数。
        domain_bounds : tuple
            计算域边界。

        Returns
        -------
        results : list of dict
            每条射线的追踪结果字典。
        """
        N = positions.shape[0]
        results = []
        k0_base = self.omega0 / C_LIGHT
        for i in range(N):
            r0 = positions[i]
            d = directions[i]
            d_norm = np.linalg.norm(d)
            if d_norm < 1e-20:
                d = np.array([1.0, 0.0])
                d_norm = 1.0
            k0 = k0_base * (d / d_norm)
            traj, k_traj, s_vals, status = self.trace_ray(
                r0, k0, density_interp_func, domain_bounds
            )
            results.append({
                'trajectory': traj,
                'k_trajectory': k_traj,
                's_vals': s_vals,
                'status': status,
                'final_position': traj[-1],
                'path_length': s_vals[-1]
            })
        return results
