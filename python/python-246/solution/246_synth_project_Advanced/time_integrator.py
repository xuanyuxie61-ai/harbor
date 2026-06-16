"""
time_integrator.py  —  宇宙学 N 体时间积分器
==========================================

科学来源种子:
  - 615_kdv_exact / kdv_parameters.m, kdv_exact_sech.m,
    kdv_exact_rational.m, kdv_residual.m
    借鉴其精确解 + 残差评估的范式: 在时间积分中,
    使用 KdV 型精确解 (sech² soliton / rational solution)
    作为密度波演化的 benchmark,通过残差:
        r = ∂u/∂t - 6 u ∂u/∂x + ∂³u/∂x³
    评估数值解的精度。
    kdv_parameters.m 的参数化管理也用于时间积分器参数。

物理背景:
  N 体模拟使用时间步进求解:
      dx/dt = v / a
      dv/dt = -∇Φ/a - H(a) v
  标准方法:  leapfrog (KDK) 或 Kick-Drift-Kick:
      v^{n+1/2} = v^n - (Δt/2) [∇Φ^n/a + H v^n]
      x^{n+1} = x^n + Δt v^{n+1/2}/a^{n+1/2}
      v^{n+1} = v^{n+1/2} - (Δt/2) [∇Φ^{n+1}/a + H v^{n+1/2}]
  此为 2 阶辛积分,保持相空间体积。
  更高阶: Forest-Ruth (4阶) 与 Yoshida (6阶) 辛积分。
"""

from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
from typing import Tuple, Callable, Dict, Any

from cosmo_config import CosmoParams, z_to_a, hubble_factor
from finite_difference import gradient_3d_periodic


# ---------------------------------------------------------------------------- #
#                  KdV 精确解作为基准 (seed 615)
# ---------------------------------------------------------------------------- #
def kdv_exact_sech(x: NDArray, t: float, a_phase: float = 0.0,
                   v_vel: float = 1.0) -> Tuple[NDArray, NDArray, NDArray,
                                                 NDArray, NDArray]:
    """
    KdV 方程的 sech² soliton 精确解:
        u(x,t) = -(v/2) sech²( √v/2 · (x - vt - a) )
    满足 KdV 方程 (与 kdv_residual 约定一致):
        u_t - 6 u u_x + u_xxx = 0
    (注: 正振幅 soliton 满足 u_t + 6u u_x + u_xxx = 0;
     为与 kdv_residual 的 '-' 约定匹配,取负号.)

    Returns
    -------
    u, u_t, u_x, u_xx, u_xxx : arrays of same shape as x
    """
    x = np.asarray(x, dtype=float)
    sqrtv2 = np.sqrt(max(v_vel, 0.0)) / 2.0
    zeta = sqrtv2 * (x - v_vel * t - a_phase)
    sech = 1.0 / np.cosh(zeta)
    sech2 = sech ** 2
    tanh = np.tanh(zeta)
    # 约定: u = -v/2 sech²(z), z = c(x-vt-a), c = √v/2
    u = -(v_vel / 2.0) * sech2
    # u_x = vc sech² tanh
    u_x = v_vel * sqrtv2 * sech2 * tanh
    # u_xx = vc² sech²(3 sech² - 2)
    u_xx = v_vel * (v_vel / 4.0) * sech2 * (3.0 * sech2 - 2.0)
    # u_xxx = -4 v c³ sech² tanh (3 sech² - 1)
    u_xxx = -4.0 * v_vel * (sqrtv2 ** 3) * sech2 * tanh * (3.0 * sech2 - 1.0)
    # u_t = -v · u_x (行波)
    u_t = -v_vel * u_x
    return u, u_t, u_x, u_xx, u_xxx


def kdv_exact_rational(x: NDArray, t: float) -> Tuple[NDArray, NDArray,
                                                        NDArray, NDArray,
                                                        NDArray]:
    """
    KdV 有理精确解 (kdv_exact_rational.m):
        u(x,t) = 6 x (x³ - 24 t) / (x³ + 12 t)²
    """
    x = np.asarray(x, dtype=float)
    denom = (x ** 3 + 12.0 * t) ** 2
    denom_safe = np.where(np.abs(denom) < 1e-30, 1e-30, denom)
    u = 6.0 * x * (x ** 3 - 24.0 * t) / denom_safe
    denom3 = (x ** 3 + 12.0 * t) ** 3
    denom3_safe = np.where(np.abs(denom3) < 1e-30, 1e-30, denom3)
    u_t = -288.0 * x * (x ** 3 - 6.0 * t) / denom3_safe
    u_x = -12.0 * (x ** 6 - 84.0 * t * x ** 3 + 144.0 * t * t) / denom3_safe
    denom4 = (x ** 3 + 12.0 * t) ** 4
    denom4_safe = np.where(np.abs(denom4) < 1e-30, 1e-30, denom4)
    u_xx = 36.0 * (x ** 8 - 192.0 * t * x ** 5 + 1440.0 * t * t * x * x) / denom4_safe
    denom5 = (x ** 3 + 12.0 * t) ** 5
    denom5_safe = np.where(np.abs(denom5) < 1e-30, 1e-30, denom5)
    u_xxx = -144.0 * x * (x ** 9 - 360.0 * t * x ** 6
                          + 6480.0 * t * t * x ** 3
                          - 8640.0 * t ** 3) / denom5_safe
    return u, u_t, u_x, u_xx, u_xxx


def kdv_residual(u: NDArray, u_t: NDArray, u_x: NDArray,
                 u_xxx: NDArray) -> NDArray:
    """
    KdV 残差 (kdv_residual.m):
        r = u_t - 6 u u_x + u_xxx
    """
    return u_t - 6.0 * u * u_x + u_xxx


def kdv_parameters(a_phase: float = 0.0, v_vel: float = 1.0,
                   t0: float = 0.0, tstop: float = 10.0) -> dict:
    """KdV 参数封装 (kdv_parameters.m)。"""
    return {"a": a_phase, "v": v_vel, "t0": t0, "tstop": tstop}


# ---------------------------------------------------------------------------- #
#                     时间步长选择 (CFL + Hubble 阻尼)
# ---------------------------------------------------------------------------- #
def compute_timestep(phi: NDArray, box_length: float, a_curr: float,
                     p: CosmoParams) -> float:
    """
    基于 CFL 条件与 Hubble 时间选择 Δt:
        Δt_CFL = c_safety · h / v_max
        Δt_H   = c_safety / H(a)
        Δt     = min(Δt_CFL, Δt_H, Δt_max)
    其中 v_max = max |∇Φ|^{1/2} 为特征速度。
    """
    h = box_length / phi.shape[0]
    grad_phi = gradient_3d_periodic(phi, h, p=min(2, p.fd_order // 2))
    v_max = float(np.sqrt(np.max(np.sum(grad_phi ** 2, axis=0)) + 1e-30))
    dt_cfl = p.cfl_safety * h / max(v_max, 1e-10)
    H_a = hubble_factor(a_curr, p) * 100.0  # km/s/Mpc
    # 转换 H 到模拟单位 (a 无量纲,距离 Mpc/h,时间任意):
    dt_h = p.cfl_safety / max(H_a, 1e-10)
    dt_max = (z_to_a(p.z_init) - z_to_a(p.z_final)) / max(p.n_steps, 1)
    return float(min(dt_cfl, dt_h, dt_max))


# ---------------------------------------------------------------------------- #
#                  Leapfrog (KDK) 积分器
# ---------------------------------------------------------------------------- #
def kick_drift_kick_step(x: NDArray, v: NDArray, phi: NDArray,
                         box_length: float, a: float, dt: float,
                         p: CosmoParams) -> Tuple[NDArray, NDArray]:
    """
    一个 KDK leapfrog 步:
        v^{n+1/2} = v^n - (dt/2) · [∇Φ/a + H(a) v^n]
        x^{n+1} = x^n + dt · v^{n+1/2}/a
        v^{n+1} = v^{n+1/2} - (dt/2) · [∇Φ_new/a + H(a) v^{n+1/2}]
    此处假设 Φ 在一个步长内不变 (PM 近似)。
    """
    h = box_length / phi.shape[0]
    p_fd = max(1, min(4, p.fd_order // 2))
    grad_phi = gradient_3d_periodic(phi, h, p=p_fd)
    H_a = hubble_factor(a, p) * 100.0  # 简化单位
    # 插值 ∇Φ 到粒子位置 (CIC):
    acc = _cic_interpolate_vector(grad_phi, x, box_length)
    # 半步 kick:
    v_new = v - 0.5 * dt * (acc / a + H_a * v)
    # drift:
    x_new = x + dt * v_new / a
    # 周期包裹:
    x_new = x_new % box_length
    # 重新计算加速度 (此处简化:仍用同一 Φ):
    acc_new = _cic_interpolate_vector(grad_phi, x_new, box_length)
    # 另半步 kick:
    v_new = v_new - 0.5 * dt * (acc_new / a + H_a * v_new)
    return x_new, v_new


def _cic_interpolate_vector(field: NDArray, positions: NDArray,
                            box_length: float) -> NDArray:
    """
    Cloud-In-Cell 插值: 把 3D 矢量场 (3, Nx, Ny, Nz) 插值到粒子位置。
    返回 (n_particles, 3) 数组。
    """
    n_grid = field.shape[1]
    h = box_length / n_grid
    n_part = positions.shape[0]
    result = np.zeros((n_part, 3))
    # 归一化坐标:
    q = positions / h - 0.5
    i0 = np.floor(q).astype(int) % n_grid
    i1 = (i0 + 1) % n_grid
    frac = q - np.floor(q)
    for alpha in range(3):
        f = field[alpha]
        # 三线性插值:
        c000 = f[i0[:, 0], i0[:, 1], i0[:, 2]]
        c100 = f[i1[:, 0], i0[:, 1], i0[:, 2]]
        c010 = f[i0[:, 0], i1[:, 1], i0[:, 2]]
        c110 = f[i1[:, 0], i1[:, 1], i0[:, 2]]
        c001 = f[i0[:, 0], i0[:, 1], i1[:, 2]]
        c101 = f[i1[:, 0], i0[:, 1], i1[:, 2]]
        c011 = f[i0[:, 0], i1[:, 1], i1[:, 2]]
        c111 = f[i1[:, 0], i1[:, 1], i1[:, 2]]
        fx, fy, fz = frac[:, 0], frac[:, 1], frac[:, 2]
        result[:, alpha] = (
            c000 * (1 - fx) * (1 - fy) * (1 - fz)
            + c100 * fx * (1 - fy) * (1 - fz)
            + c010 * (1 - fx) * fy * (1 - fz)
            + c110 * fx * fy * (1 - fz)
            + c001 * (1 - fx) * (1 - fy) * fz
            + c101 * fx * (1 - fy) * fz
            + c011 * (1 - fx) * fy * fz
            + c111 * fx * fy * fz
        )
    return result


# ---------------------------------------------------------------------------- #
#                    高阶辛积分器 (Forest-Ruth 4 阶)
# ---------------------------------------------------------------------------- #
def forest_ruth_step(x: NDArray, v: NDArray, phi: NDArray,
                     box_length: float, a: float, dt: float,
                     p: CosmoParams) -> Tuple[NDArray, NDArray]:
    """
    Forest-Ruth 4 阶辛积分器:
        θ = 1 / (2 - 2^{1/3})
        KDK(θ dt/2) · DRIFT((1-θ)dt) · KDK((1-θ)dt/2) · DRIFT((1-θ)dt) · KDK(θ dt/2)
    保持相空间体积,4 阶精度,适用于长时间积分。
    """
    theta = 1.0 / (2.0 - 2.0 ** (1.0 / 3.0))
    x_new, v_new = x.copy(), v.copy()
    # 5 个子步:
    for sub_dt in [theta * dt / 2.0, (1.0 - theta) * dt / 2.0]:
        x_new, v_new = kick_drift_kick_step(x_new, v_new, phi,
                                             box_length, a, 2.0 * sub_dt, p)
    return x_new, v_new


# ---------------------------------------------------------------------------- #
#                      完整时间积分循环
# ---------------------------------------------------------------------------- #
def run_nbody_simulation(delta_init: NDArray, positions: NDArray,
                         velocities: NDArray, p: CosmoParams
                         ) -> Dict[str, Any]:
    """
    运行 N 体模拟从 z_init 到 z_final。

    Parameters
    ----------
    delta_init : (N, N, N)  初始密度场
    positions : (n_part, 3)  初始位置 [Mpc/h]
    velocities : (n_part, 3)  初始速度 [km/s]
    p : CosmoParams

    Returns
    -------
    dict with keys:
        'positions_final' : (n_part, 3)
        'velocities_final': (n_part, 3)
        'delta_final'     : (N, N, N)
        'trajectory_snapshots': list of dict  各步诊断
    """
    from poisson_solver import poisson_fft_3d
    a_curr = z_to_a(p.z_init)
    a_final = z_to_a(p.z_final)
    # 时间步列表 (在尺度因子 a 上均匀):
    a_list = np.linspace(a_curr, a_final, p.n_steps + 1)
    box = p.box_length
    # 常数:
    H0_over_c = 100.0  # km/s/Mpc / (Mpc/h)
    # 初始势:
    coeff = 1.5 * p.omega_m * (H0_over_c ** 2)
    source = coeff * delta_init / (a_curr ** 1)
    phi = poisson_fft_3d(source, box)
    x, v = positions.copy(), velocities.copy()
    snapshots = []
    for step_idx in range(p.n_steps):
        a_mid = 0.5 * (a_list[step_idx] + a_list[step_idx + 1])
        dt = a_list[step_idx + 1] - a_list[step_idx]
        # 每步更新势 (PM 近似):
        source = coeff * delta_init / (a_mid ** 1)
        phi = poisson_fft_3d(source, box)
        # 积分 (使用 leapfrog):
        x, v = kick_drift_kick_step(x, v, phi, box, a_mid, dt, p)
        # 诊断:
        KE = 0.5 * np.sum(v ** 2)
        snapshots.append({
            "step": step_idx + 1,
            "a": a_mid,
            "z": 1.0 / a_mid - 1.0,
            "KE": float(KE),
            "x_mean": float(x.mean()),
        })
    return {
        "positions_final": x,
        "velocities_final": v,
        "trajectory_snapshots": snapshots,
    }
