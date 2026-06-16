"""
拉格朗日抛射物示踪粒子 (from 744_md + 1103_QPV_Particle_Tracking).

744_md: 分子动力学速度 Verlet 积分 + 对势.
1103_QPV: 粒子轨迹分析 (MSD, looping path length).

类比：在超新星爆发中，我们对激波后的"流体包" (pocket) 做
拉格朗日追踪，观察其被抛射至星际介质的轨迹.
同时计算 MSD (均方位移) 和轨迹长度，诊断湍流混合效率.

运动方程 (简化 1D 径向):
  dr_p/dt = v_p
  dv_p/dt = - GM/r_p^2  +  (1/ρ_p) dP/dr |_{r_p}  +  a_drag

对无阻力抛射物：
  dv_p/dt = -GM/r_p^2 + (1/ρ) dP/dr
  若 P → 0 (远场): dv_p/dt = -GM/r_p^2, 得逃逸速度 v_esc = √(2GM/r).

对粒子-流体耦合 (drag):
  a_drag = -(v_p - v_fluid) / t_stop
  t_stop = (ρ_p d_p^2) / (18 μ_fluid)  (Stokes drag)

MSD (from 1103_QPV_F5):
  MSD(Δt) = <|r(t+Δt) - r(t)|^2>
  对扩散过程: MSD ∝ Δt^α,  α = 1 (正常扩散), α > 1 (超扩散).

Looping path length (from 1103_F2):
  L = Σ |r(t+dt) - r(t)|
  归一化 looping = L / |r_final - r_initial|
  looping > 1 表示轨迹存在缠绕 (对流混合的标志).
"""
from __future__ import annotations
import math
import numpy as np

import constants as C


class TracerParticles:
    """1D 径向拉格朗日示踪粒子集合."""

    def __init__(self, n_particles: int = 50, r_init: np.ndarray | None = None,
                 seed: int = 42):
        self.n_particles = int(n_particles)
        self.rng = np.random.default_rng(seed)
        if r_init is None:
            # 默认分布在 gain region (1e7 - 3e7 cm)
            self.r = self.rng.uniform(1.0e7, 3.0e7, size=self.n_particles)
        else:
            if len(r_init) != n_particles:
                raise ValueError("r_init 长度必须匹配 n_particles")
            self.r = np.asarray(r_init, dtype=np.float64).copy()
        self.v = np.zeros(self.n_particles, dtype=np.float64)
        self.history_r = [self.r.copy()]
        self.history_v = [self.v.copy()]
        self.M_proto = 1.4 * C.M_SUN

    def velocity_verlet_step(self, dt: float, r_mesh: np.ndarray,
                             v_fluid: np.ndarray, P: np.ndarray,
                             rho: np.ndarray, drag_time: float = 1.0e-3):
        """速度 Verlet 积分 (from 744_md).

        r(t+dt) = r(t) + v(t) dt + 0.5 a(t) dt^2
        v(t+dt) = v(t) + 0.5 (a(t) + a(t+dt)) dt
        """
        # 插值流体场到粒子位置
        v_fluid_p = np.interp(self.r, r_mesh, v_fluid)
        P_interp = np.interp(self.r, r_mesh, P)
        rho_interp = np.interp(self.r, r_mesh, rho)
        rho_interp = np.maximum(rho_interp, C.TINY_RHO)
        # 压强梯度 (中心差分)
        dPdr = np.gradient(P, r_mesh)
        dPdr_p = np.interp(self.r, r_mesh, dPdr)
        # 当前加速度
        a_grav = -C.G_GRAV * self.M_proto / np.maximum(self.r ** 2, 1.0e20)
        a_press = dPdr_p / rho_interp
        a_drag = -(self.v - v_fluid_p) / max(drag_time, 1.0e-10)
        a = a_grav + a_press + a_drag
        # Verlet 位置更新
        r_new = self.r + self.v * dt + 0.5 * a * dt ** 2
        # 新加速度 (在新位置重新插值)
        v_fluid_p_new = np.interp(r_new, r_mesh, v_fluid)
        P_interp_new = np.interp(r_new, r_mesh, P)
        rho_interp_new = np.interp(r_new, r_mesh, rho)
        rho_interp_new = np.maximum(rho_interp_new, C.TINY_RHO)
        dPdr_p_new = np.interp(r_new, r_mesh, dPdr)
        a_grav_new = -C.G_GRAV * self.M_proto / np.maximum(r_new ** 2, 1.0e20)
        a_press_new = dPdr_p_new / rho_interp_new
        a_drag_new = -(self.v - v_fluid_p_new) / max(drag_time, 1.0e-10)
        a_new = a_grav_new + a_press_new + a_drag_new
        # 加速度限幅 (避免极端压强梯度导致的数值爆炸)
        a_max = 1.0e18  # cm/s^2
        a = np.clip(a, -a_max, a_max)
        a_new = np.clip(a_new, -a_max, a_max)
        # 重新计算位置 (已限幅的加速度)
        r_new = self.r + self.v * dt + 0.5 * a * dt ** 2
        r_new = np.maximum(r_new, 1.0e5)
        # 速度更新
        v_new = self.v + 0.5 * (a + a_new) * dt
        # 速度限幅 (物理：不超过光速的 10%)
        v_max = 0.1 * C.C_LIGHT
        v_new = np.clip(v_new, -v_max, v_max)
        self.r = r_new
        self.v = v_new
        self.history_r.append(self.r.copy())
        self.history_v.append(self.v.copy())

    def mean_square_displacement(self, lag_max: int = 20) -> np.ndarray:
        """MSD(Δt) = <(r(t+Δt) - r(t))^2> (from 1103_QPV)."""
        n_step = len(self.history_r)
        msd = np.zeros(min(lag_max + 1, n_step), dtype=np.float64)
        for lag in range(1, min(lag_max + 1, n_step)):
            diffs = []
            for p in range(self.n_particles):
                for t0 in range(n_step - lag):
                    dr = self.history_r[t0 + lag][p] - self.history_r[t0][p]
                    diffs.append(dr ** 2)
            msd[lag] = np.mean(diffs) if diffs else 0.0
        return msd

    def looping_path_length(self) -> np.ndarray:
        """Looping path length 指标 (from 1103_F2).

        looping_i = L_i / |r_i(T) - r_i(0)|
        L_i = Σ_t |r_i(t+1) - r_i(t)|
        """
        n_step = len(self.history_r)
        looping = np.zeros(self.n_particles, dtype=np.float64)
        for p in range(self.n_particles):
            L = 0.0
            for t in range(n_step - 1):
                L += abs(self.history_r[t + 1][p] - self.history_r[t][p])
            net = abs(self.history_r[-1][p] - self.history_r[0][p])
            looping[p] = L / max(net, 1.0e5)
        return looping

    def escape_fraction(self, r_escape: float = 1.0e13) -> float:
        """逃逸率：最终位置超过 r_escape 的粒子比例."""
        n_escape = int(np.sum(self.r > r_escape))
        return n_escape / self.n_particles


def escape_velocity(M: float, r: float) -> float:
    """逃逸速度 v_esc = sqrt(2 G M / r)."""
    return math.sqrt(2.0 * C.G_GRAV * max(M, 0.0) / max(r, 1.0))
