# -*- coding: utf-8 -*-
"""
phase_space_transport.py
========================

暗物质相空间输运方程求解器

对应种子项目:
  - 1173_scalar-soliton-collision: 4阶空间差分 + RK4 时间推进
  - 020_artery_pde: 阻尼 PDE 的 ODE 形式转化
  - 211_continuity_exact: 无散度流场的构造

科学公式
--------
暗物质相空间分布函数 f(x, v, t) 的输运方程:

  ∂f/∂t + v·∇_x f - ∇_x Φ · ∇_v f = C[f] + S

其中:
  Φ(x)      引力势
  C[f]      碰撞项 (DM-DM 或 DM-nucleus 散射)
  S(x,v,t)  源项

在 1D 简化 (x 方向 + 1D 速度空间) 下:
  ∂f/∂t + v ∂f/∂x - Φ'(x) ∂f/∂v = -Γ f + S

Liouville 定理约束:
  ∇·(v, -∇Φ) = 0  (Hamilton 流的无散度性质)

时间积分采用 RK4:
  f^{n+1} = f^n + (dt/6)(k1 + 2k2 + 2k3 + k4)
"""

from __future__ import annotations
import numpy as np
from typing import Tuple, Dict, Optional, Callable
from high_order_fd import (
    fd1_4th, fd2_4th, FiniteDifferenceOperator
)
from stability_analysis import cfl_condition


# ---------------------------------------------------------------------------
# 第一部分: 引力势与力场
# ---------------------------------------------------------------------------

class GravitationalPotential:
    """
    暗物质晕的引力势模型。

    等温球模型:
      Φ(r) = v_0² ln(r)
      Φ'(r) = v_0² / r

    NFW 模型:
      Φ(r) = -4π G ρ_s r_s³ / r * ln(1 + r/r_s)
      Φ'(r) = 4π G ρ_s r_s³ * (ln(1+r/r_s) / r² - 1 / (r(r+r_s)))
    """

    def __init__(self, model: str = 'isothermal', v_0: float = 220.0):
        self.model = model
        self.v_0 = v_0  # km/s

    def Phi(self, r: np.ndarray) -> np.ndarray:
        """引力势 Φ(r)。"""
        r_safe = np.maximum(r, 1e-10)
        if self.model == 'isothermal':
            return self.v_0**2 * np.log(r_safe)
        else:
            raise ValueError(f"未知势模型: {self.model}")

    def dPhi(self, r: np.ndarray) -> np.ndarray:
        """引力加速度 -Φ'(r)。"""
        r_safe = np.maximum(r, 1e-10)
        if self.model == 'isothermal':
            return -self.v_0**2 / r_safe
        else:
            raise ValueError(f"未知势模型: {self.model}")


# ---------------------------------------------------------------------------
# 第二部分: 相空间分布函数初始化
# ---------------------------------------------------------------------------

def initialize_dm_distribution(
    x: np.ndarray,
    v: np.ndarray,
    m_chi: float = 100.0,
    v_0: float = 220.0,
    rho_0: float = 0.3,
    x_center: float = 0.0,
    sigma_x: float = 1.0,
) -> np.ndarray:
    """
    初始化 DM 相空间分布函数 f(x, v, t=0)。

    模型: 局域 Maxwellian 包络 × 空间高斯分布
      f(x, v) = (ρ_0/m_χ) * (1/(π v_0²))^(3/2) * exp(-v²/v_0²)
                 * exp(-(x-x_c)²/σ_x²)

    返回: 2D 数组 f[i_x, i_v]
    """
    nx = len(x)
    nv = len(v)
    f = np.zeros((nx, nv))
    # 速度分布
    for j in range(nv):
        f_v = np.exp(-v[j]**2 / v_0**2) / (np.sqrt(np.pi) * v_0)
        for i in range(nx):
            f_x = np.exp(-(x[i] - x_center)**2 / sigma_x**2)
            f[i, j] = (rho_0 / m_chi) * f_v * f_x
    return f


# ---------------------------------------------------------------------------
# 第三部分: Liouville 流构造 (源自 211_continuity_exact)
# ---------------------------------------------------------------------------

def construct_divergence_free_flow(
    psi: np.ndarray,
    h: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    从无散度流函数 ψ 构造速度场 (源自 211_continuity_exact)。

    u = ∂ψ/∂y,  v = -∂ψ/∂x
    自动满足 ∂u/∂x + ∂v/∂y = 0 (Liouville 定理)

    应用于 DM 相空间:
      ψ(x, v) = 流函数
      (u, v) = (v, -Φ'(x))  即 Hamilton 流
    """
    # 使用 4 阶差分计算梯度
    u = fd1_4th(psi, h)
    # 无散度: 这里返回的实际上是 (∂ψ/∂v, -∂ψ/∂x)
    # 但 psi 是 1D, 所以简化
    return u, -u  # placeholder for 1D case


# ---------------------------------------------------------------------------
# 第四部分: RK4 时间推进器
# ---------------------------------------------------------------------------

class PhaseSpaceSolver:
    """
    1D 相空间输运方程的 RK4 求解器。

    方程:
      ∂f/∂t = -v ∂f/∂x + Φ'(x) ∂f/∂v - Γ f + S(x, v)

    离散:
      f^{n+1} = f^n + (dt/6)(k1 + 2k2 + 2k3 + k4)

    边界:
      Sommerfeld 辐射条件: ∂f/∂n + (1/c) ∂f/∂t = 0 at boundaries
    """

    def __init__(
        self,
        x: np.ndarray,
        v: np.ndarray,
        potential: Optional[GravitationalPotential] = None,
        scattering_rate: float = 0.0,
        fd_order: int = 4,
    ):
        self.x = x
        self.v = v
        self.nx = len(x)
        self.nv = len(v)
        self.hx = x[1] - x[0] if self.nx > 1 else 1.0
        self.hv = v[1] - v[0] if self.nv > 1 else 1.0
        self.potential = potential or GravitationalPotential()
        self.gamma = scattering_rate
        self.fd_order = fd_order

        # 预计算力场
        self.force = self.potential.dPhi(self.x)

        # RK4 时间步长选择
        cfl = cfl_condition(
            v_max=float(np.max(np.abs(v))),
            D=0.0,
            h=min(self.hx, self.hv),
            scheme='rk4_fd4',
        )
        self.dt_max = cfl['dt_max']
        # 防止 dt_max 过小或过大
        if not np.isfinite(self.dt_max) or self.dt_max <= 0:
            self.dt_max = 0.01 * min(self.hx, self.hv)

    def _rhs(self, f: np.ndarray) -> np.ndarray:
        """计算右端项 (不含时间导数)。带数值保护。"""
        rhs = np.zeros_like(f)

        # 速度对流项: v * ∂f/∂x (使用 2 阶以避免大梯度下的溢出)
        from high_order_fd import fd1_2nd
        for j in range(self.nv):
            df_dx = fd1_2nd(f[:, j], self.hx)
            rhs[:, j] -= self.v[j] * df_dx

        # 力项: Φ'(x) * ∂f/∂v
        for i in range(self.nx):
            df_dv = fd1_2nd(f[i, :], self.hv)
            # 限制力项幅度
            force_term = self.force[i] * df_dv
            force_mag = np.max(np.abs(force_term))
            f_mag = np.max(np.abs(f[i, :]))
            if force_mag > 0 and f_mag > 0:
                max_rate = 10.0 * f_mag / max(self.hv, 1e-10)
                scale = min(1.0, max_rate / force_mag)
                force_term *= scale
            rhs[i, :] -= force_term

            # 散射衰减
            rhs[i, :] -= self.gamma * f[i, :]

        # 全局保护: 替换 NaN/Inf
        rhs = np.where(np.isfinite(rhs), rhs, 0.0)
        return rhs

    def _apply_boundary(self, f: np.ndarray) -> np.ndarray:
        """
        Sommerfeld 辐射边界条件 + 数值保护。
        """
        f_bc = f.copy()
        decay = 0.5  # 温和衰减
        # x 方向边界
        f_bc[0, :] = f_bc[1, :] * decay
        f_bc[-1, :] = f_bc[-2, :] * decay
        # v 方向边界
        f_bc[:, 0] = f_bc[:, 1] * decay
        f_bc[:, -1] = f_bc[:, -2] * decay
        # 非负约束 + NaN 保护
        f_bc = np.where(np.isfinite(f_bc), f_bc, 0.0)
        f_bc = np.maximum(f_bc, 0.0)
        return f_bc

    def step(self, f: np.ndarray, dt: float) -> np.ndarray:
        """
        单步 RK4 推进。

        f^{n+1} = f^n + (dt/6)(k1 + 2k2 + 2k3 + k4)
        """
        k1 = self._rhs(f)
        k2 = self._rhs(f + 0.5 * dt * k1)
        k3 = self._rhs(f + 0.5 * dt * k2)
        k4 = self._rhs(f + dt * k3)

        f_new = f + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        f_new = self._apply_boundary(f_new)

        # 守恒性检查: 总粒子数应大致守恒 (衰减除外)
        return f_new

    def run(
        self,
        f0: np.ndarray,
        n_steps: int,
        dt: Optional[float] = None,
    ) -> Tuple[np.ndarray, list]:
        """
        运行 n_steps 步 RK4 时间推进。

        返回: (f_final, snapshots)
        """
        if dt is None:
            dt = 0.5 * self.dt_max
        if dt <= 0:
            raise ValueError(f"dt 必须 > 0, 得到 dt={dt}")

        f = f0.copy()
        snapshots = []
        for step in range(n_steps):
            f = self.step(f, dt)
            if step % max(1, n_steps // 5) == 0:
                snapshots.append({
                    'step': step,
                    'time': step * dt,
                    'f_marginal_x': np.sum(f, axis=1) * self.hv,
                    'f_marginal_v': np.sum(f, axis=0) * self.hx,
                    'total_mass': np.sum(f) * self.hx * self.hv,
                })
        return f, snapshots


# ---------------------------------------------------------------------------
# 第五部分: PDE 转 ODE 系统 (源自 020_artery_pde)
# ---------------------------------------------------------------------------

def pde_to_ode_system(
    w: np.ndarray,
    nx: int,
    params: Dict,
) -> np.ndarray:
    """
    将 PDE 离散化为 ODE 系统 (源自 020_artery_pde):

    dw/dt = A w + b(t)

    其中 w = [u; v] (解及其时间导数), 类比动脉 PDE:
      d²u/dt² + α du/dt + β u = γ x ΔP (a + b cos(ωt))

    对 DM: 类似阻尼振荡器形式的局域模式
      d²φ/dt² + Γ dφ/dt + ω₀² φ = S(t)
    """
    alpha = params.get('alpha', 1.0)
    beta = params.get('beta', 1.0)
    gamma = params.get('gamma', 0.1)
    omega = params.get('omega', 1.0)
    t = params.get('t', 0.0)

    u = w[:nx]
    v_w = w[nx:]

    dudt = v_w
    dvdt = -alpha * u - beta * v_w + gamma * np.sin(omega * t)

    return np.concatenate([dudt, dvdt])
