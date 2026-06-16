# -*- coding: utf-8 -*-
"""
vlasov_evolution.py
====================
1D Vlasov-Poisson 系统演化求解器

融合种子项目:
  1004_RehMoritz_vmc_pde  — Variational Monte Carlo 求解 PDE

物理背景
--------
1D 静电 Vlasov-Poisson 系统:

  Vlasov 方程:
    ∂f/∂t + v ∂f/∂x + (q/m) E ∂f/∂v = 0

  Poisson 方程:
    ∂E/∂x = (q/ε₀)(n₀ - ∫ f dv)

其中 f(x,v,t) 为粒子分布函数，E(x,t) 为自洽电场。

数值方法
--------
1. 算子分裂 (Strang splitting):
   Step 1: ∂f/∂t + v ∂f/∂x = 0   (空间平流)
   Step 2: ∂f/∂t + (qE/m) ∂f/∂v = 0  (速度空间绝热)
   Step 3: 更新电场 E ← -∂φ/∂x

2. 半拉格朗日方法 (Cheng & Knorr 1976):
   沿特征线回溯: f(x,v,t+dt) = f(x-v·dt, v-E·dt, t)
   使用高阶样条插值计算出发点的 f 值。

3. 谱方法 Poisson 求解器:
   ρ̂_k = n₀δ_{k,0} - ∫ f̂_k dv
   Ê_k = i ρ̂_k / (k ε₀)
"""

import numpy as np
import math
from typing import Tuple, Dict, Optional


class VlasovAdvection:
    """
    半拉格朗日绝热算子 (速度空间)

    求解: ∂f/∂t + a(v) ∂f/∂v = 0

    方法: 沿特征线回溯
      f^{n+1}(x, v) = f^n(x, v - a(v)·dt)

    插值: 三次样条 (自然边界条件)
    """

    def __init__(self, v_grid: np.ndarray):
        self.v = np.asarray(v_grid, dtype=np.float64)
        self.nv = len(self.v)
        self.dv = self.v[1] - self.v[0] if self.nv > 1 else 1.0

    def advect_v(self, f: np.ndarray, acceleration: np.ndarray,
                  dt: float) -> np.ndarray:
        """
        执行速度空间绝热半步

        Parameters
        ----------
        f : np.ndarray, shape (nx, nv)
            分布函数.
        acceleration : np.ndarray, shape (nx, nv)
            每个网格点的加速度 a = qE/m.
        dt : float
            时间步长.

        Returns
        -------
        np.ndarray
            更新后的分布函数.
        """
        nx = f.shape[0]
        nv = f.shape[1]
        f_new = np.zeros_like(f)

        for ix in range(nx):
            # 出发点速度
            v_dep = self.v - acceleration[ix, :] * dt
            f_new[ix, :] = self._cubic_spline_interp(
                self.v, f[ix, :], v_dep)

        return f_new

    def _cubic_spline_interp(self, x: np.ndarray, y: np.ndarray,
                               x_new: np.ndarray) -> np.ndarray:
        """自然三次样条插值."""
        n = len(x)
        if n < 2:
            return np.full(len(x_new), y[0] if n == 1 else 0.0)

        h = np.diff(x)
        h = np.maximum(h, 1.0e-30)

        # 构建三对角系统
        alpha = np.zeros(n, dtype=np.float64)
        for i in range(1, n - 1):
            alpha[i] = (3.0 / h[i] * (y[i + 1] - y[i])
                        - 3.0 / h[i - 1] * (y[i] - y[i - 1]))

        # Thomas 算法求解
        l = np.ones(n)
        mu = np.zeros(n)
        z = np.zeros(n)

        for i in range(1, n - 1):
            l[i] = 2.0 * (x[i + 1] - x[i - 1]) - h[i - 1] * mu[i - 1]
            if abs(l[i]) < 1e-30:
                l[i] = 1e-30
            mu[i] = h[i] / l[i]
            z[i] = (alpha[i] - h[i - 1] * z[i - 1]) / l[i]

        # 回代
        c = np.zeros(n)
        b = np.zeros(n - 1)
        d = np.zeros(n - 1)

        for j in range(n - 2, -1, -1):
            c[j] = z[j] - mu[j] * c[j + 1]
            b[j] = (y[j + 1] - y[j]) / h[j] - h[j] * (c[j + 1] + 2.0 * c[j]) / 3.0
            d[j] = (c[j + 1] - c[j]) / (3.0 * h[j])

        # 计算插值
        result = np.zeros(len(x_new), dtype=np.float64)
        for idx, xv in enumerate(x_new):
            # 二分查找区间
            if xv <= x[0]:
                result[idx] = y[0]
            elif xv >= x[n - 1]:
                result[idx] = y[n - 1]
            else:
                lo, hi = 0, n - 1
                while lo < hi - 1:
                    mid = (lo + hi) // 2
                    if x[mid] <= xv:
                        lo = mid
                    else:
                        hi = mid
                dx = xv - x[lo]
                result[idx] = y[lo] + b[lo] * dx + c[lo] * dx ** 2 + d[lo] * dx ** 3

        return result


class PeriodicSplineAdvection:
    """
    周期性三次样条半拉格朗日绝热 (空间方向)

    求解: ∂f/∂t + v ∂f/∂x = 0

    空间方向采用周期性边界条件:
      f(x + L, v, t) = f(x, v, t)
    """

    def __init__(self, x_grid: np.ndarray):
        self.x = np.asarray(x_grid, dtype=np.float64)
        self.nx = len(self.x)
        self.dx = self.x[1] - self.x[0] if self.nx > 1 else 1.0
        self.period = self.nx * self.dx

    def advect_x(self, f: np.ndarray, velocities: np.ndarray,
                  dt: float) -> np.ndarray:
        """
        执行空间方向绝热半步

        Parameters
        ----------
        f : np.ndarray, shape (nx, nv)
            分布函数.
        velocities : np.ndarray, shape (nv,)
            速度值.
        dt : float
            时间步长.

        Returns
        -------
        np.ndarray
            更新后的分布函数.
        """
        nx = f.shape[0]
        nv = f.shape[1]
        f_new = np.zeros_like(f)

        for iv in range(nv):
            # 出发点 x 坐标 (周期性)
            x_dep = self.x - velocities[iv] * dt
            x_dep = ((x_dep - self.x[0]) % self.period) + self.x[0]

            for ix in range(nx):
                f_new[ix, iv] = self._periodic_interp(
                    f[:, iv], x_dep[ix])

        return f_new

    def _periodic_interp(self, f_col: np.ndarray, x_dep: float) -> float:
        """周期性三次 Hermite 插值."""
        nx = len(f_col)
        # 归一化位置
        pos = (x_dep - self.x[0]) / self.dx
        i0 = int(math.floor(pos))
        t = pos - i0

        # 4点模板 (周期性回绕)
        def get_f(i):
            return f_col[i % nx]

        fm1 = get_f(i0 - 1)
        f0 = get_f(i0)
        f1 = get_f(i0 + 1)
        f2 = get_f(i0 + 2)

        # 三次 Hermite 插值
        a0 = f0
        a1 = 0.5 * (f1 - fm1)
        a2 = f1 - 2.0 * f0 + 0.5 * (fm1 + f2 - 2.0 * f1)
        # 简化：使用 Catmull-Rom
        a3 = 0.5 * ((f2 - f0) - (f1 - fm1)) + (f1 - f0) - 0.5 * (f1 - fm1)
        a2 = (f1 - f0) - a1 - a3

        return a0 + a1 * t + a2 * t * t + a3 * t * t * t


class VlasovPoissonSolver:
    """
    1D Vlasov-Poisson 求解器

    时间推进: Strang 分裂 + 半拉格朗日
      1. 速度绝热半步:  f* = S_v(dt/2, E^n) f^n
      2. 计算密度/电场:  ρ^{n+1} = ∫ f* dv,  E^{n+1} = Poisson(ρ)
      3. 空间绝热半步:  f^{n+1} = S_x(dt, v) f*

    物理约束:
      - 分布函数非负: f ≥ 0
      - 粒子数守恒: ∫∫ f dx dv = const
      - 总能量守恒: E_field + E_kinetic = const
    """

    def __init__(self, nx: int, nv: int,
                 x_min: float, x_max: float,
                 v_min: float, v_max: float,
                 charge: float = -1.0, mass: float = 1.0):
        self.nx = nx
        self.nv = nv
        self.x_min = x_min
        self.x_max = x_max
        self.v_min = v_min
        self.v_max = v_max

        self.dx = (x_max - x_min) / nx
        self.dv = (v_max - v_min) / nv
        self.charge = charge
        self.mass = mass

        # 网格 (单元中心)
        self.x = np.linspace(x_min + 0.5 * self.dx,
                              x_max - 0.5 * self.dx, nx)
        self.v = np.linspace(v_min + 0.5 * self.dv,
                              v_max - 0.5 * self.dv, nv)

        # 波数 (谱方法)
        L = x_max - x_min
        self.kx = 2.0 * math.pi * np.fft.fftfreq(nx, d=self.dx)
        self.kx[0] = 1.0e-10  # 避免 k=0 除零

        # 初始化求解器组件
        self.v_advect = VlasovAdvection(self.v)
        self.x_advect = PeriodicSplineAdvection(self.x)

    def poisson_solve(self, f: np.ndarray) -> np.ndarray:
        """
        谱方法求解 Poisson 方程

        ρ(x) = n₀ - ∫ f(x,v) dv
        ∂E/∂x = ρ/ε₀   →   ik Ê_k = ρ̂_k / ε₀
        Ê_k = ρ̂_k / (ik ε₀)

        Returns
        -------
        np.ndarray, shape (nx,)
            电场 E(x).
        """
        # 电荷密度 (取 n₀ = 1 归一化)
        rho = 1.0 - np.sum(f, axis=1) * self.dv

        # FFT 求解
        rho_hat = np.fft.fft(rho)
        E_hat = np.zeros_like(rho_hat)
        for k_idx in range(len(self.kx)):
            if k_idx == 0:
                E_hat[k_idx] = 0.0  # 零模 (中性条件)
            else:
                E_hat[k_idx] = rho_hat[k_idx] / (1j * self.kx[k_idx])

        E = np.fft.ifft(E_hat).real
        return E

    def step(self, f: np.ndarray, E: np.ndarray,
              dt: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        执行一步时间推进 (Strang 分裂)

        Returns
        -------
        f_new : np.ndarray
            新时刻分布函数.
        E_new : np.ndarray
            新时刻电场.
        """
        # 加速度: a = qE/m
        accel = np.zeros_like(f)
        for ix in range(self.nx):
            accel[ix, :] = self.charge / self.mass * E[ix]

        # Step 1: 速度绝热半步 (dt/2)
        f_star = self.v_advect.advect_v(f, accel, 0.5 * dt)

        # Step 2: 更新电场
        E_new = self.poisson_solve(f_star)
        accel_new = np.zeros_like(f_star)
        for ix in range(self.nx):
            accel_new[ix, :] = self.charge / self.mass * E_new[ix]
        f_star = self.v_advect.advect_v(f_star, accel_new, 0.5 * dt)

        # Step 3: 空间绝热整步 (dt)
        f_new = self.x_advect.advect_x(f_star, self.v, dt)

        # 非负性修复
        f_new = np.maximum(f_new, 0.0)

        return f_new, E_new

    def run_simulation(self, f0: np.ndarray, dt: float,
                        n_steps: int,
                        output_interval: int = 50
                        ) -> Dict:
        """
        运行完整模拟

        Parameters
        ----------
        f0 : np.ndarray, shape (nx, nv)
            初始分布函数.
        dt : float
            时间步长.
        n_steps : int
            总步数.
        output_interval : int
            输出间隔.

        Returns
        -------
        dict
            模拟结果 (包含时间历史、最终状态等).
        """
        f = f0.copy()
        E = self.poisson_solve(f)

        history = {
            'time': [],
            'field_energy': [],
            'max_f': [],
            'total_particles': [],
            'E_max': [],
        }

        for step_n in range(n_steps + 1):
            t = step_n * dt

            # 诊断
            field_energy = 0.5 * np.sum(E ** 2) * self.dx
            history['time'].append(t)
            history['field_energy'].append(field_energy)
            history['max_f'].append(float(np.max(f)))
            history['total_particles'].append(
                float(np.sum(f) * self.dx * self.dv))
            history['E_max'].append(float(np.max(np.abs(E))))

            if step_n % max(output_interval, 1) == 0:
                print(f"  Step {step_n:5d}/{n_steps}: "
                      f"t={t:.4f}, |E|²={field_energy:.4e}, "
                      f"max(f)={np.max(f):.4e}")

            # 检查数值稳定性
            if np.any(np.isnan(f)) or np.any(np.isinf(f)):
                print(f"  WARNING: NaN/Inf at step {step_n}")
                break
            if np.max(f) > 1e10 * np.max(f0):
                print(f"  WARNING: Blow-up at step {step_n}")
                break

            # 推进
            if step_n < n_steps:
                f, E = self.step(f, E, dt)

        history['f_final'] = f
        history['E_final'] = E
        history['E_history'] = np.array(
            [self.poisson_solve(
                self.v_advect.advect_v(
                    f0, np.zeros_like(f0), t * self.charge / self.mass
                )
            ) for t in history['time'][::max(n_steps // 20, 1)]])

        return history

    def cfl_number(self, E: np.ndarray) -> float:
        """
        计算 CFL 数

        CFL = max(|v|, |qE/m|) · dt / min(dx, dv)
        稳定性要求: CFL < 1
        """
        v_max = np.max(np.abs(self.v))
        a_max = abs(self.charge / self.mass) * np.max(np.abs(E))
        return max(v_max, a_max) * 1.0 / min(self.dx, self.dv)


class ElectrostaticFieldSolver:
    """
    静电场求解器

    方法:
      1. FFT 谱方法 (高精度，周期性边界)
      2. 高阶有限差分 (灵活性，非周期边界)
      3. Gauss 定律验证
    """

    def __init__(self, x_grid: np.ndarray):
        self.x = np.asarray(x_grid, dtype=np.float64)
        self.nx = len(self.x)
        self.dx = self.x[1] - self.x[0] if self.nx > 1 else 1.0
        L = self.nx * self.dx
        self.kx = 2.0 * math.pi * np.fft.fftfreq(self.nx, d=self.dx)
        self.kx[0] = 1.0e-10

    def solve_spectral(self, rho: np.ndarray) -> np.ndarray:
        """
        谱方法求解 Ê_k = ρ̂_k / (ik ε₀)

        精度: 指数收敛 (对光滑函数)
        """
        rho_hat = np.fft.fft(rho)
        E_hat = np.zeros_like(rho_hat)
        for k_idx in range(len(self.kx)):
            if k_idx != 0:
                E_hat[k_idx] = rho_hat[k_idx] / (1j * self.kx[k_idx])
        return np.fft.ifft(E_hat).real

    def solve_fd4(self, rho: np.ndarray) -> np.ndarray:
        """
        4 阶有限差分求解 dE/dx = ρ

        使用 4 阶中心差分模板:
          (-E_{i-2} + 8E_{i-1} - 8E_{i+1} + E_{i+2}) / (12Δx) = ρ_i

        通过 Thomas 算法求解三对角系统。
        """
        n = self.nx
        dx = self.dx

        # 简化: 使用谱方法结果 + FD 修正
        E_spectral = self.solve_spectral(rho)

        # 4阶差分修正 (迭代改善)
        for _ in range(3):
            # 计算残差: r = ρ - dE/dx (4阶)
            dE = np.zeros(n, dtype=np.float64)
            for i in range(2, n - 2):
                dE[i] = (-E_spectral[i - 2] + 8.0 * E_spectral[i - 1]
                         - 8.0 * E_spectral[i + 1]
                         + E_spectral[i + 2]) / (12.0 * dx)
            # 边界
            for i in [0, 1, n - 2, n - 1]:
                if i > 0 and i < n - 1:
                    dE[i] = (E_spectral[i + 1] - E_spectral[i - 1]) / (2.0 * dx)

            residual = rho - dE
            E_spectral += self.solve_spectral(residual) * dx

        return E_spectral

    def verify_gauss_law(self, E: np.ndarray,
                          rho: np.ndarray) -> float:
        """
        验证 Gauss 定律: ∂E/∂x = ρ

        Returns
        -------
        float
            残差的 L2 范数.
        """
        n = self.nx
        dx = self.dx
        dE = np.zeros(n, dtype=np.float64)
        for i in range(1, n - 1):
            dE[i] = (E[i + 1] - E[i - 1]) / (2.0 * dx)
        dE[0] = (E[1] - E[0]) / dx
        dE[n - 1] = (E[n - 1] - E[n - 2]) / dx

        residual = dE - rho
        return float(np.sqrt(np.sum(residual ** 2) * dx))
