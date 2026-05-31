"""
euler_reactive_solver.py
二维可压缩反应 Euler 方程数值求解器
融合来源：978_r8crs（稀疏矩阵存储与矩阵-向量乘法）
"""
import numpy as np
from combustion_utils import check_positive, sound_speed_from_prho
from reaction_kinetics import ReactiveState, euler_flux_x, euler_flux_y, chemical_source_term


class SparseCRS:
    r"""
    压缩稀疏行 (Compressed Sparse Row, CSR/CRS) 格式矩阵。
    用于隐式时间推进时的 Jacobian 存储。
    融合来源：978_r8crs。
    """

    def __init__(self, m, n, row, col, val):
        self.m = m
        self.n = n
        self.row = np.asarray(row, dtype=int)
        self.col = np.asarray(col, dtype=int)
        self.val = np.asarray(val, dtype=float)
        self.nz = len(val)

    def multiply(self, x):
        r"""
        计算 y = A * x，其中 A 为 CRS 格式矩阵。
        对应 r8crs_mv。
        """
        x = np.asarray(x, dtype=float)
        if x.shape[0] != self.n:
            raise ValueError(f"x size {x.shape[0]} != matrix cols {self.n}")
        y = np.zeros(self.m, dtype=float)
        for i in range(self.m):
            for k in range(self.row[i], self.row[i + 1]):
                j = self.col[k]
                y[i] += self.val[k] * x[j]
        return y

    def multiply_transpose(self, x):
        r"""
        计算 y = A^T * x。
        对应 r8crs_mtv。
        """
        x = np.asarray(x, dtype=float)
        if x.shape[0] != self.m:
            raise ValueError(f"x size {x.shape[0]} != matrix rows {self.m}")
        y = np.zeros(self.n, dtype=float)
        for i in range(self.m):
            for k in range(self.row[i], self.row[i + 1]):
                j = self.col[k]
                y[j] += self.val[k] * x[i]
        return y


class ReactiveEulerSolver:
    r"""
    二维反应 Euler 方程求解器。

    方程:
        dU/dt + dF(U)/dx + dG(U)/dy = omega(U)

    采用显式 TVD Runge-Kutta 3 阶时间推进 + 空间 Lax-Friedrichs 通量。
    """

    def __init__(self, nx, ny, dx, dy, gamma=1.4, Q=2.5e6,
                 A=1.0e8, Ea=8.314e4, n_order=1.0, W_mol=0.029):
        check_positive(nx, "nx")
        check_positive(ny, "ny")
        check_positive(dx, "dx")
        check_positive(dy, "dy")
        self.nx = nx
        self.ny = ny
        self.dx = dx
        self.dy = dy
        self.gamma = gamma
        self.Q = Q
        self.A = A
        self.Ea = Ea
        self.n_order = n_order
        self.W_mol = W_mol
        self.U = np.zeros((nx, ny, 5))

    def initialize_cj_planar_wave(self, D, rho0, p0, u0=0.0, v0=0.0,
                                   lambda0=0.0, width_factor=5.0):
        r"""
        初始化一维 CJ 平面爆轰波（沿 x 方向）。
        波前在左侧（x=0 处），波后处于 CJ 平衡态。
        使用 tanh 型初始剖面:
            lambda(x) = 0.5 * (1 + tanh((x - x0) / delta))
        """
        from combustion_utils import cj_detonation_velocity, sound_speed_from_prho
        check_positive(D, "D")
        check_positive(rho0, "rho0")
        check_positive(p0, "p0")

        a0 = sound_speed_from_prho(p0, rho0, self.gamma)
        M = D / a0
        p_ratio = 1.0 + 2.0 * self.gamma / (self.gamma + 1.0) * (M * M - 1.0)
        rho_ratio = (self.gamma + 1.0) * M * M / ((self.gamma - 1.0) * M * M + 2.0)
        p1 = p0 * p_ratio
        rho1 = rho0 * rho_ratio
        u1 = D * (1.0 - rho0 / rho1)

        # 比内能
        e0 = p0 / ((self.gamma - 1.0) * rho0) + self.Q
        e1 = p1 / ((self.gamma - 1.0) * rho1)

        x0 = self.nx * self.dx * 0.3
        delta = width_factor * self.dx

        for i in range(self.nx):
            x = i * self.dx
            # tanh 过渡
            s = 0.5 * (1.0 + np.tanh((x - x0) / delta))
            rho = rho0 + (rho1 - rho0) * s
            u = u0 + (u1 - u0) * s
            v = v0
            lam = lambda0 + (1.0 - lambda0) * s
            e = e0 + (e1 - e0) * s
            state = ReactiveState(rho, u, v, e, lam)
            self.U[i, :, :] = state.to_conservative()

    def _rhs(self, U):
        r"""
        计算 dU/dt = -dF/dx - dG/dy + omega。
        """
        return self._spatial_rhs(U) + self._source_rhs(U)

    def _spatial_rhs(self, U):
        r"""
        空间导数右端项（无源项）。
        """
        nx, ny, nvar = U.shape
        dUdt = np.zeros_like(U)

        for i in range(nx - 1):
            for j in range(ny):
                stateL = ReactiveState.from_conservative(U[i, j], self.gamma, self.Q)
                stateR = ReactiveState.from_conservative(U[i + 1, j], self.gamma, self.Q)
                FL = euler_flux_x(stateL, self.gamma, self.Q, self.W_mol)
                FR = euler_flux_x(stateR, self.gamma, self.Q, self.W_mol)
                aL = sound_speed_from_prho(stateL.pressure(self.gamma, self.Q, self.W_mol), stateL.rho, self.gamma)
                aR = sound_speed_from_prho(stateR.pressure(self.gamma, self.Q, self.W_mol), stateR.rho, self.gamma)
                alpha = max(abs(stateL.u) + aL, abs(stateR.u) + aR, 1.0e-12)
                F_num = 0.5 * (FL + FR) - 0.5 * alpha * (U[i + 1, j] - U[i, j])
                dUdt[i, j] -= F_num / self.dx
                dUdt[i + 1, j] += F_num / self.dx

        for i in range(nx):
            for j in range(ny - 1):
                stateL = ReactiveState.from_conservative(U[i, j], self.gamma, self.Q)
                stateR = ReactiveState.from_conservative(U[i, j + 1], self.gamma, self.Q)
                GL = euler_flux_y(stateL, self.gamma, self.Q, self.W_mol)
                GR = euler_flux_y(stateR, self.gamma, self.Q, self.W_mol)
                aL = sound_speed_from_prho(stateL.pressure(self.gamma, self.Q, self.W_mol), stateL.rho, self.gamma)
                aR = sound_speed_from_prho(stateR.pressure(self.gamma, self.Q, self.W_mol), stateR.rho, self.gamma)
                alpha = max(abs(stateL.v) + aL, abs(stateR.v) + aR, 1.0e-12)
                G_num = 0.5 * (GL + GR) - 0.5 * alpha * (U[i, j + 1] - U[i, j])
                dUdt[i, j] -= G_num / self.dy
                dUdt[i, j + 1] += G_num / self.dy

        return dUdt

    def _source_rhs(self, U):
        r"""
        化学反应源项。
        """
        nx, ny, nvar = U.shape
        dUdt = np.zeros_like(U)
        for i in range(nx):
            for j in range(ny):
                state = ReactiveState.from_conservative(U[i, j], self.gamma, self.Q)
                dUdt[i, j] += chemical_source_term(
                    state, self.gamma, self.Q, self.A, self.Ea, self.n_order, self.W_mol
                )
        return dUdt

    def step_rk3(self, dt):
        r"""
        TVD Runge-Kutta 3 阶时间步进:
            U^(1) = U^n + dt * L(U^n)
            U^(2) = 3/4 U^n + 1/4 U^(1) + 1/4 dt * L(U^(1))
            U^(n+1) = 1/3 U^n + 2/3 U^(2) + 2/3 dt * L(U^(2))
        """
        check_positive(dt, "dt")
        U0 = self.U.copy()

        L1 = self._rhs(U0)
        U1 = U0 + dt * L1

        L2 = self._rhs(U1)
        U2 = 0.75 * U0 + 0.25 * U1 + 0.25 * dt * L2

        L3 = self._rhs(U2)
        self.U = (1.0 / 3.0) * U0 + (2.0 / 3.0) * U2 + (2.0 / 3.0) * dt * L3

        # 数值鲁棒性：密度和内能截断
        for i in range(self.nx):
            for j in range(self.ny):
                self.U[i, j, 0] = max(self.U[i, j, 0], 1.0e-9)
                self.U[i, j, 4] = max(0.0, min(self.U[i, j, 4], self.U[i, j, 0]))

    def compute_cfl_dt(self, cfl=0.3):
        r"""
        基于 CFL 条件计算时间步长:
            dt = CFL * min(dx, dy) / max(|u|+a, |v|+a)
        """
        max_speed = 1.0e-12
        for i in range(self.nx):
            for j in range(self.ny):
                state = ReactiveState.from_conservative(self.U[i, j], self.gamma, self.Q)
                p = state.pressure(self.gamma, self.Q, self.W_mol)
                a = sound_speed_from_prho(p, state.rho, self.gamma)
                speed = max(abs(state.u) + a, abs(state.v) + a)
                if speed > max_speed:
                    max_speed = speed
        dt = cfl * min(self.dx, self.dy) / max_speed
        return dt

    def advance(self, t_final, cfl=0.3, n_print=10):
        r"""
        推进到 t_final，自动 CFL 时间步长。
        """
        t = 0.0
        step = 0
        while t < t_final:
            dt = self.compute_cfl_dt(cfl)
            if t + dt > t_final:
                dt = t_final - t
            self.step_rk3(dt)
            t += dt
            step += 1
            if n_print > 0 and step % n_print == 0:
                print(f"  Step {step:5d}, t={t:.6e}, dt={dt:.6e}")
        return t, step
