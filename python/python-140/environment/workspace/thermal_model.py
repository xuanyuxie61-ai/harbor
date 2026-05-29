"""
thermal_model.py
反应器传热模型与 PDE 求解模块
基于有限差分法求解反应器内的热传导-对流-反应耦合方程，
利用带状矩阵直接求解器处理隐式离散化产生的大型稀疏线性系统。
原项目映射:
  - 973_r8cb (紧凑带状矩阵 LU 分解与求解)
"""

import numpy as np
from utils import safe_divide, check_bounds, cond_number_estimate


class BandedMatrixSolver:
    """
    紧凑带状矩阵（Compact Banded Matrix）直接求解器。
    映射自 r8cb_np_fa.m 和 r8cb_np_sl.m。
    
    带状矩阵存储格式:
        A 为 (ml + mu + 1) x n 矩阵，
        其中 A[mu + i - j, j] = A_full[i, j]
    """

    def __init__(self, n, ml, mu):
        self.n = n
        self.ml = ml
        self.mu = mu
        self.m = mu + 1  # 主对角线行索引

    def factor(self, a):
        """
        无选主元的 LU 分解（R8CB_NP_FA）。
        
        参数:
            a: 带状矩阵, shape (ml+mu+1, n)
        返回:
            a_lu: LU 因子
            info: 0 表示成功
        """
        a_lu = np.array(a, dtype=np.float64, copy=True)
        n = self.n
        ml = self.ml
        mu = self.mu
        m = self.m
        info = 0
        ju = 0

        for k in range(n - 1):
            if abs(a_lu[m - 1, k]) < 1e-15:
                info = k + 1
                raise ValueError(f"零主元出现在第 {k} 步，矩阵可能奇异")

            lm = min(ml, n - k - 1)
            a_lu[m:m + lm, k] = -a_lu[m:m + lm, k] / a_lu[m - 1, k]

            ju = min(max(ju, mu + k), n - 1)
            mm = m
            for j in range(k + 1, ju + 1):
                mm -= 1
                a_lu[mm:mm + lm, j] += a_lu[mm - 1, j] * a_lu[m:m + lm, k]

        if abs(a_lu[m - 1, n - 1]) < 1e-15:
            info = n
            raise ValueError("最后一个主元为零，矩阵奇异")

        return a_lu, info

    def solve(self, a_lu, b):
        """
        利用 LU 因子求解线性系统 A x = b（R8CB_NP_SL）。
        
        参数:
            a_lu: LU 因子, shape (ml+mu+1, n)
            b: 右端项, shape (n,)
        返回:
            x: 解向量
        """
        n = self.n
        ml = self.ml
        mu = self.mu
        m = self.m
        x = np.array(b, dtype=np.float64, copy=True)

        # 前代求解 L y = b
        # L 为单位下三角，非对角元存储为负的乘数
        for k in range(n - 1):
            lm = min(ml, n - k - 1)
            if lm > 0:
                x[k + 1:k + 1 + lm] += a_lu[m:m + lm, k] * x[k]

        # 回代求解 U x = y
        # U 为上三角，对角元在 a_lu[m-1, :]，上对角元在 a_lu[:m-1, :]
        for k in range(n - 1, -1, -1):
            # 先减去已求出的右侧变量贡献
            um = min(mu, n - k - 1)
            for j in range(k + 1, k + 1 + um):
                row = k - j + mu  # U[k, j] 在带状存储中的行索引
                x[k] -= a_lu[row, j] * x[j]
            x[k] /= a_lu[m - 1, k]

        return x


class ThermalReactorModel:
    """
    反应器一维传热模型。
    
    控制方程（非稳态热传导-对流-反应热源耦合）:
        ρ * Cp * ∂T/∂t = k_eff * ∂²T/∂x² - ρ * Cp * u * ∂T/∂x + Q_rxn(x, t)
    
    其中:
        ρ: 堆积密度 [kg/m³]
        Cp: 比热容 [J/(kg·K)]
        k_eff: 有效导热系数 [W/(m·K)]
        u: 气流速度 [m/s]
        Q_rxn: 反应放热源项 [W/m³]
    
    离散化（隐式欧拉 + 中心差分）:
        (T_i^{n+1} - T_i^n) / Δt = α * (T_{i+1}^{n+1} - 2T_i^{n+1} + T_{i-1}^{n+1}) / Δx²
                                     - u * (T_{i+1}^{n+1} - T_{i-1}^{n+1}) / (2Δx)
                                     + Q_i / (ρ Cp)
    
    整理为带状矩阵系统 A T^{n+1} = T^n + Δt * Q / (ρ Cp)。
    """

    def __init__(self, L=1.0, nx=50, rho=200.0, Cp=1500.0, k_eff=0.15, u=0.05):
        self.L = L
        self.nx = nx
        self.dx = L / (nx - 1)
        self.rho = rho
        self.Cp = Cp
        self.k_eff = k_eff
        self.u = u
        self.alpha = k_eff / (rho * Cp)
        self.solver = BandedMatrixSolver(nx, ml=1, mu=1)

    def build_system_matrix(self, dt):
        """
        构建隐式离散化的带状矩阵（R8CB 格式）。
        对于三对角系统，ml=1, mu=1，带状矩阵 shape 为 (3, nx)。
        
        存储映射（0-based）:
            a[0, j] = A[j-1, j]  (上对角元，来自行 j-1)
            a[1, j] = A[j, j]    (对角元)
            a[2, j] = A[j+1, j]  (下对角元，来自行 j+1)
        """
        nx = self.nx
        dx = self.dx
        alpha = self.alpha
        u = self.u

        a = np.zeros((3, nx), dtype=np.float64)

        r = alpha * dt / (dx * dx)
        p = u * dt / (2.0 * dx)

        # Dirichlet 边界
        a[1, 0] = 1.0
        a[1, nx - 1] = 1.0

        # 内部节点 — 采用迎风格式处理对流项，保证对角占优
        # u > 0 时，dT/dx ≈ (T_i - T_{i-1}) / dx
        p_up = self.u * dt / self.dx
        for i in range(1, nx - 1):
            # 行 i 的方程: -(r + p_up) T_{i-1} + (1 + 2r + p_up) T_i - r T_{i+1} = RHS
            a[2, i - 1] = -(r + p_up)       # A[i, i-1] 存储于列 i-1 的行 2
            a[1, i] = 1.0 + 2.0 * r + p_up  # A[i, i]   存储于列 i   的行 1
            a[0, i + 1] = -r                # A[i, i+1] 存储于列 i+1 的行 0

        return a

    def solve_timestep(self, T_old, dt, Q_source, T_inlet=300.0):
        """
        求解一个时间步的传热方程。
        
        参数:
            T_old: 上一时刻温度场, shape (nx,)
            dt: 时间步长
            Q_source: 反应热源项, shape (nx,)
            T_inlet: 入口温度 (Dirichlet 边界)
        返回:
            T_new: 新时刻温度场
        """
        T_old = np.asarray(T_old, dtype=np.float64)
        Q_source = np.asarray(Q_source, dtype=np.float64)
        nx = self.nx

        # 构建右端项
        b = T_old.copy()
        b[1:nx - 1] += dt * Q_source[1:nx - 1] / (self.rho * self.Cp)
        b[0] = T_inlet
        b[-1] = 350.0  # 出口 Dirichlet 温度

        # 构建并分解矩阵
        a = self.build_system_matrix(dt)
        a_lu, info = self.solver.factor(a)

        # 求解
        T_new = self.solver.solve(a_lu, b)
        T_new = check_bounds(T_new, 250.0, 1500.0, name="temperature")
        return T_new

    def simulate(self, T_init, dt, n_steps, Q_func, T_inlet=300.0):
        """
        完整的时间推进模拟。
        
        参数:
            T_init: 初始温度场
            dt: 时间步长
            n_steps: 时间步数
            Q_func: 热源函数 Q_func(t, x) -> array(nx,)
            T_inlet: 入口温度
        返回:
            t_history, T_history
        """
        T = np.asarray(T_init, dtype=np.float64)
        t_history = np.zeros(n_steps + 1, dtype=np.float64)
        T_history = np.zeros((n_steps + 1, self.nx), dtype=np.float64)
        T_history[0, :] = T

        for n in range(n_steps):
            t = n * dt
            Q = Q_func(t, self.dx * np.arange(self.nx))
            T = self.solve_timestep(T, dt, Q, T_inlet)
            t_history[n + 1] = t + dt
            T_history[n + 1, :] = T

        return t_history, T_history


def compute_reaction_heat_source(x, T, kinetics, y_mass, reaction_enthalpy=-500e3):
    """
    计算反应热源项 Q_rxn [W/m³]。
    
    公式: Q_rxn = ρ * Σ (dy_i/dt * ΔH_i)
    
    参数:
        x: 空间坐标
        T: 局部温度 [K]
        kinetics: BiomassPyrolysisKinetics 实例
        y_mass: 当前质量分数
        reaction_enthalpy: 反应焓 [J/kg]
    返回:
        Q: 热源项 [W/m³]
    """
    # TODO: 实现反应热源项计算
    raise NotImplementedError("Hole 2: 请补全 compute_reaction_heat_source 函数")
