#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
coupled_pde_system.py — 热-化学-力学耦合 PDE 系统

对应种子项目:
  - 385_fem1d_approximate: 一维有限元近似与刚度矩阵组装
  - 737_matrix_analyze: 矩阵结构分析 (对称性、带状性、Cholesky)
  - 091_biochemical_nonlinear_ode: 生化反应动力学

核心数学公式:
  热传导方程 (稳态):
      -∇·(κ(ρ) ∇T) = Q(ρ, T)        in Ω
      T = T_D                          on Γ_D
      κ(ρ) ∂T/∂n = q_N                on Γ_N

  化学扩散-反应方程:
      ∂c/∂t + v·∇c = D(ρ)∇²c + R(c, T)   in Ω
      R(c, T) = k_0 exp(-E_a/(RT)) c^n     (Arrhenius 动力学)

  线弹性平衡:
      -∇·σ(u) = f_body(ρ, T)              in Ω
      σ = C(ρ) : ε(u)
      ε(u) = ½(∇u + (∇u)^T)

  材料插值 (SIMP):
      E(ρ) = E_min + ρ^p (E_0 - E_min)
      κ(ρ) = κ_min + ρ^p (κ_0 - κ_min)
      其中 p = 3 (惩罚因子), ρ ∈ [0,1] 为密度设计变量

  弱形式 (Galerkin):
      ∫_Ω κ(ρ) ∇T·∇w dV = ∫_Ω Q w dV + ∫_{Γ_N} q_N w dS
      ∫_Ω C(ρ) ε(u):ε(w) dV = ∫_Ω f·w dV + ∫_{Γ_N} t·w dS
"""

import numpy as np
from typing import Tuple, Dict, Optional
from adaptive_rk_integrator import rk_step, butchers_rk4


# ---------------------------------------------------------------------------
# 材料属性 — SIMP 插值
# ---------------------------------------------------------------------------
class MaterialModel:
    """
    SIMP (Solid Isotropic Material with Penalization) 材料插值.

    物理量随密度 ρ 的分布:
        E(ρ)     = E_min + ρ^p · (E_0 - E_min)
        κ(ρ)     = κ_min + ρ^p · (κ_0 - κ_min)
        D(ρ)     = D_min + ρ^p · (D_0 - D_min)
        α_th(ρ)  = α_min + ρ^p · (α_0 - α_min)

    其中 p = 3 为惩罚因子 (确保 0-1 拓扑).
    """

    def __init__(self,
                 E_0: float = 200e9,        # Pa (钢)
                 E_min: float = 200e9 * 1e-9,
                 kappa_0: float = 50.0,      # W/(m·K)
                 kappa_min: float = 5e-9,
                 D_0: float = 1e-5,          # m²/s (扩散系数)
                 D_min: float = 1e-14,
                 alpha_0: float = 1.2e-5,    # 1/K (热膨胀系数)
                 alpha_min: float = 1e-14,
                 penalty: float = 3.0):
        self.E_0 = E_0
        self.E_min = E_min
        self.kappa_0 = kappa_0
        self.kappa_min = kappa_min
        self.D_0 = D_0
        self.D_min = D_min
        self.alpha_0 = alpha_0
        self.alpha_min = alpha_min
        self.p = penalty

    def young(self, rho: np.ndarray) -> np.ndarray:
        rho = np.clip(rho, 0.0, 1.0)
        return self.E_min + rho ** self.p * (self.E_0 - self.E_min)

    def conductivity(self, rho: np.ndarray) -> np.ndarray:
        rho = np.clip(rho, 0.0, 1.0)
        return self.kappa_min + rho ** self.p * (self.kappa_0 - self.kappa_min)

    def diffusivity(self, rho: np.ndarray) -> np.ndarray:
        rho = np.clip(rho, 0.0, 1.0)
        return self.D_min + rho ** self.p * (self.D_0 - self.D_min)

    def thermal_expansion(self, rho: np.ndarray) -> np.ndarray:
        rho = np.clip(rho, 0.0, 1.0)
        return self.alpha_min + rho ** self.p * (self.alpha_0 - self.alpha_min)

    def density_physical(self, rho: np.ndarray) -> np.ndarray:
        rho = np.clip(rho, 0.0, 1.0)
        return rho * 7800.0  # kg/m³ (钢)


# ---------------------------------------------------------------------------
# 生化反应动力学 (来自 091)
# ---------------------------------------------------------------------------
class BiochemicalKinetics:
    """
    催化反应动力学 — Michaelis-Menten 型.

    反应速率:
        r(c, T) = k_0 exp(-E_a / (R_g T)) · c / (K_m + c)

    生化反应网络 (来自 091):
        dc/dt = -a · r_1
        dn/dt = -b · r_1
        dp/dt = r_1 - r_2
        dd/dt = r_2

    其中:
        r_1 = r_max · c/(K_c + c) · n/(K_n + n) · p
        r_2 = ε · p
    """

    def __init__(self,
                 k0: float = 1.5e6,         # 指前因子 1/s
                 Ea: float = 50e3,           # 活化能 J/mol
                 Rg: float = 8.314,          # 气体常数 J/(mol·K)
                 Km: float = 0.05,           # Michaelis 常数 mol/m³
                 r_max: float = 2.0,         # 最大反应速率 mol/(m³·s)
                 Kc: float = 0.1,
                 Kn: float = 0.05,
                 epsilon: float = 0.3):
        self.k0 = k0
        self.Ea = Ea
        self.Rg = Rg
        self.Km = Km
        self.r_max = r_max
        self.Kc = Kc
        self.Kn = Kn
        self.epsilon = epsilon

    def rate_constant(self, T: float) -> float:
        """Arrhenius 方程: k(T) = k_0 exp(-E_a / (R_g T))"""
        if T <= 0:
            return 0.0
        return self.k0 * np.exp(-self.Ea / (self.Rg * T))

    def reaction_rate(self, c: float, T: float) -> float:
        """
        总反应速率:
            r = k(T) · c / (K_m + c)
        """
        if c <= 0:
            return 0.0
        k = self.rate_constant(T)
        return k * c / (self.Km + c)

    def biochemical_rhs(self, t: float, cnpd: np.ndarray,
                        T: float = 300.0) -> np.ndarray:
        """
        生化 ODE 右端项 (来自 091).

        化学计量矩阵:
            S = [[-a, 0], [-b, 0], [1, -1], [0, 1]]

        反应速率向量:
            r = [r_max · c/(K_c+c) · n/(K_n+n) · p,  ε·p]

        Parameters
        ----------
        t    : 时间 (未显式使用, 保持接口一致)
        cnpd : [c, n, p, d] 浓度向量
        T    : 温度

        Returns
        -------
        dcnpd/dt : shape (4,)
        """
        a, b = 1.0, 0.5
        c, n, p, d = cnpd
        # 确保非负
        c, n, p, d = max(c, 0), max(n, 0), max(p, 0), max(d, 0)

        r1 = self.r_max * c / (self.Kc + c) * n / (self.Kn + n) * p
        r2 = self.epsilon * p

        S = np.array([
            [-a, 0.0],
            [-b, 0.0],
            [1.0, -1.0],
            [0.0, 1.0]
        ])
        r_vec = np.array([r1, r2])
        return S @ r_vec


# ---------------------------------------------------------------------------
# 1D 有限元组装 (来自 385 + 737)
# ---------------------------------------------------------------------------
def fem1d_assemble(n_elements: int, L: float,
                   material: MaterialModel,
                   rho: np.ndarray,
                   T_bc_left: float = 300.0,
                   T_bc_right: float = 400.0,
                   Q_source: float = 1e3
                   ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    一维稳态热传导 FEM 组装.

    弱形式:
        ∫_0^L κ(ρ) T' w' dx = ∫_0^L Q w dx

    离散:
        K T = F
    其中 K 为刚度矩阵, F 为载荷向量.

    单元刚度矩阵 (线性单元):
        K_e = κ_e / h_e · [[1, -1], [-1, 1]]

    单元载荷向量:
        F_e = Q · h_e / 2 · [1, 1]^T

    Parameters
    ----------
    n_elements : int
    L          : float — 域长度
    material   : MaterialModel
    rho        : 密度设计变量, shape (n_elements,)
    T_bc_left, T_bc_right : 边界温度
    Q_source   : 体积热源

    Returns
    -------
    (K_global, F_global, node_x)
    """
    n_nodes = n_elements + 1
    h = L / n_elements
    node_x = np.linspace(0, L, n_nodes)

    K = np.zeros((n_nodes, n_nodes))
    F = np.zeros(n_nodes)

    for e in range(n_elements):
        rho_e = rho[e] if e < len(rho) else 0.5
        kappa_e = float(material.conductivity(np.array([rho_e]))[0])

        # 单元刚度矩阵
        ke = kappa_e / h * np.array([[1, -1], [-1, 1]])
        # 单元载荷向量
        fe = Q_source * h / 2 * np.array([1, 1])

        # 组装
        dofs = [e, e + 1]
        for i_loc in range(2):
            for j_loc in range(2):
                K[dofs[i_loc], dofs[j_loc]] += ke[i_loc, j_loc]
            F[dofs[i_loc]] += fe[i_loc]

    # 施加 Dirichlet 边界条件
    K[0, :] = 0; K[0, 0] = 1.0; F[0] = T_bc_left
    K[-1, :] = 0; K[-1, -1] = 1.0; F[-1] = T_bc_right

    return K, F, node_x


def solve_thermal_field(n_elements: int, L: float,
                        material: MaterialModel,
                        rho: np.ndarray,
                        Q_source: float = 1e3
                        ) -> Tuple[np.ndarray, np.ndarray]:
    """
    求解一维温度场.

    K T = F  →  T = K^{-1} F

    Returns
    -------
    (node_x, T_field)
    """
    K, F, node_x = fem1d_assemble(n_elements, L, material, rho,
                                  Q_source=Q_source)
    T_field = np.linalg.solve(K, F)
    return node_x, T_field


def compute_compliance(K: np.ndarray, u: np.ndarray) -> float:
    """
    结构柔度 (compliance):
        C = u^T K u

    最小化柔度等价于最大化刚度.
    """
    return float(u @ K @ u)


def compute_thermal_stress_variance(T_field: np.ndarray,
                                    T_ref: float = 300.0) -> float:
    """
    热应力方差 — 热管理目标.

    σ_th ∝ E α (T - T_ref)
    方差:  Var[σ_th] = (E α)² · Var[T - T_ref]

    目标: 最小化温度分布方差 (均匀温度分布).
    """
    dT = T_field - T_ref
    return float(np.var(dT))


# ---------------------------------------------------------------------------
# 矩阵分析工具 (来自 737)
# ---------------------------------------------------------------------------
def analyze_matrix_structure(A: np.ndarray) -> Dict[str, object]:
    """
    分析矩阵结构特性.

    检查: 对称性、正定性、带状结构、条件数.
    """
    info = {}
    n = A.shape[0]
    info['size'] = (n, A.shape[1])

    # 对称性
    if n == A.shape[1]:
        sym_err = np.linalg.norm(A - A.T) / (np.linalg.norm(A) + 1e-16)
        info['is_symmetric'] = sym_err < 1e-10
        info['symmetry_error'] = float(sym_err)

        # 正定性 (通过 Cholesky)
        try:
            np.linalg.cholesky(A)
            info['is_positive_definite'] = True
        except np.linalg.LinAlgError:
            info['is_positive_definite'] = False

    # 条件数
    try:
        info['condition_number'] = float(np.linalg.cond(A))
    except Exception:
        info['condition_number'] = np.inf

    # 带宽
    if n == A.shape[1]:
        upper_band = 0
        lower_band = 0
        for i in range(n):
            for j in range(i + 1, n):
                if abs(A[i, j]) > 1e-14:
                    upper_band = max(upper_band, j - i)
                    break
            for j in range(i - 1, -1, -1):
                if abs(A[i, j]) > 1e-14:
                    lower_band = max(lower_band, i - j)
                    break
        info['upper_bandwidth'] = upper_band
        info['lower_bandwidth'] = lower_band

    return info


# ---------------------------------------------------------------------------
# 耦合求解器 — 热-化学-力学
# ---------------------------------------------------------------------------
def coupled_solve(n_elements: int, L: float,
                  material: MaterialModel,
                  kinetics: BiochemicalKinetics,
                  rho: np.ndarray,
                  T_inlet: float = 300.0,
                  c_inlet: float = 1.0,
                  Q_source: float = 1e3
                  ) -> Dict[str, np.ndarray]:
    """
    顺序耦合求解: 热场 → 化学场 → 力学场.

    步骤:
      1. 求解温度场 T(x): -κ T'' = Q
      2. 沿流向积分化学反应: dc/dx = -r(c, T) / v
      3. 计算热应力: σ_th = E α (T - T_ref)
      4. 求解位移: -∇·σ = f_th

    Returns
    -------
    dict with keys: 'x', 'T', 'c', 'stress', 'displacement'
    """
    # 1. 热场
    node_x, T_field = solve_thermal_field(n_elements, L, material, rho,
                                          Q_source)
    # 2. 化学场 — 沿 x 方向积分
    c_field = np.zeros(n_elements + 1)
    c_field[0] = c_inlet
    v_flow = 0.01  # m/s (流速)
    h_elem = L / n_elements

    tab = butchers_rk4()

    def chem_rhs(x_local, c_arr):
        T_local = np.interp(x_local, node_x, T_field)
        c_val = max(c_arr[0], 1e-15)
        r = kinetics.reaction_rate(c_val, T_local)
        return np.array([-r / v_flow])

    for e in range(n_elements):
        x0 = e * h_elem
        c0 = np.array([max(c_field[e], 1e-15)])
        h = h_elem
        c_new = rk_step(chem_rhs, x0, c0, h, tab)
        c_field[e + 1] = max(c_new[0], 0.0)

    # 3. 热应力
    T_ref = 300.0
    E_arr = material.young(rho)
    alpha_arr = material.thermal_expansion(rho)
    stress = np.zeros(n_elements)
    for e in range(n_elements):
        T_avg = 0.5 * (T_field[e] + T_field[e + 1])
        stress[e] = E_arr[e] * alpha_arr[e] * (T_avg - T_ref)

    # 4. 位移 (简化: 一维)
    n_nodes = n_elements + 1
    u_field = np.zeros(n_nodes)
    for i in range(1, n_nodes):
        u_field[i] = u_field[i - 1] + alpha_arr[min(i - 1, len(alpha_arr) - 1)] * (T_field[i] - T_ref) * h_elem

    return {
        'x': node_x,
        'T': T_field,
        'c': c_field,
        'stress': stress,
        'displacement': u_field
    }
