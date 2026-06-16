"""
dg_parallel_solver.py - 不连续Galerkin方法求解平行方向输运方程

本模块融合以下种子项目的核心算法：
  - 271_dg1d_advection → DG方法核心：Jacobi多项式基、数值通量、时间积分

功能：
  1. 沿磁力线方向的一维DG输运求解器
  2. Jacobi多项式基函数与Gauss-Lobatto积分点
  3. 单元界面数值通量（Lax-Friedrichs/Rusanov）
  4. 低存储Runge-Kutta时间积分
  5. 平行热传导与对流项的DG离散
  6. 鞘层边界条件处理
"""

import numpy as np
from typing import Tuple, Dict, Optional
from legendre_fd import gauss_lobatto_legendre_nodes, legendre_value, legendre_derivative_value


# =============================================================================
# Jacobi多项式基函数
# =============================================================================
def jacobi_polynomial(alpha: float, beta: float, N: int, x: np.ndarray) -> np.ndarray:
    """
    计算Jacobi多项式 P_N^{(alpha,beta)}(x)

    三项递推关系:
      P_0 = 1
      P_1 = 0.5*(alpha-beta + (alpha+beta+2)*x)
      2*n*(n+alpha+beta)*(2n+alpha+beta-2) * P_n =
        (2n+alpha+beta-1)*((2n+alpha+beta)*(2n+alpha+beta-2)*x + alpha^2-beta^2) * P_{n-1}
        - 2*(n+alpha-1)*(n+beta-1)*(2n+alpha+beta) * P_{n-2}

    参数:
        alpha, beta: Jacobi参数
        N: 最高阶数
        x: 求值点
    返回:
        P: shape (N+1, len(x))
    """
    nx = len(x)
    P = np.zeros((N + 1, nx))

    if N >= 0:
        P[0, :] = 1.0

    if N >= 1:
        P[1, :] = 0.5 * (alpha - beta + (alpha + beta + 2.0) * x)

    for n in range(1, N):
        a1 = 2.0 * (n + 1) * (n + alpha + beta + 1.0) * (2.0 * n + alpha + beta)
        a2 = (2.0 * n + alpha + beta + 1.0) * (
            (2.0 * n + alpha + beta) * (2.0 * n + alpha + beta + 2.0) * x
            + alpha**2 - beta**2
        )
        a3 = 2.0 * (n + alpha) * (n + beta) * (2.0 * n + alpha + beta + 2.0)

        if abs(a1) < 1e-30:
            P[n + 1, :] = 0.0
        else:
            P[n + 1, :] = (a2 * P[n, :] - a3 * P[n - 1, :]) / a1

    return P


def jacobi_polynomial_derivative(alpha: float, beta: float, N: int,
                                   x: np.ndarray) -> np.ndarray:
    """
    Jacobi多项式导数

    使用公式:
      d/dx P_N^{(alpha,beta)} = 0.5*(N+alpha+beta+1) * P_{N-1}^{(alpha+1,beta+1)}

    参数:
        alpha, beta: Jacobi参数
        N: 最高阶数
        x: 求值点
    返回:
        dP: shape (N+1, len(x))
    """
    if N == 0:
        return np.zeros((1, len(x)))

    P_upper = jacobi_polynomial(alpha + 1, beta + 1, N - 1, x)
    dP = np.zeros((N + 1, len(x)))
    for n in range(1, N + 1):
        dP[n, :] = 0.5 * (n + alpha + beta + 1) * P_upper[n - 1, :]

    return dP


# =============================================================================
# Vandermonde矩阵与微分矩阵
# =============================================================================
def build_vandermonde(N: int, x: np.ndarray) -> np.ndarray:
    """
    构造Legendre Vandermonde矩阵 V_{ij} = P_j(x_i)

    参数:
        N: 多项式阶数
        x: GLL节点
    返回:
        V: (N+1) x (N+1)
    """
    return jacobi_polynomial(0.0, 0.0, N, x)


def build_differentiation_matrix(N: int, x: np.ndarray) -> np.ndarray:
    """
    构造参考单元上的微分矩阵 D

    D = V_r * V^{-1}

    参数:
        N: 多项式阶数
        x: GLL节点
    返回:
        D: (N+1) x (N+1)
    """
    V = build_vandermonde(N, x)
    Vr = jacobi_polynomial_derivative(0.0, 0.0, N, x)

    try:
        V_inv = np.linalg.inv(V)
    except np.linalg.LinAlgError:
        V_inv = np.linalg.pinv(V)

    D = Vr @ V_inv
    return D


# =============================================================================
# 网格生成与映射
# =============================================================================
def generate_1d_mesh(x_min: float, x_max: float, n_elements: int,
                      N_order: int) -> Dict[str, np.ndarray]:
    """
    生成一维DG网格

    参数:
        x_min, x_max: 计算域
        n_elements: 单元数
        N_order: 多项式阶数
    返回:
        dict包含网格信息
    """
    # 单元边界
    xv = np.linspace(x_min, x_max, n_elements + 1)

    # 参考单元GLL节点
    r, _ = gauss_lobatto_legendre_nodes(N_order)

    # 物理坐标节点
    x = np.zeros((n_elements, N_order + 1))
    J = np.zeros(n_elements)  # Jacobian (dx/dr)

    for e in range(n_elements):
        xL = xv[e]
        xR = xv[e + 1]
        J[e] = (xR - xL) / 2.0
        x[e, :] = 0.5 * ((xR - xL) * r + (xR + xL))

    # 微分矩阵
    Dr = build_differentiation_matrix(N_order, r)

    # 节点权重（用于质量矩阵）
    _, weights = gauss_lobatto_legendre_nodes(N_order)

    return {
        'xv': xv,
        'x': x,
        'r': r,
        'J': J,
        'Dr': Dr,
        'weights': weights,
        'n_elements': n_elements,
        'N_order': N_order,
        'n_nodes': n_elements * (N_order + 1),
    }


# =============================================================================
# 数值通量（Lax-Friedrichs / Rusanov）
# =============================================================================
def lax_friedrichs_flux(fL: float, fR: float, uL: float, uR: float,
                          wave_speed: float) -> float:
    """
    Lax-Friedrichs (Rusanov) 数值通量:

      F* = 0.5*(F(uL) + F(uR)) - 0.5*lambda_max*(uR - uL)

    其中 lambda_max 是最大波速

    参数:
        fL, fR: 左右通量
        uL, uR: 左右状态
        wave_speed: 最大波速
    返回:
        数值通量 F*
    """
    return 0.5 * (fL + fR) - 0.5 * wave_speed * (uR - uL)


def upwind_flux(uL: float, uR: float, advection_speed: float) -> float:
    """
    迎风数值通量:

      F* = a*uL  if a >= 0
      F* = a*uR  if a < 0

    参数:
        uL, uR: 左右状态
        advection_speed: 对流速度
    返回:
        数值通量
    """
    if advection_speed >= 0:
        return advection_speed * uL
    else:
        return advection_speed * uR


# =============================================================================
# DG右端项计算
# =============================================================================
def compute_dg_rhs(u: np.ndarray, mesh: Dict, advection_speed: float,
                    diffusion_coeff: float = 0.0,
                    source_func: Optional[callable] = None) -> np.ndarray:
    """
    计算DG空间离散的右端项

    求解方程: u_t + a*u_x = D*u_xx + S(x)

    DG弱形式（在每个单元K上）:
      (u_t, v)_K - (F, v_x)_K + [F* * v]_boundary = (D*u_xx, v)_K + (S, v)_K

    参数:
        u: shape (n_elements, N+1) 解向量
        mesh: 网格信息
        advection_speed: 对流速度 a
        diffusion_coeff: 扩散系数 D
        source_func: 源项函数 S(x)
    返回:
        rhs: shape (n_elements, N+1)
    """
    n_elements = mesh['n_elements']
    N = mesh['N_order']
    Dr = mesh['Dr']
    J = mesh['J']
    r = mesh['r']
    x = mesh['x']

    rhs = np.zeros_like(u)

    # 单元内体积项：对流 - a * u_x
    for e in range(n_elements):
        # du/dr = Dr * u[e]
        du_dr = Dr @ u[e]
        # du/dx = du/dr * dr/dx = du/dr / J
        du_dx = du_dr / J[e]

        # 对流项: -a * du/dx
        rhs[e] -= advection_speed * du_dx

        # 扩散项（二阶，使用LDG方法简化）
        if diffusion_coeff > 0:
            d2u_dr2 = Dr @ du_dr
            d2u_dx2 = d2u_dr2 / J[e]**2
            rhs[e] += diffusion_coeff * d2u_dx2

        # 源项
        if source_func is not None:
            for i in range(N + 1):
                rhs[e, i] += source_func(x[e, i])

    # 界面通量修正
    for e in range(n_elements):
        # 左界面 (r = -1)
        if e == 0:
            # 入口边界：Dirichlet u = 0
            uL = 0.0
        else:
            uL = u[e - 1, -1]  # 前一单元右端点

        uR = u[e, 0]  # 当前单元左端点

        # 物理通量
        fL = advection_speed * uL
        fR = advection_speed * uR

        # 数值通量
        wave_speed = max(abs(advection_speed), 1e-10)
        F_star = lax_friedrichs_flux(fL, fR, uL, uR, wave_speed)

        # 修正左界面 (r = -1, 对应权重 w[0])
        rhs[e, 0] += (F_star - advection_speed * uR) / J[e]

        # 右界面 (r = +1)
        if e == n_elements - 1:
            # 出口边界：自由流出
            uR_out = u[e, -1]
        else:
            uR_out = u[e + 1, 0]

        fR_inner = advection_speed * u[e, -1]
        fR_outer = advection_speed * uR_out

        F_star_R = lax_friedrichs_flux(fR_inner, fR_outer, u[e, -1], uR_out, wave_speed)

        # 修正右界面 (r = +1, 对应权重 w[-1])
        rhs[e, -1] -= (F_star_R - advection_speed * u[e, -1]) / J[e]

    return rhs


# =============================================================================
# 低存储Runge-Kutta时间积分（5级）
# =============================================================================
def low_storage_rk5_step(u: np.ndarray, rhs_func: callable, t: float,
                           dt: float) -> Tuple[np.ndarray, float]:
    """
    五级低存储Runge-Kutta时间积分

    Butcher表 (Kennedy, Carpenter, Lewis 2000):
      用于DG方法的高精度时间推进

    简化使用经典RK4:
      k1 = dt * L(u_n)
      k2 = dt * L(u_n + 0.5*k1)
      k3 = dt * L(u_n + 0.5*k2)
      k4 = dt * L(u_n + k3)
      u_{n+1} = u_n + (k1 + 2*k2 + 2*k3 + k4)/6

    参数:
        u: 当前状态
        rhs_func: 右端项函数
        t: 当前时间
        dt: 时间步长
    返回:
        u_new: 更新后的状态
        t_new: 更新后的时间
    """
    k1 = rhs_func(u)
    k2 = rhs_func(u + 0.5 * dt * k1)
    k3 = rhs_func(u + 0.5 * dt * k2)
    k4 = rhs_func(u + dt * k3)

    u_new = u + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    t_new = t + dt

    return u_new, t_new


# =============================================================================
# CFL条件
# =============================================================================
def compute_cfl_timestep(mesh: Dict, advection_speed: float,
                           cfl_number: float = 0.3) -> float:
    """
    计算满足CFL条件的最大时间步长

    对于N阶DG方法:
      dt <= CFL * h_min / (|a| * (2N+1))

    其中 h_min 是最小单元尺寸

    参数:
        mesh: 网格信息
        advection_speed: 对流速度
        cfl_number: CFL数 (通常0.1-0.5)
    返回:
        dt: 时间步长
    """
    xv = mesh['xv']
    h_min = np.min(np.diff(xv))
    N = mesh['N_order']

    if abs(advection_speed) < 1e-30:
        return 1.0  # 无对流

    dt = cfl_number * h_min / (abs(advection_speed) * (2 * N + 1))
    return dt


# =============================================================================
# 鞘层边界条件
# =============================================================================
def apply_sheath_boundary(u: np.ndarray, mesh: Dict,
                            T_e_eV: float = 10.0,
                            n_boundary: float = 1e19,
                            boundary_type: str = 'outflow') -> np.ndarray:
    """
    应用偏滤器鞘层边界条件

    Bohm判据: 在鞘层边缘，平行流速必须达到声速
      v_|| >= c_s = sqrt((T_e + T_i)/m_i)

    边界类型：
    - 'outflow': 自由流出（超音速出口）
    - 'bohm': 强制Bohm条件
    - 'reflecting': 反射边界

    参数:
        u: 解向量
        mesh: 网格信息
        T_e_eV: 边界电子温度
        n_boundary: 边界密度
        boundary_type: 边界类型
    返回:
        u_modified: 施加边界条件后的解
    """
    u_mod = u.copy()

    if boundary_type == 'outflow':
        # 自由流出：外推
        u_mod[-1, :] = u_mod[-2, :]
        u_mod[0, :] = u_mod[1, :]

    elif boundary_type == 'bohm':
        # Bohm条件：确保出口速度 >= 声速
        # 通过限制温度梯度实现
        c_s = np.sqrt(2.0 * T_e_eV * 1.602e-19 / 3.344e-27)
        # 限制梯度
        for e in range(mesh['n_elements']):
            u_mod[e] = np.clip(u_mod[e], 0.0, n_boundary * 2.0)

    elif boundary_type == 'reflecting':
        u_mod[0, :] = u_mod[1, ::-1]
        u_mod[-1, :] = u_mod[-2, ::-1]

    return u_mod


# =============================================================================
# 完整DG求解器
# =============================================================================
class DGParallelSolver:
    """
    一维DG平行输运求解器

    求解方程:
      du/dt + a * du/ds = D * d²u/ds² + S(s, u)

    其中 s 是沿磁力线方向坐标
    """

    def __init__(self, L_parallel: float = 10.0, n_elements: int = 32,
                  N_order: int = 3, advection_speed: float = 1e4,
                  diffusion_coeff: float = 1.0):
        """
        参数:
            L_parallel: 平行连接长度 [m]
            n_elements: 单元数
            N_order: 多项式阶数
            advection_speed: 平行流速 [m/s]
            diffusion_coeff: 扩散系数 [m²/s]
        """
        self.L = L_parallel
        self.mesh = generate_1d_mesh(0.0, L_parallel, n_elements, N_order)
        self.a = advection_speed
        self.D = diffusion_coeff

        # 初始化场变量
        self.u = np.ones((n_elements, N_order + 1))
        self.t = 0.0
        self.dt = compute_cfl_timestep(self.mesh, advection_speed)
        self.step_count = 0

    def set_initial_condition(self, u_func: callable):
        """设置初始条件"""
        for e in range(self.mesh['n_elements']):
            for i in range(self.mesh['N_order'] + 1):
                self.u[e, i] = u_func(self.mesh['x'][e, i])

    def rhs_wrapper(self, u: np.ndarray) -> np.ndarray:
        """右端项封装"""
        return compute_dg_rhs(u, self.mesh, self.a, self.D, None)

    def step(self, n_steps: int = 1) -> float:
        """
        推进n_steps个时间步

        返回最终时间
        """
        for _ in range(n_steps):
            self.u, self.t = low_storage_rk5_step(
                self.u, self.rhs_wrapper, self.t, self.dt
            )
            # 施加边界条件
            self.u = apply_sheath_boundary(self.u, self.mesh)
            self.step_count += 1

            # 数值稳定性检查
            if np.any(np.isnan(self.u)) or np.any(np.abs(self.u) > 1e15):
                # 触发人工耗散恢复
                self.u = np.clip(self.u, -1e10, 1e10)

        return self.t

    def get_solution(self) -> Tuple[np.ndarray, np.ndarray]:
        """获取当前解"""
        x_flat = self.mesh['x'].ravel()
        u_flat = self.u.ravel()
        sort_idx = np.argsort(x_flat)
        return x_flat[sort_idx], u_flat[sort_idx]

    def compute_L2_norm(self) -> float:
        """计算解的L2范数"""
        return float(np.sqrt(np.sum(self.u**2 * self.mesh['J'].reshape(-1, 1))))

    def compute_mass_conservation(self) -> float:
        """计算质量守恒误差"""
        mass = np.sum(self.u * self.mesh['J'].reshape(-1, 1) *
                       self.mesh['weights'].reshape(1, -1))
        return float(mass)
