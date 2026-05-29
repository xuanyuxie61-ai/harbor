"""
master_equation.py - 量子点-微腔系统主方程求解模块

融合原项目 208_conservation_ode（守恒量保持的 ODE 系统）
与 764_midpoint（隐式中点积分法）的核心算法，
用于求解开放量子系统的 Lindblad 主方程：

    d rho/dt = -i/hbar [H, rho] + L[rho]

其中耗散超算符：
    L[rho] = sum_k gamma_k ( L_k rho L_k^dagger - 0.5 {L_k^dagger L_k, rho} )

本模块特别关注单光子发射动力学：
    - 二能级量子点（激发态 |e>，基态 |g>）与单模腔场耦合
    - Jaynes-Cummings 哈密顿量
    - 耗散通道：量子点自发辐射 (gamma_dot)、腔衰减 (kappa)
"""

import numpy as np
from typing import Callable, Tuple, Dict
from utils import validate_array_1d, validate_array_2d


# 物理常数
H_BAR = 1.054571817e-34


def jaynes_cummings_hamiltonian(
    omega_c: float,
    omega_dot: float,
    g_coupling: float,
    n_photon_cutoff: int = 5,
) -> np.ndarray:
    """
    构建 Jaynes-Cummings 哈密顿量矩阵（旋波近似下）：
    
        H = hbar omega_c a^dagger a + hbar omega_dot sigma_+ sigma_-
            + hbar g (a^dagger sigma_- + a sigma_+)
    
    在基矢 {|g,n>, |e,n>} 下，n = 0, ..., N-1，总维度 2N。
    """
    if n_photon_cutoff < 1:
        raise ValueError("n_photon_cutoff must be >= 1")
    # TODO Hole 1: 实现 Jaynes-Cummings 哈密顿量矩阵构建
    # 注意：基矢排序约定必须与 main.py 中的 jump operators 保持一致
    # 基矢: {|g,n>, |e,n>} (n = 0, ..., n_photon_cutoff-1)，总维度 2N
    # H = hbar omega_c a^dagger a + hbar omega_dot sigma_+ sigma_-
    #     + hbar g (a^dagger sigma_- + a sigma_+)
    raise NotImplementedError("Hole 1: 请实现 jaynes_cummings_hamiltonian 函数体")


def lindblad_dissipator(
    rho: np.ndarray,
    L: np.ndarray,
) -> np.ndarray:
    """
    计算单个 Lindblad 耗散项：
    
        D[L](rho) = L rho L^dagger - 0.5 (L^dagger L rho + rho L^dagger L)
    """
    rho = validate_array_2d(rho, "rho")
    L = validate_array_2d(L, "L")
    if rho.shape != L.shape:
        raise ValueError("rho and L must have same shape")
    Ld = L.conj().T
    term = L @ rho @ Ld
    anti = Ld @ L @ rho + rho @ Ld @ L
    return term - 0.5 * anti


def lindblad_master_equation_rhs(
    rho: np.ndarray,
    H: np.ndarray,
    jump_operators: list,
    gamma_rates: np.ndarray,
) -> np.ndarray:
    """
    计算主方程右侧：
    
        d rho/dt = -i/hbar [H, rho] + sum_k gamma_k D[L_k](rho)
    """
    rho = validate_array_2d(rho, "rho")
    H = validate_array_2d(H, "H")
    if rho.shape != H.shape:
        raise ValueError("rho and H must have same shape")
    comm = H @ rho - rho @ H
    drho = -1j / H_BAR * comm
    for Lk, gk in zip(jump_operators, gamma_rates):
        drho += gk * lindblad_dissipator(rho, Lk)
    return drho


def vectorize_density_matrix(rho: np.ndarray) -> np.ndarray:
    """
    将密度矩阵按列堆叠为向量（矢量化），用于线性化主方程。
    
        vec(rho) = [rho_00, rho_10, ..., rho_{N-1,0}, rho_01, ...]^T
    """
    rho = validate_array_2d(rho, "rho")
    return rho.T.ravel()


def unvectorize_density_matrix(vec: np.ndarray, dim: int) -> np.ndarray:
    """将向量还原为密度矩阵。"""
    vec = validate_array_1d(vec, "vec")
    if vec.size != dim * dim:
        raise ValueError("Vector size incompatible with dimension")
    return vec.reshape((dim, dim)).T


def build_liouvillian_superoperator(
    H: np.ndarray,
    jump_operators: list,
    gamma_rates: np.ndarray,
) -> np.ndarray:
    """
    构建 Liouvillian 超算符 L，使得 vec(d rho/dt) = L vec(rho)。
    
    利用恒等式：
        vec(A X B) = (B^T \otimes A) vec(X)
    
    因此：
        L = -i/hbar (I \otimes H - H^T \otimes I)
            + sum_k gamma_k [ L_k^* \otimes L_k
                             - 0.5 (I \otimes L_k^dagger L_k + (L_k^T L_k^*) \otimes I) ]
    """
    dim = H.shape[0]
    I = np.eye(dim, dtype=complex)
    L_sup = -1j / H_BAR * (np.kron(I, H) - np.kron(H.T, I))
    for Lk, gk in zip(jump_operators, gamma_rates):
        Ld = Lk.conj().T
        term1 = np.kron(Lk.T.conj(), Lk)
        term2 = 0.5 * (np.kron(I, Ld @ Lk) + np.kron((Lk.T @ Lk.conj()), I))
        L_sup += gk * (term1 - term2)
    return L_sup


def solve_steady_state(
    H: np.ndarray,
    jump_operators: list,
    gamma_rates: np.ndarray,
) -> np.ndarray:
    """
    求解稳态密度矩阵：L vec(rho_ss) = 0，且 tr(rho_ss) = 1。
    
    方法：将迹约束替换 L 的最后一行，求解线性方程组。
    """
    dim = H.shape[0]
    L_sup = build_liouvillian_superoperator(H, jump_operators, gamma_rates)
    # 替换最后一行为迹约束
    trace_constraint = np.zeros(dim * dim, dtype=complex)
    for i in range(dim):
        trace_constraint[i * dim + i] = 1.0
    L_sup[-1, :] = trace_constraint
    b = np.zeros(dim * dim, dtype=complex)
    b[-1] = 1.0
    # 使用最小二乘求解以避免奇异性
    vec_rho, residuals, rank, s = np.linalg.lstsq(L_sup, b, rcond=None)
    rho_ss = unvectorize_density_matrix(vec_rho, dim)
    # Hermitian 化与归一化
    rho_ss = 0.5 * (rho_ss + rho_ss.conj().T)
    tr = np.trace(rho_ss)
    if abs(tr) > 1e-15:
        rho_ss /= tr
    return rho_ss


def midpoint_integration_ode(
    f: Callable[[np.ndarray], np.ndarray],
    y0: np.ndarray,
    t_span: Tuple[float, float],
    n_steps: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    隐式中点法积分常微分方程：
    
        y_{n+1} = y_n + h f( (y_n + y_{n+1})/2 )
    
    源自 764_midpoint 的核心思想。对线性系统可解析求解；
    对非线性系统使用不动点迭代。
    
    此处假设 f(y) = M y 为线性，则：
        y_{n+1} = (I - h/2 M)^{-1} (I + h/2 M) y_n
    """
    y0 = validate_array_1d(y0, "y0")
    dim = y0.size
    dt = (t_span[1] - t_span[0]) / n_steps
    t_vals = np.linspace(t_span[0], t_span[1], n_steps + 1)
    y_vals = np.zeros((n_steps + 1, dim), dtype=complex)
    y_vals[0, :] = y0

    # 估计雅可比矩阵 M 的一步前向差分
    eps = 1e-8
    M = np.zeros((dim, dim), dtype=complex)
    for j in range(dim):
        e_j = np.zeros(dim, dtype=complex)
        e_j[j] = 1.0
        M[:, j] = (f(e_j * eps) - f(np.zeros(dim, dtype=complex))) / eps

    I = np.eye(dim, dtype=complex)
    LHS = I - 0.5 * dt * M
    RHS = I + 0.5 * dt * M
    try:
        inv_LHS = np.linalg.inv(LHS)
        propagator = inv_LHS @ RHS
    except np.linalg.LinAlgError:
        # 若奇异，使用伪逆
        propagator = np.linalg.pinv(LHS) @ RHS

    for n in range(n_steps):
        y_vals[n + 1, :] = propagator @ y_vals[n, :]
    return t_vals, y_vals


def solve_master_equation_time_evolution(
    H: np.ndarray,
    jump_operators: list,
    gamma_rates: np.ndarray,
    rho0: np.ndarray,
    t_span: Tuple[float, float],
    n_steps: int,
) -> Dict[str, np.ndarray]:
    """
    数值求解 Lindblad 主方程的时间演化，返回密度矩阵随时间的轨迹。
    """
    rho0 = validate_array_2d(rho0, "rho0")
    dim = H.shape[0]
    if rho0.shape != (dim, dim):
        raise ValueError("rho0 shape incompatible with H")

    def rhs_vec(y_vec: np.ndarray) -> np.ndarray:
        rho = unvectorize_density_matrix(y_vec, dim)
        drho = lindblad_master_equation_rhs(rho, H, jump_operators, gamma_rates)
        return vectorize_density_matrix(drho)

    y0 = vectorize_density_matrix(rho0)
    t_vals, y_traj = midpoint_integration_ode(rhs_vec, y0, t_span, n_steps)

    rho_traj = []
    for k in range(n_steps + 1):
        rho_k = unvectorize_density_matrix(y_traj[k, :], dim)
        # Hermitian 化
        rho_k = 0.5 * (rho_k + rho_k.conj().T)
        rho_traj.append(rho_k)

    return {
        "t": t_vals,
        "rho_traj": rho_traj,
    }


def excited_state_population(rho: np.ndarray) -> float:
    """提取量子点激发态占据概率（假设 |e,0> 为第二个基矢）。"""
    # TODO Hole 2: 实现激发态占据概率提取
    # 注意：索引必须与 jaynes_cummings_hamiltonian 中的基矢排序一致
    raise NotImplementedError("Hole 2: 请实现 excited_state_population 函数体")


def cavity_photon_number(rho: np.ndarray, n_cutoff: int) -> float:
    """计算腔中平均光子数。"""
    rho = validate_array_2d(rho, "rho")
    dim = rho.shape[0]
    if dim != 2 * n_cutoff:
        raise ValueError("Density matrix dimension mismatch with n_cutoff")
    n_avg = 0.0
    for n in range(n_cutoff):
        idx_g = 2 * n
        idx_e = 2 * n + 1
        n_avg += n * (np.real(rho[idx_g, idx_g]) + np.real(rho[idx_e, idx_e]))
    return float(n_avg)


def check_trace_conservation(rho_traj: list, tol: float = 1e-6) -> bool:
    """
    校验密度矩阵轨迹的迹守恒（源自 conservation_ode 的守恒量检测思想）。
    """
    for k, rho in enumerate(rho_traj):
        tr = np.trace(rho)
        if abs(tr - 1.0) > tol:
            return False
    return True
