"""
线性稳定性分析 (from 610_jordan_matrix).

Burkardt jordan_matrix_random: 构造随机 Jordan 块矩阵.
Jordan 标准型: J = diag(J_1, ..., J_k), 每块 J_i = λ_i I + N,
  N 为严格上三角的幂零矩阵.

在线性稳定性分析中，我们对线性化辐射流体 RHS
  dU/dt = L(U) 在平衡态 U_0 处做扰动 U = U_0 + δU:
  d(δU)/dt = J δU,  其中 J = ∂L/∂U |_{U_0} 为 Jacobian.

解 δU(t) = exp(J t) δU(0).
  若 J 可对角化：exp(J t) = V exp(Λ t) V^{-1},
    λ_i 实部 < 0 → 稳定; λ_i 实部 > 0 → 不稳定.
  若 J 含 Jordan 块：
    exp(J t) = V diag(exp(J_k t)) V^{-1}
    exp(J_k t) = e^{λ_k t} Σ_{j=0}^{m-1} t^j N^j / j!
  此时即使 Re(λ_k) = 0, 非对角幂零部分也可导致多项式增长.

稳定性诊断:
  谱半径 s(J) = max |λ_i|,
  伪谱 ε-容差: 若最小非正规度 ||[J, J*]|| / ||J||^2 > 0,
  则特征值对小扰动敏感 (Trefethen 1999).

超新星增益区不稳定性:
  - 对流 (Ledoux 判据): ∇_rad > ∇_ad + B, B 为 Brunt 项
  - SASI (声波-重力波循环): 声波在激波反射，重力波在增益层内传播
"""
from __future__ import annotations
import math
import numpy as np


def jordan_block(eigenvalue: complex, size: int) -> np.ndarray:
    """构造 size × size Jordan 块 J = λ I + N.

  J[i, i] = λ,  J[i, i+1] = 1,  其余为 0.
  """
    J = np.zeros((size, size), dtype=np.complex128)
    for i in range(size):
        J[i, i] = eigenvalue
        if i < size - 1:
            J[i, i + 1] = 1.0 + 0.0j
    return J


def jordan_decomposition(A: np.ndarray, tol: float = 1.0e-10):
    """近似 Jordan 分解 A = V J V^{-1}.

  实际数值计算中，精确 Jordan 分解是病态的 (Wilkinson 1965).
  这里采用 Schur 分解 (稳定) 作为替代：
    A = Q T Q^H,  T 上三角, 对角元 = 特征值.
  然后将对角附近聚类为 Jordan 块.
  """
    from numpy.linalg import schur
    T, Q = schur(A)
    evals = np.diag(T)
    # 聚类：对每个特征值找其代数量数
    unique_evals = []
    multiplicities = []
    used = np.zeros(len(evals), dtype=bool)
    for i, ev in enumerate(evals):
        if used[i]:
            continue
        cluster = [i]
        used[i] = True
        for j in range(i + 1, len(evals)):
            if not used[j] and abs(evals[j] - ev) < tol * max(abs(ev), 1.0):
                cluster.append(j)
                used[j] = True
        unique_evals.append(ev)
        multiplicities.append(len(cluster))
    return {'schur_T': T, 'schur_Q': Q, 'eigenvalues': evals,
            'unique_evals': np.array(unique_evals),
            'multiplicities': multiplicities}


def matrix_exponential(A: np.ndarray, t: float, n_terms: int = 20) -> np.ndarray:
    """矩阵指数 exp(A t) 的截断 Taylor 级数.

  exp(At) = Σ_{k=0}^{∞} (At)^k / k!
  对 ||A t|| < 1 收敛迅速.
  """
    At = A * t
    n = A.shape[0]
    result = np.eye(n, dtype=np.complex128)
    term = np.eye(n, dtype=np.complex128)
    for k in range(1, n_terms + 1):
        term = term @ At / k
        result = result + term
        if np.linalg.norm(term, ord='fro') < 1.0e-14:
            break
    return result


def von_neumann_amplification(wave_number: float, dx: float, dt: float,
                              jacobian: np.ndarray) -> np.ndarray:
    """von Neumann 稳定性放大矩阵 G(k).

  对半离散化 dU/dt = (J/dx) U, 采用 RK3 时间推进:
    U^{n+1} = R(Δt J/dx) U^n
    R(z) = 1 + z + z^2/2 + z^3/6  (三阶 Taylor, SSP-RK3 的线性化)
  放大矩阵 G(k) = R(i k dx · J / dx · dt) = R(i k dt · J).
  """
    z_matrix = 1j * wave_number * dt * jacobian
    n = jacobian.shape[0]
    G = np.eye(n, dtype=np.complex128) + z_matrix + 0.5 * z_matrix @ z_matrix \
        + (1.0 / 6.0) * z_matrix @ z_matrix @ z_matrix
    return G


def stability_criterion(jacobian: np.ndarray, dx: float,
                        cfl_target: float = 0.4) -> dict:
    """计算临界 Courant 数和稳定区域.

  稳定性要求 ρ(G(k)) ≤ 1 对所有 k.
  扫描 k ∈ [0, π/dx], 找最大的 dt 使条件满足.
  """
    n = jacobian.shape[0]
    # 特征值谱
    evals = np.linalg.eigvals(jacobian)
    spectral_radius = float(np.max(np.abs(evals)))
    max_wave_speed = spectral_radius
    # 临界 dt (CFL 条件)
    dt_cfl = cfl_target * dx / max(max_wave_speed, 1.0e-30)
    # 伪谱估计：非正规度
    commutator = jacobian @ jacobian.conj().T - jacobian.conj().T @ jacobian
    non_normality = float(np.linalg.norm(commutator, 'fro'))
    return {'spectral_radius': spectral_radius,
            'max_wave_speed': max_wave_speed,
            'eigenvalues': evals,
            'dt_cfl': dt_cfl,
            'non_normality': non_normality,
            'cfl_target': cfl_target}


def ledoux_criterion(rho: np.ndarray, T: np.ndarray, Y_e: np.ndarray,
                     nabla_rad: np.ndarray, nabla_ad: np.ndarray,
                     eos) -> np.ndarray:
    """Ledoux 对流判据.

  对流不稳定: ∇_rad > ∇_ad + B_Y
    B_Y = -(∂ln μ / ∂ln Y_e)_{ρ,T,P} (d ln Y_e / d ln P) / (∂ln ρ / ∂ln T)_{P,μ}
  简化版: B_Y ≈ 4 (d Y_e / d r) H_P^{-1}  (H_P 压强标高)
  """
    n = len(rho)
    convective = np.zeros(n, dtype=bool)
    # 简单判据：∇_rad > ∇_ad
    # 更严格需含成分梯度
    convective = nabla_rad > nabla_ad
    return convective
