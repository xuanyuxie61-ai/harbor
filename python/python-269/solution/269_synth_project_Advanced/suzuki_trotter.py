"""
suzuki_trotter.py — Suzuki-Trotter 分解与量子态时间演化
============================================================

薛定谔方程的时间演化:
    |ψ(t)⟩ = e^{-iHt/ℏ} |ψ(0)⟩

对于 H = T + V (动能 + 势能), Suzuki-Trotter 分解:

一阶 Trotter:
    e^{-i(T+V)dt} ≈ e^{-iTdt} · e^{-iVdt} + O(dt²)

二阶 Suzuki-Trotter (Strang splitting):
    e^{-i(T+V)dt} ≈ e^{-iTdt/2} · e^{-iVdt} · e^{-iTdt/2} + O(dt³)

四阶 Suzuki:
    S₄(dt) = S₂(p·dt)² · S₂((1-4p)·dt) · S₂(p·dt)²
    其中 p = 1/(4 - 4^{1/3})

关键性质:
- 酉性: ||U||₂ = 1 (保模)
- 保辛: 保持相空间体积 (Liouville 定理)
- 时间可逆: U(-dt) = U†(dt)

在量子霍尔效应中, Trotter 分解用于:
1. 波包的实时演化 (观察边缘态传播)
2. Kubo 公式计算 (线性响应)
3. 拓扑不变量的数值计算

参考文献:
    [1] Suzuki, M. Commun. Math. Phys. 51, 183 (1976)
    [2] Hatano, N. & Suzuki, M. "Finding Exponential Product Formulas"
        cond-mat/0506374
"""

import numpy as np
from scipy import sparse
from scipy.linalg import expm
from typing import Tuple, Callable, Optional


def suzuki_trotter_2nd(H_T: sparse.csr_matrix,
                        H_V: sparse.csr_matrix,
                        dt: float) -> sparse.csr_matrix:
    """二阶 Suzuki-Trotter 分解 (Strang splitting)

    U₂(dt) = e^{-iT·dt/2} · e^{-iV·dt} · e^{-iT·dt/2}

    误差: ||U_exact - U₂|| = O(dt³) per step, O(dt²) global

    对于对角势 V, e^{-iVdt} 是对角矩阵, 计算代价 O(N).
    对于动能 T (稀疏), e^{-iTdt/2} 需要矩阵指数.

    Args:
        H_T: 动能部分 (稀疏矩阵)
        H_V: 势能部分 (稀疏矩阵)
        dt: 时间步长
    Returns:
        近似时间演化算符 U(dt)
    """
    # 动能半步步 (稀疏矩阵指数 - 使用 Padé 近似)
    exp_T_half = _sparse_matrix_exp(-0.5j * dt * H_T)
    # 势能全步
    exp_V_full = _sparse_matrix_exp(-1.0j * dt * H_V)

    # U = e^{-iTdt/2} · e^{-iVdt} · e^{-iTdt/2}
    U = exp_T_half @ exp_V_full @ exp_T_half
    return U


def suzuki_trotter_4th(H_T: sparse.csr_matrix,
                        H_V: sparse.csr_matrix,
                        dt: float) -> sparse.csr_matrix:
    """四阶 Suzuki-Trotter 分解

    S₄(dt) = S₂(p·dt)² · S₂((1-4p)·dt) · S₂(p·dt)²
    p = 1/(4 - 4^{1/3}) ≈ 0.41449

    误差: O(dt⁵) per step, O(dt⁴) global

    Args:
        H_T, H_V: 动能和势能部分
        dt: 时间步长
    Returns:
        近似时间演化算符
    """
    p = 1.0 / (4.0 - 4.0 ** (1.0 / 3.0))

    S2_p = suzuki_trotter_2nd(H_T, H_V, p * dt)
    S2_14p = suzuki_trotter_2nd(H_T, H_V, (1.0 - 4.0 * p) * dt)

    U = (S2_p @ S2_p) @ S2_14p @ (S2_p @ S2_p)
    return U


def evolve_step(H: sparse.csr_matrix, psi: np.ndarray, dt: float,
                method: str = 'suzuki_trotter') -> np.ndarray:
    """执行单步时间演化

    使用指数算符直接作用: |ψ(t+dt)⟩ = e^{-iHdt}|ψ(t)⟩

    对于小矩阵, 直接使用 expm.
    对于大矩阵, 使用 Krylov 子空间方法 (Lanczos).

    Args:
        H: 哈密顿量 (完整)
        psi: 当前波函数
        dt: 时间步长
        method: 方法 ('expm_direct', 'suzuki_trotter', 'lanczos')
    Returns:
        演化后的波函数
    """
    N = len(psi)

    if method == 'expm_direct':
        # 直接矩阵指数 (小矩阵)
        U = expm(-1j * dt * H.toarray())
        return U @ psi

    elif method == 'suzuki_trotter':
        # 分解为 T + V (对角势能部分 + 其余)
        V_diag = np.diag(H.toarray()).real
        H_V = sparse.diags(V_diag)
        H_T = H - H_V
        H_T = (H_T + H_T.conj().T) / 2.0  # 确保厄米

        U = suzuki_trotter_2nd(H_T, H_V, dt)
        return U @ psi

    elif method == 'lanczos':
        # Lanczos 近似 (大矩阵)
        return _lanczos_exp(H, psi, dt)

    else:
        raise ValueError(f"未知方法: {method}")


def _sparse_matrix_exp(A: sparse.csr_matrix) -> np.ndarray:
    """计算稀疏矩阵的指数 e^A

    对于小矩阵: 使用 scipy.linalg.expm
    对于大矩阵: 使用 Padé 近似 + 缩放-平方

    Args:
        A: 稀疏矩阵
    Returns:
        e^A (稠密矩阵)
    """
    N = A.shape[0]
    if N <= 200:
        return expm(A.toarray())
    else:
        # 缩放-平方法
        norm_A = sparse.linalg.norm(A, 'fro')
        s = max(0, int(np.ceil(np.log2(norm_A / 1.0))))
        A_scaled = A / (2.0 ** s)
        # Padé(6,6) 近似
        result = _pade_approx(A_scaled.toarray(), order=6)
        for _ in range(s):
            result = result @ result
        return result


def _pade_approx(A: np.ndarray, order: int = 6) -> np.ndarray:
    """Padé 近似 e^A ≈ [p/q](A)

    Padé(p,q) 近似:
        e^A ≈ [Σ_{k=0}^p c_k A^k] · [Σ_{k=0}^q d_k A^k]^{-1}
    其中 c_k = (p+q-k)!p! / ((p+q)!k!(p-k)!)

    Args:
        A: 矩阵
        order: Padé 阶数
    Returns:
        e^A 的近似
    """
    from scipy.special import factorial

    N = A.shape[0]
    I = np.eye(N)

    # 计算系数 (对角 Padé)
    p = order
    c = np.zeros(p + 1)
    for k in range(p + 1):
        c[k] = (factorial(2*p - k) * factorial(p) /
                (factorial(2*p) * factorial(k) * factorial(p - k)))

    # 计算 A^k 和多项式
    A_powers = [I]
    for k in range(1, p + 1):
        A_powers.append(A_powers[-1] @ A)

    numerator = sum(c[k] * A_powers[k] for k in range(p + 1))
    denominator = sum((-1)**k * c[k] * A_powers[k] for k in range(p + 1))

    return np.linalg.solve(denominator, numerator)


def _lanczos_exp(H: sparse.csr_matrix, psi: np.ndarray,
                 dt: float, m: int = 30) -> np.ndarray:
    """Lanczos 方法近似矩阵指数作用 e^{-iHdt}|ψ⟩

    Lanczos 迭代构建 Krylov 子空间:
        K_m = span{|ψ⟩, H|ψ⟩, H²|ψ⟩, ..., H^{m-1}|ψ⟩}

    在 K_m 中, H 约化为三对角矩阵 T_m:
        H·V_m = V_m·T_m + β_m·v_{m+1}·e_m†

    然后: e^{-iHdt}|ψ⟩ ≈ ||ψ|| · V_m · e^{-iT_m·dt} · e_1

    Args:
        H: 哈密顿量
        psi: 初始向量
        dt: 时间步长
        m: Lanczos 步数
    Returns:
        近似 e^{-iHdt}|ψ⟩
    """
    N = len(psi)
    m = min(m, N)

    norm_psi = np.linalg.norm(psi)
    if norm_psi < 1e-15:
        return np.zeros_like(psi)

    # Lanczos 迭代
    V = np.zeros((N, m + 1), dtype=complex)
    alpha = np.zeros(m)
    beta = np.zeros(m)

    V[:, 0] = psi / norm_psi

    for j in range(m):
        w = H @ V[:, j]
        if j > 0:
            w = w - beta[j-1] * V[:, j-1]
        alpha[j] = np.real(np.vdot(V[:, j], w))
        w = w - alpha[j] * V[:, j]

        if j < m - 1:
            beta[j] = np.linalg.norm(w)
            if beta[j] < 1e-14:
                m = j + 1
                break
            V[:, j+1] = w / beta[j]

    # 构建三对角矩阵
    T_m = np.diag(alpha[:m]) + np.diag(beta[:m-1], 1) + np.diag(beta[:m-1], -1)

    # 指数化三对角矩阵
    exp_T = expm(-1j * dt * T_m)

    # 重构
    result = norm_psi * V[:, :m] @ exp_T[:, 0]
    return result


def time_evolution_sequence(H: sparse.csr_matrix, psi0: np.ndarray,
                            dt: float, n_steps: int,
                            method: str = 'suzuki_trotter'
                            ) -> Tuple[np.ndarray, np.ndarray]:
    """执行多步时间演化, 记录每步的波函数和能量

    Args:
        H: 哈密顿量
        psi0: 初始波函数
        dt: 时间步长
        n_steps: 总步数
        method: 积分方法
    Returns:
        (psi_final, energies_array)
    """
    psi = psi0.copy()
    energies = np.zeros(n_steps + 1)
    energies[0] = np.real(np.vdot(psi, H @ psi))

    for step in range(n_steps):
        psi = evolve_step(H, psi, dt, method)
        energies[step + 1] = np.real(np.vdot(psi, H @ psi))

    return psi, energies
