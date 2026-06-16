"""
hexagonal_lattice_hofstadter.py — 六角格点 Hofstadter 模型
============================================================

Hofstadter 模型描述了六角格点 (蜂窝格点) 上在垂直磁场中的紧束缚电子:

    H = -t Σ_{<ij>} e^{iθ_{ij}} c†_i c_j + h.c.

其中:
    t     — 近邻跃迁振幅
    θ_{ij} = (e/ℏ)∫_{r_i}^{r_j} A·dl  — Peierls 相位

蜂窝格点结构:
    - 两个子晶格 A, B
    - 每个格点有 3 个最近邻
    - 最近邻矢量: δ₁ = (1,0)a, δ₂ = (-1/2, √3/2)a, δ₃ = (1/2, √3/2)a

Hofstadter 蝴蝶:
    当磁通 per plaquette φ = p/q (有理数) 时,
    原始能带分裂为 q 个子带, 形成分形能谱结构.

    对于蜂窝格点, φ = Φ/(Φ₀) = BA_hex/(2πℏ/e)
    其中 A_hex = (3√3/2)a² 是六角形面积.

Chern 数与拓扑:
    每个子带具有整数 Chern 数 C_n:
        C_n = (1/2π) ∫_{BZ} F_n(k) d²k
    其中 F_n(k) = ∇×A_n(k) 是 Berry 曲率.

    TKNN 不变量: σ_xy = (e²/h) Σ_{filled} C_n

参考文献:
    [1] Hofstadter, D. R. Phys. Rev. B 14, 2239 (1976)
    [2] Thouless, D. J. et al. PRL 49, 405 (1982)
    [3] Fukui, T. et al. JPSJ 74, 1674 (2005)
"""

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as splinalg
from typing import Tuple, List, Dict, Any, Optional
from physical_constants import peierls_phase


def generate_honeycomb_flake(radius: float, a: float = 1.0
                              ) -> Tuple[np.ndarray, np.ndarray, List]:
    """生成有限六角形蜂窝格点

    蜂窝格点 = 三角格点 A + 三角格点 B (偏移 δ₁)

    A 子晶格: R_A = n₁a₁ + n₂a₂
    B 子晶格: R_B = R_A + δ₁

    其中:
        a₁ = a(1, 0)
        a₂ = a(1/2, √3/2)
        δ₁ = a(0, 1/√3)  (实际最近邻方向之一)

    Args:
        radius: flake 半径
        a: 晶格常数
    Returns:
        (pos_A, pos_B, bonds): A/B 子晶格位置和键列表
    """
    # 基矢
    a1 = a * np.array([1.0, 0.0])
    a2 = a * np.array([0.5, np.sqrt(3.0) / 2.0])
    # 最近邻矢量
    delta1 = a * np.array([0.0, 1.0 / np.sqrt(3.0)])
    delta2 = a * np.array([-0.5, -1.0 / (2 * np.sqrt(3.0))])
    delta3 = a * np.array([0.5, -1.0 / (2 * np.sqrt(3.0))])

    pos_A = []
    pos_B = []

    # 生成 A 子晶格
    n_max = int(radius / a) + 2
    for n1 in range(-n_max, n_max + 1):
        for n2 in range(-n_max, n_max + 1):
            r_A = n1 * a1 + n2 * a2
            if np.linalg.norm(r_A) <= radius:
                pos_A.append(r_A)
                r_B = r_A + delta1
                if np.linalg.norm(r_B) <= radius:
                    pos_B.append(r_B)

    pos_A = np.array(pos_A)
    pos_B = np.array(pos_B)
    N_A = len(pos_A)
    N_B = len(pos_B)
    N_total = N_A + N_B

    # 找最近邻键 (A→B)
    bonds = []
    nn_dist = a / np.sqrt(3.0)  # 最近邻距离
    tol = 0.1 * a

    for i in range(N_A):
        for j in range(N_B):
            dist = np.linalg.norm(pos_A[i] - pos_B[j])
            if abs(dist - nn_dist) < tol:
                bonds.append((i, N_A + j))  # A[i] → B[j]

    return pos_A, pos_B, bonds


def build_hofstadter_hamiltonian(pos_A: np.ndarray, pos_B: np.ndarray,
                                  bonds: List, B: float,
                                  t: float = 1.0) -> sparse.csr_matrix:
    """构建 Hofstadter 哈密顿量

    H_{ij} = -t · exp(iθ_{ij})  (对于近邻键 <ij>)
    θ_{ij} = B · x_mid · Δy  (Landau 规范 A = (0, Bx, 0))

    Args:
        pos_A: A 子晶格位置 (N_A × 2)
        pos_B: B 子晶格位置 (N_B × 2)
        bonds: 键列表 [(i_A, j_B), ...]
        B: 磁场强度
        t: 跃迁振幅
    Returns:
        H: 稀疏哈密顿量矩阵
    """
    N_A = len(pos_A)
    N_B = len(pos_B)
    N_total = N_A + N_B

    all_pos = np.vstack([pos_A, pos_B])

    rows = []
    cols = []
    vals = []

    for (i, j) in bonds:
        r_i = all_pos[i]
        r_j = all_pos[j]

        # Peierls 相位 (Landau 规范)
        x_mid = (r_i[0] + r_j[0]) / 2.0
        dy = r_j[1] - r_i[1]
        theta = B * x_mid * dy
        hopping = -t * np.exp(1j * theta)

        # H_{ij}
        rows.append(i)
        cols.append(j)
        vals.append(hopping)

        # H_{ji} = H_{ij}*
        rows.append(j)
        cols.append(i)
        vals.append(np.conj(hopping))

    H = sparse.coo_matrix((vals, (rows, cols)),
                          shape=(N_total, N_total)).tocsr()

    # 确保厄米性
    H = (H + H.conj().T) / 2.0
    return H.tocsr()


def hofstadter_spectrum(B_values: np.ndarray, radius: float = 3.0,
                        a: float = 1.0, t: float = 1.0
                        ) -> Tuple[np.ndarray, List[np.ndarray]]:
    """计算 Hofstadter 能谱 (有限 flake)

    Args:
        B_values: 磁场值数组
        radius: flake 半径
        a: 晶格常数
        t: 跃迁振幅
    Returns:
        (B_values, spectra_list): 磁场和对应的能谱
    """
    pos_A, pos_B, bonds = generate_honeycomb_flake(radius, a)

    spectra = []
    for B in B_values:
        H = build_hofstadter_hamiltonian(pos_A, pos_B, bonds, B, t)
        N = H.shape[0]
        if N <= 100:
            evals = np.linalg.eigvalsh(H.toarray())
        else:
            k = min(N - 2, 20)
            evals = splinalg.eigsh(H, k=k, which='SA',
                                    return_eigenvectors=False)
            evals = np.sort(evals)
        spectra.append(evals)

    return B_values, spectra


def compute_chern_number(H_k: callable, k_mesh: np.ndarray,
                         band_index: int = 0) -> int:
    """使用 Fukui-Hatsugai-Suzuki 方法计算 Chern 数

    Chern 数 = (1/2π) ∫_{BZ} F(k) d²k

    FHS 离散方法:
    1. 在 k-空间网格上计算 Bloch 态 |u_n(k)⟩
    2. 计算 link variable:
       U_μ(k) = ⟨u_n(k)|u_n(k+μ̂)⟩ / |⟨u_n(k)|u_n(k+μ̂)⟩|
    3. 计算 plaquette 的 Berry 通量:
       F(k) = arg[U_x(k) · U_y(k+x̂) · U_x(k+ŷ)⁻¹ · U_y(k)⁻¹]
    4. Chern 数 = (1/2π) Σ_k F(k) ∈ ℤ

    Args:
        H_k: H(k) 函数, 返回 Bloch 哈密顿量
        k_mesh: k-空间网格 (Nk × Nk × 2)
        band_index: 能带指标
    Returns:
        Chern 数 (整数)
    """
    Nk = k_mesh.shape[0]
    dk_x = k_mesh[1, 0, 0] - k_mesh[0, 0, 0]
    dk_y = k_mesh[0, 1, 1] - k_mesh[0, 0, 1]

    # 计算每个 k 点的本征态
    evecs = np.zeros((Nk, Nk, H_k(0, 0).shape[0]), dtype=complex)
    for i in range(Nk):
        for j in range(Nk):
            H = H_k(k_mesh[i, j, 0], k_mesh[i, j, 1])
            evals, eigs = np.linalg.eigh(H)
            evecs[i, j, :] = eigs[:, band_index]

    # 计算 link variables 和 Berry 通量
    F_sum = 0.0
    for i in range(Nk):
        for j in range(Nk):
            # 四个角
            k00 = (i, j)
            k10 = ((i+1) % Nk, j)
            k01 = (i, (j+1) % Nk)

            # link variables
            U_x = np.vdot(evecs[k00], evecs[k10])
            U_y_0 = np.vdot(evecs[k00], evecs[k01])
            U_x_1 = np.vdot(evecs[k01], evecs[((i+1) % Nk, (j+1) % Nk)])
            U_y_1 = np.vdot(evecs[k10], evecs[((i+1) % Nk, (j+1) % Nk)])

            # plaquette
            F = np.angle(U_x * U_x_1 * np.conj(U_y_1) * np.conj(U_y_0))
            F_sum += F

    chern = int(round(F_sum / (2 * np.pi)))
    return chern


def bloch_hamiltonian_honeycomb(kx: float, ky: float,
                                 t: float = 1.0,
                                 delta: Optional[np.ndarray] = None
                                 ) -> np.ndarray:
    """蜂窝格点的 Bloch 哈密顿量 (无磁场)

    H(k) = [0, f(k); f*(k), 0]
    f(k) = -t Σ_j exp(ik·δ_j)

    无磁场时, 在 K, K' 点有 Dirac 锥:
        E(k) = ±|f(k)| = ±t√(1 + 4cos²(k_x a/2) + 4cos(k_x a/2)cos(√3k_y a/2))

    Args:
        kx, ky: 波矢
        t: 跃迁振幅
        delta: 最近邻矢量
    Returns:
        2×2 Bloch 哈密顿量
    """
    if delta is None:
        a = 1.0
        delta = np.array([
            [0.0, a / np.sqrt(3.0)],
            [-0.5, -a / (2 * np.sqrt(3.0))],
            [0.5, -a / (2 * np.sqrt(3.0))],
        ]) * a

    k = np.array([kx, ky])
    f_k = -t * sum(np.exp(1j * np.dot(k, d)) for d in delta)

    H = np.array([[0.0, f_k],
                  [np.conj(f_k), 0.0]], dtype=complex)
    return H
