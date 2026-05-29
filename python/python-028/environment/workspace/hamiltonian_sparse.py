"""
hamiltonian_sparse.py
=====================
核壳模型稀疏哈密顿量构建与格式转换模块

本模块实现：
1. 一维径向格点哈密顿量矩阵构建（基于有限差分法）
2. 壳模型组态空间哈密顿量的稀疏矩阵表示
3. ST (Coordinate) 格式与 CRS (Compressed Row Storage) 格式互转

数学基础：
- 径向薛定谔方程离散化：
  [-ħ²/(2M) d²/dr² + ħ²l(l+1)/(2Mr²) + V(r)] u(r) = E u(r)
  其中 u(r) = r R(r)

- 二阶导数中心差分：
  d²u/dr² ≈ (u_{i+1} - 2u_i + u_{i-1}) / Δr²

- 离散哈密顿量矩阵元：
  H_{ii}   = ħ²/(M Δr²) + V_i            (对角元)
  H_{i,i+1} = H_{i+1,i} = -ħ²/(2M Δr²)   (次对角元)
"""

import numpy as np
from math import sqrt

# 物理常数
HBARC = 197.3269804  # MeV·fm
M_NUCLEON = 939.0    # MeV/c²


def build_radial_hamiltonian_st(r_grid, potential_values, l, mass=M_NUCLEON):
    """
    构建一维径向薛定谔方程的离散哈密顿量（ST 格式）。

    方程形式：
    [-ħ²/(2M) d²/dr² + l(l+1)ħ²/(2Mr²) + V(r)] u(r) = E u(r)

    参数
    ----
    r_grid : ndarray
        径向格点，shape (N,)
    potential_values : ndarray
        对应格点上的势值 V(r)，shape (N,)
    l : int
        轨道角动量量子数
    mass : float
        核子质量

    返回
    ----
    nst : int
        非零元个数
    ist, jst : ndarray
        非零元的行、列索引
    Ast : ndarray
        非零元值
    """
    N = len(r_grid)
    if N < 3:
        raise ValueError("格点数至少为 3")
    dr = r_grid[1] - r_grid[0]
    if dr <= 0:
        raise ValueError("格点必须严格递增")

    kinetic_prefactor = HBARC ** 2 / (2.0 * mass * dr ** 2)

    ist = []
    jst = []
    Ast = []

    for i in range(N):
        r = r_grid[i]
        # 对角元：动能 + 离心势 + 中心势
        # TODO [Hole 3]: 填入离心势 V_cent 在离散哈密顿量中的表达式
        # V_cent = ħ² l(l+1) / (2M r²)，对应径向薛定谔方程的离心势项
        V_cent = 0.0  # 占位符，需要正确实现
        H_ii = 2.0 * kinetic_prefactor + potential_values[i] + V_cent
        ist.append(i)
        jst.append(i)
        Ast.append(H_ii)

        # 次对角元（右邻居）
        if i < N - 1:
            H_off = -kinetic_prefactor
            ist.append(i)
            jst.append(i + 1)
            Ast.append(H_off)

            # 对称的左邻居在遍历到 i+1 时会自动添加
            # 但 ST 格式要求显式存储所有非零元
            ist.append(i + 1)
            jst.append(i)
            Ast.append(H_off)

    return len(ist), np.array(ist), np.array(jst), np.array(Ast)


def st_to_ge(nst, ist, jst, Ast):
    """
    将 ST (Sparse Triplet) 格式转换为稠密全矩阵 (GE)。

    ST 格式以三元组 (i, j, value) 存储非零元。
    """
    if nst == 0:
        return np.zeros((0, 0))
    m = max(ist) + 1
    n = max(jst) + 1
    A = np.zeros((m, n))
    for k in range(nst):
        A[ist[k], jst[k]] = Ast[k]
    return A


def ge_to_crs(Age):
    """
    将稠密全矩阵转换为 CRS (Compressed Row Storage) 格式。

    CRS 格式：
    - row[i]：第 i 行首个非零元在 val 中的索引
    - col[j]：val[j] 对应的列索引
    - val[j]：非零元值

    该格式在稀疏矩阵-向量乘法中效率极高。
    """
    m, n = Age.shape
    row = [0]
    col = []
    val = []
    nz = 0
    for i in range(m):
        for j in range(n):
            if abs(Age[i, j]) > 1e-15:
                col.append(j)
                val.append(Age[i, j])
                nz += 1
        row.append(nz)
    return m, n, nz, np.array(row), np.array(col), np.array(val)


def st_to_crs(nst, ist, jst, Ast):
    """
    ST 格式直接转 CRS 格式（基于 st_to_ge → ge_to_crs）。
    """
    Age = st_to_ge(nst, ist, jst, Ast)
    return ge_to_crs(Age)


def crs_matvec(row, col, val, x):
    """
    CRS 格式稀疏矩阵与向量乘法 y = A x。

    复杂度 O(nz)，适用于大型壳模型计算。
    """
    n = len(row) - 1
    y = np.zeros(n)
    for i in range(n):
        for idx in range(row[i], row[i + 1]):
            y[i] += val[idx] * x[col[idx]]
    return y


def shell_model_hamiltonian_sparse(n_particles, n_orbitals, interaction_strength,
                                   single_energies, max_particles_per_orbital=2):
    """
    构建简化的壳模型组态空间哈密顿量。

    模型：
    H = Σ_i ε_i a_i^† a_i + (V/2) Σ_{ij} a_i^† a_j^† a_j a_i

    由于完整壳模型组态空间维数随粒子数指数增长，
    这里采用截断的配对近似（BCS-like 简化）。

    参数
    ----
    n_particles : int
        核子数
    n_orbitals : int
        单粒子轨道数
    interaction_strength : float
        剩余相互作用强度 (MeV)
    single_energies : ndarray
        单粒子能量，shape (n_orbitals,)
    max_particles_per_orbital : int
        每个轨道最多占据数（考虑自旋简并，通常取 2）

    返回
    ----
    row, col, val : ndarray
        CRS 格式的哈密顿量
    dim : int
        组态空间维数
    """
    # 简化：每个轨道最多 2 个粒子（自旋向上/向下）
    # 组态空间维数 = C(n_orbitals, n_particles/2) 的配对近似
    # 这里采用直接对角化小维数哈密顿量

    dim = n_orbitals * max_particles_per_orbital
    H = np.zeros((dim, dim))

    # 单粒子部分（对角）
    for i in range(n_orbitals):
        for spin in range(max_particles_per_orbital):
            idx = i * max_particles_per_orbital + spin
            H[idx, idx] = single_energies[i]

    # 配对相互作用（非对角）
    # 模拟同一条轨道内自旋相反的配对：-G P^† P
    G = interaction_strength
    for i in range(n_orbitals):
        idx_up = i * max_particles_per_orbital
        idx_down = i * max_particles_per_orbital + 1
        H[idx_up, idx_down] -= G
        H[idx_down, idx_up] -= G

    # 轨道间剩余相互作用（四极-四极耦合近似）
    for i in range(n_orbitals):
        for j in range(i + 1, n_orbitals):
            coupling = -G * 0.5 / abs(single_energies[i] - single_energies[j] + 1.0)
            for s1 in range(max_particles_per_orbital):
                for s2 in range(max_particles_per_orbital):
                    idx_i = i * max_particles_per_orbital + s1
                    idx_j = j * max_particles_per_orbital + s2
                    H[idx_i, idx_j] += coupling
                    H[idx_j, idx_i] += coupling

    # 转为 CRS
    m, n, nz, row, col, val = ge_to_crs(H)
    return row, col, val, dim


def lanczos_iteration(row, col, val, dim, n_iter, v0=None):
    """
    Lanczos 迭代算法求解稀疏矩阵的最低几个本征值。

    算法：
    1. 初始化随机向量 v₀
    2. 对 k = 1, ..., n_iter：
       w = A v_k - β_k v_{k-1}
       α_k = v_k^T w
       w = w - α_k v_k
       β_{k+1} = ||w||
       v_{k+1} = w / β_{k+1}
    3. 构建三对角矩阵 T 并对角化

    该算法在壳模型中是求基态能量的标准方法。
    """
    if v0 is None:
        v0 = np.random.randn(dim)
        v0 = v0 / np.linalg.norm(v0)

    alpha = np.zeros(n_iter)
    beta = np.zeros(n_iter + 1)

    v_prev = np.zeros(dim)
    v_curr = v0

    for k in range(n_iter):
        w = crs_matvec(row, col, val, v_curr)
        w = w - beta[k] * v_prev
        alpha[k] = np.dot(v_curr, w)
        w = w - alpha[k] * v_curr
        beta[k + 1] = np.linalg.norm(w)
        if beta[k + 1] < 1e-14:
            break
        v_prev = v_curr
        v_curr = w / beta[k + 1]

    # 构建三对角矩阵
    T = np.diag(alpha) + np.diag(beta[1:n_iter], k=1) + np.diag(beta[1:n_iter], k=-1)
    eigenvalues = np.linalg.eigvalsh(T)
    return sorted(eigenvalues)
