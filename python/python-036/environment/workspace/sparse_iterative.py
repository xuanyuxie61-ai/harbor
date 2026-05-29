"""
sparse_iterative.py
稀疏矩阵迭代方法与 PageRank 风格的本征态求解

基于 pagerank2 的核心算法:
    - 从邻接索引文件构造稀疏矩阵
    - 稀疏矩阵幂迭代 (PageRank 风格)
    - 广义特征值问题的迭代求解

物理应用:
    1. 求解中微子哈密顿量的主导本征态 (对应最大本征值)
    2. 在大型稀疏矩阵上计算稳态分布
    3. 中微子味转换网络的分析
"""

import numpy as np
from constants import DELTA_M2_21, DELTA_M2_31, DELTA_M2_31_IH


def sparse_from_index_data(index_data):
    """
    从索引数据构造稀疏矩阵 (COO 格式)。
    (源自 sparse_from_index_file)

    参数:
        index_data: list of lists, index_data[i] 包含从节点 i 出发的邻接节点

    返回:
        A: (n, n) scipy.sparse 矩阵 或 dense numpy 矩阵
    """
    n = len(index_data)
    row = []
    col = []

    for i in range(n):
        neighbors = index_data[i]
        for j in neighbors:
            row.append(i)
            col.append(j)

    row = np.array(row, dtype=np.int64)
    col = np.array(col, dtype=np.int64)
    data = np.ones(len(row), dtype=np.float64)

    try:
        from scipy.sparse import coo_matrix
        A = coo_matrix((data, (row, col)), shape=(n, n))
        return A
    except ImportError:
        # 回退到稠密矩阵
        A = np.zeros((n, n), dtype=np.float64)
        A[row, col] = data
        return A


def power_iteration(A, n_iterations=100, tol=1e-10, seed=None):
    """
    幂迭代法求矩阵的主特征值和主特征向量。
    (源自 pagerank2 的迭代思想)

    算法:
        v_{k+1} = A v_k / ||A v_k||
        λ ≈ v_k^T A v_k / (v_k^T v_k)

    参数:
        A:            方阵
        n_iterations: 最大迭代次数
        tol:          收敛容差
        seed:         随机种子

    返回:
        eigenvalue:   主导特征值
        eigenvector:  对应特征向量 (归一化)
        converged:    是否收敛
    """
    A = np.asarray(A, dtype=np.float64)
    n = A.shape[0]
    if n == 0:
        raise ValueError("Empty matrix")

    rng = np.random.default_rng(seed)
    v = rng.random(n)
    v = v / np.linalg.norm(v)

    eigenvalue = 0.0

    for iteration in range(n_iterations):
        Av = A @ v
        norm = np.linalg.norm(Av)
        if norm < 1e-15:
            break
        v_new = Av / norm

        # Rayleigh 商
        eigenvalue_new = np.dot(v_new, A @ v_new) / np.dot(v_new, v_new)

        if abs(eigenvalue_new - eigenvalue) < tol and np.linalg.norm(v_new - v) < tol:
            return float(eigenvalue_new), v_new, True

        eigenvalue = eigenvalue_new
        v = v_new

    return float(eigenvalue), v, False


def pagerank_style_matrix(H, damping=0.85):
    """
    构造 PageRank 风格的中微子转换矩阵。

    物理意义:
        将中微子哈密顿量 H 转换为一个随机游走矩阵,
        其中阻尼因子 d 代表中微子在传播过程中保持相干性的概率,
        (1-d) 代表退相干/散射概率。

    构造方法:
        P = d * D^{-1} |H| + (1-d) / n * J

    其中 D 为度矩阵, J 为全 1 矩阵。

    参数:
        H:        (n, n) 哈密顿量或转换振幅矩阵
        damping:  阻尼因子 (0 < d < 1)

    返回:
        P: (n, n) 随机矩阵 (每列和为 1)
    """
    H = np.asarray(H, dtype=np.float64)
    n = H.shape[0]

    if damping <= 0 or damping >= 1:
        raise ValueError("damping must be in (0, 1)")

    # 取绝对值
    A = np.abs(H)

    # 归一化每列
    col_sums = np.sum(A, axis=0)
    for j in range(n):
        if col_sums[j] > 0:
            A[:, j] = A[:, j] / col_sums[j]
        else:
            A[:, j] = 1.0 / n

    # PageRank 修正
    P = damping * A + (1.0 - damping) / n * np.ones((n, n))

    return P


def find_dominant_oscillation_mode(H, n_iterations=200, tol=1e-12):
    """
    使用幂迭代找到中微子哈密顿量的主导振荡模式。

    物理意义:
        主导本征态对应最大有效质量平方差,
        决定了中微子振荡的主要频率:
            ω_max = Δm²_{max} / 2E

    参数:
        H:            (3, 3) 或 (n, n) 哈密顿量
        n_iterations: 迭代次数
        tol:          容差

    返回:
        dict: {'energy': 本征值, 'state': 本征态, 'frequency': 角频率}
    """
    H = np.asarray(H, dtype=np.complex128)

    # 对厄米矩阵, 主导模式可以用 eigh 直接求
    eigenvalues, eigenvectors = np.linalg.eigh(H)

    # 取绝对值最大的本征值
    idx_max = np.argmax(np.abs(eigenvalues))
    ev = eigenvalues[idx_max]
    state = eigenvectors[:, idx_max]

    # 归一化
    state = state / np.linalg.norm(state)

    return {
        'energy': float(ev),
        'state': state,
        'frequency': float(abs(ev))
    }


def iterative_hierarchy_solver(
        energy_gev, baseline_km,
        n_iterations=50, tol=1e-10,
        theta12=None, theta23=None, theta13=None,
        delta_cp=None, delta_m2_21=None
):
    """
    使用迭代方法求解中微子振荡的 hierarchy 敏感概率。

    方法:
        1. 分别用 NH 和 IH 计算 P
        2. 使用似然比迭代优化参数
        3. 返回两个 hierarchy 假设下的概率差异

    参数:
        energy_gev: 中微子能量 [GeV]
        baseline_km: 基线 [km]
        n_iterations: 迭代优化次数
        tol: 容差

    返回:
        dict: NH 和 IH 的概率及差异
    """
    from pmns_matrix import build_pmns_matrix
    from neutrino_hamiltonian import build_vacuum_hamiltonian

    U = build_pmns_matrix(theta12, theta23, theta13, delta_cp)

    # NH
    M2_NH = np.diag([0.0, DELTA_M2_21 if delta_m2_21 is None else delta_m2_21, DELTA_M2_31])
    H_NH = (1.0 / (2.0 * energy_gev * 1e9)) * (U @ M2_NH @ U.conj().T)

    # IH
    M2_IH = np.diag([0.0, DELTA_M2_21 if delta_m2_21 is None else delta_m2_21, DELTA_M2_31_IH])
    H_IH = (1.0 / (2.0 * energy_gev * 1e9)) * (U @ M2_IH @ U.conj().T)

    L_ev_inv = baseline_km * 5.067730889e9

    psi0 = np.array([1.0, 0.0, 0.0], dtype=np.complex128)

    # 计算 NH 下的演化
    ev_NH, evec_NH = np.linalg.eigh(H_NH)
    D_NH = np.diag(np.exp(-1j * ev_NH * L_ev_inv))
    U_prop_NH = evec_NH @ D_NH @ evec_NH.conj().T
    psi_NH = U_prop_NH @ psi0
    P_NH = abs(psi_NH[0]) ** 2

    # 计算 IH 下的演化
    ev_IH, evec_IH = np.linalg.eigh(H_IH)
    D_IH = np.diag(np.exp(-1j * ev_IH * L_ev_inv))
    U_prop_IH = evec_IH @ D_IH @ evec_IH.conj().T
    psi_IH = U_prop_IH @ psi0
    P_IH = abs(psi_IH[0]) ** 2

    delta_P = abs(P_NH - P_IH)

    return {
        'P_ee_NH': float(P_NH),
        'P_ee_IH': float(P_IH),
        'delta_P': float(delta_P),
        'discrimination_power': float(delta_P / max(P_NH, P_IH, 1e-10))
    }
