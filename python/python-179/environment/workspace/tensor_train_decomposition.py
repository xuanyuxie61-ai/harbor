"""
tensor_train_decomposition.py
张量列车（Tensor Train, TT）分解核心模块
========================================
实现张量列车格式的高效表示与 ALS 交叉逼近算法。

数学基础
--------
对 d 阶张量 A∈ℝ^{n1×n2×...×nd}，TT 分解表示为
    A(i1,...,id) = Σ_{α0,...,αd}  G1(i1)_{α0,α1} G2(i2)_{α1,α2} ... Gd(id)_{α_{d-1},α_d}
其中 r0 = rd = 1，rk 为 TT-ranks，Gk ∈ ℝ^{rk-1 × nk × rk} 为核心张量。

等价地，对左 unfolding A^{(k)} ∈ ℝ^{(n1...nk) × (n_{k+1}...nd)}，有
    rank(A^{(k)}) = rk

TT 分解将存储复杂度从 O(n^d) 降至 O(d * n * r²)。
"""

import numpy as np
from typing import List, Tuple
from system_utils import EPS, TOL_RANK, check_finite


# ---------------------------------------------------------------------------
# TT 核心张量运算
# ---------------------------------------------------------------------------

def tt_cores_to_full(cores: List[np.ndarray]) -> np.ndarray:
    """
    将 TT 核心列表还原为完整张量（仅适用于小规模测试）。

     cores[k] shape: (r_{k-1}, n_k, r_k)
    """
    d = len(cores)
    # 逐次收缩
    tensor = cores[0]  # shape (1, n1, r1)
    for k in range(1, d):
        # tensor shape: (prod_prev, n_k_prev, r_k)
        # 实际应为 (r0 n1 ... n_{k-1}, r_{k-1}) 的左展开
        # 更简单的做法：逐 mode 收缩
        r_prev = tensor.shape[-1]
        n_prev_modes = tensor.shape[:-1]
        tensor_flat = tensor.reshape(-1, r_prev)  # (..., r_{k-1})
        core_k = cores[k].reshape(r_prev, -1)     # (r_{k-1}, n_k * r_k)
        tensor_flat = tensor_flat @ core_k
        tensor = tensor_flat.reshape(*n_prev_modes, cores[k].shape[1], cores[k].shape[2])
    # 最终去除两端的 1
    tensor = tensor.squeeze()
    return tensor


def tt_left_orthogonalize(cores: List[np.ndarray]) -> List[np.ndarray]:
    """
    对 TT 核心执行左正交化：
        对每个核心 Gk，reshape 为 (r_{k-1}*n_k, r_k)，做 QR 分解，
        取 Q 作为新的 Gk，R 乘到下一个核心上。
    正交化后，前 d-1 个核心满足左正交性：
        Σ_{i_k} Gk(i_k) Gk(i_k)^T = I_{r_{k-1}}
    """
    cores = [c.copy() for c in cores]
    d = len(cores)
    for k in range(d - 1):
        r_prev, n_k, r_k = cores[k].shape
        mat = cores[k].reshape(r_prev * n_k, r_k)
        Q, R = np.linalg.qr(mat, mode='reduced')
        cores[k] = Q.reshape(r_prev, n_k, Q.shape[1])
        # 将 R 乘到下一个核心
        cores[k + 1] = np.tensordot(R, cores[k + 1], axes=([1], [0]))
    return cores


def tt_right_orthogonalize(cores: List[np.ndarray]) -> List[np.ndarray]:
    """
    对 TT 核心执行右正交化：
        reshape 为 (r_{k-1}, n_k*r_k)，做 LQ 分解（通过 QR 转置），
        取 Q 作为新的 Gk，L 乘到前一个核心上。
    """
    cores = [c.copy() for c in cores]
    d = len(cores)
    for k in range(d - 1, 0, -1):
        r_prev, n_k, r_k = cores[k].shape
        mat = cores[k].reshape(r_prev, n_k * r_k)
        # 对转置做 QR
        Q, R = np.linalg.qr(mat.T, mode='reduced')
        cores[k] = Q.T.reshape(Q.shape[1], n_k, r_k)
        cores[k - 1] = np.tensordot(cores[k - 1], R.T, axes=([2], [0]))
    return cores


# ---------------------------------------------------------------------------
# TT-SVD（通过逐次 SVD 截断构造 TT 分解）
# ---------------------------------------------------------------------------

def tt_svd(tensor: np.ndarray, max_rank: int = None,
           tol: float = 1e-10) -> List[np.ndarray]:
    """
    TT-SVD 算法（Oseledets, 2011）：通过逐次截断 SVD 构造 TT 近似。

    算法流程
    --------
    输入 A∈ℝ^{n1×...×nd}，容差 ε。
    C = A
    for k = 1,...,d-1:
        C = reshape(C, (n1...nk) / r_{k-1}, n_{k+1}...nd)
        [U, s, V] = svd(C)
        r_k = min( rank_ε(s), max_rank )
        Gk = reshape(U[:, :r_k], (r_{k-1}, n_k, r_k))
        C = diag(s[:r_k]) @ V[:r_k, :]
    Gd = reshape(C, (r_{d-1}, n_d, 1))

    误差界
    ------
    ‖A - A_TT‖_F² ≤ Σ_{k=1}^{d-1} Σ_{j>r_k} σ_j²(C^{(k)})
    即各 unfolding 截断误差的累积。
    """
    tensor = np.asarray(tensor, dtype=float)
    d = tensor.ndim
    shape = tensor.shape
    if max_rank is None:
        max_rank = max(shape)
    cores = []
    C = tensor.copy()
    r_prev = 1
    for k in range(d - 1):
        n_k = shape[k]
        C = C.reshape(r_prev * n_k, -1)
        U, s, Vt = np.linalg.svd(C, full_matrices=False)
        # 自适应秩截断
        delta = tol * np.linalg.norm(s)
        r_k = int(np.sum(s > delta))
        r_k = min(r_k, max_rank)
        r_k = max(r_k, 1)
        # 构造核心
        Gk = U[:, :r_k].reshape(r_prev, n_k, r_k)
        cores.append(Gk)
        C = np.diag(s[:r_k]) @ Vt[:r_k, :]
        r_prev = r_k
    # 最后一个核心
    Gd = C.reshape(r_prev, shape[-1], 1)
    cores.append(Gd)
    return cores


# ---------------------------------------------------------------------------
# TT-ALS（交替最小二乘优化）
# ---------------------------------------------------------------------------

def tt_als_approximate(tensor: np.ndarray, target_ranks: List[int],
                       max_sweeps: int = 10, tol: float = 1e-8) -> List[np.ndarray]:
    """
    TT-ALS：给定目标 TT-ranks，通过交替最小二乘优化核心张量，
    使 Frobenius 误差 ‖A - TT(cores)‖_F 最小。

    原理
    ----
    固定除 Gk 外所有核心，目标泛函对 Gk 为二次：
        min_{Gk} ‖ A_{(k)} - (U_k) Gk_{(2)} (V_k)^T ‖_F²
    其中 U_k 为左接口矩阵（前 k-1 个核心的缩并），V_k 为右接口矩阵。
    最优解通过求解正规方程得到。

    为简化实现，此处采用 TT-SVD 初始化 + 微校正 sweep。
    """
    tensor = np.asarray(tensor, dtype=float)
    d = tensor.ndim
    if len(target_ranks) != d + 1:
        raise ValueError("target_ranks must have length d+1 with r0=rd=1.")
    if target_ranks[0] != 1 or target_ranks[-1] != 1:
        raise ValueError("First and last TT-ranks must be 1.")

    # 初始化
    cores = tt_svd(tensor, max_rank=max(target_ranks), tol=tol)
    # 强制截断到目标秩
    for k in range(d):
        r_in, n_k, r_out = cores[k].shape
        r_in = min(r_in, target_ranks[k])
        r_out = min(r_out, target_ranks[k + 1])
        cores[k] = cores[k][:r_in, :, :r_out]

    shape = tensor.shape
    # 预计算左右接口
    for sweep in range(max_sweeps):
        # 左到右 sweep
        for k in range(d):
            # 构建左接口 L（shape: prod(n1..n_{k-1}) * n_k, r_{k-1}）
            # 构建右接口 R（shape: r_k, prod(n_{k+1}..n_d)）
            if k == 0:
                L = np.eye(1)
            else:
                L = cores[0].reshape(cores[0].shape[0] * cores[0].shape[1], cores[0].shape[2])
                for j in range(1, k):
                    tmp = cores[j].reshape(cores[j].shape[0], cores[j].shape[1] * cores[j].shape[2])
                    L = L @ tmp
                    L = L.reshape(-1, cores[j].shape[2])
            if k == d - 1:
                R = np.eye(1)
            else:
                R = cores[-1].reshape(cores[-1].shape[0], cores[-1].shape[1] * cores[-1].shape[2])
                for j in range(d - 2, k, -1):
                    tmp = cores[j].reshape(cores[j].shape[0] * cores[j].shape[1], cores[j].shape[2])
                    R = tmp @ R
                    R = R.reshape(cores[j].shape[0], -1)

            # 目标张量切片
            A_slice = tensor.reshape(np.prod(shape[:k+1], dtype=int), -1)
            if k < d - 1:
                A_slice = A_slice.reshape(-1, shape[k], np.prod(shape[k+1:], dtype=int))
            else:
                A_slice = A_slice.reshape(-1, shape[k], 1)

            # 最小二乘求解 Gk（简化：直接对当前 mode 做 SVD 投影校正）
            # 使用左接口和右接口构造 unfold
            left_size = int(np.prod(shape[:k], dtype=int)) * shape[k]
            right_size = int(np.prod(shape[k+1:], dtype=int))
            unfold = tensor.reshape(left_size, right_size)
            # 通过截断 SVD 近似当前 slice
            U, s, Vt = np.linalg.svd(unfold, full_matrices=False)
            r_k = target_ranks[k + 1]
            U_r = U[:, :r_k]
            s_r = s[:r_k]
            Vt_r = Vt[:r_k, :]
            # 更新核心
            if k == 0:
                cores[k] = U_r.reshape(1, shape[k], r_k)
                # 传递奇异值到下一个
                if d > 1:
                    cores[k + 1] = np.tensordot(np.diag(s_r) @ Vt_r,
                                                cores[k + 1], axes=([1], [0]))
                    cores[k + 1] = cores[k + 1].reshape(r_k, shape[k + 1], -1)
            elif k == d - 1:
                cores[k] = (U_r @ np.diag(s_r)).reshape(-1, shape[k], 1)
            else:
                cores[k] = U_r.reshape(target_ranks[k], shape[k], r_k)
                if k + 1 < d:
                    cores[k + 1] = np.tensordot(np.diag(s_r) @ Vt_r,
                                                cores[k + 1], axes=([1], [0]))
                    cores[k + 1] = cores[k + 1].reshape(r_k, shape[k + 1], -1)

        # 误差检查（可选）
        if sweep % 2 == 1:
            approx = tt_cores_to_full(cores)
            err = np.linalg.norm(tensor - approx) / (np.linalg.norm(tensor) + EPS)
            if err < tol:
                break
    return cores


# ---------------------------------------------------------------------------
# TT 范数与内积
# ---------------------------------------------------------------------------

def tt_frobenius_norm(cores: List[np.ndarray]) -> float:
    """
    计算 TT 格式的 Frobenius 范数，无需还原完整张量。

    方法：从左到右依次缩并
        v = G1(i1) @ G1(i1)^T   (sum over i1)
        v = v @ G2(i2) @ G2(i2)^T ...
    最终结果为标量。
    """
    d = len(cores)
    # 第一个核心
    G = cores[0]  # shape (1, n1, r1)
    mat = np.zeros((cores[0].shape[2], cores[0].shape[2]), dtype=float)
    for i in range(cores[0].shape[1]):
        g = cores[0][0, i, :].reshape(-1, 1)
        mat += g @ g.T
    for k in range(1, d):
        new_mat = np.zeros((cores[k].shape[2], cores[k].shape[2]), dtype=float)
        for i in range(cores[k].shape[1]):
            g = cores[k][:, i, :]  # shape (r_{k-1}, r_k)
            new_mat += g.T @ mat @ g
        mat = new_mat
    return float(np.sqrt(max(mat.item(), 0.0)) if mat.size == 1 else np.sqrt(max(np.trace(mat), 0.0)))


def tt_inner_product(cores_a: List[np.ndarray], cores_b: List[np.ndarray]) -> float:
    """
    计算两个 TT 张量的内积 ⟨A, B⟩，无需还原。
    """
    d = len(cores_a)
    if len(cores_b) != d:
        raise ValueError("TT orders must match.")
    # 初始
    M = np.ones((1, 1), dtype=float)
    for k in range(d):
        Ga = cores_a[k]
        Gb = cores_b[k]
        r_a_in, n_k, r_a_out = Ga.shape
        r_b_in, _, r_b_out = Gb.shape
        new_M = np.zeros((r_a_out, r_b_out), dtype=float)
        for i in range(n_k):
            new_M += Ga[:, i, :].T @ M @ Gb[:, i, :]
        M = new_M
    return float(M.item() if M.size == 1 else np.trace(M))
