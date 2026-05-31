r"""
network_coupling.py
===================
级联晶体网络耦合模型 —— 融合原项目 285_digraph_adj (有向图邻接矩阵)。

在多级纠缠光源架构中，多个非线性晶体段通过线性光学网络（波导分束器、
相位调制器）级联。光子态在各级之间的演化可用有向图描述：

- 节点：各级晶体输出模式（信号/闲置通道）。
- 边：模式之间的线性耦合（分束、相位延迟）与非线性增益。

核心公式
--------
**有向图邻接矩阵** :math:`A`（:math:`N \times N`）：

.. math::
    A_{ij} = \begin{cases}
    g_{ij} & \text{若存在从节点 } j \text{ 到节点 } i \text{ 的耦合} \\
    0 & \text{否则}
    \end{cases}

其中 :math:`g_{ij}` 为复耦合系数，含振幅与相位：

.. math::
    g_{ij} = |g_{ij}| \exp(i \phi_{ij})

**状态转移矩阵**（Google/转移矩阵形式）：

.. math::
    T_{ij} = \frac{A_{ij}}{\sum_k A_{kj}}

若列和为零，则设 :math:`T_{jj}=1`（吸收态）。

**多级网络光子数演化**

设 :math:`n^{(m)}` 为第 :math:`m` 级各模式的光子数向量，
则经过 :math:`M` 级级联后：

.. math::
    n^{(M)} = T^{M} n^{(0)} + \sum_{k=0}^{M-1} T^{k} s^{(M-1-k)}

其中 :math:`s^{(k)}` 为第 :math:`k` 级 SPDC 产生源项。
"""

import numpy as np
from typing import Tuple


def build_coupling_digraph(n_stages: int,
                           coupling_strength: float = 0.1,
                           phase_noise_std: float = 0.05) -> np.ndarray:
    """
    构建级联晶体网络的邻接矩阵。

    参数
    ----
    n_stages : int
        级数，>= 1。
    coupling_strength : float
        相邻级间的耦合强度，>= 0。
    phase_noise_std : float
        相位噪声标准差。

    返回
    ----
    A : np.ndarray, shape (2*n_stages, 2*n_stages)
        有向邻接矩阵。节点顺序为 [s_1, i_1, s_2, i_2, ...]。
    """
    if n_stages < 1:
        raise ValueError("n_stages 必须至少为 1。")
    if coupling_strength < 0.0:
        raise ValueError("coupling_strength 必须非负。")

    N = 2 * n_stages
    A = np.zeros((N, N), dtype=np.complex128)

    # 每级内部：泵浦光产生信号-闲置对（自环 + 交叉）
    for stage in range(n_stages):
        s_idx = 2 * stage
        i_idx = 2 * stage + 1
        A[s_idx, s_idx] = 1.0  # 自保持
        A[i_idx, i_idx] = 1.0
        # 信号-闲置的量子关联（非经典边）
        A[s_idx, i_idx] = coupling_strength * np.exp(1j * np.random.normal(0.0, phase_noise_std))
        A[i_idx, s_idx] = coupling_strength * np.exp(-1j * np.random.normal(0.0, phase_noise_std))

    # 级间耦合：前级的信号/闲置部分注入后级
    for stage in range(n_stages - 1):
        s_curr = 2 * stage
        i_curr = 2 * stage + 1
        s_next = 2 * (stage + 1)
        i_next = 2 * (stage + 1) + 1
        t = 0.5 * coupling_strength
        A[s_next, s_curr] = t * np.exp(1j * np.random.normal(0.0, phase_noise_std))
        A[i_next, i_curr] = t * np.exp(1j * np.random.normal(0.0, phase_noise_std))

    return A


def adjacency_to_transition(A: np.ndarray) -> np.ndarray:
    """
    将邻接矩阵转化为列随机转移矩阵（Google matrix 变体）。

    参数
    ----
    A : np.ndarray
        复邻接矩阵。

    返回
    ----
    T : np.ndarray
        列随机矩阵（每列和为 1）。
    """
    N = A.shape[0]
    col_sums = np.sum(np.abs(A), axis=0)
    T = np.zeros_like(A, dtype=np.complex128)
    for j in range(N):
        if col_sums[j] > 1e-15:
            T[:, j] = A[:, j] / col_sums[j]
        else:
            T[j, j] = 1.0  # 吸收态
    return T


def network_photon_number_evolution(n_stages: int,
                                    n_initial: np.ndarray,
                                    source_terms: np.ndarray,
                                    A: np.ndarray) -> np.ndarray:
    """
    计算级联网络中各级模式的光子数演化。

    参数
    ----
    n_stages : int
    n_initial : np.ndarray, shape (2*n_stages,)
        初始光子数（各模式）。
    source_terms : np.ndarray, shape (n_stages, 2*n_stages)
        每级 SPDC 产生源项。
    A : np.ndarray
        邻接矩阵。

    返回
    ----
    n_history : np.ndarray, shape (n_stages+1, 2*n_stages)
        每步演化后的光子数。
    """
    N = 2 * n_stages
    if n_initial.shape != (N,):
        raise ValueError("n_initial 维度不匹配。")
    if source_terms.shape[1] != N:
        raise ValueError("source_terms 列数不匹配。")

    T = adjacency_to_transition(A)
    n_history = np.zeros((n_stages + 1, N), dtype=np.float64)
    n_history[0, :] = n_initial

    n_current = n_initial.astype(np.float64)
    for m in range(n_stages):
        # 线性演化 + 本地产生
        n_current = np.abs(T @ n_current) + source_terms[m, :]
        # 保证非负
        n_current = np.maximum(n_current, 0.0)
        n_history[m + 1, :] = n_current

    return n_history


def transitive_closure_digraph(A: np.ndarray, tol: float = 1e-12) -> np.ndarray:
    r"""
    计算有向图邻接矩阵的传递闭包（Warshall 算法）。

    用于判断网络中任意两模式之间是否存在间接耦合路径。

    .. math::
        C^{(0)} = A, \quad C^{(k+1)}_{ij} = C^{(k)}_{ij}
        \lor \left( C^{(k)}_{ik} \land C^{(k)}_{kj} \right)

    参数
    ----
    A : np.ndarray
    tol : float

    返回
    ----
    C : np.ndarray
        布尔传递闭包。
    """
    N = A.shape[0]
    C = (np.abs(A) > tol).astype(int)
    for k in range(N):
        for i in range(N):
            for j in range(N):
                C[i, j] = C[i, j] or (C[i, k] and C[k, j])
    return C
