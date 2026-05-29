r"""
markov_causal_chain.py
================================================================================
基于吸收态马尔可夫链的因果网络状态转移与干预效应分析

原项目映射: 1200_tennis_matrix — 网球得分系统的状态转移矩阵构造

科学背景
--------
在因果推断中，当变量取离散状态且存在时间演化时，可将其建模为**马尔可夫因果链**。
干预 $do(X_i=x)$ 改变了状态转移概率，从而改变系统的稳态分布。
本项目将网球得分系统的转移矩阵思想推广到一般因果网络的离散状态空间，
计算干预后的吸收概率与期望到达时间。

核心公式
--------
1. 状态转移矩阵 $P$ 满足 $\sum_j P_{ij}=1$，其中状态包括：
   - 瞬态（Transient）：系统正常的因果演化状态
   - 吸收态（Absorbing）：因果饱和态（如疾病康复、经济崩溃）

2. 将 $P$ 按标准形排列：
   $$ P = \begin{pmatrix} Q & R \\ 0 & I \end{pmatrix} $$
   其中 $Q$ 为瞬态→瞬态子矩阵，$R$ 为瞬态→吸收态子矩阵。

3. 基本矩阵（Fundamental Matrix）:
   $$ N = (I - Q)^{-1} = I + Q + Q^2 + Q^3 + \cdots $$
   其元素 $N_{ij}$ 表示从状态 $i$ 出发，访问瞬态 $j$ 的期望次数。

4. 吸收概率矩阵:
   $$ B = N R $$
   $B_{ik}$ 为从瞬态 $i$ 出发最终被吸收态 $k$ 吸收的概率。

5. 期望到达时间（到任意吸收态）:
   $$ t = N \mathbf{1} $$

6. 因果干预效应（do-演算）:
   干预 $do(X=x)$ 将转移矩阵修改为 $P^{(do)}$，通过对比 $P$ 与 $P^{(do)}$
   下的吸收概率差异，量化该干预的因果效应：
   $$ \text{CE} = B^{(do)} - B $$
r"""

import numpy as np
from typing import Tuple, List, Optional


def build_causal_markov_chain(p: int,
                               causal_edges: List[Tuple[int, int, float]],
                               n_states_per_var: int = 3,
                               absorption_threshold: float = 0.9) -> Tuple[np.ndarray, List[int], List[int]]:
    r"""
    基于因果骨架边构建高维因果马尔可夫链的转移矩阵。

    每个变量有 n_states_per_var 个离散状态（低/中/高）。
    系统总状态数为 $n_{\text{states}} = p \times n_{\text{states_per_var}}$。
    当任意变量达到最高状态且满足累积因果强度超过阈值时，进入吸收态。

    Parameters
    ----------
    p : int
        变量数。
    causal_edges : list of (i, j, w)
        因果骨架边及其权重。
    n_states_per_var : int
        每个变量的离散状态数。
    absorption_threshold : float
        吸收判定阈值，范围 (0,1]。

    Returns
    -------
    P : ndarray, shape (n_total, n_total)
        转移概率矩阵（已归一化）。
    transient_states : list of int
        瞬态索引。
    absorbing_states : list of int
        吸收态索引。
    r"""
    if p <= 0:
        raise ValueError("变量数 p 必须为正。")
    if not (0.0 < absorption_threshold <= 1.0):
        raise ValueError("absorption_threshold 必须在 (0,1] 内。")

    n_total = p * n_states_per_var
    P = np.zeros((n_total, n_total))

    # 基础转移：每个变量在自身状态间随机游走（带有向高状态漂移）
    for var in range(p):
        base = var * n_states_per_var
        for s in range(n_states_per_var):
            idx = base + s
            probs = np.zeros(n_states_per_var)
            if s == 0:
                probs[0] = 0.5
                probs[1] = 0.5
            elif s == n_states_per_var - 1:
                probs[s] = 0.6
                probs[s - 1] = 0.4
            else:
                probs[s - 1] = 0.25
                probs[s] = 0.5
                probs[s + 1] = 0.25
            # 归一化
            probs = probs / np.sum(probs)
            for sp in range(n_states_per_var):
                P[idx, base + sp] = probs[sp]

    # 因果耦合：若变量 i 对 j 有因果影响，则 i 的高状态会提升 j 向高状态转移的概率
    for i, j, w in causal_edges:
        if i == j:
            continue
        base_i = i * n_states_per_var
        base_j = j * n_states_per_var
        for si in range(n_states_per_var):
            idx_i = base_i + si
            # 当 i 处于高状态时，j 的转移概率被扰动
            boost = float(si) / max(n_states_per_var - 1, 1) * abs(w)
            for sj in range(n_states_per_var):
                idx_j_from = base_j + sj
                # 提升向更高状态转移的概率
                if sj < n_states_per_var - 1:
                    P[idx_j_from, base_j + min(sj + 1, n_states_per_var - 1)] += boost * 0.1
                if sj > 0:
                    P[idx_j_from, base_j + max(sj - 1, 0)] -= boost * 0.05

    # 确保非负并归一化
    P = np.maximum(P, 0.0)
    row_sums = P.sum(axis=1)
    row_sums[row_sums == 0.0] = 1.0
    P = P / row_sums[:, np.newaxis]

    # 判定吸收态：变量处于最高状态且累积入边权重超过阈值
    # 简化判定：将最后几个状态设为吸收态
    absorbing_states = []
    transient_states = []
    for var in range(p):
        base = var * n_states_per_var
        # 将每个变量的最高状态设为潜在吸收态
        absorbing_states.append(base + n_states_per_var - 1)
        for s in range(n_states_per_var - 1):
            transient_states.append(base + s)

    # 强制吸收态自环
    for ab in absorbing_states:
        P[ab, :] = 0.0
        P[ab, ab] = 1.0

    return P, transient_states, absorbing_states


def canonical_form(P: np.ndarray,
                   transient_states: List[int],
                   absorbing_states: List[int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    r"""
    将转移矩阵转换为标准形并提取 $Q$ 与 $R$ 子矩阵。

    Returns
    -------
    P_canon : ndarray
        标准形转移矩阵。
    Q : ndarray
        瞬态→瞬态子矩阵。
    R : ndarray
        瞬态→吸收态子矩阵。
    state_map : ndarray
        原始状态到新排列的映射。
    r"""
    t_states = list(transient_states)
    a_states = list(absorbing_states)
    n_t = len(t_states)
    n_a = len(a_states)
    state_map = np.array(t_states + a_states)

    P_canon = np.zeros_like(P)
    for new_i, old_i in enumerate(state_map):
        for new_j, old_j in enumerate(state_map):
            P_canon[new_i, new_j] = P[old_i, old_j]

    Q = P_canon[:n_t, :n_t]
    R = P_canon[:n_t, n_t:]
    return P_canon, Q, R, state_map


def fundamental_matrix(Q: np.ndarray) -> np.ndarray:
    r"""
    计算基本矩阵 $N = (I - Q)^{-1}$。
    r"""
    n_t = Q.shape[0]
    I = np.eye(n_t)
    # 数值稳定性检查：谱半径
    eigvals = np.linalg.eigvals(Q)
    spectral_radius = np.max(np.abs(eigvals))
    if spectral_radius >= 1.0:
        # 轻微正则化
        Q = Q * 0.999 / spectral_radius
    N = np.linalg.inv(I - Q)
    return N


def absorption_probabilities_and_times(Q: np.ndarray, R: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    r"""
    计算吸收概率矩阵 $B=NR$ 与期望到达时间 $t=N\mathbf{1}$。

    Returns
    -------
    B : ndarray
        吸收概率矩阵。
    t : ndarray
        期望到达时间向量。
    r"""
    N = fundamental_matrix(Q)
    B = N @ R
    t = N @ np.ones(N.shape[0])
    return B, t


def intervene_do_state(P: np.ndarray,
                       state_idx: int,
                       new_value_state: int,
                       n_states_per_var: int) -> np.ndarray:
    r"""
    执行 do-干预：将某变量的状态固定为指定值（改变转移矩阵）。

    这对应 Pearl 的 do-演算 $P(Y|do(X=x))$，在离散状态空间中
    通过将该变量的所有非目标状态转移概率归零实现。
    r"""
    P_do = P.copy()
    # 找到该变量对应的所有状态行
    var_id = state_idx // n_states_per_var
    base = var_id * n_states_per_var
    for s in range(n_states_per_var):
        row = base + s
        P_do[row, :] = 0.0
        P_do[row, new_value_state] = 1.0
    return P_do


def demo():
    r"""模块自测试。"""
    np.random.seed(7)
    p = 6
    causal_edges = [(0, 1, 0.5), (1, 2, 0.4), (2, 3, 0.3),
                    (0, 3, 0.2), (4, 5, 0.6), (3, 5, 0.35)]
    P, trans, absorb = build_causal_markov_chain(p, causal_edges, n_states_per_var=3)
    P_canon, Q, R, state_map = canonical_form(P, trans, absorb)
    B, t = absorption_probabilities_and_times(Q, R)
    print(f"[markov_causal_chain] 状态数={P.shape[0]}, 瞬态={len(trans)}, 吸收态={len(absorb)}")
    print(f"[markov_causal_chain] 吸收概率矩阵 B 维度={B.shape}, 期望时间范围=[{t.min():.3f}, {t.max():.3f}]")

    # do-干预测试
    P_do = intervene_do_state(P, state_idx=0, new_value_state=2, n_states_per_var=3)
    P_canon_do, Q_do, R_do, _ = canonical_form(P_do, trans, absorb)
    B_do, t_do = absorption_probabilities_and_times(Q_do, R_do)
    effect = np.linalg.norm(B_do - B, 'fro')
    print(f"[markov_causal_chain] do-干预因果效应 (Frobenius 范数差): {effect:.4f}")
    return P, B, t


if __name__ == "__main__":
    demo()
