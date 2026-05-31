"""
transition_network.py
离子跃迁网络分析模块

基于种子项目 286_digraph_arc 的核心算法：
- digraph_arc_is_eulerian: 有向图欧拉回路判定
- digraph_arc_degree: 节点入度/出度分析
- 图连通性与邻接矩阵转换

在离子通道问题中的应用：
将离子的离散空间运动建模为状态转移网络（Markov 状态模型）：
- 节点：通道内的离散结合位点（S0, S1, S2, S3, S4）
- 有向边：离子从一个位点跃迁到相邻位点
- 欧拉路径分析：判断是否存在连续的离子传导路径
- 度分析：评估各结合位点的离子流量平衡

KcsA 钾通道选择性滤器包含 4 个连续的 K+ 结合位点（S1-S4），
加上胞外入口 S0 和胞内入口 S_ext，形成一维跃迁链。
"""

import numpy as np


class TransitionNetwork:
    """
    离子跃迁网络（Markov 状态模型）。
    """
    def __init__(self, n_states, state_names=None):
        self.n_states = n_states
        self.state_names = state_names or [f"S{i}" for i in range(n_states)]
        # 转移速率矩阵 K，K[i,j] 表示从 i -> j 的速率
        self.K = np.zeros((n_states, n_states))
        # 稳态占据概率
        self.pi = np.ones(n_states) / n_states

    def add_transition(self, i, j, rate):
        """
        添加从状态 i 到 j 的跃迁速率（单位：s^{-1}）。
        """
        if i < 0 or i >= self.n_states or j < 0 or j >= self.n_states:
            raise IndexError("状态索引越界")
        self.K[i, j] = rate

    def compute_degrees(self):
        """
        计算每个节点的入度、出度和净流量（源自 digraph_arc_degree.m 思想）。

        indegree[i] = Σ_j K[j,i]   （流入速率之和）
        outdegree[i] = Σ_j K[i,j]  （流出速率之和）
        """
        indegree = np.sum(self.K, axis=0)
        outdegree = np.sum(self.K, axis=1)
        return indegree, outdegree

    def is_eulerian_path(self):
        """
        判断网络是否存在欧拉路径（源自 digraph_arc_is_eulerian.m）。

        对于离子传导网络，欧拉路径意味着存在一条不重复的连续离子
        穿透路径，这是高效离子通道的必要条件。

        判定条件：
            - 所有节点的 |indegree - outdegree| <= 1
            - 恰好 0 或 2 个节点的 |indegree - outdegree| == 1
        """
        indegree, outdegree = self.compute_degrees()
        diff = indegree - outdegree

        n_plus = np.sum(diff == 1)
        n_minus = np.sum(diff == -1)
        n_zero = np.sum(diff == 0)

        if n_plus == 0 and n_minus == 0:
            return 2  # 闭合欧拉回路
        elif n_plus == 1 and n_minus == 1:
            return 1  # 开放欧拉路径
        else:
            return 0  # 非欧拉

    def steady_state_probability(self, max_iter=1000, tol=1e-12):
        """
        通过主方程的稳态求解占据概率：
            dπ/dt = K^T π = 0,  Σ_i π_i = 1

        采用幂迭代法。
        """
        # 构建转移概率矩阵（行归一化）
        P = np.zeros_like(self.K)
        for i in range(self.n_states):
            row_sum = np.sum(self.K[i, :])
            if row_sum > 0:
                P[i, :] = self.K[i, :] / row_sum
            else:
                P[i, i] = 1.0

        pi = np.ones(self.n_states) / self.n_states
        for _ in range(max_iter):
            pi_new = pi @ P
            pi_new = pi_new / np.sum(pi_new)
            if np.max(np.abs(pi_new - pi)) < tol:
                break
            pi = pi_new

        self.pi = pi
        return pi

    def mean_first_passage_time(self, target):
        """
        计算从各状态到目标状态的平均首次通过时间（MFPT）。

        方程：
            Σ_j K_{ij} τ_j = -1   (i ≠ target)
            τ_target = 0

        其中 τ_i 为从状态 i 首次到达 target 的平均时间。
        """
        n = self.n_states
        # 构建速率矩阵 Q（主方程矩阵）
        Q = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    Q[i, j] = self.K[j, i]  # 注意转置
            Q[i, i] = -np.sum(self.K[:, i])

        # 移除 target 行/列
        idx = [i for i in range(n) if i != target]
        Q_reduced = Q[np.ix_(idx, idx)]
        b = -np.ones(n - 1)

        tau_reduced = np.linalg.solve(Q_reduced, b)
        tau = np.zeros(n)
        for k, i in enumerate(idx):
            tau[i] = tau_reduced[k]
        return tau

    def conductivity(self, entry_state, exit_state):
        """
        估算离子电导（pS 量级），基于稳态流。

        单通道电流：
            I = e * π_entry * k_entry->next * (1 - π_exit)

        电导：
            G = I / V
        """
        pi = self.steady_state_probability()
        rate_out = np.sum(self.K[exit_state, :])
        # 简化：稳态通量 ≈ π_entry * rate_entry
        flux = pi[entry_state] * np.sum(self.K[entry_state, :])
        e_charge = 1.602176634e-19
        # 假设驱动力 100 mV
        V = 0.1
        I = e_charge * flux
        G = I / V  # Siemens
        return G * 1e12  # 转为 pS


def build_kcsa_k_channel_network(k_on=1e8, k_off=1e7, k_hop=5e7):
    """
    构建 KcsA 钾通道选择性滤器的一维跃迁网络。

    状态定义：
        S0: 滤器入口（胞外侧）
        S1-S4: 滤器内部 4 个 K+ 结合位点
        S5: 滤器出口（腔体侧）

    跃迁：相邻位点之间的单离子跳跃（knock-on 机制）。
    """
    net = TransitionNetwork(6, ["S0", "S1", "S2", "S3", "S4", "S5"])

    # 入口 -> S1
    net.add_transition(0, 1, k_on)
    net.add_transition(1, 0, k_off)

    # 内部跃迁（knock-on）
    for i in range(1, 5):
        net.add_transition(i, i + 1, k_hop)
        net.add_transition(i + 1, i, k_hop * 0.5)  # 反向较慢

    # S5 -> 出口（腔体）
    net.add_transition(5, 4, k_hop * 0.3)

    return net


def build_na_leaky_network(k_on=1e8, k_off=1e8, k_hop=1e6):
    """
    Na+ 在 KcsA 中的泄漏网络：
    Na+ 由于水合壳层无法脱去，结合较弱（k_off 大），
    内部跳跃速率也低，导致传导效率远低于 K+。
    """
    net = TransitionNetwork(6, ["S0", "S1", "S2", "S3", "S4", "S5"])
    net.add_transition(0, 1, k_on)
    net.add_transition(1, 0, k_off)
    for i in range(1, 5):
        net.add_transition(i, i + 1, k_hop)
        net.add_transition(i + 1, i, k_hop)
    net.add_transition(5, 4, k_hop)
    return net
