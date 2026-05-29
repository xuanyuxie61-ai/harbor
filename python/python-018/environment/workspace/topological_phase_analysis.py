"""
topological_phase_analysis.py

基于种子项目 1095_snakes_probability（马尔可夫链转移矩阵与
概率分布计算），实现Kitaev链拓扑相变的马尔可夫链分析。

物理模型：
    将拓扑相变理解为参数空间中的"状态转移"过程。
    定义拓扑相的状态空间 S = {平庸相, 非平庸相}，
    状态转移由化学势μ和无序W驱动。

    转移矩阵P的元素P_{ij}表示从相i转移到相j的概率，
    可通过费米能级处的态密度和能隙信息估计。

    长期行为由稳态分布π满足 π = πP 决定，
    对应系统的热力学极限下的拓扑相概率。

核心公式：
    绕数（winding number）作为Z₂拓扑不变量：
        ν = (1/2πi) ∮ dk Tr[H_k^{-1} ∂_k H_k]

    对于Kitaev链，简化为：
        ν = sgn(Δ) * [sgn(μ+2t) - sgn(μ-2t)] / 2

    拓扑非平庸相：ν = ±1 （当 |μ| < 2t, Δ ≠ 0）
    拓扑平庸相：ν = 0   （当 |μ| > 2t）
"""

import numpy as np
from typing import Tuple, Optional


class TopologicalMarkovChain:
    """
    拓扑相变的马尔可夫链模型。

    状态定义：
        状态0：拓扑平庸相 (Trivial, ν=0)
        状态1：拓扑非平庸相 (Topological, ν=±1)
        状态2：临界点/金属相 (Critical, 能隙闭合)

    转移概率由以下物理量决定：
        P_{01} ∝ exp(-E_gap / k_B T)  （热激发驱动相变）
        P_{10} ∝ W^2 / Δ^2            （无序驱动相变）
    """

    def __init__(self, temperature: float = 0.01,
                 disorder_strength: float = 0.0,
                 delta: float = 0.8,
                 t: float = 1.0):
        """
        初始化拓扑马尔可夫链。

        Args:
            temperature: 有效温度 (k_B T / t)
            disorder_strength: 无序强度 W/t
            delta: 超导能隙 Δ/t
            t: 跃迁强度（能量单位）
        """
        self.T = max(temperature, 1e-10)
        self.W = max(disorder_strength, 0.0)
        self.delta = delta
        self.t = t
        self.num_states = 3

    def _compute_transition_matrix(self, mu: float) -> np.ndarray:
        """
        构建拓扑相变的转移矩阵。

        转移概率设计原则：
            1) 从当前相到自身的概率最高（相的稳定性）
            2) 能隙越小，越容易发生相变
            3) 无序增强相变概率

        参数μ决定当前能隙：
            E_gap(μ) = | |Δ| * sqrt(1 - (μ/2t)^2) |  （均匀Kitaev链近似）
        """
        n = self.num_states
        P = np.zeros((n, n))

        # 计算能隙（近似公式）
        ratio = mu / (2.0 * self.t)
        if abs(ratio) < 1.0:
            egap = abs(self.delta) * np.sqrt(max(1.0 - ratio ** 2, 0.0))
        else:
            egap = 0.0

        # 热涨落因子
        thermal_factor = np.exp(-egap / self.T)
        disorder_factor = min(self.W / (abs(self.delta) + 1e-15), 1.0)

        # 状态0 (Trivial) -> 状态1 (Topological)
        # 仅在 |μ|<2t 时可能，概率正比于能隙闭合程度
        if abs(mu) < 2.0 * abs(self.t):
            p01 = 0.1 * thermal_factor + 0.05 * disorder_factor
        else:
            p01 = 0.01 * thermal_factor

        # 状态1 (Topological) -> 状态0 (Trivial)
        p10 = 0.1 * thermal_factor + 0.2 * disorder_factor

        # 到临界状态2的概率
        p02 = 0.05 * thermal_factor
        p12 = 0.05 * thermal_factor
        p20 = 0.3
        p21 = 0.3

        # 填充转移矩阵并归一化
        P[0, 1] = min(p01, 0.5)
        P[0, 2] = min(p02, 0.3)
        P[0, 0] = 1.0 - P[0, 1] - P[0, 2]

        P[1, 0] = min(p10, 0.5)
        P[1, 2] = min(p12, 0.3)
        P[1, 1] = 1.0 - P[1, 0] - P[1, 2]

        P[2, 0] = p20
        P[2, 1] = p21
        P[2, 2] = 1.0 - P[2, 0] - P[2, 1]

        # 数值鲁棒性：确保每行和为1
        for i in range(n):
            row_sum = np.sum(P[i, :])
            if abs(row_sum) > 1e-15:
                P[i, :] /= row_sum
            else:
                P[i, i] = 1.0

        return P

    def steady_state_distribution(self, mu: float,
                                   max_iter: int = 1000,
                                   tol: float = 1e-12) -> np.ndarray:
        """
        计算给定μ下的稳态分布 π。

        通过迭代求解 π^{(n+1)} = π^{(n)} P 直至收敛。
        稳态分布满足：
            π_j = Σ_i π_i P_{ij}
            Σ_j π_j = 1

        这对应于参数μ缓慢变化时，系统在各拓扑相中
        停留时间的比例。
        """
        P = self._compute_transition_matrix(mu)
        pi = np.ones(self.num_states) / self.num_states

        for _ in range(max_iter):
            pi_new = pi @ P
            if np.linalg.norm(pi_new - pi, ord=1) < tol:
                return pi_new
            pi = pi_new

        return pi

    def phase_transition_probability(self, mu_path: np.ndarray,
                                      initial_phase: int = 0) -> np.ndarray:
        """
        沿化学势路径计算处于拓扑相的概率。

        模拟从初始相initial_phase出发，沿μ路径逐步演化，
        计算在每一步处于拓扑非平庸相（状态1）的概率。
        """
        n_steps = len(mu_path)
        prob_topo = np.zeros(n_steps)

        # 初始分布
        dist = np.zeros(self.num_states)
        if 0 <= initial_phase < self.num_states:
            dist[initial_phase] = 1.0
        else:
            dist = np.ones(self.num_states) / self.num_states

        for i, mu in enumerate(mu_path):
            P = self._compute_transition_matrix(mu)
            dist = dist @ P
            prob_topo[i] = dist[1]

        return prob_topo

    def winding_number(self, mu: float) -> int:
        """
        计算Kitaev链的Z₂绕数拓扑不变量。

        对于均匀Kitaev链，绕数有解析表达式：
            ν(μ,Δ,t) = sgn(Δ) * [sgn(μ+2t) - sgn(μ-2t)] / 2

        结果：
            ν = 0  （|μ| > 2|t|，平庸相）
            ν = ±1 （|μ| < 2|t|，非平庸相，符号由Δ决定）
        """
        # === HOLE 2 START ===
        # TODO: 实现Kitaev链的Z₂绕数拓扑不变量计算
        #
        # 科学背景：对于均匀Kitaev链，绕数有解析表达式：
        #     ν(μ, Δ, t) = sgn(Δ) * [sgn(μ + 2t) - sgn(μ - 2t)] / 2
        #
        # 结果：
        #     ν = 0  （|μ| > 2|t|，平庸相）
        #     ν = ±1 （|μ| < 2|t|，非平庸相，符号由Δ决定）
        #
        # 需要完成的任务：
        # 1. 根据 self.delta 的符号计算 sgn_delta
        # 2. 根据 mu + 2t 和 mu - 2t 的符号分别计算 sgn_mu_plus 和 sgn_mu_minus
        # 3. 计算绕数 nu = sgn_delta * (sgn_mu_plus - sgn_mu_minus) / 2
        # 4. 返回 float 类型（返回值已从 int 改为 float，调用方需要同步适配）
        raise NotImplementedError("Hole 2: 请实现Z₂绕数拓扑不变量计算公式")
        # === HOLE 2 END ===

    def topological_phase_diagram_markov(self,
                                          mu_vals: np.ndarray,
                                          w_vals: np.ndarray) -> np.ndarray:
        """
        构建(μ, W)参数空间中的拓扑相图。

        返回矩阵 topo_prob[i,j] = 在(μ_i, W_j)处处于拓扑相的概率。
        """
        n_mu = len(mu_vals)
        n_w = len(w_vals)
        topo_prob = np.zeros((n_mu, n_w))

        for j, w in enumerate(w_vals):
            self.W = w
            for i, mu in enumerate(mu_vals):
                pi = self.steady_state_distribution(mu)
                topo_prob[i, j] = pi[1]

        return topo_prob

    def compute_entanglement_entropy(self, mu: float,
                                      subsystem_size: int,
                                      n_sites: int = 100) -> float:
        """
        计算拓扑相的纠缠熵（简化模型）。

        在Kitaev链的拓扑非平庸相中，将系统分为左右两部分，
        纠缠熵满足：
            S_A = (ln 2)/2 * N_MZM
        其中N_MZM为跨越边界的马约拉纳零能模数目。

        对于开边界Kitaev链，N_MZM = 2（两端各一个），
        因此 S_A → ln(2)/2 在拓扑相中。
        """
        nu = self.winding_number(mu)
        if nu == 0:
            # 平庸相：纠缠熵呈面积律衰减
            s = 0.1 * np.exp(-subsystem_size / 10.0)
        else:
            # 拓扑相：存在常数项（拓扑纠缠熵）
            s = 0.5 * np.log(2.0) + 0.05 * np.exp(-subsystem_size / 5.0)

        return float(s)

    def correlation_length_critical_exponent(self,
                                              mu_vals: np.ndarray) -> np.ndarray:
        """
        计算相变点附近的关联长度临界指数。

        在相变点 μ_c = ±2t 附近，关联长度发散：
            ξ(μ) ~ |μ - μ_c|^{-ν}
        对于Kitaev链，ν = 1（平均场指数）。
        """
        xi = np.zeros_like(mu_vals)
        for i, mu in enumerate(mu_vals):
            dist = min(abs(mu - 2.0 * self.t), abs(mu + 2.0 * self.t))
            if dist < 1e-6:
                xi[i] = 1e6
            else:
                xi[i] = abs(self.t) / dist
        return xi


def demo():
    """演示拓扑相马尔可夫链分析。"""
    tmc = TopologicalMarkovChain(
        temperature=0.05, disorder_strength=0.2, delta=0.8, t=1.0
    )

    mu_path = np.linspace(-3.0, 3.0, 61)
    prob_topo = tmc.phase_transition_probability(mu_path, initial_phase=0)

    print("Topological phase probability along μ path:")
    for mu, p in zip(mu_path[::10], prob_topo[::10]):
        nu = tmc.winding_number(mu)
        print(f"  μ={mu:+.2f}, ν={nu:2d}, P_topo={p:.4f}")

    # 纠缠熵
    for mu in [-0.5, 0.5, 3.0]:
        s = tmc.compute_entanglement_entropy(mu, subsystem_size=10)
        print(f"Entanglement entropy at μ={mu}: S={s:.4f}")


if __name__ == "__main__":
    demo()
