"""
load_markov.py
基于马尔可夫链的负荷预测模型
融合种子项目：markov_letters（双字母频率统计 → 状态转移矩阵）
"""

import numpy as np
from typing import Optional, Tuple


class LoadMarkovModel:
    """
    基于离散时间马尔可夫链（DTMC）的电力负荷随机预测模型。

    状态空间：将连续负荷 P 离散化为 N_s 个区间（状态）：
        S_k = [P_k, P_{k+1}),  k = 0, 1, ..., N_s-1

    状态转移概率矩阵（One-step Transition Matrix）：
        P_{ij} = Pr( X_{t+1} = j | X_t = i )

    满足：
        0 ≤ P_{ij} ≤ 1,   Σ_j P_{ij} = 1   ∀i

    Chapman-Kolmogorov 方程（n 步转移）：
        P^{(n)} = P^n

    稳态分布 π 满足：
        π = π·P,   Σ_i π_i = 1

    在智能电网中，该模型用于短期负荷预测和风电/光伏出力场景生成，
    为经济调度和备用容量规划提供随机性输入。
    """

    def __init__(self, n_states: int):
        self.n_states = n_states
        self.P = np.eye(n_states, dtype=np.float64)
        self.steady_state: Optional[np.ndarray] = None

    def fit(self, load_series: np.ndarray) -> None:
        """
        从历史负荷序列估计转移矩阵（MLE）。

        计数估计：
            N_{ij} = #{ t | X_t = i, X_{t+1} = j }
            P̂_{ij} = N_{ij} / Σ_k N_{ik}
        """
        load_series = np.array(load_series, dtype=np.float64)
        if len(load_series) < 2:
            raise ValueError("load_series must have at least 2 points")

        # 自动划分状态边界（等频分箱）
        sorted_load = np.sort(load_series)
        n = len(sorted_load)
        self.state_edges = np.zeros(self.n_states + 1)
        for k in range(self.n_states + 1):
            idx = int(np.clip(k * n / self.n_states, 0, n - 1))
            self.state_edges[k] = sorted_load[idx]
        self.state_edges[-1] = sorted_load[-1] + 1e-6

        # 映射到状态索引
        states = np.digitize(load_series, self.state_edges) - 1
        states = np.clip(states, 0, self.n_states - 1)

        count = np.zeros((self.n_states, self.n_states), dtype=np.float64)
        for t in range(len(states) - 1):
            i, j = int(states[t]), int(states[t + 1])
            count[i, j] += 1.0

        # 拉普拉斯平滑避免零概率
        count += 1e-3
        row_sums = count.sum(axis=1, keepdims=True)
        row_sums[row_sums < 1e-12] = 1.0
        self.P = count / row_sums

        # 计算稳态分布
        self._compute_steady_state()

    def _compute_steady_state(self) -> None:
        """
        通过特征值分解求解稳态分布。

        由于 P 是随机矩阵，其最大特征值为 1，对应左特征向量即为稳态分布。
        """
        w, v = np.linalg.eig(self.P.T)
        idx = np.argmin(np.abs(w - 1.0))
        pi = np.real(v[:, idx])
        pi = np.abs(pi)
        pi = pi / np.sum(pi)
        self.steady_state = pi

    def predict(self, current_state: int, n_steps: int) -> np.ndarray:
        """
        预测 n 步后的状态概率分布：
            p^{(n)} = e_{current_state} · P^n
        """
        if current_state < 0 or current_state >= self.n_states:
            raise ValueError("current_state out of range")
        p_n = np.eye(self.n_states, dtype=np.float64)
        P_power = self.P.copy()
        # 快速幂
        while n_steps > 0:
            if n_steps % 2 == 1:
                p_n = p_n @ P_power
            P_power = P_power @ P_power
            n_steps //= 2
        return p_n[current_state]

    def generate_trajectory(self, initial_state: int, n_steps: int,
                            rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """
        基于蒙特卡洛采样生成一条负荷状态轨迹。
        """
        if rng is None:
            rng = np.random.default_rng(seed=42)
        traj = np.zeros(n_steps, dtype=np.int32)
        traj[0] = initial_state
        for t in range(1, n_steps):
            traj[t] = rng.choice(self.n_states, p=self.P[traj[t - 1]])
        return traj

    def entropy_rate(self) -> float:
        """
        马尔可夫链的熵率（Entropy Rate）：
            H(P) = - Σ_i π_i Σ_j P_{ij} log P_{ij}

        衡量负荷序列的不可预测性。熵率越高，负荷波动越随机，
        对调度算法的鲁棒性要求越高。
        """
        if self.steady_state is None:
            self._compute_steady_state()
        H = 0.0
        for i in range(self.n_states):
            for j in range(self.n_states):
                p = self.P[i, j]
                if p > 1e-12:
                    H -= self.steady_state[i] * p * np.log2(p)
        return float(H)

    def n_step_correlation(self, n: int) -> float:
        """
        n 步自相关函数（基于稳态分布的混合时间分析）：
            ρ(n) = (P^n_{ij} - π_j) / (δ_{ij} - π_j)

        反映系统记忆的衰减速度，用于确定调度窗口长度。
        """
        Pn = np.linalg.matrix_power(self.P, n)
        # 使用 Frobenius 范数度量与稳态的偏差
        diff = Pn - self.steady_state[np.newaxis, :]
        return float(np.linalg.norm(diff, 'fro'))


def load_forecast_example() -> dict:
    """
    生成示例负荷数据并训练马尔可夫模型。
    """
    # 模拟日负荷曲线（24小时，每小时一个采样点）
    t = np.linspace(0, 24, 48)
    base = 100.0
    peak1 = 60.0 * np.exp(-0.5 * ((t - 12.0) / 3.0) ** 2)
    peak2 = 40.0 * np.exp(-0.5 * ((t - 19.0) / 2.0) ** 2)
    noise = np.random.default_rng(seed=7).normal(0, 5.0, len(t))
    load = base + peak1 + peak2 + noise
    load = np.maximum(load, 20.0)

    model = LoadMarkovModel(n_states=6)
    model.fit(load)
    return {
        "load_series": load,
        "model": model
    }
