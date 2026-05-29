"""
markov_kinetics.py
马尔可夫链主方程求解器

整合原项目:
  - 778_monopoly_matrix: 马尔可夫转移矩阵与稳态分布

科学背景:
  表面催化反应网络可建模为离散状态马尔可夫过程:
  
  状态空间:
    S = {s_1, s_2, ..., s_N}
  
  每个状态表示催化剂表面的吸附构型，例如:
    s_i = (site_1_occupancy, site_2_occupancy, ..., site_M_occupancy)
  
  主方程 (Master Equation):
    dP_i/dt = Σ_j (W_{ji} P_j - W_{ij} P_i)
  
  其中:
    P_i(t): t 时刻处于状态 i 的概率
    W_{ij}: 从状态 i 到状态 j 的跃迁速率
  
  矩阵形式:
    dP/dt = W^T P
  
  稳态条件:
    W^T P_ss = 0,  Σ_i P_ss,i = 1
  
  对于详细平衡 (Detailed Balance):
    W_{ij} P_i^{eq} = W_{ji} P_j^{eq}
"""

import numpy as np
from typing import Tuple, List, Optional


class SurfaceReactionNetwork:
    """
    表面催化反应网络
    
    将表面位点占据态作为马尔可夫状态，
    定义吸附、脱附和反应事件为状态间跃迁
    """

    def __init__(self, n_sites: int, max_occupancy: int = 1):
        if n_sites < 1:
            raise ValueError("n_sites >= 1")
        self.n_sites = n_sites
        self.max_occupancy = max_occupancy
        # 状态编码: 每个位点 0=空, 1=CO, 2=O
        # 为简化计算，使用较少数量的代表性状态
        self.states = self._enumerate_representative_states()
        self.n_states = len(self.states)
        self.W = np.zeros((self.n_states, self.n_states))

    def _enumerate_representative_states(self) -> List[np.ndarray]:
        """
        枚举代表性表面构型状态
        
        对于 n_sites 个位点，完整状态空间大小为 3^n_sites。
        此处采用降维策略，只保留低覆盖度状态。
        """
        states = []
        # 空表面
        states.append(np.zeros(self.n_sites, dtype=int))
        # 单吸附态 (CO 或 O)
        for i in range(min(self.n_sites, 4)):
            s = np.zeros(self.n_sites, dtype=int)
            s[i] = 1
            states.append(s.copy())
            s[i] = 2
            states.append(s.copy())
        # 双吸附态 (邻近位点)
        for i in range(min(self.n_sites - 1, 3)):
            s = np.zeros(self.n_sites, dtype=int)
            s[i] = 1
            s[i + 1] = 2
            states.append(s.copy())
            s[i] = 2
            s[i + 1] = 1
            states.append(s.copy())
        # 去重
        unique = []
        seen = set()
        for s in states:
            key = tuple(s.tolist())
            if key not in seen:
                seen.add(key)
                unique.append(s)
        return unique

    def build_transition_matrix(self, rate_ads_co: float = 1.0,
                                rate_des_co: float = 0.1,
                                rate_ads_o: float = 0.5,
                                rate_des_o: float = 0.05,
                                rate_rxn: float = 0.2):
        """
        构建马尔可夫转移速率矩阵 W
        
        W_{ij} (i ≠ j): 从状态 i 到状态 j 的跃迁速率
        W_{ii} = -Σ_{j≠i} W_{ij} (离开状态 i 的总速率)
        
        事件类型:
          - 吸附 CO: 空位 → CO (速率 k_ads_co)
          - 脱附 CO: CO → 空位 (速率 k_des_co)
          - 吸附 O:  空位 → O  (速率 k_ads_o)
          - 脱附 O:  O → 空位  (速率 k_des_o)
          - 反应: CO + O → 空 + 空 (速率 k_rxn)
        """
        self.W = np.zeros((self.n_states, self.n_states))

        for i in range(self.n_states):
            for j in range(self.n_states):
                if i == j:
                    continue
                s_i = self.states[i]
                s_j = self.states[j]
                diff = s_j - s_i
                n_diff = np.sum(diff != 0)

                if n_diff == 1:
                    # 单粒子吸附/脱附
                    idx = np.where(diff != 0)[0][0]
                    if diff[idx] == 1 and s_i[idx] == 0:
                        # CO 吸附
                        self.W[i, j] = rate_ads_co
                    elif diff[idx] == 2 and s_i[idx] == 0:
                        # O 吸附
                        self.W[i, j] = rate_ads_o
                    elif diff[idx] == -1 and s_i[idx] == 1:
                        # CO 脱附
                        self.W[i, j] = rate_des_co
                    elif diff[idx] == -2 and s_i[idx] == 2:
                        # O 脱附
                        self.W[i, j] = rate_des_o
                elif n_diff == 2:
                    # 双粒子反应
                    idxs = np.where(diff != 0)[0]
                    if (s_i[idxs[0]] == 1 and s_i[idxs[1]] == 2 and
                        s_j[idxs[0]] == 0 and s_j[idxs[1]] == 0):
                        self.W[i, j] = rate_rxn
                    elif (s_i[idxs[0]] == 2 and s_i[idxs[1]] == 1 and
                          s_j[idxs[0]] == 0 and s_j[idxs[1]] == 0):
                        self.W[i, j] = rate_rxn

        # 对角线元素
        for i in range(self.n_states):
            self.W[i, i] = -np.sum(self.W[i, :])

    def solve_master_equation_ode(self, p0: np.ndarray, t_end: float,
                                  n_steps: int = 10000) -> Tuple[np.ndarray, np.ndarray]:
        """
        求解主方程时间演化
        
        dP/dt = W^T P
        
        使用矩阵指数或时间离散化方法
        """
        p = np.asarray(p0, dtype=float)
        if len(p) != self.n_states:
            raise ValueError("p0 长度必须等于状态数")
        if abs(np.sum(p) - 1.0) > 1e-6:
            p = p / np.sum(p)

        dt = t_end / n_steps
        trajectory = [p.copy()]
        times = [0.0]

        # 使用隐式 Euler 以保证概率守恒
        I = np.eye(self.n_states)
        M = I - dt * self.W.T
        for _ in range(n_steps):
            p = np.linalg.solve(M, p)
            p = np.maximum(p, 0.0)
            p = p / np.sum(p)
            trajectory.append(p.copy())
            times.append(times[-1] + dt)

        return np.array(trajectory), np.array(times)

    def steady_state_distribution(self) -> np.ndarray:
        """
        计算稳态概率分布
        
        求解:
          W^T P_ss = 0
          Σ P_ss = 1
        
        方法: 将方程组改写为 A P = b 求解
        """
        A = self.W.T.copy()
        A[-1, :] = 1.0  # 最后一个方程替换为概率归一化
        b = np.zeros(self.n_states)
        b[-1] = 1.0
        try:
            p_ss = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            p_ss = np.linalg.lstsq(A, b, rcond=None)[0]
        p_ss = np.maximum(p_ss, 0.0)
        p_ss = p_ss / np.sum(p_ss)
        return p_ss

    def compute_turnover_frequency(self, p_ss: np.ndarray) -> float:
        """
        计算转换频率 (Turnover Frequency, TOF)
        
        TOF = Σ_{i,j} W_{ij} P_i  (对所有反应事件求和)
        
        单位: s^{-1} per active site
        """
        tof = 0.0
        for i in range(self.n_states):
            for j in range(self.n_states):
                if i != j:
                    tof += self.W[i, j] * p_ss[i]
        return tof

    def mean_first_passage_time(self, target_state: int,
                                start_state: int) -> float:
        """
        计算从起始状态到目标状态的平均首次通过时间 (MFPT)
        
        公式:
          对于非目标状态子集 T，求解:
            Σ_{j∈T} W_{ij} τ_j = -1   (i ∈ T)
          
          MFPT = τ_{start}
        """
        n = self.n_states
        if target_state < 0 or target_state >= n:
            raise ValueError("target_state 超出范围")
        if start_state < 0 or start_state >= n:
            raise ValueError("start_state 超出范围")

        # 构建缩减矩阵 (移除目标状态)
        mask = np.ones(n, dtype=bool)
        mask[target_state] = False
        W_sub = self.W[np.ix_(mask, mask)]
        b = -np.ones(n - 1)

        try:
            tau = np.linalg.solve(W_sub, b)
        except np.linalg.LinAlgError:
            tau = np.linalg.lstsq(W_sub, b, rcond=None)[0]

        # 找到 start_state 在缩减矩阵中的索引
        idx = int(np.sum(~mask[:start_state]))
        if start_state == target_state:
            return 0.0
        return float(tau[start_state - idx])

    def entropy_production_rate(self, p_ss: np.ndarray) -> float:
        """
        计算稳态熵产生率
        
        公式 (Schnakenberg):
          σ = 0.5 * Σ_{i,j} (W_{ij} P_i - W_{ji} P_j) * ln(W_{ij} P_i / (W_{ji} P_j))
        """
        sigma = 0.0
        for i in range(self.n_states):
            for j in range(i + 1, self.n_states):
                j_i = self.W[i, j] * p_ss[i]
                i_j = self.W[j, i] * p_ss[j]
                if j_i > 1e-300 and i_j > 1e-300:
                    sigma += (j_i - i_j) * np.log(j_i / i_j)
        return sigma

    def dump_transition_matrix(self):
        """输出转移矩阵摘要"""
        print("=" * 60)
        print("马尔可夫转移速率矩阵 W (s^{-1})")
        print("=" * 60)
        print(f"状态数: {self.n_states}")
        print(f"总跃迁速率: {np.sum(self.W[self.W > 0]):.4e}")
        for i in range(self.n_states):
            row_sum = np.sum(self.W[i, :]) - self.W[i, i]
            print(f"  状态 {i}: 离开速率 = {row_sum:.4e}")
        print("=" * 60)
