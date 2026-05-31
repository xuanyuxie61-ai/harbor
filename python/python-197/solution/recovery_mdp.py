"""
recovery_mdp.py
================================================================================
高性能计算检查点容错：基于马尔可夫决策过程的恢复策略优化

融合原项目：
  - 1091_snakes_and_ladders (马尔可夫链与转移矩阵)

科学角色：
  1) 将 HPC 执行过程建模为离散时间马尔可夫链：
         状态 = {计算, 检查点写入, 检查点验证, 故障恢复, 完成}；
  2) 利用转移概率矩阵分析各状态的稳态分布与期望 hitting time；
  3) 通过值迭代求解最优恢复策略（选择重启级别以最小化期望完工时间）。
================================================================================
"""

import numpy as np


class CheckpointMDP:
    """
    检查点-恢复 MDP。
    状态: 0=Compute, 1=Checkpoint, 2=Verify, 3=Recover, 4=Done
    动作: 0=内存恢复, 1=本地恢复, 2=远程恢复
    """

    STATES = ["Compute", "Checkpoint", "Verify", "Recover", "Done"]
    ACTIONS = ["Memory", "Local", "Remote"]

    def __init__(self, p_fault: float, p_fault_during_ckpt: float,
                 recover_probs: np.ndarray, step_costs: np.ndarray):
        """
        p_fault: 每步计算阶段故障概率
        p_fault_during_ckpt: 检查点写入阶段故障概率
        recover_probs: shape (3,)，各恢复动作成功概率
        step_costs: shape (5, 3)，cost[state, action]
        """
        self.n_states = 5
        self.n_actions = 3
        self.p_fault = max(0.0, min(1.0, p_fault))
        self.p_fault_during_ckpt = max(0.0, min(1.0, p_fault_during_ckpt))
        self.recover_probs = np.asarray(recover_probs, dtype=float)
        self.step_costs = np.asarray(step_costs, dtype=float)
        self._build_transition_matrices()

    def _build_transition_matrices(self):
        """构造每个动作下的转移概率矩阵 P[a][s, s']。"""
        self.P = np.zeros((self.n_actions, self.n_states, self.n_states))
        for a in range(self.n_actions):
            P = self.P[a]
            # Compute -> Compute (无故障) 或 Recover (故障)
            P[0, 0] = 1.0 - self.p_fault
            P[0, 3] = self.p_fault
            # Checkpoint -> Verify (成功) 或 Recover (故障)
            P[1, 2] = 1.0 - self.p_fault_during_ckpt
            P[1, 3] = self.p_fault_during_ckpt
            # Verify -> Compute (成功) 或 Done (若已收敛)
            P[2, 0] = 0.9
            P[2, 4] = 0.1
            # Recover -> Compute (恢复成功) 或 Recover (再次故障)
            succ = self.recover_probs[a]
            P[3, 0] = succ
            P[3, 3] = 1.0 - succ
            # Done -> Done (吸收态)
            P[4, 4] = 1.0

    def value_iteration(self, gamma: float = 0.95, tol: float = 1.0e-8, max_iter: int = 10000):
        """
        值迭代求解最优策略。
        返回 (V, policy)。
        """
        V = np.zeros(self.n_states)
        policy = np.zeros(self.n_states, dtype=int)
        for _ in range(max_iter):
            V_old = V.copy()
            for s in range(self.n_states):
                q_vals = np.zeros(self.n_actions)
                for a in range(self.n_actions):
                    q_vals[a] = self.step_costs[s, a] + gamma * np.dot(self.P[a, s, :], V_old)
                V[s] = np.min(q_vals)
                policy[s] = np.argmin(q_vals)
            if np.max(np.abs(V - V_old)) < tol:
                break
        return V, policy

    def stationary_distribution(self, action: int = 0) -> np.ndarray:
        """
        对给定动作，计算马尔可夫链的稳态分布（忽略 Done 吸收态后的子链）。
        """
        P = self.P[action][:4, :4]
        # 归一化行
        row_sums = P.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        P = P / row_sums
        # 求左特征向量
        w, v = np.linalg.eig(P.T)
        idx = np.argmin(np.abs(w - 1.0))
        pi = np.real(v[:, idx])
        pi = np.abs(pi)
        pi = pi / np.sum(pi)
        return pi

    def expected_time_to_done(self, action: int = 0, max_steps: int = 10000) -> float:
        """
        从 Compute 状态出发，期望多少步到达 Done（使用模拟）。
        """
        P = self.P[action]
        state = 0
        total_cost = 0.0
        for step in range(max_steps):
            if state == 4:
                break
            total_cost += self.step_costs[state, action]
            state = np.random.choice(self.n_states, p=P[state, :])
        return total_cost
