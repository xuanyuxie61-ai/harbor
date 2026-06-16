"""
slds_phonon.py — 声子动力学状态切换跟踪 (SLDS)
===============================================

融合种子项目:
  - 1239_ynffsy_DurationModulatedDynamics:
    SLDS 模型: z_t ~ Cat(softmax(R*x_{t-1}+r))
    x_t ~ N(A_{z_t}*x_{t-1}+b_{z_t}, Sigma_{z_t})
    DSUP ratio = dsup / (dsup + ip)
    熵: H = -sum p*log(p)
  - 818_normal_ode: ODE 参考解

物理背景:
  在非谐晶格动力学中, 声子模式间的耦合导致
  动力学状态随时间切换 (如从准谐到非谐区域)。

  我们将 SLDS 框架应用于声子模式振幅的时间演化:
    离散状态 z_t: 声子占据的能级区域
    连续状态 x_t: 声子模式振幅 (复数)

  DSUP ratio 衡量动力学更新过程中确定性 vs 随机性的贡献:
    DSUP = ||(M-I)*x + b|| / (||(M-I)*x + b|| + ||K*innovation||)

  当 DSUP ≈ 1: 动力学由确定性方程主导 (准谐近似有效)
  当 DSUP ≈ 0: 动力学由随机涨落主导 (强非谐区域)
"""

import numpy as np
from typing import Tuple, Dict, List


class PhononSLDS:
    """
    声子动力学切换线性动力系统。

    将 DurationModulatedDynamics 的 SLDS 框架
    映射到声子模式振幅演化。

    状态方程:
      z_t ~ Categorical(softmax(R @ x_{t-1} + r))  # 离散状态切换
      x_t = A_{z_t} @ x_{t-1} + b_{z_t} + noise    # 连续演化
      y_t = C @ x_t + d + noise                     # 观测 (模式振幅)
    """

    def __init__(
        self,
        n_modes: int,
        n_discrete_states: int = 3,
        phonon_frequencies: np.ndarray = None,
    ):
        self.n_modes = n_modes
        self.n_states = n_discrete_states
        self.dim = n_modes

        # 初始化动力学参数
        self._init_dynamics_matrices(phonon_frequencies)

    def _init_dynamics_matrices(self, frequencies: np.ndarray = None):
        """
        初始化各离散状态下的动力学矩阵。

        状态 0: 准谐区域 (低频, 小振幅)
          A_0 ≈ cos(omega*dt), 弱阻尼
        状态 1: 弱非谐区域 (中等振幅)
          A_1 ≈ cos(omega*dt) + 非谐修正
        状态 2: 强非谐区域 (大振幅, 声子-声子散射)
          A_2 ≈ 衰减矩阵 (强阻尼)
        """
        self.As = []
        self.bs = []
        self.Sigmas = []

        if frequencies is None:
            frequencies = np.linspace(0.5, 5.0, self.n_modes)

        for s in range(self.n_states):
            if s == 0:
                # 准谐: 近酉矩阵
                omega_dt = frequencies * 0.01
                A = np.diag(np.cos(omega_dt))
                # 加入小的模间耦合
                coupling = 0.01 * np.random.randn(self.n_modes, self.n_modes)
                A += coupling
                b = np.zeros(self.n_modes)
                Sigma = 0.001 * np.eye(self.n_modes)
            elif s == 1:
                # 弱非谐: 带修正的动力学
                omega_dt = frequencies * 0.01
                A = np.diag(np.cos(omega_dt) * 0.99)
                coupling = 0.05 * np.random.randn(self.n_modes, self.n_modes)
                A += coupling
                b = 0.001 * np.ones(self.n_modes)
                Sigma = 0.01 * np.eye(self.n_modes)
            else:
                # 强非谐: 强阻尼
                omega_dt = frequencies * 0.01
                A = np.diag(np.cos(omega_dt) * 0.9)
                coupling = 0.1 * np.random.randn(self.n_modes, self.n_modes)
                A += coupling
                b = 0.01 * np.ones(self.n_modes)
                Sigma = 0.1 * np.eye(self.n_modes)

            self.As.append(A)
            self.bs.append(b)
            self.Sigmas.append(Sigma)

        # 状态转移矩阵 (依赖于连续状态)
        self.R = 0.1 * np.random.randn(self.n_states, self.n_modes)
        self.r = np.zeros(self.n_states)

        # 观测矩阵
        self.C = np.eye(self.n_modes) + 0.01 * np.random.randn(self.n_modes, self.n_modes)
        self.d = np.zeros(self.n_modes)
        self.obs_noise = 0.01 * np.eye(self.n_modes)

    def sample_trajectory(
        self, n_steps: int, x0: np.ndarray = None, seed: int = 42,
    ) -> Dict[str, np.ndarray]:
        """
        生成声子动力学轨迹。

        返回:
            results: 包含 discrete_states, continuous_states, observations
        """
        rng = np.random.RandomState(seed)

        if x0 is None:
            x0 = 0.1 * rng.randn(self.n_modes)

        x_traj = np.zeros((n_steps, self.n_modes))
        z_traj = np.zeros(n_steps, dtype=int)
        y_traj = np.zeros((n_steps, self.n_modes))

        x = x0.copy()
        z = 0  # 初始状态: 准谐

        for t in range(n_steps):
            # 观测
            y = self.C @ x + self.d + rng.multivariate_normal(
                np.zeros(self.n_modes), self.obs_noise
            )
            y_traj[t] = y
            x_traj[t] = x
            z_traj[t] = z

            # 状态转移
            logits = self.R @ x + self.r
            probs = np.exp(logits - np.max(logits))
            probs /= np.sum(probs)
            z = rng.choice(self.n_states, p=probs)

            # 连续演化
            x_new = self.As[z] @ x + self.bs[z]
            x_new += rng.multivariate_normal(np.zeros(self.n_modes), self.Sigmas[z])
            x = x_new

        return {
            'discrete_states': z_traj,
            'continuous_states': x_traj,
            'observations': y_traj,
        }

    def compute_dsup_ratio(
        self, x_prev: np.ndarray, x_curr: np.ndarray,
        y_curr: np.ndarray, state: int,
    ) -> float:
        """
        计算 DSUP 比率 (融合 DurationModulatedDynamics)。

        DSUP = dsup / (dsup + ip)

        其中:
          dsup = ||(A-I)*x + b||  (动力学更新量)
          ip = ||K * (y - C*(A*x+b) - d)||  (观测修正量)
          K = Kalman 增益

        物理含义:
          DSUP ≈ 1: 模式演化由晶格动力学方程主导
          DSUP ≈ 0: 模式演化由热涨落/测量噪声主导
        """
        A = self.As[state]
        b = self.bs[state]

        # 预测
        x_pred = A @ x_prev + b

        # DSUP 部分: 动力学确定性更新
        dsup = np.linalg.norm((A - np.eye(self.n_modes)) @ x_prev + b)

        # Innovation 部分: 观测与预测的差异
        innovation = y_curr - self.C @ x_pred - self.d

        # 简化的 Kalman 增益
        P = self.Sigmas[state]
        S = self.C @ P @ self.C.T + self.obs_noise
        K = P @ self.C.T @ np.linalg.inv(S + 1e-10 * np.eye(self.n_modes))

        ip = np.linalg.norm(K @ innovation)

        denom = dsup + ip
        if denom < 1e-30:
            return 0.5
        return dsup / denom

    def compute_entropy(self, state_probs: np.ndarray) -> float:
        """
        计算离散状态的 Shannon 熵 (融合 DurationModulatedDynamics)。

        H = -sum_z p(z) * log(p(z))

        H = 0: 系统处于确定状态
        H = log(n_states): 完全随机
        """
        p = state_probs[state_probs > 1e-30]
        return -np.sum(p * np.log(p))

    def fit_to_phonon_data(
        self,
        observations: np.ndarray,
        n_iterations: int = 20,
    ) -> Dict[str, float]:
        """
        简化 EM 拟合 (融合 DurationModulatedDynamics 的 Laplace-EM)。

        E步: 推断离散/连续状态的后验
        M步: 更新动力学参数

        返回:
            fit_results: ELBO, 收敛信息
        """
        n_steps, n_obs = observations.shape
        elbos = []

        for iteration in range(n_iterations):
            # E步: 前向-后向 (简化版)
            log_likelihood = 0.0
            for t in range(n_steps):
                y = observations[t]
                # 简化: 计算观测与最近状态的匹配度
                best_ll = -np.inf
                for s in range(self.n_states):
                    y_pred = self.C @ (self.As[s] @ (observations[max(0, t - 1)]) + self.bs[s]) + self.d
                    residual = y - y_pred
                    ll = -0.5 * np.sum(residual ** 2) / np.trace(self.obs_noise)
                    best_ll = max(best_ll, ll)
                log_likelihood += best_ll

            elbos.append(log_likelihood)

            # M步: 简化参数更新
            for s in range(self.n_states):
                self.Sigmas[s] *= 0.99  # 衰减正则化

            if len(elbos) > 1 and abs(elbos[-1] - elbos[-2]) < 1e-6:
                break

        return {
            'elbos': np.array(elbos),
            'final_elbo': elbos[-1],
            'n_iterations': len(elbos),
            'converged': len(elbos) < n_iterations,
        }


def compute_phonon_lifetime_from_slds(
    slds: PhononSLDS,
    trajectory: Dict[str, np.ndarray],
) -> np.ndarray:
    """
    从 SLDS 轨迹提取声子寿命。

    声子寿命 tau_n 与状态驻留时间相关:
      1/tau_n = sum_z p(z) / tau_z
    其中 tau_z 是状态 z 的特征时间尺度。
    """
    z_traj = trajectory['discrete_states']
    n_steps = len(z_traj)

    # 计算每个状态的驻留时间
    state_durations = {}
    current_state = z_traj[0]
    start = 0

    for t in range(1, n_steps):
        if z_traj[t] != current_state:
            dur = t - start
            if current_state not in state_durations:
                state_durations[current_state] = []
            state_durations[current_state].append(dur)
            current_state = z_traj[t]
            start = t

    # 最后一段
    dur = n_steps - start
    if current_state not in state_durations:
        state_durations[current_state] = []
    state_durations[current_state].append(dur)

    # 平均驻留时间 -> 声子寿命
    lifetimes = np.zeros(slds.n_modes)
    for s in range(slds.n_states):
        if s in state_durations and len(state_durations[s]) > 0:
            mean_dur = np.mean(state_durations[s])
            # 分配到该状态主导的模式
            lifetimes += mean_dur / slds.n_states

    return lifetimes
