# -*- coding: utf-8 -*-
"""
adaptive_resolution.py
======================================================================
自适应波长分辨率控制器 —— 基于 RL 的网格优化

物理背景:
    系外行星大气光谱反演中, 波长分辨率的选择对计算效率
    和反演精度有重大影响:
    (1) 强吸收线区需要高分辨率 (R ~ 100000)
    (2) 连续谱区可以低分辨率 (R ~ 1000)
    (3) 最优分辨率分布取决于大气成分和温度结构

    本模块将自适应网格选择建模为马尔可夫决策过程 (MDP):
        State  : 当前网格配置 + 局部梯度信息
        Action : 细化 / 粗化 / 保持
        Reward : -(|error| + alpha * n_cells)

    使用 Q-learning 的简化版本 (无深度学习框架依赖),
    基于 1021_uiuc-ae598-rl 项目的 DQN 思想,
    但采用表格型 Q 函数以减小依赖。

    物理约束:
        - 总点数不超过 N_max
        - 最小间距 >= dx_min (避免病态)
        - 最大间距 <= dx_max (避免欠采样)

数学公式:
    误差估计:
        e_i = |f''(x_i)| * dx_i^2 / 12                    (1)
    最优间距 (等分布原则):
        dx_i ~ |f''(x_i)|^{-1/2}                           (2)

    Q-learning 更新:
        Q(s,a) <- Q(s,a) + alpha [r + gamma max_a' Q(s',a') - Q(s,a)]

依赖: numpy
======================================================================
"""

import numpy as np
from typing import Tuple, Dict, Optional, List


class AdaptiveResolutionController:
    """
    自适应分辨率控制器 (简化版 RL)。

    将 1021 项目的 DQN 简化为表格型 Q-learning:
        State: (gradient_class, cell_size_class)
        Action: 0=coarsen, 1=keep, 2=refine
    """

    # 离散化参数
    N_GRADIENT_BINS = 5
    N_SIZE_BINS = 4
    N_ACTIONS = 3

    def __init__(
        self,
        max_cells: int = 200,
        min_dx: float = 1.0e-4,
        max_dx: float = 0.1,
        learning_rate: float = 0.1,
        gamma: float = 0.95,
        epsilon_start: float = 0.8,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.99,
        rng_seed: int = 42,
    ):
        self.max_cells = max_cells
        self.min_dx = min_dx
        self.max_dx = max_dx
        self.alpha = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.rng = np.random.default_rng(rng_seed)

        self.q_table = np.zeros(
            (self.N_GRADIENT_BINS, self.N_SIZE_BINS, self.N_ACTIONS),
            dtype=np.float64,
        )
        self.visit_count = np.zeros_like(self.q_table, dtype=int)

        self._validate()

    def _validate(self) -> None:
        if self.max_cells < 20:
            raise ValueError(f"最大网格数过少: {self.max_cells}")
        if self.min_dx <= 0:
            raise ValueError(f"最小间距必须为正: {self.min_dx}")
        if self.max_dx <= self.min_dx:
            raise ValueError(f"最大间距 {self.max_dx} 必须大于最小间距 {self.min_dx}")

    def _discretize_state(
        self, gradient_mag: float, dx: float
    ) -> Tuple[int, int]:
        """将连续状态离散化。"""
        grad_bins = np.logspace(-6, 0, self.N_GRADIENT_BINS + 1)
        g_class = int(np.searchsorted(grad_bins, max(gradient_mag, 1.0e-10))) - 1
        g_class = max(0, min(g_class, self.N_GRADIENT_BINS - 1))

        dx_log = np.log10(max(dx, self.min_dx))
        dx_min_log = np.log10(self.min_dx)
        dx_max_log = np.log10(self.max_dx)
        s_class = int(
            (dx_log - dx_min_log) / (dx_max_log - dx_min_log) * self.N_SIZE_BINS
        )
        s_class = max(0, min(s_class, self.N_SIZE_BINS - 1))

        return g_class, s_class

    def select_action(self, state: Tuple[int, int]) -> int:
        """epsilon-greedy 动作选择。"""
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(0, self.N_ACTIONS))
        q_values = self.q_table[state[0], state[1], :]
        return int(np.argmax(q_values))

    def update(
        self,
        state: Tuple[int, int],
        action: int,
        reward: float,
        next_state: Tuple[int, int],
    ) -> None:
        """Q-learning 更新。"""
        current_q = self.q_table[state[0], state[1], action]
        max_next_q = np.max(self.q_table[next_state[0], next_state[1], :])
        td_target = reward + self.gamma * max_next_q
        self.q_table[state[0], state[1], action] += self.alpha * (
            td_target - current_q
        )
        self.visit_count[state[0], state[1], action] += 1

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_end)

    def adapt_grid(
        self,
        x_grid: np.ndarray,
        f_values: np.ndarray,
        n_training_episodes: int = 20,
    ) -> np.ndarray:
        """
        对给定网格进行自适应优化。

        物理意义: 根据函数 f 的梯度分布调整网格密度,
        使得在强梯度区 (吸收线中心) 网格更密,
        在平缓区 (连续谱) 网格更稀。
        """
        x = np.asarray(x_grid, dtype=np.float64).copy()
        f = np.asarray(f_values, dtype=np.float64).copy()

        # 确保 f 长度与 x 一致
        if len(f) != len(x):
            f = np.interp(x, np.linspace(x[0], x[-1], len(f)), f)

        for episode in range(n_training_episodes):
            n = len(x)
            if n < 3:
                break

            # 再次确保一致
            if len(f) != len(x):
                f = np.interp(x, np.linspace(x[0], x[-1], len(f)), f)
                if len(f) != len(x):
                    f = f[:len(x)] if len(f) > len(x) else np.pad(f, (0, len(x) - len(f)))

            # 梯度估计
            try:
                df = np.abs(np.gradient(f, x))
            except ValueError:
                df = np.abs(np.gradient(f))
            df_max = np.max(df)
            df = df / max(df_max, 1.0e-30)

            for i in range(1, n - 1):
                if i + 1 >= len(x):
                    break
                dx = x[i + 1] - x[i]
                if dx < 1.0e-30:
                    continue

                state = self._discretize_state(df[i], dx)
                action = self.select_action(state)

                # 执行动作 (确保长度一致)
                new_x = x.copy()
                new_f = f.copy()

                if action == 2 and len(x) < self.max_cells:
                    if dx > 2.0 * self.min_dx and i + 1 < len(f):
                        x_mid = 0.5 * (x[i] + x[i + 1])
                        f_mid = 0.5 * (f[i] + f[i + 1])
                        new_x = np.insert(x, i + 1, x_mid)
                        new_f = np.insert(f, i + 1, f_mid)
                elif action == 0 and len(x) > 20:
                    new_x = np.delete(x, i)
                    new_f = np.delete(f, i)

                # 安全校验: 长度必须一致
                if len(new_x) != len(new_f):
                    min_len = min(len(new_x), len(new_f))
                    new_x = new_x[:min_len]
                    new_f = new_f[:min_len]

                # 计算奖励
                if len(new_x) > 2:
                    try:
                        dx_arr = np.diff(new_x)
                        dx_arr = np.maximum(dx_arr, 1.0e-30)
                        grad_f = np.abs(np.gradient(new_f, new_x))
                        interp_error = float(np.max(grad_f) * np.max(dx_arr) ** 2)
                    except (ValueError, FloatingPointError):
                        interp_error = 0.0
                else:
                    interp_error = 0.0
                reward = -interp_error - 0.001 * len(new_x)

                # 下一个状态
                if len(new_x) > 2:
                    try:
                        new_df = np.abs(np.gradient(new_f, new_x))
                        new_df = new_df / max(np.max(new_df), 1.0e-30)
                        i_next = min(i, len(new_x) - 2)
                        dx_next = new_x[i_next + 1] - new_x[i_next]
                        next_state = self._discretize_state(
                            new_df[min(i_next, len(new_df) - 1)],
                            max(dx_next, self.min_dx),
                        )
                    except (ValueError, FloatingPointError):
                        next_state = state
                else:
                    next_state = state

                self.update(state, action, reward, next_state)

                x = new_x
                f = new_f

            self.decay_epsilon()

        return x


def build_adaptive_wavelength_grid(
    wl_min_um: float,
    wl_max_um: float,
    reference_spectrum: np.ndarray,
    reference_grid: np.ndarray,
    max_cells: int = 150,
    target_error: float = 1.0e-4,
) -> np.ndarray:
    """
    构建自适应波长网格 (高层接口)。
    """
    controller = AdaptiveResolutionController(max_cells=max_cells)
    uniform_grid = np.linspace(wl_min_um, wl_max_um, 50)
    adapted = controller.adapt_grid(uniform_grid, reference_spectrum, n_training_episodes=15)
    return np.sort(adapted)
