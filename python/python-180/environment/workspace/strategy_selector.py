"""
strategy_selector.py
基于局部特征的数值格式自适应选择策略

融合种子项目:
  - 1022_reversi_game: 离散状态空间上的博弈决策与贪心评估

科学背景:
  在 SPDE 数值积分中，解场的不同区域具有不同的数学特征：
    - 平滑区: 高阶中心差分或谱方法足够精确
    - 梯度层/波前: 需要迎风或 DG 格式抑制数值振荡
    - 刚性区: 需要隐式或半隐式时间积分

  将每个空间节点视为 "棋盘格子"，其状态为局部物理指标向量:
      s_i = [Pe_i, CFL_i, |du/dx|_i, |d2u/dx2|_i]
  其中:
      Pe_i = |v| h_i / epsilon          (局部 Peclet 数)
      CFL_i = |v| dt / h_i              (Courant-Friedrichs-Lewy 数)

  决策规则 (类比 reversi 的合法落子与翻转评估):
    1. 若 Pe_i < 1.5 且 CFL_i < 0.5: 选择 "centered_diffusion"
    2. 若 Pe_i >= 1.5 且 CFL_i < 1.0: 选择 "upwind_advection"
    3. 若 CFL_i >= 1.0: 选择 "lax_wendroff" 或缩减时间步长
    4. 若 |d2u/dx2| 极大: 标记为 "shock_layer"，启用限制器

  每个决策的 "得分" 由局部截断误差估计决定:
      score = - log10( |estimated_local_error| + 1e-16 )
  贪心策略选择得分最高的格式组合。
"""

import numpy as np
from typing import List, Tuple


class LocalState:
    """
    局部状态描述符。
    """

    def __init__(self,
                 peclet: float,
                 cfl: float,
                 grad: float,
                 curvature: float):
        self.peclet = peclet
        self.cfl = cfl
        self.grad = grad
        self.curvature = curvature


class NumericalStrategy:
    """
    可用的数值策略。
    """

    STRATEGIES = ["centered", "upwind", "lax_wendroff", "shock_capturing"]

    def __init__(self, name: str):
        if name not in self.STRATEGIES:
            raise ValueError(f"Unknown strategy {name}")
        self.name = name

    def estimate_error(self, state: LocalState) -> float:
        """
        基于局部状态估计该策略的截断误差。
        """
        if self.name == "centered":
            # 二阶中心差分误差: O(h^2) * |u^{(4)}|，但对流主导时不稳定
            if state.peclet > 2.0:
                return 1e3  # 惩罚不稳定
            return state.cfl ** 2 + state.curvature * 1e-3
        elif self.name == "upwind":
            # 一阶迎风误差: O(h)
            return state.cfl + state.grad * 1e-2
        elif self.name == "lax_wendroff":
            # Lax-Wendroff 二阶，但有色散误差
            return state.cfl ** 2 + state.grad * 1e-3 + abs(state.cfl - 1.0) * 1e-2
        elif self.name == "shock_capturing":
            # 高耗散，适用于激波层
            if state.curvature < 10.0:
                return 1.0  # 不必要的高耗散
            return 0.1 * state.curvature
        return 1.0


class StrategySelector:
    """
    自适应策略选择器，类比 reversi 的贪心最优落子。
    """

    def __init__(self):
        self.strategies = [NumericalStrategy(s) for s in NumericalStrategy.STRATEGIES]

    def evaluate_state(self,
                       u: np.ndarray,
                       x: np.ndarray,
                       v: float,
                       epsilon: float,
                       dt: float) -> List[LocalState]:
        """
        计算每个节点的局部状态。
        """
        nx = len(u)
        states = []
        for i in range(nx):
            hp = x[min(i + 1, nx - 1)] - x[i] if i < nx - 1 else x[i] - x[max(i - 1, 0)]
            hm = x[i] - x[max(i - 1, 0)] if i > 0 else hp
            h = 0.5 * (hp + hm)
            pe = abs(v) * h / epsilon if epsilon > 0 else 1e6
            cfl = abs(v) * dt / h if h > 0 else 0.0

            if 0 < i < nx - 1:
                grad = (u[i + 1] - u[i - 1]) / (hp + hm)
                curvature = abs((u[i + 1] - 2 * u[i] + u[i - 1]) / (0.5 * (hp + hm)) ** 2)
            else:
                grad = 0.0
                curvature = 0.0

            states.append(LocalState(pe, cfl, abs(grad), curvature))
        return states

    def select_best_strategies(self,
                               u: np.ndarray,
                               x: np.ndarray,
                               v: float,
                               epsilon: float,
                               dt: float) -> List[str]:
        """
        为每个节点选择最优策略。
        """
        states = self.evaluate_state(u, x, v, epsilon, dt)
        best = []
        for state in states:
            scores = [(s.name, -np.log10(s.estimate_error(state) + 1e-16)) for s in self.strategies]
            scores.sort(key=lambda item: item[1], reverse=True)
            best.append(scores[0][0])
        return best

    def aggregate_recommendation(self,
                                 u: np.ndarray,
                                 x: np.ndarray,
                                 v: float,
                                 epsilon: float,
                                 dt: float) -> str:
        """
        综合所有节点的建议选择全局格式（多数表决）。
        """
        best_per_node = self.select_best_strategies(u, x, v, epsilon, dt)
        from collections import Counter
        c = Counter(best_per_node)
        return c.most_common(1)[0][0]
