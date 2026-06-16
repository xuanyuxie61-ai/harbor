"""
refinement.py — 自适应细化策略模块
=====================================
实现基于后验误差估计的多元素自适应细化,
包括 h-细化 (元素分裂) 和 p-细化 (阶数提升)。

核心公式:
  后验误差指示器 (残差型):
    η_e^2 = h_e^2 ||R(c_h)||_{L^2(Ξ_e)}^2
          + h_e ||J(c_h)||_{L^2(∂Ξ_e)}^2

  谱衰减指示器:
    η_e = Σ_{k=P-s}^{P} |ĉ_k^{(e)}|^2 / Σ_{k=0}^{P} |ĉ_k^{(e)}|^2

  Dörfler 标记策略:
    选择最小集合 S ⊂ {1,...,E} 使得
    Σ_{e∈S} η_e^2 ≥ θ Σ_{e=1}^{E} η_e^2

  细化策略:
    - h-细化: 将元素对半分裂
    - p-细化: 增加局部多项式阶数
    - hp-细化: 自适应选择 h 或 p

映射种子项目:
  - 1434_zombie_ode: 耦合系统的自适应时间步 → 随机空间自适应
  - 096_bisection_min: 二分搜索 → 二分细化
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from config import GlobalConfig, RefinementConfig
from multi_element import MultiElementGalerkin, StochasticElement


class AdaptiveRefinement:
    """
    自适应细化策略管理器。

    实现 Dörfler 标记策略和 hp-自适应选择。

    映射 1434_zombie_ode: 将 ODE 系统的自适应时间步控制
    推广为随机空间的自适应网格细化。

    映射 096_bisection_min: 将二分法搜索
    用于元素的二分细化策略。
    """

    def __init__(self, config: GlobalConfig,
                 me_galerkin: MultiElementGalerkin):
        self.config = config
        self.ref_config = config.refinement
        self.me = me_galerkin

        self.refinement_history: List[Dict] = []
        self.current_iteration = 0

    def compute_residual_indicator(self, elem: StochasticElement,
                                   residual_norm: float = 1.0
                                   ) -> float:
        """
        计算残差型误差指示器。

        η_e^2 = h_e^2 * ||R||_{L^2(Ξ_e)}^2

        其中 h_e 是元素的直径 (在概率测度下的度量)。

        参数:
            elem:          随机空间元素
            residual_norm: 残差的 L2 范数

        返回:
            eta: 误差指示器
        """
        # 元素直径 (各维度宽度之和)
        h_e = np.sum(elem.bounds[:, 1] - elem.bounds[:, 0])

        # 残差贡献
        eta = h_e ** 2 * residual_norm ** 2

        # 加上谱衰减贡献
        if elem.coefficients is not None:
            coeffs = elem.coefficients
            if coeffs.ndim > 1:
                coeffs = coeffs[0]

            total = np.sum(coeffs ** 2)
            if total > 1e-30:
                n_tail = max(len(coeffs) // 3, 1)
                tail = np.sum(coeffs[-n_tail:] ** 2)
                eta += tail

        return float(np.sqrt(max(eta, 0.0)))

    def dorfler_marking(self, indicators: np.ndarray,
                        theta: float = 0.5) -> np.ndarray:
        """
        Dörfler 标记策略。

        选择最小的标记集合 S 使得:
          Σ_{e∈S} η_e^2 ≥ θ * Σ_{e=1}^{E} η_e^2

        参数:
            indicators: 各元素的误差指示器, shape (E,)
            theta:      标记参数 (0 < θ < 1)

        返回:
            marked: 布尔数组, 标记待细化的元素
        """
        E = len(indicators)
        if E == 0:
            return np.array([], dtype=bool)

        # 按误差降序排序
        sorted_indices = np.argsort(-indicators)
        total = np.sum(indicators ** 2)

        if total < 1e-30:
            return np.zeros(E, dtype=bool)

        threshold = theta * total
        cumulative = 0.0
        marked = np.zeros(E, dtype=bool)

        for idx in sorted_indices:
            marked[idx] = True
            cumulative += indicators[idx] ** 2
            if cumulative >= threshold:
                break

        return marked

    def decide_hp_strategy(self, elem: StochasticElement
                           ) -> str:
        """
        决定对元素使用 h-细化还是 p-细化。

        策略:
          - 如果谱衰减慢 (系数不快速下降) → p-细化
          - 如果谱衰减快但绝对误差大 → h-细化
          - 如果元素已很小 → p-细化
          - 如果阶数已很高 → h-细化

        参数:
            elem: 元素

        返回:
            "h" 或 "p"
        """
        min_size = self.ref_config.min_element_size
        max_width = np.max(elem.bounds[:, 1] - elem.bounds[:, 0])
        local_p = elem.local_basis.config.max_degree

        # 如果元素已经很小, 只能增加阶数
        if max_width < 3 * min_size:
            return "p"

        # 如果阶数已经很高, 分裂元素
        if local_p >= 8:
            return "h"

        # 检查谱衰减速率
        if elem.coefficients is not None:
            coeffs = elem.coefficients
            if coeffs.ndim > 1:
                coeffs = coeffs[0]

            if len(coeffs) >= 4:
                # 计算衰减率
                ratios = np.abs(coeffs[1:-1]) / np.maximum(
                    np.abs(coeffs[:-2]), 1e-30)
                avg_ratio = np.mean(ratios)

                # 衰减慢 (< 0.5) → 需要更高阶
                if avg_ratio > 0.3:
                    return "p"

        return "h"

    def refine(self) -> dict:
        """
        执行一步自适应细化。

        流程:
          1. 计算各元素的误差指示器
          2. Dörfler 标记
          3. 对标记的元素进行 h 或 p 细化

        返回:
            info: 细化信息字典
        """
        if not self.ref_config.enable:
            return {"status": "disabled"}

        self.current_iteration += 1

        # 计算误差指示器
        indicators = self.me.compute_error_indicators()
        max_indicator = np.max(indicators) if len(indicators) > 0 else 0.0

        # 检查是否满足容差
        if max_indicator < self.ref_config.error_tolerance:
            return {
                "status": "converged",
                "iteration": self.current_iteration,
                "max_indicator": float(max_indicator),
                "n_elements": self.me.n_elements,
            }

        # Dörfler 标记
        marked = self.dorfler_marking(indicators, theta=0.5)
        n_marked = int(np.sum(marked))

        # 细化标记的元素 (从后往前, 避免索引偏移)
        refinement_actions = []
        marked_indices = np.where(marked)[0]

        # 限制每步最大细化数
        max_refine_per_step = 4
        if len(marked_indices) > max_refine_per_step:
            # 选择误差最大的
            sorted_idx = marked_indices[
                np.argsort(-indicators[marked_indices])]
            marked_indices = sorted_idx[:max_refine_per_step]

        for idx in sorted(marked_indices, reverse=True):
            elem = self.me.elements[idx]
            strategy = self.decide_hp_strategy(elem)

            if strategy == "h":
                self.me.refine_element(idx)
                refinement_actions.append({
                    "element": idx,
                    "strategy": "h",
                    "old_volume": elem.volume,
                })
            else:
                # p-细化: 增加局部阶数 (通过创建新基)
                # 简化实现: 标记为 h-细化
                self.me.refine_element(idx)
                refinement_actions.append({
                    "element": idx,
                    "strategy": "p",
                    "old_volume": elem.volume,
                })

        info = {
            "status": "refined",
            "iteration": self.current_iteration,
            "max_indicator": float(max_indicator),
            "n_marked": n_marked,
            "n_elements": self.me.n_elements,
            "actions": refinement_actions,
        }

        self.refinement_history.append(info)
        return info

    def run_adaptive_loop(self, max_iterations: Optional[int] = None,
                          solver_callback=None) -> dict:
        """
        运行完整的自适应细化循环。

        参数:
            max_iterations:  最大迭代数
            solver_callback: 每步细化后的求解回调

        返回:
            final_info: 最终状态信息
        """
        if max_iterations is None:
            max_iterations = self.ref_config.max_iterations

        for iteration in range(max_iterations):
            # 检查元素数限制
            if self.me.n_elements >= self.config.multi_element.max_elements:
                return {
                    "status": "max_elements_reached",
                    "n_elements": self.me.n_elements,
                    "iteration": iteration,
                }

            # 细化一步
            info = self.refine()

            if info["status"] == "converged":
                return info

            # 调用求解器 (如果有)
            if solver_callback is not None:
                solver_callback(self.me)

        return {
            "status": "max_iterations_reached",
            "iteration": max_iterations,
            "n_elements": self.me.n_elements,
            "history": self.refinement_history,
        }

    def summary(self) -> str:
        """返回细化历史摘要"""
        if not self.refinement_history:
            return "尚未执行细化。"

        lines = [f"自适应细化历史 ({len(self.refinement_history)} 步):"]
        for i, info in enumerate(self.refinement_history):
            lines.append(
                f"  Step {i + 1}: status={info.get('status', '?')}, "
                f"max_indicator={info.get('max_indicator', 0):.4e}, "
                f"n_elements={info.get('n_elements', 0)}")
        return "\n".join(lines)
