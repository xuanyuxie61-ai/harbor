"""
自适应载荷步控制模块
====================
基于种子项目:
  - 743_mcnuggets_diophantine: 多维丢番图方程/回溯搜索
  - 1179_subset_sum_backtrack: 子集和问题的回溯求解

科学背景:
  在大变形非线性有限元分析中，载荷必须分步施加以避免Newton-Raphson
  迭代发散。自适应载荷步控制根据收敛行为动态调整步长：
  - 收敛快 → 增大步长
  - 不收敛 → 回溯并减小步长

  将载荷历史离散化为若干子步 {Δλ_1, Δλ_2, ..., Δλ_n}，满足:
      Σ Δλ_i = λ_max
  这类似于子集和问题或丢番图方程的解空间搜索。

  本模块实现:
  1. 基于收敛历史的自适应步长调整 (回溯法)
  2. 载荷步的整数分解规划 (受丢番图方程启发)
  3. 弧长法 (Riks/Wempner) 的简化实现，用于跨越极限点

关键公式:
  - 载荷参数: λ ∈ [0, 1]
  - 步长调整: Δλ_{new} = min(Δλ_max, max(Δλ_min, η Δλ_{old}))
    其中 η = (N_desired / N_actual)^{1/2}
  - 弧长约束: Δu^T Δu + ψ^2 Δλ^2 = Δs^2
"""

import numpy as np
from typing import List, Tuple, Optional


class AdaptiveLoadStepping:
    """
    自适应载荷步控制器。
    """
    def __init__(self, lambda_max: float = 1.0,
                 initial_step: float = 0.1,
                 min_step: float = 0.001,
                 max_step: float = 0.5,
                 desired_iterations: int = 5,
                 max_total_steps: int = 200):
        self.lambda_max = lambda_max
        self.step = initial_step
        self.min_step = min_step
        self.max_step = max_step
        self.desired_iterations = desired_iterations
        self.max_total_steps = max_total_steps
        self.current_lambda = 0.0
        self.step_history = []
        self.iter_history = []
        self.n_backtracks = 0

    def adjust_step(self, n_iterations: int, converged: bool) -> float:
        """
        根据迭代次数和收敛性调整步长。
        基于Ramm(1981)的自适应策略:
          η = sqrt(N_desired / N_actual)
          若收敛: Δλ_new = min(Δλ_max, η Δλ_old)
          若不收敛: Δλ_new = max(Δλ_min, Δλ_old / 2)
        """
        if not converged:
            self.step = max(self.min_step, self.step / 2.0)
            self.n_backtracks += 1
        else:
            ratio = np.sqrt(self.desired_iterations / max(n_iterations, 1))
            self.step = min(self.max_step, max(self.min_step, ratio * self.step))
        self.step_history.append(self.step)
        self.iter_history.append(n_iterations)
        return self.step

    def next_lambda(self) -> Tuple[float, bool]:
        """
        计算下一个载荷参数值。
        返回 (lambda_next, finished)。
        finished=True 表示载荷已全部施加。
        """
        if self.current_lambda >= self.lambda_max - 1e-12:
            return self.current_lambda, True

        remaining = self.lambda_max - self.current_lambda
        actual_step = min(self.step, remaining)
        self.current_lambda += actual_step
        # 边界处理: 避免浮点误差导致超调
        if self.current_lambda > self.lambda_max:
            self.current_lambda = self.lambda_max
        return self.current_lambda, False

    def reset_to_previous(self):
        """回溯到上一步的lambda值。"""
        if len(self.step_history) > 0:
            last_step = self.step_history.pop()
            self.iter_history.pop()
            self.current_lambda -= last_step
            if self.current_lambda < 0:
                self.current_lambda = 0.0


def integer_load_partition(total_load: int, step_sizes: List[int],
                            target_steps: int) -> Optional[List[int]]:
    """
    将总载荷分解为给定步长大小的整数组合。
    类似多维丢番图方程求解: 寻找 {n_i} 使得 Σ n_i * step_sizes[i] = total_load
    且总步数接近 target_steps。

    参数:
        total_load: 总载荷(整数表示)
        step_sizes: 可用步长列表
        target_steps: 目标步数

    返回:
        partition: 步长序列，或None若无解
    """
    step_sizes = sorted(step_sizes, reverse=True)
    partition = []
    remaining = total_load

    # 贪心+回溯策略
    def backtrack(idx: int, rem: int, current: List[int]) -> Optional[List[int]]:
        if rem == 0:
            return current
        if idx >= len(step_sizes):
            return None
        s = step_sizes[idx]
        max_count = rem // s
        for count in range(max_count, -1, -1):
            new_rem = rem - count * s
            new_current = current + [s] * count
            result = backtrack(idx + 1, new_rem, new_current)
            if result is not None:
                return result
        return None

    result = backtrack(0, total_load, [])
    if result is None:
        return None
    # 若步数过多，尝试合并
    while len(result) > target_steps and len(result) > 1:
        # 合并两个最小步
        result = sorted(result)
        combined = result[0] + result[1]
        result = [combined] + result[2:]
    return result


def arc_length_control_step(u_n: np.ndarray, du_predictor: np.ndarray,
                             dlambda_predictor: float,
                             ds: float, psi: float = 0.0) -> Tuple[float, np.ndarray]:
    """
    简化的弧长法步长控制。
    约束条件: ||Δu||^2 + ψ^2 Δλ^2 = Δs^2

    参数:
        u_n: 当前位移
        du_predictor: 位移增量预测
        dlambda_predictor: 载荷增量预测
        ds: 弧长参数
        psi: 载荷权重因子

    返回:
        dlambda, du: 修正后的增量
    """
    norm_du = np.linalg.norm(du_predictor)
    constraint = norm_du ** 2 + psi ** 2 * dlambda_predictor ** 2
    if constraint < 1e-14:
        return dlambda_predictor, du_predictor
    scale = ds / np.sqrt(constraint)
    dlambda = dlambda_predictor * scale
    du = du_predictor * scale
    return dlambda, du
