# -*- coding: utf-8 -*-
"""
stochastic_optimizer.py
=======================

随机优化算法库: 求解 SAA 近似问题

    min_{x in X}  F_N(x) = (1/N) sum_{i=1}^N f(x, xi_i)

融合的种子项目:
  - 613_jumping_bean_simulation -> 温度依赖随机跳跃 (模拟退火式扰动)
  - 773_mnist_neural            -> 小批量 SGD 训练循环
  - 022_asa_graphs              -> 排列洗牌的 batch ordering

核心算法:
  1. SGD with Polyak-Ruppert averaging
  2. Jumping-bean perturbation (模拟退火)
  3. 混合策略: SGD + 周期性 jumping-bean 探索
"""

import math
from typing import List, Callable, Tuple, Optional
from middle_square_rng import MiddleSquareRNG
from saa_objective import SAAObjective, SAASampler, RunningStatistics
from scientific_formulas import sgd_convergence_rate


class StepSizeSchedule:
    """步长策略.

    经典 Robbins-Monro 条件:
        sum eta_t = infty,  sum eta_t^2 < infty

    实现:
        eta_t = eta_0 / (1 + decay * t)^power
    """

    def __init__(self, eta0: float = 0.1, decay: float = 0.05,
                 power: float = 0.6):
        self.eta0 = eta0
        self.decay = decay
        self.power = power

    def at(self, t: int) -> float:
        return self.eta0 / ((1.0 + self.decay * t) ** self.power)


class SGDOptimizer:
    """随机梯度下降求解 SAA 问题.

    算法 (复刻 773_mnist_neural 的小批量训练结构):
        for t = 1, 2, ..., T:
            xi_batch = sampler.next_batch(B)
            g_t = grad F_{xi_batch}(x_t)    (有限差分近似)
            x_{t+1} = x_t - eta_t * g_t

    选项:
        - Polyak-Ruppert 平均化: x_bar_T = (1/T) sum x_t
        - 投影到可行域 (box constraints)
    """

    def __init__(self, objective: SAAObjective,
                 sampler: SAASampler,
                 x_init: List[float],
                 step_size: StepSizeSchedule,
                 batch_size: int = 4,
                 n_iter: int = 20,
                 rng: Optional[MiddleSquareRNG] = None,
                 use_averaging: bool = True):
        self.objective = objective
        self.sampler = sampler
        self.x = list(x_init)
        self.dim = len(x_init)
        self.step_size = step_size
        self.batch_size = batch_size
        self.n_iter = n_iter
        self.rng = rng or MiddleSquareRNG(seed=42)
        self.use_averaging = use_averaging

        # 轨迹
        self.x_history: List[List[float]] = [list(x_init)]
        self.f_history: List[float] = []
        self.grad_history: List[List[float]] = []

        # 运行统计
        self.stats = RunningStatistics()

    def _project_box(self, x: List[float],
                     lower: float = -3.0, upper: float = 3.0
                     ) -> List[float]:
        """投影到 box [lower, upper]^dim (KL 系数物理上不能太大)."""
        return [max(lower, min(upper, xi)) for xi in x]

    def run(self) -> dict:
        """执行 SGD 优化循环.

        Returns:
            dict: 最终解, 历史, 统计
        """
        x_avg = [0.0] * self.dim  # Polyak-Ruppert 累加器

        for t in range(1, self.n_iter + 1):
            eta = self.step_size.at(t)

            # 采样小批量
            xi_batch = self.sampler.next_batch(self.batch_size)

            # 有限差分梯度
            grad = self.objective.finite_difference_gradient(
                self.x, xi_batch, eps=1e-3)

            # 梯度裁剪 (防止爆炸)
            g_norm = math.sqrt(sum(g * g for g in grad))
            max_norm = 10.0
            if g_norm > max_norm:
                grad = [g * max_norm / g_norm for g in grad]

            # 更新
            for k in range(self.dim):
                self.x[k] -= eta * grad[k]

            # 投影
            self.x = self._project_box(self.x)

            # 记录
            self.x_history.append(list(self.x))
            self.grad_history.append(list(grad))

            # 评估当前目标 (用新 batch)
            xi_eval = self.sampler.next_batch(self.batch_size)
            f_val = self.objective.evaluate_batch(self.x, xi_eval)
            self.f_history.append(f_val)
            self.stats.update(f_val)

            # Polyak-Ruppert 累加
            if self.use_averaging:
                for k in range(self.dim):
                    x_avg[k] += self.x[k]

        # 平均解
        if self.use_averaging:
            x_avg = [xk / self.n_iter for xk in x_avg]
            x_avg = self._project_box(x_avg)
        else:
            x_avg = list(self.x)

        # 最终评估
        xi_final = self.sampler.next_batch(max(self.batch_size * 2, 8))
        f_final = self.objective.evaluate_batch(x_avg, xi_final)

        return {
            "x_opt": x_avg,
            "x_last": list(self.x),
            "f_final": f_final,
            "f_history": self.f_history,
            "f_avg": self.stats.mean(),
            "f_ci": self.stats.confidence_interval(0.95),
            "n_iter": self.n_iter,
            "running_averages": self.stats.running_averages(),
        }


class JumpingBeanPerturbation:
    """跳跃豆式随机扰动 (复刻 seed project 613_jumping_bean_simulation).

    灵感: 跳跃豆根据环境温度决定是否跳跃, 温度越高跳得越频繁.
    在优化中, "温度" 类比为目标函数值相对于基线的偏移:
      - 当前解差 -> "温度高" -> 大幅随机探索
      - 当前解好 -> "温度低" -> 局部微调

    数学:
      T_t = |f(x_t) - f_baseline| / (|f_baseline| + eps)
      p_jump(t) = min(1, T_t / T_max)
      if rand() < p_jump:
          x_{t+1} = x_t + sigma * T_t * N(0, I)

    这是一种自适应的 Metropolis 风格探索.
    """

    def __init__(self, rng: MiddleSquareRNG, dim: int,
                 sigma_base: float = 0.3, T_max: float = 1.0,
                 ground_temp: float = 0.1):
        self.rng = rng
        self.dim = dim
        self.sigma_base = sigma_base
        self.T_max = max(T_max, 1e-8)
        self.ground_temp = ground_temp

    def ground_temperature(self, x: List[float]) -> float:
        """地面温度函数 (复刻 jumping_bean_simulation.m 中
        ground_temperature(x, y) 的空间依赖温度场).
        此处用 ||x||^2 作为"远离原点越热"的温度场.
        """
        return sum(xi * xi for xi in x) / self.dim

    def step(self, x: List[float], f_current: float,
             f_baseline: float) -> List[float]:
        """单步跳跃豆扰动."""
        T = abs(f_current - f_baseline) / (abs(f_baseline) + 1e-8)
        T = max(T, self.ground_temp)
        T = min(T, self.T_max)

        p_jump = min(1.0, T / self.T_max)
        if self.rng.next_uniform() < p_jump:
            # 跳跃
            sigma = self.sigma_base * T
            new_x = []
            for k in range(self.dim):
                noise = self.rng.next_gaussian() * sigma
                new_x.append(x[k] + noise)
            return new_x
        else:
            # 温度调节: 向 ground_temperature 靠近
            g_temp = self.ground_temperature(x)
            # 如果当前位置温度高于地面, 小幅向原点收缩
            scale = 0.95 if g_temp > self.ground_temp else 1.0
            return [xi * scale for xi in x]


class HybridSGAJumpingBean:
    """混合优化器: SGD + Jumping-Bean 探索.

    交替执行:
      1. SGD 阶段: 利用梯度做局部 exploitation
      2. JB 阶段:  跳跃豆式全局 exploration
    每 K_sgd 步 SGD 后接一次 JB 扰动.
    """

    def __init__(self, sgd: SGDOptimizer,
                 jb: JumpingBeanPerturbation,
                 k_sgd: int = 5,
                 rng: Optional[MiddleSquareRNG] = None):
        self.sgd = sgd
        self.jb = jb
        self.k_sgd = k_sgd
        self.rng = rng or MiddleSquareRNG(seed=123)
        self.best_x = list(sgd.x)
        self.best_f = float('inf')
        self.f_baseline = 1.0

    def run(self) -> dict:
        """执行混合优化."""
        f_best_history = []
        sgd_phase_history = []

        phase = 0
        total_iter = self.sgd.n_iter

        for t in range(1, total_iter + 1):
            eta = self.sgd.step_size.at(t)
            xi_batch = self.sgd.sampler.next_batch(self.sgd.batch_size)
            grad = self.sgd.objective.finite_difference_gradient(
                self.sgd.x, xi_batch, eps=1e-3)

            # 梯度裁剪
            g_norm = math.sqrt(sum(g * g for g in grad))
            if g_norm > 10.0:
                grad = [g * 10.0 / g_norm for g in grad]

            # SGD 步
            for k in range(self.sgd.dim):
                self.sgd.x[k] -= eta * grad[k]
            self.sgd.x = self.sgd._project_box(self.sgd.x)

            # 评估
            xi_eval = self.sgd.sampler.next_batch(self.sgd.batch_size)
            f_val = self.sgd.objective.evaluate_batch(self.sgd.x, xi_eval)
            sgd_phase_history.append(f_val)
            self.sgd.stats.update(f_val)

            # 跟踪最优
            if f_val < self.best_f:
                self.best_f = f_val
                self.best_x = list(self.sgd.x)

            # 周期性 JB 扰动
            if t % self.k_sgd == 0:
                self.sgd.x = self.jb.step(
                    self.sgd.x, f_val, self.f_baseline)
                self.sgd.x = self.sgd._project_box(self.sgd.x)

            f_best_history.append(self.best_f)

        return {
            "x_opt": self.best_x,
            "f_best": self.best_f,
            "f_best_history": f_best_history,
            "sgd_phase_history": sgd_phase_history,
            "mean_f": self.sgd.stats.mean(),
        }
