#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
nsga2_optimizer.py — NSGA-II 多目标进化优化器

对应种子项目:
  - 909_predator_prey_ode_period: 竞争-合作动态映射到选择压力
  - 906_pram_view: PRAM 并行分块映射到种群并行评估

核心算法 (Deb et al. 2002):
  1. 初始化种群 P_0 (大小 N)
  2. 非支配排序 + 拥挤距离 → 锦标赛选择
  3. SBX 交叉 + 多项式变异 → 子代 Q_t
  4. 合并 R_t = P_t ∪ Q_t
  5. 快速非支配排序 → 分层 F_1, F_2, ...
  6. 依次填充下一前沿, 直到超过 N
  7. 超出部分用拥挤距离截断

SBX 交叉 (Simulated Binary Crossover):
    对每个变量:
        u ~ U(0,1)
        β_q = (2u)^{1/(η+1)}           if u ≤ 0.5
              (1/(2(1-u)))^{1/(η+1)}    if u > 0.5
        c1 = 0.5((p1+p2) - β_q(p2-p1))
        c2 = 0.5((p1+p2) + β_q(p2-p1))

多项式变异:
    u ~ U(0,1)
    δ = (2u)^{1/(η_m+1)} - 1           if u < 0.5
        1 - (2(1-u))^{1/(η_m+1)}        if u ≥ 0.5
    child = parent + δ · (upper - lower)
"""

import numpy as np
from typing import Callable, Tuple, Dict, List
from pareto_core import (
    fast_non_dominated_sort, crowding_distance,
    extract_pareto_front, hypervolume_2d, spacing_metric
)


# ---------------------------------------------------------------------------
# 遗传算子
# ---------------------------------------------------------------------------
def tournament_select(pop: np.ndarray, fitness: np.ndarray,
                      fronts: List[List[int]],
                      cd: np.ndarray,
                      n_select: int,
                      tournament_size: int = 2,
                      seed: int = 42) -> np.ndarray:
    """
    锦标赛选择 — 基于非支配层和拥挤距离.

    比较规则:
        1. 层号小的优先 (更靠近 Pareto 前沿)
        2. 层号相同时, 拥挤距离大的优先

    Parameters
    ----------
    pop           : shape (N, D)
    fitness       : shape (N, M)
    fronts        : 非支配排序结果
    cd            : shape (N,) 拥挤距离
    n_select      : 选择数量
    tournament_size : 锦标赛大小

    Returns
    -------
    selected : shape (n_select, D)
    """
    rng = np.random.default_rng(seed)
    N = pop.shape[0]

    # 建立 rank 映射
    rank = np.zeros(N, dtype=int)
    for r, front in enumerate(fronts):
        for idx in front:
            rank[idx] = r

    selected = np.zeros((n_select, pop.shape[1]))
    for i in range(n_select):
        # 随机选取 tournament_size 个候选
        candidates = rng.choice(N, size=min(tournament_size, N), replace=False)
        best = candidates[0]
        for c in candidates[1:]:
            if rank[c] < rank[best]:
                best = c
            elif rank[c] == rank[best] and cd[c] > cd[best]:
                best = c
        selected[i] = pop[best]

    return selected


def sbx_crossover(parent1: np.ndarray, parent2: np.ndarray,
                  eta: float = 20.0,
                  lower: np.ndarray = None,
                  upper: np.ndarray = None,
                  rng: np.random.Generator = None
                  ) -> Tuple[np.ndarray, np.ndarray]:
    """
    SBX (Simulated Binary Crossover) 交叉算子.

    对每个维度独立进行:
        u ~ U(0,1)
        β_q = (2u)^{1/(η+1)}           if u ≤ 0.5
              (1/(2(1-u)))^{1/(η+1)}    if u > 0.5
        c1 = 0.5((p1+p2) - β_q(p2-p1))
        c2 = 0.5((p1+p2) + β_q(p2-p1))

    Parameters
    ----------
    parent1, parent2 : shape (D,)
    eta              : 分布指数 (越大, 子代越接近父代)
    lower, upper     : 变量边界

    Returns
    -------
    (child1, child2)
    """
    if rng is None:
        rng = np.random.default_rng()
    D = len(parent1)
    u = rng.uniform(0, 1, D)

    beta = np.where(
        u <= 0.5,
        (2 * u) ** (1.0 / (eta + 1)),
        (1.0 / (2 * (1 - u))) ** (1.0 / (eta + 1))
    )

    child1 = 0.5 * ((parent1 + parent2) - beta * (parent2 - parent1))
    child2 = 0.5 * ((parent1 + parent2) + beta * (parent2 - parent1))

    # 边界处理
    if lower is not None and upper is not None:
        child1 = np.clip(child1, lower, upper)
        child2 = np.clip(child2, lower, upper)

    return child1, child2


def polynomial_mutation(individual: np.ndarray,
                        eta_m: float = 20.0,
                        lower: np.ndarray = None,
                        upper: np.ndarray = None,
                        mutation_rate: float = None,
                        rng: np.random.Generator = None
                        ) -> np.ndarray:
    """
    多项式变异算子.

    对每个维度以概率 p_m 进行:
        u ~ U(0,1)
        δ = (2u)^{1/(η_m+1)} - 1           if u < 0.5
            1 - (2(1-u))^{1/(η_m+1)}        if u ≥ 0.5
        child_i = parent_i + δ · (upper_i - lower_i)

    默认 p_m = 1/D.
    """
    if rng is None:
        rng = np.random.default_rng()
    D = len(individual)
    if mutation_rate is None:
        mutation_rate = 1.0 / D

    child = individual.copy()
    for i in range(D):
        if rng.uniform() >= mutation_rate:
            continue
        if lower is None or upper is None:
            continue
        u = rng.uniform()
        if u < 0.5:
            delta = (2 * u) ** (1.0 / (eta_m + 1)) - 1
        else:
            delta = 1 - (2 * (1 - u)) ** (1.0 / (eta_m + 1))
        child[i] += delta * (upper[i] - lower[i])
        child[i] = np.clip(child[i], lower[i], upper[i])

    return child


# ---------------------------------------------------------------------------
# NSGA-II 主循环
# ---------------------------------------------------------------------------
class NSGA2Optimizer:
    """
    NSGA-II 多目标优化器.

    参数:
        pop_size        : 种群大小
        n_generations   : 进化代数
        eta_c           : SBX 交叉分布指数
        eta_m           : 多项式变异分布指数
        crossover_prob  : 交叉概率
        lower, upper    : 变量边界
    """

    def __init__(self,
                 pop_size: int = 40,
                 n_generations: int = 15,
                 eta_c: float = 20.0,
                 eta_m: float = 20.0,
                 crossover_prob: float = 0.9,
                 lower: np.ndarray = None,
                 upper: np.ndarray = None,
                 seed: int = 42):
        self.pop_size = pop_size
        self.n_generations = n_generations
        self.eta_c = eta_c
        self.eta_m = eta_m
        self.crossover_prob = crossover_prob
        self.lower = lower
        self.upper = upper
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        # 记录
        self.history: Dict[str, list] = {
            'generation': [],
            'hypervolume': [],
            'spacing': [],
            'n_evals': [],
            'n_front': []
        }

    def optimize(self,
                 evaluator: Callable[[np.ndarray], np.ndarray]
                 ) -> Tuple[np.ndarray, np.ndarray]:
        """
        执行 NSGA-II 优化.

        Parameters
        ----------
        evaluator : callable — 输入 (D,) 设计变量,
                    返回 (M,) 目标向量

        Returns
        -------
        (pareto_set, pareto_front)
            pareto_set  : shape (K, D) — Pareto 最优设计
            pareto_front: shape (K, M) — Pareto 前沿
        """
        D = len(self.lower) if self.lower is not None else 1
        M = 3  # 默认三目标

        # 初始化种群
        pop = self._initialize_population(D)
        fitness = np.array([evaluator(ind) for ind in pop])
        M = fitness.shape[1]
        n_evals = self.pop_size

        for gen in range(self.n_generations):
            # 生成子代
            offspring = self._create_offspring(pop, fitness)
            # 评估子代
            off_fitness = np.array([evaluator(ind) for ind in offspring])
            n_evals += len(offspring)

            # 合并
            merged_pop = np.vstack([pop, offspring])
            merged_fitness = np.vstack([fitness, off_fitness])

            # 环境选择
            pop, fitness = self._environmental_selection(
                merged_pop, merged_fitness
            )

            # 记录
            pf = extract_pareto_front(fitness)
            self.history['generation'].append(gen)
            self.history['n_front'].append(len(pf))
            self.history['n_evals'].append(n_evals)

            if M == 2:
                ref = np.max(pf, axis=0) * 1.1
                hv = hypervolume_2d(pf[:, :2], ref[:2])
                self.history['hypervolume'].append(hv)
            else:
                self.history['hypervolume'].append(0.0)

            sp = spacing_metric(pf) if len(pf) > 1 else 0.0
            self.history['spacing'].append(sp)

            if (gen + 1) % 5 == 0 or gen == 0:
                print(f"  [NSGA-II] Gen {gen+1:3d}/{self.n_generations}: "
                      f"|PF|={len(pf):3d}, spacing={sp:.4f}, "
                      f"evals={n_evals}")

        # 最终 Pareto 前沿
        pf = extract_pareto_front(fitness)
        fronts = fast_non_dominated_sort(fitness)
        front1_idx = fronts[0]
        pareto_set = pop[front1_idx]
        pareto_front = fitness[front1_idx]

        return pareto_set, pareto_front

    def _initialize_population(self, D: int) -> np.ndarray:
        """初始化种群 — 在边界内均匀随机."""
        pop = np.zeros((self.pop_size, D))
        for i in range(self.pop_size):
            if self.lower is not None and self.upper is not None:
                pop[i] = self.rng.uniform(self.lower, self.upper)
            else:
                pop[i] = self.rng.uniform(0, 1, D)
        return pop

    def _create_offspring(self, pop: np.ndarray,
                          fitness: np.ndarray) -> np.ndarray:
        """通过选择、交叉、变异产生子代."""
        N, D = pop.shape
        fronts = fast_non_dominated_sort(fitness)
        cd = np.zeros(N)
        for front in fronts:
            if len(front) > 0:
                cd[front] = crowding_distance(fitness, front)

        # 锦标赛选择
        selected = tournament_select(pop, fitness, fronts, cd,
                                     N, tournament_size=2,
                                     seed=self.seed + N)

        offspring = np.zeros_like(selected)
        for i in range(0, N - 1, 2):
            if self.rng.uniform() < self.crossover_prob:
                c1, c2 = sbx_crossover(
                    selected[i], selected[min(i + 1, N - 1)],
                    eta=self.eta_c,
                    lower=self.lower, upper=self.upper,
                    rng=self.rng
                )
            else:
                c1 = selected[i].copy()
                c2 = selected[min(i + 1, N - 1)].copy()

            offspring[i] = polynomial_mutation(
                c1, eta_m=self.eta_m,
                lower=self.lower, upper=self.upper,
                rng=self.rng
            )
            if i + 1 < N:
                offspring[i + 1] = polynomial_mutation(
                    c2, eta_m=self.eta_m,
                    lower=self.lower, upper=self.upper,
                    rng=self.rng
                )

        return offspring

    def _environmental_selection(self, pop: np.ndarray,
                                 fitness: np.ndarray
                                 ) -> Tuple[np.ndarray, np.ndarray]:
        """
        环境选择 — 保留最优 N 个个体.

        按非支配层依次填充, 最后一层用拥挤距离截断.
        """
        N = self.pop_size
        fronts = fast_non_dominated_sort(fitness)

        new_pop = []
        new_fit = []
        for front in fronts:
            if len(new_pop) + len(front) <= N:
                for idx in front:
                    new_pop.append(pop[idx])
                    new_fit.append(fitness[idx])
            else:
                # 需要截断
                remaining = N - len(new_pop)
                cd = crowding_distance(fitness, front)
                sorted_by_cd = np.argsort(-cd)
                for k in range(remaining):
                    idx = front[sorted_by_cd[k]]
                    new_pop.append(pop[idx])
                    new_fit.append(fitness[idx])
                break

        return np.array(new_pop), np.array(new_fit)
