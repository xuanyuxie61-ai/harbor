"""
orbit_optimization.py

基于 box_behnken (Box-Behnken 实验设计) 与
backtrack_binary_rc (二进制回溯搜索) 核心算法，
实现小行星轨道参数的敏感性分析与最优轨道搜索。

科学背景：
近小行星轨道的稳定性高度依赖于初始轨道参数的选取。
本项目使用：
1. Box-Behnken 设计：在多参数空间中高效采样，评估各参数对轨道寿命的影响。
2. 二进制回溯搜索：在离散参数空间中寻找满足约束的最优轨道配置。

目标泛函（轨道品质函数）：
    J(a, e, i, ω, Ω) = w1 * T_lifetime(a,e,i) − w2 * Δv_maintenance(a,e,i)
                        − w3 * P_collision(a,e,i)

其中：
- T_lifetime: 轨道在 Lidov-Kozai 共振前维持的时长
- Δv_maintenance: 年均轨道维持速度增量
- P_collision: 与小行星表面碰撞的概率
"""

import numpy as np
from typing import Callable, List, Tuple, Optional


class OrbitOptimizationError(Exception):
    pass


def box_behnken_design(dim_num: int, ranges: np.ndarray) -> np.ndarray:
    """
    Box-Behnken 实验设计。
    基于 box_behnken.m 的核心算法。

    对于 dim_num 个因素，总试验数为:
        N = dim_num * 2^{dim_num-1} + 1

    每个因素取三水平：低、中、高。
    设计特点：
    - 第一个点为所有因素的中心水平
    - 对每个因素，固定该因素为中心水平，其余因素取高低两水平的所有组合

    参数:
        dim_num: 因素个数
        ranges: (dim_num, 2) 每因素的 [min, max]

    返回:
        design: (N, dim_num) 设计矩阵
    """
    if ranges.shape != (dim_num, 2):
        raise OrbitOptimizationError("ranges 形状必须为 (dim_num, 2)")
    if np.any(ranges[:, 1] <= ranges[:, 0]):
        raise OrbitOptimizationError("范围下限必须小于上限")

    # 计算试验数
    x_num = dim_num * (2 ** (dim_num - 1)) + 1
    design = np.zeros((x_num, dim_num))

    j = 0
    design[j, :] = (ranges[:, 0] + ranges[:, 1]) / 2.0

    for i in range(dim_num):
        # 第 i 个因素固定为中值
        j += 1
        design[j, :] = ranges[:, 0].copy()
        design[j, i] = (ranges[i, 0] + ranges[i, 1]) / 2.0

        # 二进制回溯生成其余因素的高低组合
        while True:
            last_low = -1
            for i2 in range(dim_num):
                if design[j, i2] == ranges[i2, 0]:
                    last_low = i2
            if last_low == -1:
                break
            j += 1
            design[j, :] = design[j - 1, :].copy()
            design[j, last_low] = ranges[last_low, 1]
            for i2 in range(last_low + 1, dim_num):
                if design[j, i2] == ranges[i2, 1]:
                    design[j, i2] = ranges[i2, 0]

    # 清理多余行（实际产生数可能小于理论值，因中心点去重）
    # 但实际算法应恰好产生 x_num 行
    actual = j + 1
    if actual < x_num:
        design = design[:actual, :]
    return design


def backtrack_binary_rc(
    n: int,
    reject: bool,
    n2: int,
    choice: np.ndarray
) -> Tuple[int, np.ndarray]:
    """
    二进制回溯的反向通信实现。
    基于 backbin_rc.m。

    参数:
        n: 解向量长度
        reject: 当前部分解是否应被拒绝
        n2: 当前部分解长度（首次调用设为 -1）
        choice: (n,) 当前解向量，元素为 0 或 1

    返回:
        n2_new, choice_new
    """
    choice = choice.copy()
    if n2 == -1:
        choice[:] = -1
        n2 = 0
        choice[n2] = 1
    elif n2 == n - 1 or reject:
        while n2 > 0:
            if choice[n2] == 1:
                choice[n2] = 0
                break
            choice[n2] = -1
            n2 -= 1
        if n2 == 0:
            if choice[0] == 1:
                choice[0] = 0
            else:
                choice[0] = -1
                n2 = -1
    else:
        n2 += 1
        choice[n2] = 1
    return n2, choice


def optimize_orbit_binary_backtrack(
    n_params: int,
    evaluate: Callable[[np.ndarray], float],
    max_evals: int = 10000
) -> Tuple[np.ndarray, float]:
    """
    使用二进制回溯搜索在离散 {0,1}^n 空间中寻找使 evaluate 最大化的配置。
    每个二进制位代表一个参数的两种选择（如低/高倾角）。

    参数:
        n_params: 二进制变量数
        evaluate: 评分函数，输入二进制向量，输出标量评分
        max_evals: 最大评估次数

    返回:
        best_choice: 最优二进制配置
        best_score: 最优评分
    """
    choice = np.full(n_params, -1, dtype=int)
    n2 = -1
    best_score = -np.inf
    best_choice = np.zeros(n_params, dtype=int)
    eval_count = 0

    while eval_count < max_evals:
        n2, choice = backtrack_binary_rc(n_params, False, n2, choice)
        if n2 == -1:
            break

        if n2 == n_params - 1:
            # 完整解
            score = evaluate(choice)
            eval_count += 1
            if score > best_score:
                best_score = score
                best_choice = choice.copy()
            n2, choice = backtrack_binary_rc(n_params, False, n2, choice)
            if n2 == -1:
                break

    return best_choice, best_score


class OrbitSensitivityAnalysis:
    """
    基于 Box-Behnken 设计的轨道参数敏感性分析器。
    """

    def __init__(
        self,
        param_names: List[str],
        param_ranges: np.ndarray,
        objective_func: Callable[[np.ndarray], float]
    ):
        """
        参数:
            param_names: 参数名称列表
            param_ranges: (dim, 2) 参数范围
            objective_func: 目标函数 J(params)
        """
        self.param_names = param_names
        self.param_ranges = param_ranges
        self.objective = objective_func
        self.dim = len(param_names)

    def run_analysis(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        执行 Box-Behnken 设计并计算每个试验点的目标函数值。

        返回:
            design: (N, dim) 设计矩阵
            responses: (N,) 响应值
            main_effects: (dim,) 主效应估计
        """
        design = box_behnken_design(self.dim, self.param_ranges)
        n = design.shape[0]
        responses = np.zeros(n)
        for i in range(n):
            responses[i] = self.objective(design[i])

        # 主效应估计：中心水平 vs 高低平均之差
        main_effects = np.zeros(self.dim)
        for d in range(self.dim):
            high_mask = np.abs(design[:, d] - self.param_ranges[d, 1]) < 1e-12
            low_mask = np.abs(design[:, d] - self.param_ranges[d, 0]) < 1e-12
            if np.sum(high_mask) > 0 and np.sum(low_mask) > 0:
                main_effects[d] = np.mean(responses[high_mask]) - np.mean(responses[low_mask])

        return design, responses, main_effects

    def find_optimal_from_design(self) -> Tuple[np.ndarray, float]:
        """
        从 Box-Behnken 设计结果中选择最优参数组合。
        """
        design, responses, _ = self.run_analysis()
        idx = np.argmax(responses)
        return design[idx], responses[idx]


def compute_orbit_quality_score(
    params: np.ndarray,
    simulate_lifetime_func: Callable[[np.ndarray], float],
    simulate_dv_func: Callable[[np.ndarray], float],
    simulate_collision_prob_func: Callable[[np.ndarray], float],
    weights: Optional[np.ndarray] = None
) -> float:
    """
    计算轨道品质综合评分。

    参数:
        params: (dim,) 轨道参数 [半长轴, 偏心率, 倾角, ...]
        simulate_lifetime_func: 返回轨道寿命 (s)
        simulate_dv_func: 返回年均 Δv (km/s)
        simulate_collision_prob_func: 返回碰撞概率
        weights: (3,) 权重 [w_lifetime, w_dv, w_collision]

    返回:
        score: 综合评分（越大越好）
    """
    if weights is None:
        weights = np.array([1.0, -0.5, -10.0])

    lifetime = simulate_lifetime_func(params)
    dv = simulate_dv_func(params)
    p_coll = simulate_collision_prob_func(params)

    # 边界处理
    lifetime = max(lifetime, 0.0)
    dv = max(dv, 0.0)
    p_coll = np.clip(p_coll, 0.0, 1.0)

    score = (
        weights[0] * np.log1p(lifetime / 86400.0) +
        weights[1] * dv * 1e3 +
        weights[2] * p_coll
    )
    return score
