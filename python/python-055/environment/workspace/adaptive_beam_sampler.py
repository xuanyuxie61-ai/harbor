"""
adaptive_beam_sampler.py
基于种子项目 264_cvtp（周期边界 CVT 迭代）、
293_disk_grid（圆盘网格生成）与 626_knapsack_random（随机子集选择），
构建多波束声纳自适应采样与最优波束子集选择模块。

科学背景：在海底地形测绘中，测线间距与波束开角直接影响
空间分辨率与作业效率。利用重心 Voronoi 镶嵌（Centroidal Voronoi
Tessellation, CVT）可在给定区域内生成均匀且最优的采样点分布，
使每个采样点到其最近生成元的二阶矩最小化：

    F({z_i}) = Σ_i ∫_{V_i} ρ(x) · ||x - z_i||² dA

其中 V_i 为 Voronoi 单元，z_i 为生成元，ρ(x) 为密度函数。

对于多波束声纳，将每个波束 footprint 中心视为生成元，
CVT 优化等价于最小化覆盖不均匀度。

此外，受作业时间/能耗约束，需从全部候选波束中选择最优子集。
这被建模为带权 0-1 背包问题：

    max Σ_{i∈S} v_i,  s.t. Σ_{i∈S} w_i ≤ W

其中 v_i 为波束 i 的信息增益（覆盖面积 × 地形复杂度），
w_i 为能耗/时间成本，W 为总预算。
"""

import numpy as np


class DiskGridGenerator:
    """
    圆盘内网格生成器（源自 disk_grid.m）。
    """

    @staticmethod
    def disk_grid_count(n: int) -> int:
        """
        计算将半径分为 n 个子区间时圆盘内的网格点数。
        """
        if n < 0:
            return 0
        ng = 1
        for j in range(1, n + 1):
            i = 0
            while True:
                i += 1
                x = 2.0 * i / (2.0 * n + 1.0)
                y = 2.0 * j / (2.0 * n + 1.0)
                if x ** 2 + y ** 2 > 1.0:
                    break
                if j == 0:
                    ng += 2
                else:
                    ng += 4
        return ng

    @staticmethod
    def disk_grid(n: int, r: float = 1.0, c: np.ndarray = None) -> np.ndarray:
        """
        在圆盘内生成规则网格点。

        参数:
            n: 半径方向子区间数
            r: 圆盘半径
            c: 圆心坐标 (2,)
        返回:
            网格点数组，形状 (ng, 2)
        """
        if c is None:
            c = np.zeros(2)
        c = np.asarray(c, dtype=np.float64)

        points = []
        # 中心点
        points.append([c[0], c[1]])

        for j in range(1, n + 1):
            y = c[1] + r * 2.0 * j / (2.0 * n + 1.0)
            # 上/下对称
            y_vals = [y]
            if j > 0:
                y_vals.append(2.0 * c[1] - y)

            for y_val in y_vals:
                # x=0 列
                points.append([c[0], y_val])
                i = 0
                while True:
                    i += 1
                    x = c[0] + r * 2.0 * i / (2.0 * n + 1.0)
                    if (x - c[0]) ** 2 + (y_val - c[1]) ** 2 > r ** 2 + 1e-12:
                        break
                    # 四象限对称
                    points.append([x, y_val])
                    points.append([2.0 * c[0] - x, y_val])

        return np.array(points, dtype=np.float64)


class CVTBeamOptimizer:
    """
    基于 CVT 的波束位置优化器（源自 cvtp_iteration.m）。
    """

    def __init__(self, bounds: tuple, density_func=None):
        """
        参数:
            bounds: ((xmin, xmax), (ymin, ymax))
            density_func: 密度函数 ρ(x,y)，None 表示均匀密度
        """
        self.bounds = bounds
        self.density_func = density_func

    def _region_sampler(self, n_samples: int) -> np.ndarray:
        """在区域内均匀采样。"""
        x_min, x_max = self.bounds[0]
        y_min, y_max = self.bounds[1]
        pts = np.random.rand(n_samples, 2)
        pts[:, 0] = x_min + (x_max - x_min) * pts[:, 0]
        pts[:, 1] = y_min + (y_max - y_min) * pts[:, 1]
        return pts

    def _find_closest(self, generators: np.ndarray, point: np.ndarray) -> int:
        """找到最近的生成元索引。"""
        diff = generators - point
        dists = np.sum(diff ** 2, axis=1)
        return int(np.argmin(dists))

    def optimize(
        self,
        n_generators: int,
        n_samples: int = 5000,
        n_iterations: int = 20,
        init_points: np.ndarray = None
    ) -> np.ndarray:
        """
        执行 Lloyd 迭代优化 CVT。

        参数:
            n_generators: 生成元数量（波束数）
            n_samples:    每轮迭代的蒙特卡洛采样数
            n_iterations: Lloyd 迭代次数
            init_points:  初始生成元位置（可选）
        返回:
            优化后的生成元位置，形状 (n_generators, 2)
        """
        if init_points is not None:
            generators = np.asarray(init_points, dtype=np.float64).copy()
        else:
            generators = self._region_sampler(n_generators)

        for it in range(n_iterations):
            samples = self._region_sampler(n_samples)
            new_generators = np.zeros_like(generators)
            counts = np.zeros(n_generators)

            for s in samples:
                idx = self._find_closest(generators, s)
                new_generators[idx] += s
                counts[idx] += 1.0

            for j in range(n_generators):
                if counts[j] > 0:
                    new_generators[j] /= counts[j]
                else:
                    # 空单元回退到随机位置
                    new_generators[j] = self._region_sampler(1)[0]

            change = np.linalg.norm(new_generators - generators, 'fro')
            generators = new_generators
            if change < 1e-6:
                break

        # 裁剪到边界
        x_min, x_max = self.bounds[0]
        y_min, y_max = self.bounds[1]
        generators[:, 0] = np.clip(generators[:, 0], x_min, x_max)
        generators[:, 1] = np.clip(generators[:, 1], y_min, y_max)

        return generators


class KnapsackBeamSelector:
    """
    基于 0-1 背包思想的波束子集选择器（源自 knapsack_random.m）。
    """

    @staticmethod
    def random_subset(n: int, seed: int = None) -> np.ndarray:
        """
        生成 n 个元素的随机子集（二进制向量表示）。

        参数:
            n: 元素总数
            seed: 随机种子
        返回:
            二进制数组，1 表示选中
        """
        rng = np.random.default_rng(seed)
        return rng.integers(0, 2, size=n)

    @staticmethod
    def greedy_select(values: np.ndarray, weights: np.ndarray, capacity: float) -> np.ndarray:
        """
        贪心算法求解背包问题：按价值密度排序选择。

        参数:
            values:  各波束信息增益
            weights: 各波束成本
            capacity: 总预算
        返回:
            二进制选中数组
        """
        n = len(values)
        density = values / (weights + 1e-15)
        order = np.argsort(-density)

        selected = np.zeros(n, dtype=int)
        total_weight = 0.0
        for idx in order:
            if total_weight + weights[idx] <= capacity:
                selected[idx] = 1
                total_weight += weights[idx]
        return selected

    @staticmethod
    def random_search_select(
        values: np.ndarray,
        weights: np.ndarray,
        capacity: float,
        n_trials: int = 200,
        seed: int = 55
    ) -> np.ndarray:
        """
        随机搜索求解背包问题：生成多个随机子集，选取满足约束的最优解。

        参数:
            values:  各波束信息增益
            weights: 各波束成本
            capacity: 总预算
            n_trials: 随机试验次数
            seed: 随机种子
        返回:
            最优二进制选中数组
        """
        rng = np.random.default_rng(seed)
        n = len(values)
        best_value = -1.0
        best_subset = np.zeros(n, dtype=int)

        for _ in range(n_trials):
            subset = rng.integers(0, 2, size=n)
            total_weight = np.dot(subset, weights)
            if total_weight <= capacity:
                total_value = np.dot(subset, values)
                if total_value > best_value:
                    best_value = total_value
                    best_subset = subset.copy()

        return best_subset

    @staticmethod
    def optimize_beam_subset(
        beam_info_gains: np.ndarray,
        beam_costs: np.ndarray,
        time_budget: float,
        method: str = "greedy"
    ) -> dict:
        """
        优化波束子集选择。

        参数:
            beam_info_gains: 各候选波束的信息增益数组
            beam_costs:      各候选波束的观测时间/能耗成本
            time_budget:     总时间预算
            method:          "greedy" 或 "random_search"
        返回:
            包含选中掩码、总增益、总成本的字典
        """
        if method == "greedy":
            selected = KnapsackBeamSelector.greedy_select(
                beam_info_gains, beam_costs, time_budget
            )
        elif method == "random_search":
            selected = KnapsackBeamSelector.random_search_select(
                beam_info_gains, beam_costs, time_budget
            )
        else:
            raise ValueError("method 必须是 'greedy' 或 'random_search'")

        total_gain = float(np.dot(selected, beam_info_gains))
        total_cost = float(np.dot(selected, beam_costs))

        return {
            'selected': selected,
            'total_gain': total_gain,
            'total_cost': total_cost,
            'n_selected': int(np.sum(selected)),
        }


class AdaptiveSamplingPlanner:
    """
    自适应采样规划器：综合 CVT 与背包优化，
    在有限资源下实现海底地形的最优覆盖采样。
    """

    def __init__(self, survey_area: tuple):
        """
        参数:
            survey_area: ((xmin, xmax), (ymin, ymax))
        """
        self.survey_area = survey_area
        self.cvt_optimizer = CVTBeamOptimizer(survey_area)

    def plan_survey(
        self,
        n_beams: int = 64,
        time_budget: float = 3600.0,
        n_cvt_iter: int = 15
    ) -> dict:
        """
        规划完整测量方案。

        返回:
            包含最优波束位置、选中子集、覆盖效率等的字典
        """
        # 1. 用 CVT 生成候选波束位置
        beam_positions = self.cvt_optimizer.optimize(
            n_generators=n_beams,
            n_iterations=n_cvt_iter
        )

        # 2. 计算每个波束的信息增益（基于覆盖面积与位置分散度）
        info_gains = np.zeros(n_beams)
        for i in range(n_beams):
            # 简化的信息增益：到最近邻的距离平方（避免冗余）
            dists = np.linalg.norm(beam_positions - beam_positions[i], axis=1)
            dists = dists[dists > 1e-12]
            if len(dists) > 0:
                info_gains[i] = np.min(dists) ** 2
            else:
                info_gains[i] = 1.0

        # 3. 计算每个波束的成本（距离中心越远成本越高，模拟传播损耗）
        center = np.array([
            (self.survey_area[0][0] + self.survey_area[0][1]) / 2.0,
            (self.survey_area[1][0] + self.survey_area[1][1]) / 2.0,
        ])
        dists_to_center = np.linalg.norm(beam_positions - center, axis=1)
        beam_costs = 10.0 + 0.5 * dists_to_center  # 基础成本 + 距离成本

        # 4. 背包优化选择子集
        result = KnapsackBeamSelector.optimize_beam_subset(
            info_gains, beam_costs, time_budget, method="greedy"
        )

        selected_positions = beam_positions[result['selected'] == 1]

        return {
            'all_positions': beam_positions,
            'selected_positions': selected_positions,
            'selected_mask': result['selected'],
            'total_gain': result['total_gain'],
            'total_cost': result['total_cost'],
            'n_selected': result['n_selected'],
            'coverage_efficiency': result['total_gain'] / (result['total_cost'] + 1e-15),
        }
