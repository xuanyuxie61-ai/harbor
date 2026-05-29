"""
layout_optimizer.py
风电场微观选址优化器

融合源项目：
- 246_cvt_1d_sampling: 质心Voronoi镶嵌（CVT）采样优化（风机空间布局优化）
- 793_nearest_neighbor: 最近邻搜索（风机间距约束校验）
- 1180_subset_sum_brute: 子集和暴力搜索（容量组合优化）
"""

import numpy as np
from typing import List, Tuple, Optional, Callable


class LayoutOptimizer:
    """
    风电场布局优化器。

    优化目标：
        max J(x, y) = Σ_i P_i(u_eff(x_i, y_i)) - λ · C_cable(x, y)

    约束条件：
        1. 最小间距：d_{ij} ≥ d_min = k·D  （通常 k = 3~10）
        2. 边界约束：(x_i, y_i) ∈ Ω
        3. 容量约束：Σ_i P_rated ≤ P_max_grid

    采用 CVT (Centroidal Voronoi Tessellation) 思想优化风机空间分布，
    源自 246_cvt_1d_sampling 的 Lloyd 算法。
    """

    def __init__(self, n_turbines: int = 10,
                 domain: Tuple[float, float, float, float] = (0.0, 5000.0, 0.0, 5000.0),
                 min_spacing: float = 500.0,
                 rated_power: float = 5.0,
                 max_grid_capacity: float = 100.0):
        """
        Parameters
        ----------
        n_turbines : int
            风机数量。
        domain : Tuple[float, float, float, float]
            (xmin, xmax, ymin, ymax)。
        min_spacing : float
            最小风机间距 [m]。
        rated_power : float
            单风机额定功率 [MW]。
        max_grid_capacity : float
            电网最大接纳容量 [MW]。
        """
        if n_turbines <= 0:
            raise ValueError("风机数量必须为正")
        if min_spacing <= 0:
            raise ValueError("最小间距必须为正")
        self.n_turbines = n_turbines
        self.xmin, self.xmax, self.ymin, self.ymax = domain
        self.d_min = min_spacing
        self.rated_power = rated_power
        self.max_grid_capacity = max_grid_capacity
        self.positions = np.zeros((n_turbines, 2))

    def initialize_random(self, seed: Optional[int] = None):
        """随机初始化风机位置。"""
        if seed is not None:
            rng = np.random.default_rng(seed)
        else:
            rng = np.random.default_rng()
        self.positions[:, 0] = rng.uniform(self.xmin, self.xmax, self.n_turbines)
        self.positions[:, 1] = rng.uniform(self.ymin, self.ymax, self.n_turbines)

    def initialize_grid(self):
        """按规则网格初始化。"""
        nx = int(np.ceil(np.sqrt(self.n_turbines * (self.xmax - self.xmin) / (self.ymax - self.ymin))))
        ny = int(np.ceil(self.n_turbines / nx))
        x = np.linspace(self.xmin + 100, self.xmax - 100, nx)
        y = np.linspace(self.ymin + 100, self.ymax - 100, ny)
        X, Y = np.meshgrid(x, y)
        pts = np.column_stack([X.ravel(), Y.ravel()])
        self.positions = pts[:self.n_turbines]

    def _nearest_neighbor_indices(self) -> List[Tuple[int, int]]:
        """
        查找每台风机的最近邻索引。

        源自 793_nearest_neighbor 的最近邻搜索思想。

        Returns
        -------
        List[Tuple[int, int]]
            每台风机的最近邻索引 (self_index, neighbor_index)。
        """
        n = self.n_turbines
        nn_pairs = []
        for i in range(n):
            min_dist = float('inf')
            min_j = -1
            for j in range(n):
                if i == j:
                    continue
                d = np.linalg.norm(self.positions[i] - self.positions[j])
                if d < min_dist:
                    min_dist = d
                    min_j = j
            nn_pairs.append((i, min_j))
        return nn_pairs

    def check_spacing_constraints(self) -> Tuple[bool, List[Tuple[int, int, float]]]:
        """
        检查所有风机对是否满足最小间距约束。

        Returns
        -------
        ok : bool
            是否全部满足。
        violations : List[Tuple[int, int, float]]
            违反约束的风机对及其距离。
        """
        violations = []
        n = self.n_turbines
        for i in range(n):
            for j in range(i + 1, n):
                d = np.linalg.norm(self.positions[i] - self.positions[j])
                if d < self.d_min:
                    violations.append((i, j, d))
        return len(violations) == 0, violations

    def repair_spacing(self, max_iterations: int = 1000) -> bool:
        """
        通过迭代 repulsion 修复间距约束。

        对于违反间距约束的风机对，沿连线方向推开：
            Δx_i = -α · (x_i - x_j) / d · (d_min - d)
            Δx_j = +α · (x_i - x_j) / d · (d_min - d)
        """
        alpha = 0.3
        for _ in range(max_iterations):
            ok, violations = self.check_spacing_constraints()
            if ok:
                return True
            for i, j, d in violations:
                if d < 1e-6:
                    # 随机扰动
                    theta = np.random.uniform(0, 2 * np.pi)
                    dx = self.min_spacing * 0.5 * np.cos(theta)
                    dy = self.min_spacing * 0.5 * np.sin(theta)
                    self.positions[i] += np.array([dx, dy])
                    self.positions[j] -= np.array([dx, dy])
                else:
                    direction = (self.positions[i] - self.positions[j]) / d
                    delta = alpha * (self.d_min - d) * direction
                    self.positions[i] += delta
                    self.positions[j] -= delta

            # 边界裁剪
            self.positions[:, 0] = np.clip(self.positions[:, 0], self.xmin, self.xmax)
            self.positions[:, 1] = np.clip(self.positions[:, 1], self.ymin, self.ymax)

        return False

    def cvt_optimize(self, objective_func: Callable[[np.ndarray], float],
                     n_samples: int = 5000,
                     n_iterations: int = 50,
                     step_size: float = 0.3) -> np.ndarray:
        """
        基于 CVT (Centroidal Voronoi Tessellation) 思想的布局优化。

        算法步骤（源自 246_cvt_1d_sampling 的 Lloyd 算法）：
            1. 在域内随机采样 n_samples 个点
            2. 对每个采样点，找到最近的风机（Voronoi 区域归属）
            3. 将风机位置移动到其 Voronoi 区域的质心
            4. 重复迭代

        同时考虑目标函数梯度进行修正：
            x_new = (1 - β)·x_centroid + β·(x + α·∇J)

        Parameters
        ----------
        objective_func : Callable[[np.ndarray], float]
            目标函数 J(positions)，返回标量（越大越好）。
        n_samples : int
            采样点数。
        n_iterations : int
            CVT 迭代次数。
        step_size : float
            梯度步长。

        Returns
        -------
        np.ndarray
            优化后的风机位置，形状 (n_turbines, 2)。
        """
        rng = np.random.default_rng(42)
        n = self.n_turbines

        for it in range(n_iterations):
            # 采样
            samples = np.column_stack([
                rng.uniform(self.xmin, self.xmax, n_samples),
                rng.uniform(self.ymin, self.ymax, n_samples)
            ])

            # 对每个采样点找最近风机（Voronoi 归属）
            assignments = np.zeros(n_samples, dtype=int)
            for s_idx, s in enumerate(samples):
                dists = np.linalg.norm(self.positions - s, axis=1)
                assignments[s_idx] = int(np.argmin(dists))

            # 计算每个 Voronoi 区域的质心
            new_positions = np.zeros_like(self.positions)
            counts = np.zeros(n)
            for s_idx, assign in enumerate(assignments):
                new_positions[assign] += samples[s_idx]
                counts[assign] += 1

            for i in range(n):
                if counts[i] > 0:
                    new_positions[i] /= counts[i]
                else:
                    # 没有采样点归属，随机移动
                    new_positions[i] = rng.uniform(
                        [self.xmin, self.ymin], [self.xmax, self.ymax]
                    )

            # 混合 CVT 质心与梯度上升
            beta = 0.7
            # 数值梯度
            current_obj = objective_func(self.positions)
            grad = np.zeros_like(self.positions)
            eps = 10.0
            for i in range(n):
                for dim in range(2):
                    pos_plus = self.positions.copy()
                    pos_plus[i, dim] += eps
                    # 边界处理
                    pos_plus[i, dim] = np.clip(pos_plus[i, dim],
                                               self.xmin if dim == 0 else self.ymin,
                                               self.xmax if dim == 0 else self.ymax)
                    obj_plus = objective_func(pos_plus)
                    grad[i, dim] = (obj_plus - current_obj) / eps

            # 更新位置
            self.positions = (1 - beta) * new_positions + \
                             beta * (self.positions + step_size * grad)

            # 边界裁剪
            self.positions[:, 0] = np.clip(self.positions[:, 0], self.xmin, self.xmax)
            self.positions[:, 1] = np.clip(self.positions[:, 1], self.ymin, self.ymax)

            # 修复间距
            self.repair_spacing(max_iterations=50)

        return self.positions

    def min_spacing(self) -> float:
        """计算当前布局的最小风机间距 [m]。"""
        dist = self.pairwise_distances()
        n = self.n_turbines
        if n <= 1:
            return float('inf')
        np.fill_diagonal(dist, float('inf'))
        return float(np.min(dist))

    def pairwise_distances(self) -> np.ndarray:
        """计算所有风机间的 pairwise 距离矩阵。"""
        n = self.n_turbines
        if n == 0:
            return np.zeros((0, 0))
        dist = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                d = np.linalg.norm(self.positions[i] - self.positions[j])
                dist[i, j] = d
                dist[j, i] = d
        return dist

    def capacity_subset_optimization(self, available_capacities: List[float],
                                      target_capacity: float) -> Tuple[bool, List[int]]:
        """
        风机容量组合优化。

        融合 1180_subset_sum_brute 的子集和暴力搜索思想。

        给定可用风机容量列表，选择子集使得总容量尽可能接近目标容量，
        同时不超过电网最大接纳容量。

        数学描述：
            求子集 S ⊆ {1, ..., n}，使得
                max |Σ_{i∈S} c_i - target_capacity|
                s.t. Σ_{i∈S} c_i ≤ max_grid_capacity

        Parameters
        ----------
        available_capacities : List[float]
            各风机可用容量 [MW]。
        target_capacity : float
            目标总容量 [MW]。

        Returns
        -------
        found : bool
            是否找到可行解。
        chosen_indices : List[int]
            选中的风机索引。
        """
        n = len(available_capacities)
        best_diff = float('inf')
        best_subset = []

        # 枚举所有 2^n 个子集（对于小 n 可行）
        max_search = min(n, 15)
        for mask in range(1 << max_search):
            subset = []
            total = 0.0
            for i in range(max_search):
                if mask & (1 << i):
                    subset.append(i)
                    total += available_capacities[i]

            if total > self.max_grid_capacity:
                continue

            diff = abs(total - target_capacity)
            if diff < best_diff:
                best_diff = diff
                best_subset = subset

        if best_subset:
            return True, best_subset
        return False, []

    def compute_aep(self, wind_speeds: np.ndarray, wind_directions: np.ndarray,
                    power_func: Callable[[float], float],
                    wake_func: Callable[[int, float, float], float]) -> float:
        """
        计算年发电量 AEP [MWh]。

        Parameters
        ----------
        wind_speeds : np.ndarray
            各工况风速 [m/s]。
        wind_directions : np.ndarray
            各工况风向 [度]。
        power_func : Callable[[float], float]
            功率曲线函数 P(u) [MW]。
        wake_func : Callable[[int, float, float], float]
            尾流有效风速函数 wake_func(i, u0, theta) -> u_eff。

        Returns
        -------
        float
            年发电量 [MWh]。
        """
        n_cases = len(wind_speeds)
        if n_cases == 0:
            return 0.0
        hours_per_year = 8760.0
        hours_per_case = hours_per_year / n_cases

        aep = 0.0
        for u0, theta in zip(wind_speeds, wind_directions):
            case_power = 0.0
            for i in range(self.n_turbines):
                u_eff = wake_func(i, u0, theta)
                case_power += power_func(u_eff)
            aep += case_power * hours_per_case

        return aep
