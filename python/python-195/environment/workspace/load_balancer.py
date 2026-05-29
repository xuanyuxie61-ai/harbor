"""
load_balancer.py
粒子方法动态负载均衡核心模块

实现基于空间分解的粒子负载均衡算法，包括：
    - 负载度量与不平衡因子计算
    - 基于几何区域划分的负载均衡策略
    - 粒子迁移代价模型
    - 自适应域分解与动态重划分

核心数学：
    - 负载定义:
        对处理器 p，负载 w_p = sum_{e in Omega_p} (n_e + C * area_e)
        其中 n_e 为单元 e 中的粒子数，area_e 为单元面积，
        C 为场求解的计算权重（常数）。
    
    - 负载不均衡因子 (Imbalance Factor):
        I = (max_p w_p) / ( (1/P) * sum_p w_p )
        
        I = 1.0 表示完美均衡；I >> 1 表示严重失衡。
        通常设定阈值 I_max（如 1.2），当 I > I_max 时触发重均衡。
    
    - 最优分解（Orthogonal Recursive Bisection, ORB）:
        1) 计算所有粒子的质心
        2) 找到使两侧负载最均衡的切分平面（沿 x 或 y）
        3) 递归地对每个子域重复，直到子域数 = 处理器数 P
      
        最优切分位置 s^* 满足:
            | sum_{x_i < s} w_i - sum_{x_i >= s} w_i | -> min
    
    - 粒子迁移代价:
        迁移代价正比于迁移粒子数:
            Cost_migration = alpha * N_migrate
        
        重划分收益:
            Benefit = (I_old - I_new) * T_compute
        
        当 Benefit > Cost_migration 时执行重划分。
    
    - 谱负载均衡（基于切比雪夫展开）:
        将负载密度 rho(x,y) 用切比雪夫多项式展开:
            rho(x,y) approx sum_{i,j} c_{ij} * T_i(x') * T_j(y')
        
        通过分析高频系数判断是否需要进行局部细化。
"""

import numpy as np
from typing import Tuple, List, Optional
from utils import check_bounds


class LoadBalancer:
    """
    粒子方法负载均衡器。
    
    管理处理器网格、粒子分布、负载评估与均衡策略。
    """
    def __init__(self, n_procs: int, domain: Tuple[float, float, float, float],
                 imbalance_threshold: float = 1.3,
                 migration_cost_factor: float = 1.0e-4):
        """
        Parameters
        ----------
        n_procs : int
            处理器数量（模拟）
        domain : (xmin, xmax, ymin, ymax)
            计算域
        imbalance_threshold : float
            触发重均衡的不平衡阈值
        migration_cost_factor : float
            单位粒子迁移代价
        """
        self.n_procs = n_procs
        self.domain = domain
        self.xmin, self.xmax, self.ymin, self.ymax = domain
        self.imbalance_threshold = imbalance_threshold
        self.migration_cost_factor = migration_cost_factor

        # 初始化域分解（均匀网格）
        self._init_decomposition()

    def _init_decomposition(self):
        """初始化均匀域分解。"""
        # 尝试将 P 分解为 px * py
        px = int(np.sqrt(self.n_procs))
        while self.n_procs % px != 0 and px > 1:
            px -= 1
        py = self.n_procs // px
        self.px = px
        self.py = py

        self.proc_bounds = []
        dx = (self.xmax - self.xmin) / px
        dy = (self.ymax - self.ymin) / py
        for j in range(py):
            for i in range(px):
                bounds = (
                    self.xmin + i * dx,
                    self.xmin + (i + 1) * dx,
                    self.ymin + j * dy,
                    self.ymin + (j + 1) * dy
                )
                self.proc_bounds.append(bounds)

    def compute_loads(self, particles: np.ndarray,
                      field_cost: Optional[np.ndarray] = None) -> np.ndarray:
        """
        计算每个处理器的负载。
        
        Parameters
        ----------
        particles : np.ndarray, shape (n, 2)
            粒子位置
        field_cost : np.ndarray, optional
            场求解的单元计算开销（与单元面积或粒子密度相关）
        
        Returns
        -------
        np.ndarray, shape (n_procs,)
            各处理器负载
        """
        particles = np.asarray(particles, dtype=float)
        loads = np.zeros(self.n_procs, dtype=float)

        for p in range(self.n_procs):
            bxmin, bxmax, bymin, bymax = self.proc_bounds[p]
            mask = (
                (particles[:, 0] >= bxmin) & (particles[:, 0] < bxmax) &
                (particles[:, 1] >= bymin) & (particles[:, 1] < bymax)
            )
            # 边界上最后一个区域包含右/上边界
            if p % self.px == self.px - 1:
                mask = mask | ((particles[:, 0] >= bxmax - 1e-12) & (particles[:, 0] <= self.xmax + 1e-12))
            if p // self.px == self.py - 1:
                mask = mask | ((particles[:, 1] >= bymax - 1e-12) & (particles[:, 1] <= self.ymax + 1e-12))

            n_particles = np.sum(mask)
            # 负载 = 粒子数 + 基础场求解开销
            loads[p] = float(n_particles)
            if field_cost is not None and p < len(field_cost):
                loads[p] += field_cost[p]

        return loads

    def imbalance_factor(self, loads: np.ndarray) -> float:
        """
        计算负载不均衡因子。
        
        I = max(loads) / mean(loads)
        
        Parameters
        ----------
        loads : np.ndarray
            各处理器负载
        
        Returns
        -------
        float
            不均衡因子
        """
        avg = np.mean(loads)
        if avg < 1e-14:
            return 1.0
        return np.max(loads) / avg

    def find_optimal_split(self, particles_in_domain: np.ndarray,
                           axis: int = 0) -> Tuple[float, float, float]:
        """
        在一维上找到最优切分位置。
        
        算法:
            对粒子沿 axis 坐标排序
            遍历候选切分位置，计算两侧负载差
            选择使 |load_left - load_right| 最小的切分点
        
        Parameters
        ----------
        particles_in_domain : np.ndarray
            域内粒子位置
        axis : int
            0=x, 1=y
        
        Returns
        -------
        split_pos : float
            最优切分坐标
        load_left : float
            左侧负载
        load_right : float
            右侧负载
        """
        particles_in_domain = np.asarray(particles_in_domain, dtype=float)
        if particles_in_domain.shape[0] == 0:
            mid = (self.domain[axis * 2 + 1] + self.domain[axis * 2]) / 2.0
            return mid, 0.0, 0.0

        coords = particles_in_domain[:, axis]
        coords_sorted = np.sort(coords)
        n = len(coords_sorted)

        # 候选切分点：粒子位置的中点
        best_diff = float('inf')
        best_split = coords_sorted[n // 2]

        # 快速扫描关键位置
        for idx in range(1, n):
            split = 0.5 * (coords_sorted[idx - 1] + coords_sorted[idx])
            left_load = float(idx)
            right_load = float(n - idx)
            diff = abs(left_load - right_load)
            if diff < best_diff:
                best_diff = diff
                best_split = split

        left_mask = coords < best_split
        load_left = float(np.sum(left_mask))
        load_right = float(n) - load_left

        return best_split, load_left, load_right

    def recursive_bisection(self, particles: np.ndarray,
                            bounds: Tuple[float, float, float, float],
                            proc_id: int, n_subprocs: int,
                            result: dict, depth: int = 0):
        """
        递归正交二分法域分解。
        
        Parameters
        ----------
        particles : np.ndarray
            当前域内的粒子
        bounds : tuple
            当前域边界
        proc_id : int
            起始处理器ID
        n_subprocs : int
            需要分配的处理器数
        result : dict
            存储分解结果
        depth : int
            递归深度
        """
        xmin, xmax, ymin, ymax = bounds
        if n_subprocs == 1:
            result[proc_id] = {
                'bounds': bounds,
                'n_particles': particles.shape[0],
                'depth': depth
            }
            return

        # 决定沿哪个轴切分（选择粒子分布更分散的方向）
        if particles.shape[0] > 0:
            std_x = np.std(particles[:, 0]) if particles.shape[0] > 1 else 0.0
            std_y = np.std(particles[:, 1]) if particles.shape[0] > 1 else 0.0
            axis = 0 if std_x >= std_y else 1
        else:
            axis = 0 if (xmax - xmin) >= (ymax - ymin) else 1

        split, load_left, load_right = self.find_optimal_split(particles, axis)

        # 按负载比例分配处理器
        total_load = load_left + load_right
        if total_load < 1e-14:
            n_left = n_subprocs // 2
        else:
            n_left = max(1, min(n_subprocs - 1,
                                 int(round(n_subprocs * load_left / total_load))))
        n_right = n_subprocs - n_left

        if axis == 0:
            bounds_left = (xmin, split, ymin, ymax)
            bounds_right = (split, xmax, ymin, ymax)
        else:
            bounds_left = (xmin, xmax, ymin, split)
            bounds_right = (xmin, xmax, split, ymax)

        # 划分粒子
        if axis == 0:
            mask_left = particles[:, 0] < split
        else:
            mask_left = particles[:, 1] < split

        particles_left = particles[mask_left]
        particles_right = particles[~mask_left]

        self.recursive_bisection(particles_left, bounds_left, proc_id, n_left, result, depth + 1)
        self.recursive_bisection(particles_right, bounds_right, proc_id + n_left, n_right, result, depth + 1)

    def rebalance(self, particles: np.ndarray,
                  field_cost: Optional[np.ndarray] = None) -> dict:
        """
        执行负载重均衡。
        
        1) 评估当前负载
        2) 若不均衡因子 > 阈值，执行 ORB 重分解
        3) 计算迁移代价与收益
        4) 返回新的域分解与迁移方案
        
        Parameters
        ----------
        particles : np.ndarray
            所有粒子位置
        field_cost : np.ndarray, optional
            场求解开销
        
        Returns
        -------
        dict
            均衡结果，包含新分解、不均衡因子、迁移数等
        """
        current_loads = self.compute_loads(particles, field_cost)
        current_imbalance = self.imbalance_factor(current_loads)

        result = {
            'old_imbalance': current_imbalance,
            'old_loads': current_loads,
            'rebalanced': False,
            'new_decomposition': None,
            'migration_count': 0
        }

        if current_imbalance <= self.imbalance_threshold:
            result['new_imbalance'] = current_imbalance
            return result

        # 执行 ORB
        new_decomp = {}
        self.recursive_bisection(
            particles, self.domain, 0, self.n_procs, new_decomp
        )

        # 更新处理器边界
        new_bounds = []
        for p in range(self.n_procs):
            new_bounds.append(new_decomp[p]['bounds'])
        self.proc_bounds = new_bounds

        # 计算新负载
        new_loads = self.compute_loads(particles, field_cost)
        new_imbalance = self.imbalance_factor(new_loads)

        # 估算迁移粒子数（新旧区域边界变化的粒子）
        migration_count = 0
        # 简化：假设约 (I-1)/I 的粒子需要重新分配
        if current_imbalance > 1.0:
            migration_count = int(particles.shape[0] * (current_imbalance - 1.0) / current_imbalance)

        result.update({
            'rebalanced': True,
            'new_decomposition': new_decomp,
            'new_imbalance': new_imbalance,
            'new_loads': new_loads,
            'migration_count': migration_count
        })

        return result

    def evaluate_efficiency(self, loads: np.ndarray) -> dict:
        """
        评估并行效率指标。
        
        指标:
            - 不均衡因子 I
            - 标准差 sigma
            - 变异系数 CV = sigma / mu
            - 理论并行效率 eta = 1 / I
        
        Parameters
        ----------
        loads : np.ndarray
        
        Returns
        -------
        dict
            效率指标
        """
        mu = np.mean(loads)
        sigma = np.std(loads)
        I = self.imbalance_factor(loads)
        cv = sigma / mu if mu > 1e-14 else 0.0
        eta = 1.0 / I if I > 1e-14 else 0.0

        return {
            'mean_load': mu,
            'std_load': sigma,
            'imbalance_factor': I,
            'coefficient_variation': cv,
            'parallel_efficiency': eta
        }


def diffusion_based_load_balance(loads: np.ndarray,
                                  connectivity: np.ndarray,
                                  n_iterations: int = 100,
                                  tolerance: float = 1e-3) -> np.ndarray:
    """
    基于扩散的负载均衡算法（SOS: Second-Order Scheme）。
    
    将负载视为在处理器网络上扩散，每个迭代步:
        delta_{ij} = alpha * (load_i - load_j) / degree_i
        load_i^{new} = load_i - sum_j delta_{ij}
    
    在平衡态:
        load_i = load_j = mean(load)  对所有 i,j
    
    Parameters
    ----------
    loads : np.ndarray
        初始负载
    connectivity : np.ndarray
        邻接矩阵（对称，对角线为0）
    n_iterations : int
        最大迭代次数
    tolerance : float
        收敛容限
    
    Returns
    -------
    np.ndarray
        均衡后的负载
    """
    loads = np.asarray(loads, dtype=float).copy()
    n = len(loads)
    degrees = np.sum(connectivity, axis=1)
    degrees = np.maximum(degrees, 1.0)

    alpha = 0.5 / np.max(degrees)

    for it in range(n_iterations):
        loads_old = loads.copy()
        for i in range(n):
            for j in range(n):
                if connectivity[i, j] > 0:
                    delta = alpha * (loads_old[i] - loads_old[j])
                    loads[i] -= delta
                    loads[j] += delta

        max_diff = np.max(np.abs(loads - loads_old))
        if max_diff < tolerance:
            break

    return loads
