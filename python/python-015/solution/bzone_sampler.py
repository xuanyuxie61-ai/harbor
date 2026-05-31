"""
bzone_sampler.py
布里渊区（Brillouin Zone）多方法k点采样

凝聚态物理背景：
在固体物理中，布里渊区是第一布里渊区的倒格矢空间区域。
对BZ的积分需要高效的采样方法：
    <O> = (1/V_BZ) \int_{BZ} O(k) d^3k

本模块融合三种核心采样算法：
1. 椭圆/椭球蒙特卡洛采样（基于331_ellipse_monte_carlo）
   - 用于Weyl节点附近的非均匀采样
   - 利用Cholesky分解在椭球区域生成均匀分布点
   
2. 重心Voronoi镶嵌CVT采样（基于259_cvt_square_nonuniform）
   - 用于非均匀密度的自适应k点分布
   - 密度加权：rho(k) ~ 1/|E_gap(k)| 或 ~ |Omega(k)|
   
3. 均匀随机采样（基于1373_uniform）
   - 用于标准均匀k点网格

数学基础：
对于变换X = U^{-1}*Y，其中A = U^T*U（Cholesky分解），
若Y在单位球内均匀分布，则X在椭球 X^T*A*X <= R^2 内均匀分布。
"""

import numpy as np
from typing import Tuple, Callable, Optional


def cholesky_factor(a: np.ndarray) -> np.ndarray:
    """
    Cholesky分解：A = U^T * U
    
    基于种子项目331_ellipse_monte_carlo中的r8po_fa思想，
    用于椭球采样中的坐标变换。
    
    Parameters
    ----------
    a : np.ndarray, shape (n, n)
        对称正定矩阵
    
    Returns
    -------
    u : np.ndarray, shape (n, n)
        上三角矩阵，满足 A = U^T @ U
    """
    a = np.asarray(a, dtype=float)
    n = a.shape[0]
    
    if a.shape[0] != a.shape[1]:
        raise ValueError("矩阵必须是方阵")
    
    # 检查对称性
    if np.max(np.abs(a - a.T)) > 1e-12:
        raise ValueError("矩阵必须是对称的")
    
    # 使用NumPy的Cholesky分解（返回下三角L，满足A = L @ L.T）
    try:
        L = np.linalg.cholesky(a)
    except np.linalg.LinAlgError:
        raise ValueError("矩阵必须是正定的")
    
    # 转换为U（上三角），U = L^T
    u = L.T
    return u


def uniform_in_sphere01_map(m: int, n: int) -> np.ndarray:
    """
    在单位球内生成n个均匀分布的m维点
    
    基于种子项目331_ellipse_monte_carlo中的uniform_in_sphere01_map。
    算法：先生成m维高斯分布点，再归一化到随机半径r = R^{1/m}。
    
    Parameters
    ----------
    m : int
        空间维度
    n : int
        点数
    
    Returns
    -------
    x : np.ndarray, shape (m, n)
    """
    x = np.random.randn(m, n)
    norms = np.linalg.norm(x, axis=0)
    
    # 避免除零
    norms = np.where(norms < 1e-15, 1.0, norms)
    
    # 半径因子：r^m ~ Uniform[0,1] => r = u^{1/m}
    u = np.random.rand(n)
    radius = u ** (1.0 / m)
    
    x = x / norms * radius
    return x


def ellipse_sample(n: int, a: np.ndarray, r: float) -> np.ndarray:
    """
    在椭球内生成n个均匀分布的点
    
    基于种子项目331_ellipse_monte_carlo中的ellipse_sample。
    
    椭球定义：X^T * A * X <= R^2，其中A为对称正定矩阵。
    
    算法：
    1. Cholesky分解：A = U^T * U
    2. 在单位球内生成点Y
    3. 缩放：Y = R * Y
    4. 解线性方程：U * X = Y
    
    Parameters
    ----------
    n : int
        点数
    a : np.ndarray, shape (m, m)
        定义椭球的正定矩阵
    r : float
        椭球半径
    
    Returns
    -------
    x : np.ndarray, shape (m, n)
    """
    m = a.shape[0]
    u = cholesky_factor(a)
    
    y = uniform_in_sphere01_map(m, n)
    y = r * y
    
    # 解 U * X = Y，即 X = U^{-1} * Y
    x = np.linalg.solve(u, y)
    return x


def ellipse_area(a: np.ndarray, r: float) -> float:
    """
    计算m维椭球的体积
    
    基于种子项目331_ellipse_monte_carlo中的ellipse_area1。
    
    二维椭球（椭圆）面积：V = pi * R^2 / sqrt(det(A))
    m维椭球体积：V = V_m * R^m / sqrt(det(A))
    其中V_m是m维单位球体积：
        V_m = pi^{m/2} / Gamma(m/2 + 1)
    
    Parameters
    ----------
    a : np.ndarray, shape (m, m)
    r : float
    
    Returns
    -------
    volume : float
    """
    m = a.shape[0]
    det_a = np.linalg.det(a)
    
    if det_a <= 0:
        raise ValueError("矩阵A的行列式必须为正")
    
    # m维单位球体积
    from scipy.special import gamma as scipy_gamma
    unit_sphere_vol = np.pi ** (m / 2.0) / scipy_gamma(m / 2.0 + 1.0)
    
    volume = unit_sphere_vol * (r ** m) / np.sqrt(det_a)
    return volume


def cvt_sampler_nonuniform(n_generators: int, sample_num: int, it_num: int,
                            density_func: Callable[[np.ndarray], np.ndarray],
                            bounds: np.ndarray = None) -> np.ndarray:
    """
    非均匀密度下的重心Voronoi镶嵌（CVT）采样
    
    基于种子项目259_cvt_square_nonuniform的核心算法。
    
    CVT迭代算法：
    1. 初始化生成器位置P
    2. 重复it_num次：
       a. 在区域内随机撒点S
       b. 计算每个点的密度d(S)
       c. 对每个生成器i，找到其Voronoi区域内的所有样本点
       d. 更新生成器位置为密度加权重心：
          P_i^{new} = sum_j d(S_j) * S_j / sum_j d(S_j)
    
    在凝聚态物理中，密度函数可取为：
        rho(k) = 1 / (|E_gap(k)| + epsilon)
    使得能隙较小（Weyl节点附近）的区域获得更高采样密度。
    
    Parameters
    ----------
    n_generators : int
        生成器数量（采样点数）
    sample_num : int
        每次迭代的样本点数
    it_num : int
        迭代次数
    density_func : callable
        密度函数，输入shape (N, m)，输出shape (N,)
    bounds : np.ndarray, shape (m, 2)
        采样区域边界，默认[-1,1]^m
    
    Returns
    -------
    generators : np.ndarray, shape (n_generators, m)
    """
    if bounds is None:
        bounds = np.array([[-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0]])
    
    m = bounds.shape[0]
    
    # 初始化生成器
    generators = np.zeros((n_generators, m))
    for dim in range(m):
        generators[:, dim] = np.random.uniform(bounds[dim, 0], bounds[dim, 1], n_generators)
    
    for it in range(it_num):
        # 生成样本点
        samples = np.zeros((sample_num, m))
        for dim in range(m):
            samples[:, dim] = np.random.uniform(bounds[dim, 0], bounds[dim, 1], sample_num)
        
        # 计算密度
        densities = density_func(samples)
        
        # 确保密度非负
        densities = np.maximum(densities, 1e-15)
        
        # 最近生成器搜索（简化的Voronoi划分）
        # 对每个样本点找到最近的生成器
        k = np.zeros(sample_num, dtype=int)
        for i in range(sample_num):
            dists = np.linalg.norm(generators - samples[i], axis=1)
            k[i] = np.argmin(dists)
        
        # 更新生成器为密度加权重心
        new_generators = np.zeros_like(generators)
        for i in range(n_generators):
            mask = (k == i)
            if np.sum(mask) == 0:
                new_generators[i] = generators[i]
            else:
                weights = densities[mask]
                weighted_sum = np.sum(weights[:, None] * samples[mask], axis=0)
                total_weight = np.sum(weights)
                new_generators[i] = weighted_sum / total_weight
        
        generators = new_generators
    
    # 边界裁剪
    for dim in range(m):
        generators[:, dim] = np.clip(generators[:, dim], bounds[dim, 0], bounds[dim, 1])
    
    return generators


def uniform_kpoint_grid(bounds: np.ndarray, grid_size: int) -> np.ndarray:
    """
    均匀k点网格采样
    
    基于种子项目1373_uniform中的r8mat_uniform_ab思想，
    生成规则网格点。
    
    Parameters
    ----------
    bounds : np.ndarray, shape (m, 2)
    grid_size : int
        每个维度的网格点数
    
    Returns
    -------
    kpoints : np.ndarray, shape (grid_size^m, m)
    """
    m = bounds.shape[0]
    
    axes = []
    for dim in range(m):
        axes.append(np.linspace(bounds[dim, 0], bounds[dim, 1], grid_size))
    
    mesh = np.meshgrid(*axes, indexing='ij')
    kpoints = np.stack([m.ravel() for m in mesh], axis=-1)
    return kpoints


def adaptive_weyl_node_sampler(n_points: int, weyl_nodes: np.ndarray,
                                node_radius: float = 0.5,
                                bz_bounds: np.ndarray = None) -> np.ndarray:
    """
    针对Weyl节点的自适应采样器
    
    策略：
    1. 在Weyl节点附近使用椭球蒙特卡洛密集采样
    2. 在远离节点处使用均匀稀疏采样
    
    基于种子项目331_ellipse_monte_carlo和1373_uniform的融合。
    
    Parameters
    ----------
    n_points : int
        总采样点数
    weyl_nodes : np.ndarray, shape (N_nodes, 3)
        Weyl节点位置
    node_radius : float
        每个Weyl节点的采样半径
    bz_bounds : np.ndarray
    
    Returns
    -------
    samples : np.ndarray, shape (n_points, 3)
    """
    if bz_bounds is None:
        bz_bounds = np.array([[-np.pi, np.pi], [-np.pi, np.pi], [-np.pi, np.pi]])
    
    n_nodes = weyl_nodes.shape[0] if weyl_nodes.ndim > 1 else 1
    
    # 分配采样点：70%在节点附近，30%全局均匀
    n_near = int(0.7 * n_points)
    n_global = n_points - n_near
    
    samples_near = []
    if n_nodes > 0:
        points_per_node = n_near // n_nodes
        for i in range(n_nodes):
            node = weyl_nodes[i] if weyl_nodes.ndim > 1 else weyl_nodes
            # 在节点附近使用椭球采样
            a = np.eye(3)  # 球形
            pts = ellipse_sample(points_per_node, a, node_radius).T
            pts += node  # 平移到节点位置
            # 边界裁剪
            for dim in range(3):
                pts[:, dim] = np.clip(pts[:, dim], bz_bounds[dim, 0], bz_bounds[dim, 1])
            samples_near.append(pts)
    
    # 全局均匀采样
    global_samples = np.zeros((n_global, 3))
    for dim in range(3):
        global_samples[:, dim] = np.random.uniform(
            bz_bounds[dim, 0], bz_bounds[dim, 1], n_global
        )
    
    if len(samples_near) > 0:
        all_samples = np.vstack(samples_near + [global_samples])
    else:
        all_samples = global_samples
    
    # 如果点数超出，随机选择
    if all_samples.shape[0] > n_points:
        idx = np.random.choice(all_samples.shape[0], n_points, replace=False)
        all_samples = all_samples[idx]
    
    return all_samples
