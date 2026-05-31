"""
mesh_generation.py
海洋区域的最优空间离散化

融合项目:
- 244_cvt_1d_lumping: 一维CVT/Voronoi分解 (Lloyd算法)
- 1341_triangulation_order1_display: 三角网格剖分

核心科学:
海洋混合参数化需要高效的空间离散化。
1D CVT (Centroidal Voronoi Tessellation) 用于垂向最优采样点分布，
2D Delaunay三角剖分用于水平面空间离散。

数学公式:
CVT能量泛函:
    E({z_i}) = Σ_j ∫_{V_j} ρ(z) · |z - z_j|² dz

其中 V_j 为Voronoi单元，ρ(z) 为密度权重函数。
最优节点为能量泛函的极小值点。
"""

import numpy as np


class CVT1D:
    """
    一维Centroidal Voronoi Tessellation
    
    使用Lloyd迭代算法求解最优节点分布。
    """
    
    def __init__(self, n_generators, z_min=-200.0, z_max=0.0,
                 density_type='chebyshev'):
        """
        初始化1D CVT
        
        参数:
            n_generators: 生成点数量
            z_min, z_max: 深度范围 [m]
            density_type: 密度函数类型
        """
        self.n = n_generators
        self.z_min = z_min
        self.z_max = z_max
        self.density_type = density_type
    
    def density_function(self, z):
        """
        密度权重函数 ρ(z)
        
        'uniform': ρ(z) = 1
        'chebyshev': ρ(z) = 1 / √(1 - s²)  其中 s ∈ [-1, 1]
        'thermocline': 在温跃层处密度权重增大
        'mixed_layer': 混合层密度权重增大
        
        参数:
            z: 深度 [m] (负值)
        
        返回:
            rho: 密度权重
        """
        z = np.asarray(z)
        
        if self.density_type == 'uniform':
            return np.ones_like(z)
        elif self.density_type == 'chebyshev':
            # 将 z ∈ [z_min, z_max] 映射到 s ∈ [-1, 1]
            s = 2.0 * (z - self.z_min) / (self.z_max - self.z_min) - 1.0
            s = np.clip(s, -0.999, 0.999)
            return 1.0 / np.sqrt(1.0 - s**2)
        elif self.density_type == 'thermocline':
            # 温跃层在 -100m 附近
            z_rel = z + 100.0
            return 1.0 + 2.0 * np.exp(-z_rel**2 / 400.0)
        elif self.density_type == 'mixed_layer':
            # 混合层在表层
            return 1.0 + 3.0 * np.exp(z / 20.0)
        else:
            return np.ones_like(z)
    
    def lloyd_iteration(self, n_samples=10000, max_iter=50, tol=1.0e-6):
        """
        Lloyd迭代算法求解CVT
        
        步骤:
        1. 生成随机样本点并按密度加权
        2. 构建Voronoi单元
        3. 计算每个单元的密度加权质心
        4. 移动生成点到质心
        5. 重复直到收敛
        
        参数:
            n_samples: 采样点数
            max_iter: 最大迭代次数
            tol: 收敛容差
        
        返回:
            generators: 最优节点位置 [m]
            energy_history: 能量迭代历史
        """
        # 初始化: Chebyshev零点
        generators = self._chebyshev_zeros()
        energy_history = []
        
        for it in range(max_iter):
            # 均匀采样
            samples = np.random.uniform(self.z_min, self.z_max, n_samples)
            
            # 密度加权
            weights = self.density_function(samples)
            
            # 构建Voronoi单元 (通过中点分割)
            midpoints = np.sort(generators)
            cell_sums = np.zeros(self.n)
            cell_weights = np.zeros(self.n)
            
            for sample, weight in zip(samples, weights):
                # 找到最近的生成点
                distances = np.abs(sample - generators)
                nearest = np.argmin(distances)
                cell_sums[nearest] += sample * weight
                cell_weights[nearest] += weight
            
            # 新质心
            new_generators = np.where(cell_weights > 1.0e-12,
                                      cell_sums / cell_weights,
                                      generators)
            
            # 边界处理
            new_generators = np.clip(new_generators, self.z_min, self.z_max)
            new_generators = np.sort(new_generators)
            
            # 计算能量
            energy = self._compute_energy(generators, samples, weights)
            energy_history.append(energy)
            
            # 收敛检查
            displacement = np.max(np.abs(new_generators - generators))
            if displacement < tol:
                break
            
            generators = new_generators
        
        return generators, energy_history
    
    def _chebyshev_zeros(self):
        """
        Chebyshev零点初始化
        
        z_i = (z_max + z_min)/2 + (z_max - z_min)/2 · cos(π(2i+1)/(2n))
        """
        i = np.arange(self.n)
        theta = np.pi * (2.0 * i + 1.0) / (2.0 * self.n)
        z = 0.5 * (self.z_max + self.z_min) + \
            0.5 * (self.z_max - self.z_min) * np.cos(theta)
        return z
    
    def _compute_energy(self, generators, samples, weights):
        """
        计算CVT能量泛函
        
        E = Σ_j ∫ ρ(z) · |z - z_j|² dz
        """
        energy = 0.0
        for sample, weight in zip(samples, weights):
            distances = (sample - generators)**2
            nearest = np.min(distances)
            energy += weight * nearest
        return energy / len(samples)


def delaunay_triangulation_2d(points):
    """
    2D Delaunay三角剖分 (简化实现)
    
    对于给定的平面点集，生成Delaunay三角形网格。
    Delaunay三角剖分最大化最小角，避免细长三角形。
    
    参数:
        points: (n, 2) 点坐标数组
    
    返回:
        triangles: (m, 3) 三角形顶点索引
    """
    points = np.asarray(points)
    n = len(points)
    
    if n < 3:
        return np.array([])
    
    # 简化实现: 对小型点集使用基本三角剖分
    # 按x坐标排序后做扫描线三角剖分
    
    idx = np.argsort(points[:, 0])
    sorted_points = points[idx]
    
    triangles = []
    
    # 简化的扇形三角剖分
    if n >= 3:
        # 找到凸包中心
        center = np.mean(sorted_points, axis=0)
        
        # 按角度排序
        angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
        angle_idx = np.argsort(angles)
        
        # 扇形三角化
        for i in range(n - 2):
            v0 = angle_idx[0]
            v1 = angle_idx[i + 1]
            v2 = angle_idx[i + 2]
            
            # 检查三角形方向 (逆时针)
            cross = (points[v1, 0] - points[v0, 0]) * (points[v2, 1] - points[v0, 1]) - \
                    (points[v1, 1] - points[v0, 1]) * (points[v2, 0] - points[v0, 0])
            
            if cross > 0:
                triangles.append([v0, v1, v2])
            else:
                triangles.append([v0, v2, v1])
    
    return np.array(triangles, dtype=int)


def triangulate_ocean_domain(x_range=(0, 10000), y_range=(0, 10000),
                              n_points=50):
    """
    对海洋水平区域进行三角剖分
    
    参数:
        x_range: x方向范围 [m]
        y_range: y方向范围 [m]
        n_points: 节点数量
    
    返回:
        nodes: 节点坐标 (n, 2)
        triangles: 三角形索引 (m, 3)
    """
    # 生成节点 (均匀+随机扰动)
    nx = int(np.sqrt(n_points))
    ny = nx
    
    x = np.linspace(x_range[0], x_range[1], nx)
    y = np.linspace(y_range[0], y_range[1], ny)
    
    nodes = []
    for i in range(nx):
        for j in range(ny):
            # 添加随机扰动
            dx = np.random.uniform(-100, 100)
            dy = np.random.uniform(-100, 100)
            nodes.append([x[i] + dx, y[j] + dy])
    
    nodes = np.array(nodes)
    
    # 边界处理: 确保边界点不动
    for i, node in enumerate(nodes):
        if node[0] < x_range[0] + 50:
            nodes[i, 0] = x_range[0]
        if node[0] > x_range[1] - 50:
            nodes[i, 0] = x_range[1]
        if node[1] < y_range[0] + 50:
            nodes[i, 1] = y_range[0]
        if node[1] > y_range[1] - 50:
            nodes[i, 1] = y_range[1]
    
    triangles = delaunay_triangulation_2d(nodes)
    
    return nodes, triangles
