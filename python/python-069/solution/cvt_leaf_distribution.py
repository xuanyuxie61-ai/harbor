"""
三维叶片空间分布优化模块：基于 cvt_3d_lumping 与 cvt_4_movie 思想，
使用 Lloyd 算法优化叶片在三维冠层中的空间分布，
以最大化光截获效率。

核心数学：
  Centroidal Voronoi Tessellation (CVT) 能量泛函：
      F(P) = sum_{i=1}^n integral_{V_i} rho(x) ||x - p_i||^2 dx
  其中 V_i 为 p_i 的 Voronoi 区域，rho(x) 为光强密度函数。

  Lloyd 迭代：
      p_i^{new} = ( integral_{V_i} rho(x) x dx ) / ( integral_{V_i} rho(x) dx )

  密度函数（光强驱动）：
      rho(x,y,z) = I(x,y,z)^gamma, gamma > 0
"""
import numpy as np
from scipy.spatial import Delaunay, cKDTree


def cvt_3d_lloyd(n_generators, it_num, s_num, rho_func, domain=((-1, 1), (-1, 1), (-1, 1))):
    """
    3D Lloyd 算法优化生成器位置。
    n_generators: 生成器数量（叶片簇中心）
    it_num: 迭代次数
    s_num: 每维采样点数
    rho_func: 密度函数 rho(x,y,z)
    domain: 三维域边界
    返回: generators (n,3), energy_history, motion_history
    """
    # 初始化生成器
    g = np.zeros((n_generators, 3), dtype=float)
    for dim in range(3):
        g[:, dim] = np.random.uniform(domain[dim][0], domain[dim][1], n_generators)

    # 采样网格
    lin = [np.linspace(domain[d][0] + 1e-6, domain[d][1] - 1e-6, s_num) for d in range(3)]
    sx, sy, sz = np.meshgrid(lin[0], lin[1], lin[2], indexing='ij')
    s_points = np.column_stack((sx.ravel(), sy.ravel(), sz.ravel()))

    # 计算密度
    rho_vals = rho_func(s_points[:, 0], s_points[:, 1], s_points[:, 2])
    rho_vals = np.clip(rho_vals, 1e-6, 1e6)

    energy_history = []
    motion_history = []

    for it in range(it_num):
        # 查找最近生成器
        tree = cKDTree(g)
        _, nearest = tree.query(s_points)

        # 计算每个 Voronoi 区域的质量和质心
        m = np.zeros(n_generators, dtype=float)
        cx = np.zeros(n_generators, dtype=float)
        cy = np.zeros(n_generators, dtype=float)
        cz = np.zeros(n_generators, dtype=float)

        np.add.at(m, nearest, rho_vals)
        np.add.at(cx, nearest, rho_vals * s_points[:, 0])
        np.add.at(cy, nearest, rho_vals * s_points[:, 1])
        np.add.at(cz, nearest, rho_vals * s_points[:, 2])

        # 避免除以零
        m_safe = np.maximum(m, 1e-14)
        g_new = np.column_stack((cx / m_safe, cy / m_safe, cz / m_safe))

        # 计算能量
        dist2 = np.sum((s_points - g[nearest, :]) ** 2, axis=1)
        energy = np.sum(rho_vals * dist2) / s_num
        energy_history.append(energy)

        # 计算平均运动
        motion = np.mean(np.sum((g_new - g) ** 2, axis=1))
        motion_history.append(motion)

        g = g_new

    return g, energy_history, motion_history


def canopy_cvt_optimization(canopy_height, crown_radius, n_clusters=50,
                            it_num=20, s_num=20, lai_max=4.5):
    """
    对森林冠层进行 CVT 优化，得到叶片簇的最优空间分布。
    返回: generators (n_clusters, 3)
    """
    def rho_func(x, y, z):
        # 光强密度：越靠近冠层顶部、越靠近中心密度越高
        z = np.asarray(z)
        r = np.sqrt(np.asarray(x) ** 2 + np.asarray(y) ** 2)
        decay_r = np.maximum(0.0, 1.0 - (r / crown_radius) ** 2)
        decay_z = np.clip(z / canopy_height, 0.0, 1.0)
        return decay_r * decay_z + 0.1

    domain = ((-crown_radius, crown_radius),
              (-crown_radius, crown_radius),
              (0.0, canopy_height))

    g, e_hist, m_hist = cvt_3d_lloyd(n_clusters, it_num, s_num, rho_func, domain)
    return g, e_hist, m_hist
