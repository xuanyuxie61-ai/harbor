"""
cvt_adaptive_grid.py
自适应 Centroidal Voronoi Tessellation (CVT) 网格生成

基于 243_cvt_1d_lloyd 项目重构:
  Lloyd 算法用于最优采样点分布
  
物理应用:
  在相空间积分中，积分节点应集中在矩阵元 |M|^2 较大的区域。
  CVT 提供了一种密度自适应的节点分布:
    - 在 |M|^2 大的区域 (近壳 Z 玻色子) 加密节点
    - 在 |M|^2 小的区域 (远壳区域) 稀疏节点
    
  密度函数: rho(x) ~ |M(x)|^2 + epsilon
  最优节点位置最小化能量:
    E = sum_j int_{V_j} rho(x) * |x - g_j|^2 dx
"""
import numpy as np
from constants import TINY

# ============================================================
# 1. Lloyd 算法 1D (映射 243_cvt_1d_lloyd)
# ============================================================
def cvt_1d_lloyd(n_generators, density_func, domain=(0.0, 1.0),
                 max_iter=200, tol=1.0e-10, init_mode="random"):
    """
    一维密度自适应 Lloyd 算法
    
    迭代步骤:
      1. 将生成点排序: g_1 <= g_2 <= ... <= g_n
      2. 计算 Voronoi 边界: b_j = (g_j + g_{j+1}) / 2
      3. 将每个生成点移动到其 Voronoi 单元的密度加权质心:
           g_j^{new} = int_{b_{j-1}}^{b_j} x * rho(x) dx / int rho(x) dx
      4. 若平均位移 < tol，收敛
    
    参数:
        n_generators: 生成点数
        density_func: 密度函数 rho(x)
        domain: (a, b)
        max_iter: 最大迭代次数
        tol: 收敛容差
        init_mode: "random" 或 "uniform"
    返回:
        generators: 最优节点位置
        energy: 最终能量
        converged: 是否收敛
    """
    a, b = domain
    
    # 初始化
    if init_mode == "uniform":
        generators = np.linspace(a, b, n_generators)
    else:
        generators = np.sort(np.random.uniform(a, b, n_generators))
    
    # 数值积分辅助函数 (使用 Simpson 规则)
    def integrate(f, lo, hi, n=100):
        if hi <= lo:
            return 0.0, 0.0
        h = (hi - lo) / n
        # 积分 f(x)
        vals = np.array([f(lo + i * h) for i in range(n + 1)])
        total = vals[0] + vals[-1]
        total += 4.0 * np.sum(vals[1:-1:2])
        total += 2.0 * np.sum(vals[2:-1:2])
        return total * h / 3.0, np.sum(vals) * h  # Simpson 和矩形近似
    
    def integrate_xf(f, lo, hi, n=100):
        if hi <= lo:
            return 0.0
        h = (hi - lo) / n
        total = 0.0
        for i in range(n + 1):
            x = lo + i * h
            fx = f(x)
            if i == 0 or i == n:
                w = 1.0
            elif i % 2 == 1:
                w = 4.0
            else:
                w = 2.0
            total += w * x * fx
        return total * h / 3.0
    
    energy_history = []
    for iteration in range(max_iter):
        generators = np.sort(generators)
        
        # 计算 Voronoi 边界
        boundaries = np.zeros(n_generators + 1)
        boundaries[0] = a
        boundaries[-1] = b
        for j in range(1, n_generators):
            boundaries[j] = (generators[j - 1] + generators[j]) / 2.0
        
        # 移动生成点到加权质心
        new_generators = np.zeros(n_generators)
        for j in range(n_generators):
            lo = boundaries[j]
            hi = boundaries[j + 1]
            mass, _ = integrate(density_func, lo, hi, n=80)
            xmass = integrate_xf(density_func, lo, hi, n=80)
            if mass > TINY:
                new_generators[j] = xmass / mass
            else:
                new_generators[j] = (lo + hi) / 2.0
        
        # 边界约束
        new_generators = np.clip(new_generators, a, b)
        
        # 计算平均位移
        avg_motion = np.mean(np.abs(new_generators - generators))
        
        # 计算能量
        energy = 0.0
        for j in range(n_generators):
            lo = boundaries[j]
            hi = boundaries[j + 1]
            n_sample = 50
            h = (hi - lo) / n_sample
            for k in range(n_sample):
                x = lo + (k + 0.5) * h
                energy += density_func(x) * (x - generators[j]) ** 2 * h
        energy_history.append(energy)
        
        generators = new_generators
        
        if avg_motion < tol:
            return generators, energy, True, iteration + 1
    
    return generators, energy_history[-1], False, max_iter


# ============================================================
# 2. 多维乘积型 CVT (逐维度 Lloyd)
# ============================================================
def cvt_nd_product(n_per_dim, density_funcs, domains, max_iter=100, tol=1.0e-8):
    """
    多维乘积型 CVT 网格: 每个维度独立运行 Lloyd 算法
    
    参数:
        n_per_dim: 每个维度的节点数列表
        density_funcs: 每个维度的密度函数列表
        domains: 每个维度的区间列表 [(a1,b1), (a2,b2), ...]
    返回:
        grid_points: 多维网格点列表
        grid_weights: 对应权重 (基于密度)
    """
    dim = len(n_per_dim)
    nodes_per_dim = []
    
    for d in range(dim):
        nodes, _, _, _ = cvt_1d_lloyd(
            n_per_dim[d], density_funcs[d], domains[d],
            max_iter=max_iter, tol=tol
        )
        nodes_per_dim.append(nodes)
    
    # 构造张量积网格
    from itertools import product
    grid_points = []
    grid_weights = []
    
    for coords in product(*nodes_per_dim):
        pt = np.array(coords)
        # 权重为各维度密度的乘积
        w = 1.0
        for d in range(dim):
            rho_val = density_funcs[d](coords[d])
            w *= max(rho_val, TINY)
        grid_points.append(pt)
        grid_weights.append(w)
    
    grid_points = np.array(grid_points)
    grid_weights = np.array(grid_weights)
    
    # 归一化权重
    if np.sum(grid_weights) > TINY:
        grid_weights = grid_weights / np.sum(grid_weights)
    
    return grid_points, grid_weights


# ============================================================
# 3. 物理自适应密度函数
# ============================================================
def make_breit_wigner_density(m0, gamma, domain):
    """
    基于 Breit-Wigner 分布的密度函数
    
    rho(m) = BW(m) + epsilon
    """
    def density(m):
        if m < domain[0] or m > domain[1]:
            return TINY
        bw = (1.0 / np.pi) * (m0 * gamma) / ((m ** 2 - m0 ** 2) ** 2 + (m0 * gamma) ** 2)
        return bw + TINY
    return density


def make_amplitude_density(amplitude_func, domain, n_sample=200):
    """
    基于振幅平方的密度函数 (通过离散采样构造)
    
    步骤:
      1. 在 domain 上均匀采样计算 |M|^2
      2. 做单调变换使密度 ~ |M|^2
    """
    samples = np.linspace(domain[0], domain[1], n_sample)
    vals = np.array([max(amplitude_func(s), 0.0) for s in samples])
    max_val = np.max(vals)
    if max_val < TINY:
        max_val = 1.0
    
    def density(x):
        if x < domain[0] or x > domain[1]:
            return TINY
        # 线性插值
        idx = np.searchsorted(samples, x)
        if idx <= 0:
            v = vals[0]
        elif idx >= n_sample:
            v = vals[-1]
        else:
            frac = (x - samples[idx - 1]) / (samples[idx] - samples[idx - 1])
            v = vals[idx - 1] + frac * (vals[idx] - vals[idx - 1])
        return v / max_val + TINY
    
    return density


# ============================================================
# 4. H->ZZ* 相空间自适应网格
# ============================================================
def adaptive_phase_space_grid(n_m1=20, n_m2=20, n_cos=10, n_phi=8):
    """
    为 H->ZZ*->4l 相空间积分生成自适应 CVT 网格
    
    变量:
      m1, m2: Z 不变质量
      cos_theta: 极角余弦
      phi: 方位角
    
    返回:
        grid: 字典包含各维度的节点和权重
    """
    from matrix_element import matrix_element_squared_hzz4l
    from constants import M_HIGGS, M_Z, GAMMA_Z
    
    m_min = 0.001
    m_max = M_HIGGS - m_min
    
    # m1 密度: Breit-Wigner 峰值在 M_Z
    rho_m1 = make_breit_wigner_density(M_Z, GAMMA_Z, (m_min, m_max))
    nodes_m1, _, _, _ = cvt_1d_lloyd(n_m1, rho_m1, (m_min, m_max), max_iter=150)
    
    # m2 密度: 类似
    rho_m2 = make_breit_wigner_density(M_Z, GAMMA_Z, (m_min, m_max))
    nodes_m2, _, _, _ = cvt_1d_lloyd(n_m2, rho_m2, (m_min, m_max), max_iter=150)
    
    # cos_theta: 均匀 (角分布相对平坦)
    nodes_cos = np.linspace(-1.0, 1.0, n_cos)
    
    # phi: 均匀
    nodes_phi = np.linspace(0.0, 2.0 * np.pi, n_phi)
    
    return {
        "m1": nodes_m1,
        "m2": nodes_m2,
        "cos_theta": nodes_cos,
        "phi": nodes_phi,
    }
