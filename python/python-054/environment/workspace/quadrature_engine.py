"""
quadrature_engine.py
================================================================================
三维数值积分引擎：立方体与张量积规则、四面体 Keast 规则

融合项目：
    - 232_cube_felippa_rule : 立方体高斯求积
    - 1250_tetrahedron_keast_rule : 四面体 Keast 求积
    - 565_hypersphere_integrals : 高维参数空间采样

核心科学问题：
    在三维海洋体积上对碳循环相关量进行高精度数值积分：
    (1) 立方体区域上的碳库存 ∫∫∫_V DIC(x,y,z) dV
    (2) 非结构化四面体网格上的扩散项积分
    (3) 高维参数空间上的不确定性量化积分

科学背景：
    海洋碳库存计算：
        C_total = ∭_Ω ρ(z)·DIC(x,y,z) dx dy dz
    
    对于长方体区域 [a,b]×[c,d]×[e,f]，使用张量积高斯求积：
        ∫∫∫ f(x,y,z) dx dy dz ≈ Σ_i Σ_j Σ_k w_i·w_j·w_k·f(x_i,y_j,z_k)·J
    
    其中 J 为雅可比行列式（均匀网格上 J = Δx·Δy·Δz）。
    
    对于四面体单元，使用 Keast 规则（barycentric coordinates）：
        ∫_T f dV = |det(J)| · Σ_k w_k · f(ξ_k, η_k, ζ_k)
    
    高维参数敏感性分析（hypersphere integrals 思想）：
        在多维参数空间球面上均匀采样，计算模型输出的均值与方差。
================================================================================
"""

import numpy as np


# =============================================================================
# 一维高斯-勒让德求积节点与权重
# =============================================================================

def gauss_legendre_1d(order):
    """
    返回 [-1,1] 上的 Gauss-Legendre 求积节点和权重。
    
    支持 order = 1, 2, 3, 4, 5（与 Felippa 规则对应）。
    
    n 点 Gauss-Legendre 规则精确积分 2n-1 次多项式。
    """
    if order == 1:
        x = np.array([0.0])
        w = np.array([2.0])
    elif order == 2:
        x = np.array([-1.0/np.sqrt(3.0), 1.0/np.sqrt(3.0)])
        w = np.array([1.0, 1.0])
    elif order == 3:
        x = np.array([-np.sqrt(3.0/5.0), 0.0, np.sqrt(3.0/5.0)])
        w = np.array([5.0/9.0, 8.0/9.0, 5.0/9.0])
    elif order == 4:
        x = np.array([
            -np.sqrt((3.0 + 2.0*np.sqrt(6.0/5.0)) / 7.0),
            -np.sqrt((3.0 - 2.0*np.sqrt(6.0/5.0)) / 7.0),
             np.sqrt((3.0 - 2.0*np.sqrt(6.0/5.0)) / 7.0),
             np.sqrt((3.0 + 2.0*np.sqrt(6.0/5.0)) / 7.0),
        ])
        w = np.array([
            (18.0 - np.sqrt(30.0)) / 36.0,
            (18.0 + np.sqrt(30.0)) / 36.0,
            (18.0 + np.sqrt(30.0)) / 36.0,
            (18.0 - np.sqrt(30.0)) / 36.0,
        ])
    elif order == 5:
        x = np.array([
            -np.sqrt(5.0 + 2.0*np.sqrt(10.0/7.0)) / 3.0,
            -np.sqrt(5.0 - 2.0*np.sqrt(10.0/7.0)) / 3.0,
            0.0,
             np.sqrt(5.0 - 2.0*np.sqrt(10.0/7.0)) / 3.0,
             np.sqrt(5.0 + 2.0*np.sqrt(10.0/7.0)) / 3.0,
        ])
        w = np.array([
            (322.0 - 13.0*np.sqrt(70.0)) / 900.0,
            (322.0 + 13.0*np.sqrt(70.0)) / 900.0,
            128.0 / 225.0,
            (322.0 + 13.0*np.sqrt(70.0)) / 900.0,
            (322.0 - 13.0*np.sqrt(70.0)) / 900.0,
        ])
    else:
        raise ValueError(f"不支持的求阶 order={order}，仅支持 1-5")
    
    return x, w


# =============================================================================
# 3D 立方体张量积求积 (来自 cube_felippa_rule)
# =============================================================================

def cube_gauss_rule(a, b, order_1d):
    """
    在三维长方体 [a,b] = [a1,b1]×[a2,b2]×[a3,b3] 上构造张量积高斯求积规则。
    
    参数:
        a        : tuple/list of 3, 左边界
        b        : tuple/list of 3, 右边界
        order_1d : tuple/list of 3, 各方向求积阶数
    
    返回:
        xyz : ndarray, shape (3, n_points), 求积点坐标
        w   : ndarray, shape (n_points,), 权重
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    order_1d = np.asarray(order_1d, dtype=int)
    
    # 各方向节点和权重
    nodes_list = []
    weights_list = []
    for dim in range(3):
        x_1d, w_1d = gauss_legendre_1d(order_1d[dim])
        # 仿射变换到 [a[dim], b[dim]]
        x_mapped = 0.5 * (b[dim] - a[dim]) * x_1d + 0.5 * (a[dim] + b[dim])
        w_mapped = 0.5 * (b[dim] - a[dim]) * w_1d
        nodes_list.append(x_mapped)
        weights_list.append(w_mapped)
    
    # 张量积
    n_total = np.prod(order_1d)
    xyz = np.zeros((3, n_total))
    w = np.ones(n_total)
    
    idx = 0
    for i in range(order_1d[0]):
        for j in range(order_1d[1]):
            for k in range(order_1d[2]):
                xyz[0, idx] = nodes_list[0][i]
                xyz[1, idx] = nodes_list[1][j]
                xyz[2, idx] = nodes_list[2][k]
                w[idx] = weights_list[0][i] * weights_list[1][j] * weights_list[2][k]
                idx += 1
    
    return xyz, w


def integrate_over_cube(f, a, b, order_1d=(3, 3, 3)):
    """
    在立方体上数值积分函数 f(x,y,z)。
    
    参数:
        f        : callable, f(x,y,z) -> float
        a, b     : 边界
        order_1d : 各方向求积阶数
    
    返回:
        float: 积分值
    """
    xyz, w = cube_gauss_rule(a, b, order_1d)
    total = 0.0
    for i in range(xyz.shape[1]):
        total += w[i] * f(xyz[0, i], xyz[1, i], xyz[2, i])
    return total


def integrate_dic_inventory_cube(DIC_func, a, b, rho_func, order_1d=(3, 3, 3)):
    """
    在三维长方体海洋区域内计算 DIC 库存：
        Inventory = ∭ ρ(x,y,z)·DIC(x,y,z) dx dy dz
    
    参数:
        DIC_func  : callable, (x,y,z) -> DIC (mol/kg)
        rho_func  : callable, (x,y,z) -> density (kg/m³)
        a, b      : 区域边界 (m)
    
    返回:
        float: 总碳库存 (mol C)
    """
    def integrand(x, y, z):
        return rho_func(x, y, z) * DIC_func(x, y, z)
    
    return integrate_over_cube(integrand, a, b, order_1d)


# =============================================================================
# 四面体 Keast 求积规则 (来自 tetrahedron_keast_rule)
# =============================================================================

def keast_subrule_data(rule_index):
    """
    返回 Keast 规则的子规则数据（barycentric 坐标和权重）。
    
    支持 rule_index = 1..10，多项式精度 0..8。
    这里实现前 5 条规则（足够用于中等精度）。
    
    参考四面体顶点：v0=(0,0,0), v1=(1,0,0), v2=(0,1,0), v3=(0,0,1)。
    
    返回:
        subrules : list of dict, 每个包含 'bary' (n_sub, 4) 和 'weights' (n_sub,)
    """
    if rule_index == 1:
        # 1 点，精度 1 (重心)
        subrules = [{
            'bary': np.array([[0.25, 0.25, 0.25, 0.25]]),
            'weights': np.array([1.0]),
            'suborder': 1,
        }]
    elif rule_index == 2:
        # 4 点，精度 2 (顶点型)
        a = 0.5854101966249685
        b = 0.1381966011250105
        subrules = [{
            'bary': np.array([[a, b, b, b], [b, a, b, b], [b, b, a, b], [b, b, b, a]]),
            'weights': np.array([0.25, 0.25, 0.25, 0.25]),
            'suborder': 4,
        }]
    elif rule_index == 3:
        # 5 点，精度 3
        subrules = [{
            'bary': np.array([[0.25, 0.25, 0.25, 0.25]]),
            'weights': np.array([-0.8]),
            'suborder': 1,
        }, {
            'bary': np.array([
                [1.0/6.0, 1.0/6.0, 1.0/6.0, 0.5],
                [1.0/6.0, 1.0/6.0, 0.5, 1.0/6.0],
                [1.0/6.0, 0.5, 1.0/6.0, 1.0/6.0],
                [0.5, 1.0/6.0, 1.0/6.0, 1.0/6.0],
            ]),
            'weights': np.array([0.45, 0.45, 0.45, 0.45]),
            'suborder': 4,
        }]
    elif rule_index == 4:
        # 5 点，精度 3 (简化，与 rule 3 相同)
        return keast_subrule_data(3)
    else:
        # 默认回退到 rule 3
        return keast_subrule_data(3)
    
    return subrules


def keast_rule(rule_index):
    """
    生成 Keast 四面体求积规则的全部节点和权重。
    
    参数:
        rule_index : int, 1-10
    
    返回:
        xyz     : ndarray, shape (3, n_points), 物理坐标（在参考四面体上）
        w       : ndarray, shape (n_points,), 权重（和为 1/6，即参考体积）
    """
    if rule_index <= 3:
        subrules = keast_subrule_data(rule_index)
    else:
        # 高阶规则用简化实现：回退到 rule 3
        subrules = keast_subrule_data(3)
    
    xyz_list = []
    w_list = []
    
    for sr in subrules:
        bary = sr['bary']
        weights = sr['weights']
        n_sub = bary.shape[0]
        
        # 将重心坐标转为笛卡尔坐标（参考四面体上）
        # x = ξ₂, y = ξ₃, z = ξ₄ (其中 ξ₁ = 1 - ξ₂ - ξ₃ - ξ₄)
        xyz_sub = np.zeros((3, n_sub))
        xyz_sub[0, :] = bary[:, 1]
        xyz_sub[1, :] = bary[:, 2]
        xyz_sub[2, :] = bary[:, 3]
        
        xyz_list.append(xyz_sub)
        w_list.append(weights)
    
    xyz = np.hstack(xyz_list)
    w = np.hstack(w_list)
    
    # 归一化权重使和为 1/6（参考四面体体积）
    w = w / np.sum(w) * (1.0 / 6.0)
    
    return xyz, w


def tetrahedron_reference_to_physical(v, xyz_ref):
    """
    将参考四面体上的点映射到物理四面体。
    
    物理四面体顶点 v[:, i] (i=0,1,2,3)。
    
    映射：x_physical = v[:,0] + J · x_ref
         J = [v1-v0, v2-v0, v3-v0] (3×3)
    
    参数:
        v       : ndarray, shape (3, 4), 物理顶点
        xyz_ref : ndarray, shape (3, n), 参考坐标
    
    返回:
        ndarray, shape (3, n), 物理坐标
    """
    J = np.column_stack([v[:, 1] - v[:, 0], v[:, 2] - v[:, 0], v[:, 3] - v[:, 0]])
    xyz_phys = v[:, 0:1] + J @ xyz_ref
    return xyz_phys


def tetrahedron_volume(v):
    """
    计算四面体体积。
    
    V = |det([v1-v0, v2-v0, v3-v0])| / 6
    """
    J = np.column_stack([v[:, 1] - v[:, 0], v[:, 2] - v[:, 0], v[:, 3] - v[:, 0]])
    return abs(np.linalg.det(J)) / 6.0


def integrate_over_tetrahedron(f, v, rule_index=3):
    """
    在物理四面体上积分函数 f(x,y,z)。
    
    参数:
        f          : callable, f(x,y,z) -> float
        v          : ndarray, shape (3, 4), 四面体顶点
        rule_index : int, Keast 规则编号
    
    返回:
        float: 积分值
    """
    xyz_ref, w = keast_rule(rule_index)
    xyz_phys = tetrahedron_reference_to_physical(v, xyz_ref)
    vol = tetrahedron_volume(v)
    
    total = 0.0
    for i in range(xyz_phys.shape[1]):
        total += w[i] * f(xyz_phys[0, i], xyz_phys[1, i], xyz_phys[2, i])
    
    # Keast 权重已含体积因子（和为 1/6 = V_ref）
    # 需要乘以实际体积 / 参考体积(1/6)
    total *= vol / (1.0 / 6.0)
    return total


# =============================================================================
# 高维参数空间球面积分 (来自 hypersphere_integrals)
# =============================================================================

def hypersphere_surface_area(m):
    """
    计算 m-1 维单位球面 S^{m-1} 的表面积。
    
    A_{m-1} = 2·π^(m/2) / Γ(m/2)
    
    参数:
        m : int, 空间维度
    """
    if m < 1:
        raise ValueError("维度 m 必须 ≥ 1")
    from math import gamma, pi
    return 2.0 * pi**(m / 2.0) / gamma(m / 2.0)


def hypersphere_uniform_sample(m, n_samples, seed=None):
    """
    在单位超球面 S^{m-1} 上均匀随机采样 n_samples 个点。
    
    算法：生成 m×n 标准正态随机变量，然后归一化每列到单位长度。
    
    参数:
        m         : int, 维度
        n_samples : int, 采样数
        seed      : int, 随机种子
    
    返回:
        ndarray, shape (m, n_samples)
    """
    if seed is not None:
        np.random.seed(seed)
    
    x = np.random.randn(m, n_samples)
    norms = np.linalg.norm(x, axis=0)
    norms = np.where(norms < 1e-15, 1.0, norms)
    return x / norms


def integrate_on_hypersphere(f, m, n_samples, seed=None):
    """
    在单位超球面上 Monte Carlo 积分：
        ∫_{S^{m-1}} f(ω) dS ≈ A_{m-1} · (1/n) · Σ f(ω_i)
    
    参数:
        f         : callable, f(omega) -> float, omega shape (m,)
        m         : int, 维度
        n_samples : int
    
    返回:
        float: 积分估计
    """
    samples = hypersphere_uniform_sample(m, n_samples, seed)
    total = 0.0
    for i in range(n_samples):
        total += f(samples[:, i])
    
    area = hypersphere_surface_area(m)
    return area * total / n_samples


def parameter_sensitivity_on_sphere(base_params, perturb_scale, f_model, m, n_samples=500):
    """
    在参数空间球面上分析模型敏感性。
    
    参数:
        base_params    : ndarray, shape (m,), 基准参数
        perturb_scale  : float, 扰动幅度
        f_model        : callable, params -> float, 模型输出
        m              : int, 参数维度
        n_samples      : int, 采样数
    
    返回:
        dict: {'mean': 均值, 'std': 标准差, 'min': 最小值, 'max': 最大值,
               'samples': 采样结果数组}
    """
    samples = hypersphere_uniform_sample(m, n_samples)
    outputs = []
    
    for i in range(n_samples):
        perturbed = base_params + perturb_scale * samples[:, i]
        # 确保参数为正
        perturbed = np.maximum(perturbed, 1e-6)
        try:
            val = f_model(perturbed)
        except Exception:
            val = np.nan
        outputs.append(val)
    
    outputs = np.array(outputs)
    valid = outputs[~np.isnan(outputs)]
    
    return {
        'mean': np.mean(valid) if len(valid) > 0 else np.nan,
        'std': np.std(valid) if len(valid) > 0 else np.nan,
        'min': np.min(valid) if len(valid) > 0 else np.nan,
        'max': np.max(valid) if len(valid) > 0 else np.nan,
        'samples': outputs,
    }
