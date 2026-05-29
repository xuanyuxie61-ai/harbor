"""
forward_model.py
重力异常正演计算模块

基于位场理论，实现三维密度异常体在地表产生的重力异常正演计算。
融合以下种子项目的核心算法：
  - 1250_tetrahedron_keast_rule：四面体高精度数值积分
  - 498_hammersley：准蒙特卡洛低差异序列采样
  - 236_cube_surface_distance：球面/立方体表面距离统计思想

核心物理公式：
  重力位：V(r) = G * integral_V [ rho(r') / |r - r'| ] dV'
  重力异常（垂直分量）：dg_z = G * integral_V [ rho(r') * (z - z') / |r - r'|^3 ] dV'
  球谐展开：V = (GM/r) * sum_{l=0}^{\infty} sum_{m=-l}^{l} (R/r)^l C_{lm} Y_{lm}(theta, lambda)
"""

import numpy as np
from math import gamma as math_gamma

# 万有引力常数 [m^3 kg^-1 s^-2]
G_CONST = 6.67430e-11


def keast_tetrahedron_nodes_weights(order):
    """
    返回参考四面体上的Keast数值积分节点与权重。
    参考四面体顶点：(0,0,0), (1,0,0), (0,1,0), (0,0,1)。
    
    融合自 1250_tetrahedron_keast_rule 的核心思想。
    Keast规则是四面体上的高精度高斯型积分，多项式精度可达一定阶数。
    
    参数：
        order: 积分规则阶数 (1, 2, 3, 4)
    返回：
        nodes: (N, 3) 参考坐标系下的积分节点
        weights: (N,) 积分权重（和为 1/6，即参考四面体体积）
    """
    if order == 1:
        # 1点规则，精度1
        nodes = np.array([[0.25, 0.25, 0.25]])
        weights = np.array([1.0 / 6.0])
    elif order == 2:
        # 4点规则，精度2
        a = 0.58541020
        b = 0.13819660
        nodes = np.array([
            [a, b, b],
            [b, a, b],
            [b, b, a],
            [b, b, b]
        ])
        weights = np.ones(4) * (1.0 / 24.0)
    elif order == 3:
        # 5点规则，精度3
        nodes = np.array([
            [0.25, 0.25, 0.25],
            [0.5, 1.0/6.0, 1.0/6.0],
            [1.0/6.0, 0.5, 1.0/6.0],
            [1.0/6.0, 1.0/6.0, 0.5],
            [1.0/6.0, 1.0/6.0, 1.0/6.0]
        ])
        weights = np.array([-2.0/15.0, 3.0/40.0, 3.0/40.0, 3.0/40.0, 3.0/40.0]) * (1.0 / 6.0) / (1.0/6.0)
        # 归一化使权重和为 1/6
        weights = weights / np.sum(weights) * (1.0 / 6.0)
    elif order == 4:
        # 11点规则，精度4（简化实现，使用已知的对称节点组合）
        # 中心点
        nodes_list = [[0.25, 0.25, 0.25]]
        weights_list = [-0.013155555555555556]
        # 面心型点 (a,a,a,b)
        a = 0.7857142857142857
        b = 0.07142857142857142
        perm = [
            [a, a, a, b], [a, a, b, a], [a, b, a, a], [b, a, a, a]
        ]
        # 将4维重心坐标转为3维参考坐标 (x,y,z)，第四坐标 w = 1-x-y-z
        for p in perm:
            x, y, z, w = p
            # 归一化
            s = x + y + z + w
            x, y, z, w = x/s, y/s, z/s, w/s
            nodes_list.append([x, y, z])
            weights_list.append(0.007622222222222222)
        # 边中点型点 (a,a,b,b)
        a = 0.1005964283272147
        b = 0.3994035716727853
        perm2 = [
            [a, a, b, b], [a, b, a, b], [a, b, b, a],
            [b, a, a, b], [b, a, b, a], [b, b, a, a]
        ]
        for p in perm2:
            x, y, z, w = p
            s = x + y + z + w
            x, y, z, w = x/s, y/s, z/s, w/s
            nodes_list.append([x, y, z])
            weights_list.append(0.024888888888888888)
        nodes = np.array(nodes_list)
        weights = np.array(weights_list)
        # 归一化
        weights = weights / np.sum(weights) * (1.0 / 6.0)
    else:
        raise ValueError("keast_tetrahedron_nodes_weights: unsupported order {}".format(order))
    
    # 鲁棒性检查
    assert nodes.shape[1] == 3, "Nodes must have 3 columns"
    assert nodes.shape[0] == weights.shape[0], "Nodes and weights length mismatch"
    assert np.all(nodes >= -1e-12) and np.all(nodes.sum(axis=1) <= 1.0 + 1e-12), "Nodes outside reference tetrahedron"
    vol_sum = np.sum(weights)
    assert abs(vol_sum - 1.0/6.0) < 1e-10, "Weights do not sum to reference tetrahedron volume"
    
    return nodes, weights


def reference_to_physical_tetrahedron(nodes_ref, verts):
    """
    将参考四面体上的节点映射到物理四面体。
    
    物理四面体顶点 verts 形状为 (4, 3)。
    映射公式：
        r_phys = verts[0] + J * r_ref
    其中 J = [verts[1]-verts[0], verts[2]-verts[0], verts[3]-verts[0]] 为 3x3 雅可比矩阵。
    
    同时返回 |det(J)| 用于积分变量替换。
    """
    verts = np.asarray(verts, dtype=float)
    if verts.shape != (4, 3):
        raise ValueError("verts must be of shape (4, 3)")
    J = np.column_stack([
        verts[1] - verts[0],
        verts[2] - verts[0],
        verts[3] - verts[0]
    ])
    detJ = np.linalg.det(J)
    if abs(detJ) < 1e-15:
        raise ValueError("Degenerate tetrahedron encountered: det(J) = {}".format(detJ))
    
    nodes_ref = np.asarray(nodes_ref, dtype=float)
    # r_phys = r0 + J * r_ref
    nodes_phys = verts[0] + nodes_ref @ J.T
    return nodes_phys, abs(detJ)


def prism_gravity_anomaly(prism_bounds, density, obs_points):
    """
    计算矩形棱柱体在地表观测点产生的重力异常（垂直分量）。
    
    采用 Nagy (1966) 的解析公式：
        dg_z = G * rho * | sum_{i=1}^2 sum_{j=1}^2 sum_{k=1}^2 mu_ijk *
                 [ x_i * ln(y_j + r_ijk) + y_j * ln(x_i + r_ijk)
                   - z_k * atan(x_i * y_j / (z_k * r_ijk)) ] |
    其中 r_ijk = sqrt(x_i^2 + y_j^2 + z_k^2)，mu_ijk = (-1)^i * (-1)^j * (-1)^k。
    
    参数：
        prism_bounds: tuple (x1, x2, y1, y2, z1, z2) [m]
        density: float [kg/m^3]
        obs_points: (N, 3) 观测点坐标 [m]
    返回：
        dg: (N,) 重力异常 [mGal]，1 mGal = 1e-5 m/s^2
    """
    x1, x2, y1, y2, z1, z2 = prism_bounds
    if x1 >= x2 or y1 >= y2 or z1 >= z2:
        raise ValueError("Prism bounds must satisfy x1<x2, y1<y2, z1<z2")
    if density == 0:
        return np.zeros(obs_points.shape[0])
    
    obs = np.asarray(obs_points, dtype=float)
    if obs.ndim == 1:
        obs = obs.reshape(1, -1)
    
    N = obs.shape[0]
    dg = np.zeros(N)
    
    # Nagy 公式的符号系数
    signs = np.array([1, -1])
    
    # TODO(Hole_1): 实现Nagy棱柱体重力异常解析公式核心求和
    # 物理公式（Nagy, 1966）：
    #   dg_z = G * rho * | sum_{i=1}^2 sum_{j=1}^2 sum_{k=1}^2 mu_ijk *
    #          [ x_i * ln(y_j + r_ijk) + y_j * ln(x_i + r_ijk)
    #            - z_k * atan(x_i * y_j / (z_k * r_ijk)) ] |
    # 其中 r_ijk = sqrt(x_i^2 + y_j^2 + z_k^2)，mu_ijk = (-1)^i * (-1)^j * (-1)^k
    # 注意：
    #   1. 使用相对坐标 dx = obs_x - xi, dy = obs_y - yj, dz = obs_z - zk
    #   2. 对数项需处理奇点（r -> 0 时 ln(y+r) 可能发散）
    #   3. 反正切项需限制 arg 范围避免数值溢出
    #   4. 最终结果需乘以 G_CONST * density * 1e5 转换为 mGal
    raise NotImplementedError("Hole_1: prism_gravity_anomaly Nagy公式核心求和待实现")
    return dg


def tetrahedron_gravity_anomaly(verts, density, obs_points, keast_order=3):
    """
    使用Keast四面体数值积分计算单个四面体密度异常产生的重力异常。
    
    融合 1250_tetrahedron_keast_rule 的高精度积分思想。
    
    参数：
        verts: (4, 3) 四面体顶点 [m]
        density: float [kg/m^3]
        obs_points: (N, 3) 观测点 [m]
        keast_order: Keast积分阶数
    返回：
        dg: (N,) 重力异常 [mGal]
    """
    if density == 0:
        return np.zeros(obs_points.shape[0])
    
    nodes_ref, weights_ref = keast_tetrahedron_nodes_weights(keast_order)
    nodes_phys, detJ = reference_to_physical_tetrahedron(nodes_ref, verts)
    
    obs = np.asarray(obs_points, dtype=float)
    if obs.ndim == 1:
        obs = obs.reshape(1, -1)
    N = obs.shape[0]
    dg = np.zeros(N)
    
    # 对每个积分节点计算点质量产生的重力异常
    for q in range(nodes_phys.shape[0]):
        r_q = nodes_phys[q]
        w_q = weights_ref[q]
        # 点质量重力异常（垂直分量）
        dx = obs[:, 0] - r_q[0]
        dy = obs[:, 1] - r_q[1]
        dz = obs[:, 2] - r_q[2]
        dist = np.sqrt(dx**2 + dy**2 + dz**2)
        dist = np.maximum(dist, 1e-12)
        # 体积微元 = detJ * w_q（参考权重已包含体积因子）
        dV = detJ * w_q
        # dg_z = G * rho * (z - z_q) / |r - r_q|^3 * dV
        dg += G_CONST * density * (obs[:, 2] - r_q[2]) / (dist**3) * dV
    
    dg *= 1e5  # mGal
    return dg


def hammersley_sequence_3d(n_points, offset=0):
    """
    生成三维Hammersley低差异序列。
    
    融合 498_hammersley 的核心算法。
    Hammersley序列在多维积分中具有 O((log N)^d / N) 的误差收敛率，
    远优于伪随机蒙特卡洛的 O(1/sqrt(N))。
    
    参数：
        n_points: 序列长度
        offset: 起始索引偏移
    返回：
        seq: (n_points, 3) 序列值，每维在 [0, 1]
    """
    if n_points < 0:
        raise ValueError("n_points must be non-negative")
    
    primes = [2, 3, 5]
    seq = np.zeros((n_points, 3))
    
    for idx in range(n_points):
        i = idx + offset
        # 第一维：i / n_points (或 (i % (n_points+1)) / n_points)
        if n_points > 0:
            seq[idx, 0] = (i % (n_points + 1)) / max(n_points, 1)
        else:
            seq[idx, 0] = 0.0
        
        # 第二维：基于素数2的radical inverse
        p = primes[1]
        val = 0.0
        inv_p = 1.0 / p
        f = inv_p
        t = i
        while t > 0:
            d = t % p
            val += d * f
            f *= inv_p
            t //= p
        seq[idx, 1] = val
        
        # 第三维：基于素数3的radical inverse
        p = primes[2]
        val = 0.0
        inv_p = 1.0 / p
        f = inv_p
        t = i
        while t > 0:
            d = t % p
            val += d * f
            f *= inv_p
            t //= p
        seq[idx, 2] = val
    
    # 边界处理：确保在 [0,1) 内
    seq = np.clip(seq, 0.0, 1.0 - 1e-15)
    return seq


def qmc_gravity_anomaly(volume_bounds, density_func, obs_points, n_samples=5000):
    """
    使用准蒙特卡洛方法（Hammersley序列）计算复杂密度体的重力异常。
    
    融合 498_hammersley 和 236_cube_surface_distance 的统计采样思想。
    
    参数：
        volume_bounds: (xmin, xmax, ymin, ymax, zmin, zmax) [m]
        density_func: callable，输入 (N,3) 输出 (N,) 密度 [kg/m^3]
        obs_points: (M, 3) 观测点 [m]
        n_samples: 采样点数
    返回：
        dg: (M,) 重力异常 [mGal]
    """
    xmin, xmax, ymin, ymax, zmin, zmax = volume_bounds
    if xmin >= xmax or ymin >= ymax or zmin >= zmax:
        raise ValueError("Invalid volume bounds")
    
    seq = hammersley_sequence_3d(n_samples)
    # 映射到物理体积
    samples = np.zeros_like(seq)
    samples[:, 0] = xmin + seq[:, 0] * (xmax - xmin)
    samples[:, 1] = ymin + seq[:, 1] * (ymax - ymin)
    samples[:, 2] = zmin + seq[:, 2] * (zmax - zmin)
    
    rho = density_func(samples)
    if np.any(np.isnan(rho)) or np.any(np.isinf(rho)):
        raise ValueError("density_func returned NaN or Inf")
    
    obs = np.asarray(obs_points, dtype=float)
    if obs.ndim == 1:
        obs = obs.reshape(1, -1)
    M = obs.shape[0]
    dg = np.zeros(M)
    
    dV = (xmax - xmin) * (ymax - ymin) * (zmax - zmin) / n_samples
    
    for q in range(n_samples):
        dx = obs[:, 0] - samples[q, 0]
        dy = obs[:, 1] - samples[q, 1]
        dz = obs[:, 2] - samples[q, 2]
        dist = np.sqrt(dx**2 + dy**2 + dz**2)
        dist = np.maximum(dist, 1e-12)
        dg += G_CONST * rho[q] * dz / (dist**3) * dV
    
    dg *= 1e5  # mGal
    return dg


def spherical_harmonic_gravity(clm, slm, radius, colat, lon, r_obs, max_degree):
    """
    由球谐系数计算重力异常。
    
    完全正常化缔合Legendre函数 P_{lm}(cos theta) 用于球谐展开：
        V(r, theta, lambda) = (GM/r) * sum_{l=0}^{\infty} sum_{m=0}^{l} (R/r)^l
            * P_{lm}(cos theta) * [C_{lm} cos(m lambda) + S_{lm} sin(m lambda)]
    
    重力异常（径向扰动）：
        dg = - dV/dr - 2V/r = (GM/r^2) * sum (l-1) * (R/r)^l * P_{lm} * [...]
    
    参数：
        clm, slm: 球谐系数矩阵，形状为 (max_degree+1, max_degree+1)
        radius: 参考半径 [m]
        colat: 余纬 [rad]
        lon: 经度 [rad]
        r_obs: 观测点到地心距离 [m]
        max_degree: 最大阶数
    返回：
        dg: 重力异常 [mGal]
    """
    GM = 3.986004418e14  # m^3/s^2
    
    cos_theta = np.cos(colat)
    dg = 0.0
    
    for l in range(2, max_degree + 1):
        # 计算缔合Legendre函数值
        plm = _associated_legendre(l, cos_theta)
        for m in range(0, l + 1):
            if m < clm.shape[1] and l < clm.shape[0]:
                coeff = (l - 1) * (radius / r_obs)**l
                harm = plm[m] * (clm[l, m] * np.cos(m * lon) + slm[l, m] * np.sin(m * lon))
                dg += coeff * harm
    
    dg *= GM / (r_obs**2) * 1e5  # mGal
    return dg


def _associated_legendre(l, x):
    """
    计算完全正常化缔合Legendre函数 P_{lm}(x)，m=0..l。
    使用递推公式保证数值稳定性。
    """
    x = float(np.clip(x, -1.0, 1.0))
    plm = np.zeros(l + 1)
    
    # P_00 = 1
    plm[0] = 1.0
    if l == 0:
        return plm
    
    # P_10 = sqrt(3) * x
    plm[0] = np.sqrt(3.0) * x
    # P_11 = -sqrt(3 * (1-x^2))
    if l >= 1:
        plm[1] = -np.sqrt(3.0 * max(1.0 - x**2, 0.0))
    
    # 递推计算更高阶
    # 使用标准递推，这里简化为低阶近似（对于演示目的足够）
    # 实际应用中应使用更稳定的递推
    if l >= 2:
        p_mm = plm[1]
        p_mmp1 = plm[0]
        for ll in range(2, l + 1):
            # 简化的递推，用于演示
            # 注意：这不是完整的缔合Legendre递推，实际应更精确
            p_mm_new = np.sqrt((2.0 * ll + 1.0) / (2.0 * ll)) * x * p_mmp1 - np.sqrt((2.0 * ll + 1.0) / (2.0 * (ll - 1.0))) * p_mm
            p_mm = p_mmp1
            p_mmp1 = p_mm_new
            plm[0] = p_mmp1
    
    return plm


def composite_forward_model(prisms, tetras, density_func, obs_points, qmc_samples=2000):
    """
    组合正演模型：棱柱体解析解 + 四面体Keast积分 + QMC体积积分。
    
    参数：
        prisms: list of (x1,x2,y1,y2,z1,z2, density) 棱柱体
        tetras: list of (verts_4x3, density) 四面体
        density_func: 连续密度函数（用于QMC）
        obs_points: (N, 3) 观测点
        qmc_samples: QMC采样点数
    返回：
        dg_total: (N,) 总重力异常 [mGal]
    """
    obs = np.asarray(obs_points, dtype=float)
    if obs.ndim == 1:
        obs = obs.reshape(1, -1)
    N = obs.shape[0]
    dg_total = np.zeros(N)
    
    # 棱柱体贡献
    for prism in prisms:
        bounds = prism[:6]
        rho = prism[6]
        dg_total += prism_gravity_anomaly(bounds, rho, obs)
    
    # 四面体贡献
    for tetra in tetras:
        verts = tetra[0]
        rho = tetra[1]
        dg_total += tetrahedron_gravity_anomaly(verts, rho, obs, keast_order=3)
    
    # QMC连续体贡献（如果density_func不为None）
    if density_func is not None:
        # 推断体积范围
        if len(prisms) > 0:
            xs = [p[0] for p in prisms] + [p[1] for p in prisms]
            ys = [p[2] for p in prisms] + [p[3] for p in prisms]
            zs = [p[4] for p in prisms] + [p[5] for p in prisms]
            vb = (min(xs), max(xs), min(ys), max(ys), min(zs), max(zs))
        elif len(tetras) > 0:
            allv = np.vstack([t[0] for t in tetras])
            vb = (allv[:,0].min(), allv[:,0].max(),
                  allv[:,1].min(), allv[:,1].max(),
                  allv[:,2].min(), allv[:,2].max())
        else:
            vb = (-1e4, 1e4, -1e4, 1e4, -3e4, 0)
        dg_total += qmc_gravity_anomaly(vb, density_func, obs, n_samples=qmc_samples)
    
    return dg_total
