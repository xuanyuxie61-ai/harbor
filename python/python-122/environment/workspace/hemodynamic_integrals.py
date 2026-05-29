"""
脑血流动力学 — 血流积分与随机采样模块

整合 tetrahedron_integrals（四面体积分）、hypersphere（超球采样）与
triangle_io（三角网格 I/O），实现脑血管网络中的血容量计算、截面采样与
网格数据管理。

科学背景:
- 脑血管树某区域的血容量可通过四面体积分计算:
    V_blood = ∫_Ω φ(r) dV
  其中 φ(r) 为血管腔指示函数（在血管内为1，组织为0）。
- 蒙特卡罗方法估计血管截面流量:
    Q ≈ (1/N) Σ_i v_i · n_i A
  其中 v_i 为超球面上均匀采样的速度向量。
- 利用超球体内均匀采样生成红细胞在血管截面上的随机位置。
"""

import numpy as np


# ---- 四面体积分 ----
def tetrahedron01_monomial_integral(e):
    """
    单位四面体上的单项式积分:
        ∫_{T_01} x^e1 y^e2 z^e3 dV
    解析公式:
        I = Π_{i=1}^3 [Π_{j=1}^{e_i} j / (k+1)] / [Π_{i=1}^3 (k_i+1)]
    其中 k 为累加计数器。
    """
    e = np.asarray(e, dtype=int)
    if np.any(e < 0):
        return 0.0
    m = len(e)
    k = 0
    integral = 1.0
    for i in range(m):
        for j in range(1, e[i] + 1):
            k += 1
            integral *= j / k
    for i in range(m):
        k += 1
        integral /= k
    return integral


def tetrahedron01_volume():
    """单位四面体体积 = 1/6。"""
    return 1.0 / 6.0


def tetrahedron01_sample(n):
    """
    在单位四面体内均匀采样 n 个点。
    四面体顶点: (0,0,0), (1,0,0), (0,1,0), (0,0,1)。
    """
    u = np.random.rand(n, 3)
    # 指数变换保证均匀分布
    x = 1.0 - u[:, 0] ** (1.0 / 3.0)
    y = (1.0 - u[:, 1] ** (1.0 / 2.0)) * (1.0 - x)
    z = u[:, 2] * (1.0 - x - y)
    return np.column_stack((x, y, z))


def integrate_blood_volume_tetrahedral(p, t):
    """
    基于四面体剖分计算血管区域血容量。
    每个四面体体积之和即为总血容量。
    """
    total_vol = 0.0
    for i in range(t.shape[0]):
        idx = t[i]
        v0, v1, v2, v3 = p[idx[0]], p[idx[1]], p[idx[2]], p[idx[3]]
        mat = np.column_stack((v1 - v0, v2 - v0, v3 - v0))
        vol = abs(np.linalg.det(mat)) / 6.0
        total_vol += vol
    return total_vol


# ---- 超球采样 ----
def hypersphere_01_surface_uniform(m, n):
    """
    在单位 m 维超球面上均匀采样 n 个点。
    方法: 生成 m 维标准正态向量并归一化。
    """
    x = np.zeros((m, n))
    for j in range(n):
        v = np.random.randn(m)
        v_norm = np.linalg.norm(v)
        if v_norm < 1e-14:
            v_norm = 1.0
        x[:, j] = v / v_norm
    return x


def hypersphere_01_interior_uniform(m, n):
    """
    在单位 m 维超球体内均匀采样 n 个点。
    方法: 先采样表面点，再按 r^(1/m) 缩放到内部。
    """
    exponent = 1.0 / m
    surface = hypersphere_01_surface_uniform(m, n)
    r = np.random.rand(n) ** exponent
    return surface * r[None, :]


def sample_vascular_cross_section(n_points, radius, center, normal):
    """
    在圆形血管截面上均匀采样 n_points 个位置。
    利用二维圆盘（超球二维版本）均匀采样后投影到三维截面平面。
    """
    normal = np.asarray(normal, dtype=float)
    normal = normal / (np.linalg.norm(normal) + 1e-14)

    # 构造截面局部坐标系
    if abs(normal[2]) < 0.9:
        tangent1 = np.cross(normal, np.array([0.0, 0.0, 1.0]))
    else:
        tangent1 = np.cross(normal, np.array([0.0, 1.0, 0.0]))
    tangent1 = tangent1 / (np.linalg.norm(tangent1) + 1e-14)
    tangent2 = np.cross(normal, tangent1)
    tangent2 = tangent2 / (np.linalg.norm(tangent2) + 1e-14)

    # 二维圆盘均匀采样
    disk = hypersphere_01_interior_uniform(2, n_points)
    points = center[None, :] + radius * (disk[0, :][:, None] * tangent1[None, :] +
                                          disk[1, :][:, None] * tangent2[None, :])
    return points


def monte_carlo_flow_rate_integral(n_samples, radius, velocity_profile_func):
    """
    蒙特卡罗积分计算血管截面流量:
        Q = ∫_A v(r) · n dA
    在截面上均匀采样 n_samples 个点，估计平均速度后乘以面积。
    """
    area = np.pi * radius ** 2
    # 采样截面位置
    disk = hypersphere_01_interior_uniform(2, n_samples)
    r_local = radius * np.sqrt(disk[0, :] ** 2 + disk[1, :] ** 2)
    r_local = np.clip(r_local, 0.0, radius)
    v_samples = velocity_profile_func(r_local)
    Q_est = area * np.mean(v_samples)
    return Q_est


# ---- 三角网格 I/O ----
def triangle_node_write(filename, node_coord, node_att=None):
    """
    写 TRIANGLE 格式的节点文件。
    """
    n_nodes = node_coord.shape[0]
    dim = node_coord.shape[1]
    with open(filename, 'w') as f:
        f.write(f"{n_nodes} {dim} 0 0\n")
        for i in range(n_nodes):
            line = f"{i} " + " ".join(f"{c:.6e}" for c in node_coord[i])
            if node_att is not None:
                line += " " + " ".join(f"{a:.6e}" for a in np.atleast_1d(node_att[i]))
            f.write(line + "\n")


def triangle_element_write(filename, element_node, element_att=None):
    """
    写 TRIANGLE 格式的单元文件。
    """
    n_elem = element_node.shape[0]
    order = element_node.shape[1]
    with open(filename, 'w') as f:
        f.write(f"{n_elem} {order} 0\n")
        for i in range(n_elem):
            line = f"{i} " + " ".join(str(idx) for idx in element_node[i])
            if element_att is not None:
                line += " " + " ".join(f"{a:.6e}" for a in np.atleast_1d(element_att[i]))
            f.write(line + "\n")


def triangle_node_read(filename):
    """
    读 TRIANGLE 格式的节点文件。
    返回节点坐标数组。
    """
    coords = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            try:
                x = float(parts[1])
                y = float(parts[2])
                coords.append([x, y])
            except ValueError:
                continue
    return np.array(coords)


def triangle_element_read(filename):
    """
    读 TRIANGLE 格式的单元文件。
    返回单元节点索引数组（0-based）。
    """
    elems = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            try:
                n1 = int(parts[1])
                n2 = int(parts[2])
                n3 = int(parts[3])
                elems.append([n1, n2, n3])
            except ValueError:
                continue
    return np.array(elems)
