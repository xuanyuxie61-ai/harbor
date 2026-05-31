"""
ellipsoid_geometry.py
================================================================================
三维椭球域几何与二维椭圆截面网格生成模块

本模块融合以下种子项目的核心算法：
  - 332_ellipsoid  : Carlson对称椭圆积分 RF/RD，椭球表面积与体积
  - 1197_tec_io    : TECPLOT风格的有限元数据 I/O 格式

科学背景
--------
在最优控制伴随方程方法中，计算域的几何描述是离散化的第一步。
对于椭球域
    Ω = { (x,y,z) | (x/a)² + (y/b)² + (z/c)² ≤ 1 }
其表面积需要计算不完全椭圆积分 E(φ,m) 与 F(φ,m)。
当退化到二维椭圆截面时，面积 S = π a b 可直接解析求得，
但模块中保留了完整的三维椭圆积分算法，以支撑未来向真三维问题的扩展。

关键公式
--------
1. Carlson 对称形式 RF(x,y,z):
   RF(x,y,z) = (1/2) ∫₀^∞ [(t+x)(t+y)(t+z)]^{-1/2} dt

2. Carlson 对称形式 RD(x,y,z):
   RD(x,y,z) = (3/2) ∫₀^∞ [(t+x)(t+y)(t+z)]^{-3/2} dt

3. 不完全椭圆积分（第一类）:
   F(φ,m) = sin(φ) · RF(cos²φ, 1−m·sin²φ, 1)

4. 不完全椭圆积分（第二类）:
   E(φ,m) = sin(φ) · RF(cos²φ, 1−m·sin²φ, 1)
            − (1/3)·m·sin³(φ)·RD(cos²φ, 1−m·sin²φ, 1)

5. 一般椭球表面积（Knud Thomsen公式）:
   设 a ≥ b ≥ c，φ = arccos(c/a)，
   m = (a²(b²−c²)) / (b²(a²−c²))
   S = 2π c² + (2π a b / sin(φ)) · (E(φ,m)·sin²φ + F(φ,m)·cos²φ)

6. 椭球体积:
   V = (4/3) π a b c
"""

import numpy as np
from scipy.spatial import Delaunay


def carlson_rf(x, y, z, tol=1.0e-12, max_iter=100):
    """
    Carlson 对称形式 RF(x,y,z)。
    使用迭代倍增定理（duplication theorem）直至参数几乎相等，
    再应用五阶泰勒展开。
    """
    x = float(x)
    y = float(y)
    z = float(z)

    if x < 0.0 or y < 0.0 or z < 0.0:
        raise ValueError("RF: 输入参数必须非负。")

    lolim = 5.0e-26
    if x + y < lolim or x + z < lolim or y + z < lolim:
        raise ValueError("RF: 参数两两之和过小，接近奇点。")

    # 迭代倍增
    for _ in range(max_iter):
        rx = np.sqrt(x)
        ry = np.sqrt(y)
        rz = np.sqrt(z)
        lam = rx * ry + rx * rz + ry * rz
        x = 0.25 * (x + lam)
        y = 0.25 * (y + lam)
        z = 0.25 * (z + lam)
        if abs(x - y) < tol and abs(x - z) < tol and abs(y - z) < tol:
            break

    e2 = (x - y) * (x - z)
    e3 = (y - z) * e2
    # 五阶泰勒展开
    c1 = 1.0 / 24.0
    c2 = 3.0 / 44.0
    c3 = 1.0 / 14.0
    u = (x + y + z) / 3.0
    val = (1.0 + c1 * e2 / u**2 - c2 * e3 / u**3 + c3 * e2**2 / u**4) / np.sqrt(u)
    return val


def carlson_rd(x, y, z, tol=1.0e-12, max_iter=100):
    """
    Carlson 对称形式 RD(x,y,z)。
    RD(x,y,z) = (3/2) ∫₀^∞ [(t+x)(t+y)(t+z)]^{-3/2} dt
    注意 RD 对 z 有额外的权重。
    """
    x = float(x)
    y = float(y)
    z = float(z)

    if x < 0.0 or y < 0.0 or z < 0.0:
        raise ValueError("RD: 输入参数必须非负。")

    lolim = 5.0e-26
    if x + y < lolim or z < lolim:
        raise ValueError("RD: 参数接近奇点。")

    sigma = 0.0
    fac = 1.0
    for _ in range(max_iter):
        rx = np.sqrt(x)
        ry = np.sqrt(y)
        rz = np.sqrt(z)
        lam = rx * ry + rx * rz + ry * rz
        sigma += fac / (rz * (z + lam))
        fac *= 0.25
        x = 0.25 * (x + lam)
        y = 0.25 * (y + lam)
        z = 0.25 * (z + lam)
        if abs(x - y) < tol and abs(x - z) < tol and abs(y - z) < tol:
            break

    e2 = (x - y) * (x - z)
    e3 = (y - z) * e2
    c1 = 1.0 / 24.0
    c2 = 3.0 / 44.0
    c3 = 1.0 / 14.0
    u = (x + y + 3.0 * z) / 5.0
    val = (1.0 + c1 * e2 / u**2 - c2 * e3 / u**3 + c3 * e2**2 / u**4) / u**1.5
    val *= 3.0
    val += 6.0 * sigma
    return val


def elliptic_inc_fm(phi, m):
    """
    不完全椭圆积分第一类 F(φ, m)。
    F(φ,m) = sin(φ) * RF(cos²φ, 1 − m·sin²φ, 1)
    """
    s = np.sin(phi)
    c = np.cos(phi)
    if abs(s) < 1.0e-15:
        return phi
    val = s * carlson_rf(c * c, 1.0 - m * s * s, 1.0)
    return val


def elliptic_inc_em(phi, m):
    """
    不完全椭圆积分第二类 E(φ, m)。
    E(φ,m) = sin(φ)*RF(...) − (1/3)*m*sin³(φ)*RD(...)
    """
    s = np.sin(phi)
    c = np.cos(phi)
    if abs(s) < 1.0e-15:
        return phi
    ss = s * s
    rf_val = carlson_rf(c * c, 1.0 - m * ss, 1.0)
    rd_val = carlson_rd(c * c, 1.0 - m * ss, 1.0)
    val = s * rf_val - (1.0 / 3.0) * m * ss * s * rd_val
    return val


def ellipsoid_surface_area(a, b, c):
    """
    计算一般椭球 (x/a)² + (y/b)² + (z/c)² = 1 的表面积。
    使用 Knud Thomsen / John D. Cook 公式，基于不完全椭圆积分。
    对退化情形（两轴相等或三轴相等）给出解析解。
    """
    abc = np.array([a, b, c], dtype=float)
    abc = np.sort(abc)[::-1]  # a ≥ b ≥ c
    a, b, c = abc

    # 退化情形检查
    if abs(a - b) < 1.0e-12 and abs(b - c) < 1.0e-12:
        # 球体
        return 4.0 * np.pi * a * a
    if abs(a - b) < 1.0e-12 and b > c:
        # 旋转椭球（扁球）
        e2 = 1.0 - (c * c) / (a * a)
        e = np.sqrt(e2)
        return 2.0 * np.pi * a * a * (1.0 + (1.0 - e2) / e * np.arctanh(e))
    if a > b and abs(b - c) < 1.0e-12:
        # 旋转椭球（长球）
        e2 = 1.0 - (b * b) / (a * a)
        e = np.sqrt(e2)
        return 2.0 * np.pi * b * b * (1.0 + a / (b * e) * np.arcsin(e))

    # 一般情形
    phi = np.arccos(c / a)
    m = (a * a * (b * b - c * c)) / (b * b * (a * a - c * c))
    # 边界检查
    if m < 0.0:
        m = 0.0
    if m > 1.0:
        m = 1.0
    e_val = elliptic_inc_em(phi, m)
    f_val = elliptic_inc_fm(phi, m)
    s = 2.0 * np.pi * c * c + (2.0 * np.pi * a * b / np.sin(phi)) * (
        e_val * np.sin(phi)**2 + f_val * np.cos(phi)**2
    )
    return s


def ellipsoid_volume(a, b, c):
    """椭球体积 V = (4/3) π a b c"""
    return (4.0 / 3.0) * np.pi * a * b * c


def ellipse_area_2d(a, b):
    """
    二维椭圆截面面积 S = π a b。
    这是三维椭球在 z=0 平面的退化面积。
    """
    return np.pi * a * b


def generate_ellipse_mesh_2d(a, b, n_boundary=32, n_inner=80, seed=42):
    """
    生成二维椭圆域 {(x,y) | (x/a)² + (y/b)² ≤ 1} 的三角形网格。

    参数
    ----
    a, b : 椭圆半轴长度
    n_boundary : 边界节点数
    n_inner : 内部节点数
    seed : 随机种子，保证可复现

    返回
    ----
    nodes : (N_nodes, 2) 节点坐标数组
    elements : (N_elements, 3) 三角形单元（逆时针）
    boundary_nodes : 边界节点索引列表
    """
    rng = np.random.default_rng(seed)

    # 1) 边界节点：在椭圆周长上均匀分布（参数化）
    theta = np.linspace(0.0, 2.0 * np.pi, n_boundary, endpoint=False)
    x_bnd = a * np.cos(theta)
    y_bnd = b * np.sin(theta)
    boundary_nodes = list(range(n_boundary))

    # 2) 内部节点： rejection sampling 保证在椭圆内
    # 使用极坐标变换提高效率：r = sqrt(u)，u~Uniform(0,1)
    x_in = []
    y_in = []
    batch = 0
    while len(x_in) < n_inner and batch < 100:
        u = rng.random(n_inner * 2)
        r = np.sqrt(u[:n_inner * 2 // 2])
        t = u[n_inner * 2 // 2:] * 2.0 * np.pi
        # 极坐标到笛卡尔，注意椭圆不是圆，需要 rejection
        xi = a * r * np.cos(t)
        yi = b * r * np.sin(t)
        for xx, yy in zip(xi, yi):
            if len(x_in) >= n_inner:
                break
            # 极坐标采样本身已保证在椭圆内，因为 r 是归一化的
            x_in.append(xx)
            y_in.append(yy)
        batch += 1

    # 如果仍有不足，补充均匀网格点
    if len(x_in) < n_inner:
        nx = int(np.sqrt(n_inner)) + 2
        xs = np.linspace(-a, a, nx)
        ys = np.linspace(-b, b, nx)
        for xx in xs:
            for yy in ys:
                if len(x_in) >= n_inner:
                    break
                if (xx / a) ** 2 + (yy / b) ** 2 < 0.99:
                    x_in.append(xx)
                    y_in.append(yy)

    x_in = np.array(x_in[:n_inner])
    y_in = np.array(y_in[:n_inner])

    # 3) 合并节点
    nodes_bnd = np.column_stack((x_bnd, y_bnd))
    nodes_in = np.column_stack((x_in, y_in))
    nodes = np.vstack((nodes_bnd, nodes_in))

    # 4) Delaunay 三角化
    tri = Delaunay(nodes)
    elements = tri.simplices.copy()

    # 5) 过滤：只保留中心在椭圆内的三角形（边界三角形可能跨越域外）
    centroid = np.mean(nodes[elements], axis=1)
    inside = ((centroid[:, 0] / a) ** 2 + (centroid[:, 1] / b) ** 2) <= 1.05
    elements = elements[inside]

    # 6) 确保逆时针方向
    for e in range(elements.shape[0]):
        i, j, k = elements[e]
        xi, yi = nodes[i]
        xj, yj = nodes[j]
        xk, yk = nodes[k]
        area2 = (xj - xi) * (yk - yi) - (xk - xi) * (yj - yi)
        if area2 < 0:
            elements[e, 1], elements[e, 2] = elements[e, 2], elements[e, 1]

    # 7) 识别真正的边界边和边界节点
    edge_count = {}
    for e in elements:
        edges = [(e[0], e[1]), (e[1], e[2]), (e[2], e[0])]
        for v1, v2 in edges:
            key = tuple(sorted((int(v1), int(v2))))
            edge_count[key] = edge_count.get(key, 0) + 1

    boundary_edges = [e for e, c in edge_count.items() if c == 1]
    # 重新确认 boundary_nodes
    boundary_nodes = sorted(set([n for edge in boundary_edges for n in edge]))

    return nodes, elements, boundary_nodes


def compute_element_areas(nodes, elements):
    """
    计算每个三角形单元的有向面积。
    area = 0.5 * | (x2-x1)(y3-y1) - (x3-x1)(y2-y1) |
    """
    p1 = nodes[elements[:, 0]]
    p2 = nodes[elements[:, 1]]
    p3 = nodes[elements[:, 2]]
    areas = 0.5 * np.abs((p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1])
                         - (p3[:, 0] - p1[:, 0]) * (p2[:, 1] - p1[:, 1]))
    return areas


def identify_boundary_edges(elements):
    """
    识别网格的边界边（只属于一个三角形的边）。
    返回边界边的列表 [(i,j), ...]
    """
    edge_count = {}
    for e in elements:
        edges = [(e[0], e[1]), (e[1], e[2]), (e[2], e[0])]
        for v1, v2 in edges:
            key = tuple(sorted((int(v1), int(v2))))
            edge_count[key] = edge_count.get(key, 0) + 1
    boundary_edges = [e for e, c in edge_count.items() if c == 1]
    return boundary_edges


def write_tecplot_mesh(filename, nodes, elements, node_data=None, var_names=None):
    """
    简化版 TECPLOT 文件写入器，融合 1197_tec_io 的格式思想。
    将网格和节点数据写入 .tec 文件，便于外部软件读取。
    """
    n_nodes = nodes.shape[0]
    n_elements = elements.shape[0]

    with open(filename, 'w') as f:
        f.write('TITLE = "Ellipse FEM Mesh"\n')
        if node_data is not None and var_names is not None:
            vars_str = ', '.join(['X', 'Y'] + list(var_names))
        else:
            vars_str = 'X, Y'
        f.write(f'VARIABLES = "{vars_str}"\n')
        f.write(f'ZONE N={n_nodes}, E={n_elements}, DATAPACKING=POINT, ZONETYPE=FETRIANGLE\n')
        for i in range(n_nodes):
            line = f"{nodes[i, 0]:.12e} {nodes[i, 1]:.12e}"
            if node_data is not None:
                for j in range(node_data.shape[1]):
                    line += f" {node_data[i, j]:.12e}"
            f.write(line + '\n')
        for e in elements:
            f.write(f"{e[0]+1} {e[1]+1} {e[2]+1}\n")


def read_tecplot_mesh(filename):
    """
    简化版 TECPLOT 文件读取器。
    读取节点坐标、单元连接和节点数据。
    """
    nodes = []
    elements = []
    data_started = False
    elem_started = False
    n_nodes_expected = 0
    n_elements_expected = 0
    node_dim = 2

    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.upper().startswith('VARIABLES='):
                parts = line.split('=')[1]
                # 粗略计数变量数
                var_count = parts.count(',') + 1
                node_dim = var_count
                continue
            if line.upper().startswith('ZONE'):
                # 解析 N=..., E=...
                parts = line.upper().split(',')
                for p in parts:
                    if 'N=' in p:
                        n_nodes_expected = int(p.split('=')[1].strip())
                    if 'E=' in p:
                        n_elements_expected = int(p.split('=')[1].strip())
                continue
            if n_nodes_expected > 0 and len(nodes) < n_nodes_expected:
                vals = [float(v) for v in line.split()]
                nodes.append(vals)
                if len(nodes) == n_nodes_expected:
                    elem_started = True
                continue
            if elem_started and len(elements) < n_elements_expected:
                vals = [int(v) - 1 for v in line.split()]
                elements.append(vals)
                continue

    nodes = np.array(nodes, dtype=float)
    if nodes.shape[1] > 2:
        node_data = nodes[:, 2:]
        nodes = nodes[:, :2]
    else:
        node_data = None
    elements = np.array(elements, dtype=int)
    return nodes, elements, node_data
