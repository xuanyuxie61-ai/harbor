"""
mesh_integrals.py
四面体网格处理与体积积分模块

融合原项目:
  - 1235_tet_mesh_quad: 四面体网格数值积分 → 计算DNA-蛋白质复合体自由能体积积分
  - 1241_tet_mesh_to_xml: 网格转XML → 分子网格结构输出
  - 1197_tec_io: TEC文件解析 → 读取分子模拟轨迹数据

科学背景:
  在分子动力学中，溶剂化自由能 ΔG_solv 可通过体积积分估算:
    ΔG_solv = ∫_Ω [ ρ(r) · φ(r) / 2 ] dV
  其中 ρ(r) 为电荷密度，φ(r) 为静电势，Ω 为溶剂区域。
  四面体网格将计算域离散化，积分近似为:
    ∫_Ω f(r) dV ≈ Σ_e (V_e / 4) · Σ_{i=1}^{4} f(r_i)
  其中 V_e 为四面体单元体积，r_i 为顶点坐标。
"""

import numpy as np
import io


def tetrahedron_volume(p0: np.ndarray, p1: np.ndarray,
                       p2: np.ndarray, p3: np.ndarray) -> float:
    """
    计算四面体体积 (有符号)

    公式:
        V = (1/6) | (p1-p0) · [(p2-p0) × (p3-p0)] |
    """
    v0 = p1 - p0
    v1 = p2 - p0
    v2 = p3 - p0
    vol = np.dot(v0, np.cross(v1, v2)) / 6.0
    return abs(vol)


def integrate_over_tet_mesh(nodes: np.ndarray, elements: np.ndarray,
                            values: np.ndarray) -> tuple:
    """
    基于 tet_mesh_quad 思想的四面体网格数值积分

    参数:
        nodes: shape (N_node, 3) 顶点坐标
        elements: shape (N_elem, 4) 每个四面体的4个顶点索引 (0-based)
        values: shape (N_node, M) 每个顶点的M维函数值

    Returns:
        quad: shape (M,) 各分量的积分估计
        volume_sum: 网格总体积
    """
    n_nodes = nodes.shape[0]
    n_elem = elements.shape[0]
    value_dim = values.shape[1] if values.ndim > 1 else 1

    if values.shape[0] != n_nodes:
        raise ValueError("values row count must match node count")

    if elements.shape[1] not in (4, 10):
        raise ValueError("Only 4-node or 10-node tet meshes are supported")

    # 确保 values 是二维
    if values.ndim == 1:
        values = values.reshape(-1, 1)

    quad = np.zeros(value_dim, dtype=float)
    volume_sum = 0.0

    for e in range(n_elem):
        idx = elements[e, :4]
        p0, p1, p2, p3 = nodes[idx[0]], nodes[idx[1]], nodes[idx[2]], nodes[idx[3]]
        vol = tetrahedron_volume(p0, p1, p2, p3)
        volume_sum += vol

        # 四点平均值乘以体积
        avg_val = np.mean(values[idx, :], axis=0)
        quad += avg_val * vol

    return quad, volume_sum


def build_tet_mesh_around_filament(bead_coords: np.ndarray,
                                   radial_divisions: int = 6,
                                   angular_divisions: int = 8,
                                   outer_radius_nm: float = 3.0) -> tuple:
    """
    围绕核蛋白丝骨架构建同心圆柱壳四面体网格

    参数:
        bead_coords: shape (N, 3) 骨架珠子坐标
        radial_divisions: 径向分层数
        angular_divisions: 周向分段数
        outer_radius_nm: 外半径 (nm)

    Returns:
        nodes, elements
    """
    n_beads = bead_coords.shape[0]
    if n_beads < 2:
        raise ValueError("Need at least 2 beads")

    # 生成圆柱壳节点
    nodes_list = []
    # 中心线节点
    for i in range(n_beads):
        nodes_list.append(bead_coords[i])

    # 外层节点
    for r_layer in range(1, radial_divisions + 1):
        r = outer_radius_nm * r_layer / radial_divisions
        for i in range(n_beads):
            # 近似法平面
            if i == 0:
                tangent = bead_coords[1] - bead_coords[0]
            elif i == n_beads - 1:
                tangent = bead_coords[-1] - bead_coords[-2]
            else:
                tangent = bead_coords[i + 1] - bead_coords[i - 1]
            tangent = tangent / (np.linalg.norm(tangent) + 1e-12)

            # 找一个垂直于tangent的向量
            arb = np.array([1.0, 0.0, 0.0])
            if abs(np.dot(tangent, arb)) > 0.9:
                arb = np.array([0.0, 1.0, 0.0])
            perp1 = np.cross(tangent, arb)
            perp1 /= np.linalg.norm(perp1) + 1e-12
            perp2 = np.cross(tangent, perp1)

            for a in range(angular_divisions):
                theta = 2.0 * np.pi * a / angular_divisions
                offset = r * (np.cos(theta) * perp1 + np.sin(theta) * perp2)
                nodes_list.append(bead_coords[i] + offset)

    nodes = np.array(nodes_list)

    # 生成四面体单元 (简化: 每个小段构建锥形四面体)
    elements = []
    n_layers = radial_divisions
    nodes_per_ring = 1 + angular_divisions * n_layers  # per bead index

    for i in range(n_beads - 1):
        base0 = i * nodes_per_ring
        base1 = (i + 1) * nodes_per_ring
        # 中心到第一层的连接 (简化为四面体扇)
        for a in range(angular_divisions):
            n0 = base0
            n1 = base1
            n2 = base0 + 1 + a
            n3 = base0 + 1 + (a + 1) % angular_divisions
            elements.append([n0, n1, n2, n3])

    elements = np.array(elements, dtype=int)
    return nodes, elements


def parse_tec_like_data(text_data: str) -> dict:
    """
    基于 tec_io 思想的简化TEC格式解析器，用于读取分子轨迹数据

    期望格式:
        TITLE="DNA filament"
        VARIABLES="X","Y","Z","potential"
        ZONE N=100, E=50, ZONETYPE=FETETRAHEDRON
        ...节点数据...
        ...单元数据...
    """
    lines = text_data.strip().splitlines()
    dim_num = 3
    node_num = 0
    element_num = 0
    element_order = 4
    node_data_num = 0

    in_zone = False
    data_start = 0
    element_start = 0

    for idx, line in enumerate(lines):
        line_stripped = line.strip()
        if line_stripped.upper().startswith('VARIABLES='):
            # 解析变量名
            var_part = line_stripped.split('=', 1)[1]
            vars_raw = [v.strip().strip('"').strip("'") for v in var_part.split(',')]
            # 前三个假设为X,Y,Z
            dim_num = 0
            node_data_num = 0
            for v in vars_raw:
                if v.upper() in ('X', 'Y', 'Z'):
                    dim_num += 1
                else:
                    node_data_num += 1
        elif line_stripped.upper().startswith('ZONE'):
            in_zone = True
            # 解析 N=..., E=...
            parts = line_stripped.replace(' ', '').split(',')
            for p in parts:
                if p.upper().startswith('N='):
                    node_num = int(p.split('=', 1)[1])
                elif p.upper().startswith('E='):
                    element_num = int(p.split('=', 1)[1])
                elif p.upper().startswith('ZONETYPE='):
                    etype = p.split('=', 1)[1].upper()
                    if 'TETRAHEDRON' in etype:
                        element_order = 4
                    elif 'TRIANGLE' in etype:
                        element_order = 3
                    elif 'BRICK' in etype:
                        element_order = 8
            data_start = idx + 1
            element_start = data_start + node_num

    if node_num <= 0 or data_start == 0:
        raise ValueError("Failed to parse TEC data header")

    # 读取节点坐标和数据
    node_coord = np.zeros((dim_num, node_num), dtype=float)
    node_data = np.zeros((node_data_num, node_num), dtype=float)

    total_vals_per_node = dim_num + node_data_num
    for i in range(node_num):
        line = lines[data_start + i].strip()
        if not line or line.startswith('#'):
            continue
        vals = [float(v) for v in line.split()]
        if len(vals) >= total_vals_per_node:
            node_coord[:, i] = vals[:dim_num]
            if node_data_num > 0:
                node_data[:, i] = vals[dim_num:dim_num + node_data_num]

    # 读取单元
    element_node = np.zeros((element_order, element_num), dtype=int)
    for e in range(element_num):
        line = lines[element_start + e].strip()
        if not line or line.startswith('#'):
            continue
        vals = [int(v) for v in line.split()]
        if len(vals) >= element_order:
            element_node[:, e] = vals[:element_order]
            # 检测1-based并转换
            if np.min(element_node[:, e]) > 0:
                element_node[:, e] -= 1

    return {
        'dim_num': dim_num,
        'node_num': node_num,
        'element_num': element_num,
        'element_order': element_order,
        'node_data_num': node_data_num,
        'node_coord': node_coord,
        'element_node': element_node,
        'node_data': node_data
    }


def write_xml_mesh(nodes: np.ndarray, elements: np.ndarray,
                   filename: str = "filament_mesh.xml") -> None:
    """
    基于 tet_mesh_to_xml 思想，将四面体网格写入简化XML格式
    """
    n_nodes = nodes.shape[0]
    n_elem = elements.shape[0]

    with io.open(filename, 'w', encoding='utf-8') as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write('<molecular_mesh>\n')
        f.write(f'  <vertices size="{n_nodes}">\n')
        for i in range(n_nodes):
            x, y, z = nodes[i]
            f.write(f'    <vertex index="{i}" x="{x:.8g}" y="{y:.8g}" z="{z:.8g}"/>\n')
        f.write('  </vertices>\n')
        f.write(f'  <cells size="{n_elem}">\n')
        for e in range(n_elem):
            v = elements[e, :4]
            f.write(f'    <tetrahedron index="{e}" v0="{v[0]}" v1="{v[1]}" '
                    f'v2="{v[2]}" v3="{v[3]}"/>\n')
        f.write('  </cells>\n')
        f.write('</molecular_mesh>\n')


def estimate_solvation_free_energy(nodes: np.ndarray, elements: np.ndarray,
                                   charge_density: np.ndarray,
                                   potential: np.ndarray) -> float:
    """
    估算溶剂化自由能 ΔG_solv = (1/2) ∫ ρ(r) φ(r) dV

    参数:
        charge_density: shape (N_node,) 节点电荷密度 (e/nm³)
        potential: shape (N_node,) 节点静电势 (kJ/mol/e)

    Returns:
        solvation_energy: 溶剂化自由能 (kJ/mol)
    """
    integrand = 0.5 * charge_density * potential
    quad, vol = integrate_over_tet_mesh(nodes, elements, integrand)
    return float(quad[0]) if np.isscalar(quad) or quad.ndim > 0 else float(quad)
