"""
mesh_quality.py
博士级大变形非线性有限元分析 — 网格质量检查与定向修正模块

融合原项目:
  - 1344_triangulation_orient: 三角剖分定向修正

核心数学:
  1. 四面体单元体积（有向）:
     V = (1/6) * det([x2-x1, x3-x1, x4-x1])
     若 V < 0，说明节点编号为顺时针，需交换两节点修正。

  2. 网格质量度量:
     - 体积-边长比 (Volume-Length ratio):
       η = 216 * V^2 / (sum_{i<j} |e_{ij}|^2)^3
       理想正则四面体 η = 1
     - 最小角/最大角比

  3. 定向修正算法:
     对每个单元计算有向体积，若 V < 0:
       交换节点2和节点3: [n1,n2,n3,n4] -> [n1,n3,n2,n4]
       重新计算 V，确认 V > 0
"""

import numpy as np


def tetrahedron_signed_volume(nodes, element):
    """
    计算单个四面体的有向体积

    数学:
      V = (1/6) * det(J)
      J = [x2-x1, x3-x1, x4-x1]  (3x3)
    """
    x1, x2, x3, x4 = nodes[element]
    J = np.column_stack([x2 - x1, x3 - x1, x4 - x1])
    return np.linalg.det(J) / 6.0


def orient_elements(mesh, tol=1e-12):
    """
    修正四面体网格的单元定向，确保所有单元体积为正

    源自原项目 1344_triangulation_orient (triangulation_orient)

    输入/输出:
        mesh: TetMesh 对象（原地修改）
    返回:
        neg_count: 被修正的负体积单元数
        zero_count: 零体积单元数
    """
    neg_count = 0
    zero_count = 0

    for e in range(mesh.n_elements):
        vol = tetrahedron_signed_volume(mesh.nodes, mesh.elements[e])

        if vol < -tol:
            # 交换节点 1 和 2 (0-based 的索引 1 和 2)
            mesh.elements[e][1], mesh.elements[e][2] = mesh.elements[e][2], mesh.elements[e][1]
            neg_count += 1

            # 验证修正
            vol_new = tetrahedron_signed_volume(mesh.nodes, mesh.elements[e])
            if vol_new < -tol:
                raise RuntimeError(f"Orientation fix failed for element {e}")

        elif abs(vol) <= tol:
            zero_count += 1

    return neg_count, zero_count


def compute_mesh_quality_metrics(mesh):
    """
    计算网格质量指标

    返回字典:
      - min_volume: 最小体积
      - max_volume: 最大体积
      - mean_volume: 平均体积
      - min_quality: 最小质量指标
      - mean_quality: 平均质量指标

    质量指标 (Volume-Length ratio):
      η_e = 216 * V_e^2 / L_e^6
      其中 L_e^2 = sum_{i<j} ||x_i - x_j||^2
      对于正四面体，η_e = 1
    """
    volumes = np.zeros(mesh.n_elements)
    qualities = np.zeros(mesh.n_elements)

    for e in range(mesh.n_elements):
        idx = mesh.elements[e]
        x = mesh.nodes[idx]
        vol = tetrahedron_signed_volume(mesh.nodes, idx)
        volumes[e] = abs(vol)

        # 计算6条边的平方和
        edge_sq_sum = 0.0
        pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
        for i, j in pairs:
            edge_sq_sum += np.sum((x[i] - x[j]) ** 2)

        if edge_sq_sum > 1e-15:
            qualities[e] = 216.0 * (abs(vol) ** 2) / (edge_sq_sum ** 3)
        else:
            qualities[e] = 0.0

    return {
        'min_volume': float(np.min(volumes)),
        'max_volume': float(np.max(volumes)),
        'mean_volume': float(np.mean(volumes)),
        'min_quality': float(np.min(qualities)),
        'mean_quality': float(np.mean(qualities)),
        'volumes': volumes,
        'qualities': qualities,
    }


def check_mesh_validity(mesh, min_quality_tol=1e-4):
    """
    全面检查网格有效性

    边界条件检查:
      1. 无零体积/负体积单元
      2. 质量指标在合理范围
      3. 节点坐标无 NaN/Inf
    """
    neg, zero = orient_elements(mesh)
    metrics = compute_mesh_quality_metrics(mesh)

    valid = True
    issues = []

    if zero > 0:
        valid = False
        issues.append(f"Zero-volume elements: {zero}")

    if metrics['min_quality'] < min_quality_tol:
        valid = False
        issues.append(f"Poor element quality: min={metrics['min_quality']:.2e}")

    if not np.all(np.isfinite(mesh.nodes)):
        valid = False
        issues.append("Non-finite node coordinates detected")

    return valid, issues, metrics
