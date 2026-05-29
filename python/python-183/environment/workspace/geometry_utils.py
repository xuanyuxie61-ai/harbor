r"""
geometry_utils.py
================================================================================
三维几何数据处理工具：点云读写、三角曲面法向量计算与 STL 格式转换

原项目映射:
- 1425_xyzf_display — XYZ/XYZF 点云与面片数据读取
- 1296_tri_surface_to_stla — TRI_SURFACE 到 ASCII STL 的转换

科学背景
--------
在神经影像学（如 fMRI/DTI）中，因果推断需要在三维脑皮层表面上进行。
本模块提供：
1. 点云与三角网格数据的 I/O 与处理
2. 三角面片法向量计算（用于定义因果场的表面法向梯度）
3. 网格拓扑结构分析（边邻接、顶点度数）

这些几何操作为后续在三维流形上建立因果结构方程模型提供离散几何基础。

核心公式
--------
1. 三角形法向量：
   $$ \mathbf{n} = (\mathbf{p}_2 - \mathbf{p}_1) \times (\mathbf{p}_3 - \mathbf{p}_1) $$
   单位化：$\hat{\mathbf{n}} = \mathbf{n} / \|\mathbf{n}\|$。

2. 顶点法向量（面积加权平均）：
   $$ \mathbf{N}_v = \sum_{T\ni v} A_T \,\hat{\mathbf{n}}_T $$

3. 三角形面积（三维）：
   $$ A = \frac{1}{2}\|\mathbf{v}_1 \times \mathbf{v}_2\| $$

4. 叉积：
   $$ \mathbf{a}\times\mathbf{b} = (a_y b_z - a_z b_y,\; a_z b_x - a_x b_z,\; a_x b_y - a_y b_x) $$
r"""

import numpy as np
from typing import Tuple, List


def cross_product(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    r"""
    三维叉积 $v_1 \times v_2$。
    r"""
    return np.array([
        v1[1] * v2[2] - v1[2] * v2[1],
        v1[2] * v2[0] - v1[0] * v2[2],
        v1[0] * v2[1] - v1[1] * v2[0]
    ], dtype=float)


def triangle_normal(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> np.ndarray:
    r"""
    计算三角形的单位法向量（右手定则）。
    r"""
    v1 = p2 - p1
    v2 = p3 - p1
    n = cross_product(v1, v2)
    norm = np.linalg.norm(n)
    if norm < 1e-14:
        return np.array([0.0, 0.0, 1.0])
    return n / norm


def compute_face_normals(points: np.ndarray, faces: np.ndarray) -> np.ndarray:
    r"""
    计算所有三角面片的单位法向量。

    Parameters
    ----------
    points : ndarray, shape (n_nodes, 3)
    faces : ndarray, shape (n_faces, 3)
        每个面由三个顶点索引组成。

    Returns
    -------
    normals : ndarray, shape (n_faces, 3)
    r"""
    n_faces = faces.shape[0]
    normals = np.zeros((n_faces, 3))
    for f in range(n_faces):
        p1 = points[faces[f, 0]]
        p2 = points[faces[f, 1]]
        p3 = points[faces[f, 2]]
        normals[f] = triangle_normal(p1, p2, p3)
    return normals


def compute_vertex_normals(points: np.ndarray, faces: np.ndarray) -> np.ndarray:
    r"""
    计算顶点法向量（面积加权相邻面法向量平均）。
    r"""
    n_nodes = points.shape[0]
    vnormals = np.zeros((n_nodes, 3))
    areas = np.zeros(n_nodes)
    for f in range(faces.shape[0]):
        p1 = points[faces[f, 0]]
        p2 = points[faces[f, 1]]
        p3 = points[faces[f, 2]]
        n = cross_product(p2 - p1, p3 - p1)
        area = 0.5 * np.linalg.norm(n)
        for v in faces[f]:
            vnormals[v] += n  # 未归一化法向量加权
            areas[v] += area
    for v in range(n_nodes):
        if areas[v] > 0.0:
            vnormals[v] = vnormals[v] / np.linalg.norm(vnormals[v])
        else:
            vnormals[v] = np.array([0.0, 0.0, 1.0])
    return vnormals


def mesh_edge_list(faces: np.ndarray) -> List[Tuple[int, int]]:
    r"""
    从三角面片提取无向边列表（每条边仅出现一次）。
    r"""
    edge_set = set()
    for f in range(faces.shape[0]):
        a, b, c = faces[f]
        edges = [(min(a, b), max(a, b)), (min(b, c), max(b, c)), (min(c, a), max(c, a))]
        for e in edges:
            edge_set.add(e)
    return sorted(list(edge_set))


def vertex_degree(faces: np.ndarray, n_nodes: int) -> np.ndarray:
    r"""
    计算每个顶点的度数（连接的边数）。
    r"""
    deg = np.zeros(n_nodes, dtype=int)
    for f in range(faces.shape[0]):
        for v in faces[f]:
            deg[v] += 2  # 每个面给顶点贡献 2 条边（近似）
    # 修正：每个顶点被多个面共享，真实度数为相邻不同顶点数
    adj = [set() for _ in range(n_nodes)]
    for f in range(faces.shape[0]):
        a, b, c = faces[f]
        adj[a].add(b)
        adj[a].add(c)
        adj[b].add(a)
        adj[b].add(c)
        adj[c].add(a)
        adj[c].add(b)
    deg = np.array([len(s) for s in adj])
    return deg


def stla_string(points: np.ndarray, faces: np.ndarray) -> str:
    r"""
    生成 ASCII STL 格式的字符串表示（不写入文件，避免 I/O 依赖）。

    原项目 1296_tri_surface_to_stla 核心逻辑的纯数据版本。
    r"""
    normals = compute_face_normals(points, faces)
    lines = ["solid CausalMesh"]
    for f in range(faces.shape[0]):
        n = normals[f]
        lines.append(f"  facet normal {n[0]:.6e} {n[1]:.6e} {n[2]:.6e}")
        lines.append("    outer loop")
        for v in faces[f]:
            p = points[v]
            lines.append(f"      vertex {p[0]:.6e} {p[1]:.6e} {p[2]:.6e}")
        lines.append("    endloop")
        lines.append("  endfacet")
    lines.append("endsolid CausalMesh")
    return "\n".join(lines)


def generate_icosphere_nodes(radius: float = 1.0, subdivisions: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    r"""
    生成二十面体球面网格节点与面片（用于测试因果场在曲面上的传播）。

    黄金比例构造初始 12 个顶点，再递归细分。
    r"""
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    points = np.array([
        [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
        [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
        [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1]
    ], dtype=float)
    points = radius * points / np.linalg.norm(points, axis=1, keepdims=True)

    faces = np.array([
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]
    ], dtype=int)

    # 简单细分一次
    if subdivisions > 0:
        for _ in range(subdivisions):
            new_faces = []
            edge_mid = {}
            def get_mid(a, b):
                nonlocal points
                key = tuple(sorted([a, b]))
                if key not in edge_mid:
                    mid = (points[a] + points[b]) / 2.0
                    mid = radius * mid / np.linalg.norm(mid)
                    edge_mid[key] = len(points)
                    points = np.vstack([points, mid])
                return edge_mid[key]

            for f in faces:
                a, b, c = f
                ab = get_mid(a, b)
                bc = get_mid(b, c)
                ca = get_mid(c, a)
                new_faces.extend([[a, ab, ca], [ab, b, bc], [ca, bc, c], [ab, bc, ca]])
            faces = np.array(new_faces, dtype=int)

    return points, faces


def demo():
    r"""模块自测试。"""
    points, faces = generate_icosphere_nodes(radius=1.0, subdivisions=1)
    normals = compute_face_normals(points, faces)
    vnormals = compute_vertex_normals(points, faces)
    edges = mesh_edge_list(faces)
    deg = vertex_degree(faces, points.shape[0])
    print(f"[geometry_utils] 二十面体球: 节点={points.shape[0]}, 面片={faces.shape[0]}, 边={len(edges)}")
    print(f"[geometry_utils] 顶点度数范围: [{deg.min()}, {deg.max()}]")
    print(f"[geometry_utils] 面法向量示例: {normals[0].round(4)}")
    print(f"[geometry_utils] 顶点法向量示例: {vnormals[0].round(4)}")
    stl_str = stla_string(points[:3], faces[:1])
    print(f"[geometry_utils] STL 字符串长度: {len(stl_str)}")
    return points, faces


if __name__ == "__main__":
    demo()
