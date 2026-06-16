"""
mesh_basis.py
=============
有限元网格生成与重心基函数 —— 离散化优化域

融合种子项目:
  - 548_human_mesh2d: 2D 三角网格生成 (Delaunay 型)
  - 382_fem_to_xml: FEM 网格数据序列化
  - 371_fem_basis: 1D/2D/3D/MD 重心基函数

核心公式:
  1. 重心坐标: λ_i(x) = area(Δ_i)/area(Δ), Σ λ_i = 1
  2. Lagrange 基函数: L_I(x) = Π_{q=1}^{m+1} Π_{p=0}^{i_q-1} (d·λ_q - p)/(i_q - p)
  3. 三角面积: 2A = |x₁(y₂-y₃) + x₂(y₃-y₁) + x₃(y₁-y₂)|
  4. 雅可比矩阵: J = [x₂-x₁, x₃-x₁; y₂-y₁, y₃-y₁], det(J) = 2A
"""

import numpy as np
from typing import Tuple, List, Optional
import json


# ---------------------------------------------------------------------------
# 1. 2D Delaunay 型三角网格生成 (源自 548_human_mesh2d)
# ---------------------------------------------------------------------------

def generate_boundary_points(n_boundary: int, shape: str = "circle") -> np.ndarray:
    """生成边界点集.
    shape: "circle" | "square" | "human_like"

    对 "human_like": 使用参数化人体轮廓
      x(t) = 0.3 sin(t) + 0.1 sin(3t)
      y(t) = cos(t) + 0.2 cos(2t)
    """
    t = np.linspace(0, 2 * np.pi, n_boundary, endpoint=False)
    if shape == "circle":
        x = np.cos(t)
        y = np.sin(t)
    elif shape == "square":
        # 单位正方形边界
        quarter = n_boundary // 4
        pts = []
        for i in range(quarter):
            pts.append([i / quarter, 0.0])
        for i in range(quarter):
            pts.append([1.0, i / quarter])
        for i in range(quarter):
            pts.append([1.0 - i / quarter, 1.0])
        for i in range(quarter):
            pts.append([0.0, 1.0 - i / quarter])
        return np.array(pts)
    elif shape == "human_like":
        # 参数化人体轮廓 (Fourier 级数近似)
        x = 0.3 * np.sin(t) + 0.1 * np.sin(3 * t) + 0.05 * np.cos(5 * t)
        y = np.cos(t) + 0.2 * np.cos(2 * t) + 0.1 * np.sin(4 * t)
    else:
        x = np.cos(t)
        y = np.sin(t)
    return np.column_stack([x, y])


def delaunay_triangulation_2d(vertices: np.ndarray) -> np.ndarray:
    """Delaunay 三角剖分 (使用 scipy 或 Bowyer-Watson 备选).
    输入: vertices (N, 2)
    输出: triangles (M, 3) 顶点索引

    空圆性质: 每个三角形的外接圆不包含其他顶点.
    """
    n_pts = len(vertices)
    if n_pts < 3:
        return np.array([], dtype=int).reshape(0, 3)

    try:
        from scipy.spatial import Delaunay
        tri = Delaunay(vertices)
        return tri.simplices.astype(int)
    except ImportError:
        pass

    # 备选: Bowyer-Watson
    return _bowyer_watson(vertices)


def _bowyer_watson(vertices: np.ndarray) -> np.ndarray:
    """Bowyer-Watson 增量 Delaunay 三角剖分."""
    n_pts = len(vertices)

    # 超级三角形
    mins = vertices.min(axis=0) - 1.0
    maxs = vertices.max(axis=0) + 1.0
    dx = maxs[0] - mins[0]
    dy = maxs[1] - mins[1]
    dmax = max(dx, dy)
    mid_x = 0.5 * (mins[0] + maxs[0])
    mid_y = 0.5 * (mins[1] + maxs[1])

    super_tri = np.array([
        [mid_x - 20 * dmax, mid_y - dmax],
        [mid_x, mid_y + 20 * dmax],
        [mid_x + 20 * dmax, mid_y - dmax]
    ])
    all_pts = np.vstack([vertices, super_tri])

    triangles = [(n_pts, n_pts + 1, n_pts + 2)]

    for i in range(n_pts):
        pt = all_pts[i]
        bad_tris = []
        for tri_idx, tri in enumerate(triangles):
            if _in_circumcircle(pt, all_pts[tri[0]], all_pts[tri[1]], all_pts[tri[2]]):
                bad_tris.append(tri_idx)

        boundary = []
        for tri_idx in bad_tris:
            tri = triangles[tri_idx]
            for edge in [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])]:
                shared = False
                for other_idx in bad_tris:
                    if other_idx == tri_idx:
                        continue
                    other_tri = triangles[other_idx]
                    if edge[0] in other_tri and edge[1] in other_tri:
                        shared = True
                        break
                if not shared:
                    boundary.append(edge)

        for tri_idx in sorted(bad_tris, reverse=True):
            triangles.pop(tri_idx)

        for edge in boundary:
            triangles.append((i, edge[0], edge[1]))

    result = []
    for tri in triangles:
        if tri[0] < n_pts and tri[1] < n_pts and tri[2] < n_pts:
            result.append(tri)

    if len(result) == 0:
        return np.array([], dtype=int).reshape(0, 3)
    return np.array(result, dtype=int)


def _in_circumcircle(p, a, b, c) -> bool:
    """判断点 p 是否在三角形 (a,b,c) 的外接圆内.
    使用行列式判据:
    | ax-px  ay-py  (ax-px)²+(ay-py)² |
    | bx-px  by-py  (bx-px)²+(by-py)² | > 0  ⇒ 在内
    | cx-px  cy-py  (cx-px)²+(cy-py)² |
    """
    ax, ay = a[0] - p[0], a[1] - p[1]
    bx, by = b[0] - p[0], b[1] - p[1]
    cx, cy = c[0] - p[0], c[1] - p[1]
    det = (ax * (by * (cx ** 2 + cy ** 2) - cy * (bx ** 2 + by ** 2))
           - ay * (bx * (cx ** 2 + cy ** 2) - cx * (bx ** 2 + by ** 2))
           + (ax ** 2 + ay ** 2) * (bx * cy - by * cx))
    return det > 0


# ---------------------------------------------------------------------------
# 2. 三角形几何量计算
# ---------------------------------------------------------------------------

def triangle_area_2d(v1, v2, v3) -> float:
    """三角形面积 (有符号).
    2A = (x₂-x₁)(y₃-y₁) - (x₃-x₁)(y₂-y₁)
    """
    return 0.5 * ((v2[0] - v1[0]) * (v3[1] - v1[1])
                  - (v3[0] - v1[0]) * (v2[1] - v1[1]))


def triangle_jacobian_2d(v1, v2, v3) -> Tuple[np.ndarray, float]:
    """计算三角形等参映射的雅可比矩阵.
    J = [[x₂-x₁, x₃-x₁],
         [y₂-y₁, y₃-y₁]]
    det(J) = 2·Area

    返回 (J, det_J).
    """
    J = np.array([[v2[0] - v1[0], v3[0] - v1[0]],
                  [v2[1] - v1[1], v3[1] - v1[1]]])
    det_J = J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0]
    return J, det_J


def barycentric_coords_2d(point, v1, v2, v3) -> Tuple[float, float, float]:
    """计算点 p 关于三角形 (v1,v2,v3) 的重心坐标 (λ₁,λ₂,λ₃).
    λ_i = area(p, v_j, v_k) / area(v1, v2, v3),  i,j,k 循环.
    约束: λ₁+λ₂+λ₃ = 1.
    """
    A = triangle_area_2d(v1, v2, v3)
    if abs(A) < 1e-300:
        return 1.0 / 3, 1.0 / 3, 1.0 / 3
    l1 = triangle_area_2d(point, v2, v3) / A
    l2 = triangle_area_2d(v1, point, v3) / A
    l3 = 1.0 - l1 - l2
    return l1, l2, l3


# ---------------------------------------------------------------------------
# 3. FEM 基函数 (源自 371_fem_basis)
# ---------------------------------------------------------------------------

def fem_basis_1d(degree: int, node_index: int, x: float) -> float:
    """1D Lagrange 基函数.
    在参考单元 [0,1] 上, 对 degree 次多项式, 有 degree+1 个节点.
    节点: x_i = i/degree, i=0,...,degree.
    L_i(x) = Π_{j≠i} (x - x_j)/(x_i - x_j)
    """
    if degree <= 0:
        return 1.0
    nodes = [j / degree for j in range(degree + 1)]
    result = 1.0
    for j in range(degree + 1):
        if j != node_index:
            denom = nodes[node_index] - nodes[j]
            if abs(denom) < 1e-300:
                continue
            result *= (x - nodes[j]) / denom
    return result


def fem_basis_2d(i: int, j: int, k: int, x: float, y: float) -> float:
    """2D 三角形 Lagrange 基函数 (源自 371_fem_basis/fem_basis_2d).

    给定次数 D = i+j+k, 基函数:
      L_{ijk}(x,y) = [Π_{p=0}^{i-1} (Dx - p)] · [Π_{p=0}^{j-1} (Dy - p)]
                     · [Π_{p=0}^{k-1} (D(x+y) - (D-p))] / 归一化常数

    使用重心坐标 λ₁=1-x-y, λ₂=x, λ₃=y.
    """
    d = i + j + k
    if d == 0:
        return 1.0

    result = 1.0
    norm = 1.0

    # λ₂ = x 的因子
    for p in range(i):
        result *= (d * x - p)
        norm *= (i - p)

    # λ₃ = y 的因子
    for p in range(j):
        result *= (d * y - p)
        norm *= (j - p)

    # λ₁ = 1-x-y 的因子
    for p in range(k):
        result *= (d * (1.0 - x - y) - p)
        norm *= (k - p)

    if abs(norm) < 1e-300:
        return 0.0
    return result / norm


def fem_basis_3d(i: int, j: int, k: int, l: int,
                 x: float, y: float, z: float) -> float:
    """3D 四面体 Lagrange 基函数 (源自 371_fem_basis/fem_basis_3d).

    D = i+j+k+l, 四面体顶点:
      V₁=(0,0,0), V₂=(1,0,0), V₃=(0,1,0), V₄=(0,0,1).
    重心坐标: λ₁=1-x-y-z, λ₂=x, λ₃=y, λ₄=z.
    """
    d = i + j + k + l
    if d == 0:
        return 1.0

    result = 1.0
    norm = 1.0
    lam = [1.0 - x - y - z, x, y, z]
    indices = [k, i, j, l]  # 对应 λ₁, λ₂, λ₃, λ₄

    for q, idx in enumerate(indices):
        for p in range(idx):
            result *= (d * lam[q] - p)
            norm *= (idx - p)

    if abs(norm) < 1e-300:
        return 0.0
    return result / norm


def fem_basis_md(m: int, i_bary: np.ndarray, x: np.ndarray) -> float:
    """M 维单纯形 Lagrange 基函数 (源自 371_fem_basis/fem_basis_md).

    输入:
      m: 空间维数
      i_bary: 重心索引 (m+1,), 满足 Σ i_bary = D (多项式次数)
      x: 求值点 (m,)

    重心坐标: λ_q = x_q (q=1..m), λ_{m+1} = 1 - Σ x_q.
    """
    # 增广 x 向量
    lam = np.zeros(m + 1)
    lam[:m] = x
    lam[m] = 1.0 - np.sum(x)

    d = np.sum(i_bary)
    if d == 0:
        return 1.0

    result = 1.0
    for q in range(m + 1):
        for p in range(i_bary[q]):
            denom = i_bary[q] - p
            if abs(denom) < 1e-300:
                continue
            result *= (d * lam[q] - p) / denom
    return result


# ---------------------------------------------------------------------------
# 4. 网格数据序列化 (源自 382_fem_to_xml)
# ---------------------------------------------------------------------------

def mesh_to_dict(vertices: np.ndarray, elements: np.ndarray) -> dict:
    """将 FEM 网格转为字典表示 (XML 格式的 Python 等价).
    源自 382_fem_to_xml: 将节点坐标和单元连接转为结构化数据.
    """
    mesh_data = {
        "vertices": vertices.tolist(),
        "elements": elements.tolist(),
        "n_vertices": len(vertices),
        "n_elements": len(elements),
        "spatial_dim": vertices.shape[1] if len(vertices) > 0 else 0,
    }
    return mesh_data


def mesh_to_json(mesh_data: dict, filename: str = "") -> str:
    """将网格数据序列化为 JSON 字符串.
    源自 382_fem_to_xml: DOLFIN XML 的 Python/JSON 等价.
    """
    json_str = json.dumps(mesh_data, indent=2)
    if filename:
        with open(filename, 'w') as f:
            f.write(json_str)
    return json_str


# ---------------------------------------------------------------------------
# 5. 高斯求积规则 (在参考三角形/四面体上)
# ---------------------------------------------------------------------------

def gauss_triangle(degree: int) -> Tuple[np.ndarray, np.ndarray]:
    """三角形上的 Gauss 求积点与权重.
    degree: 精确到 degree 次多项式.

    对 degree ≤ 2 使用已知公式, 更高阶使用 Dunavant 规则.
    """
    if degree <= 1:
        # 1 点规则 (精确到 1 次)
        pts = np.array([[1.0 / 3, 1.0 / 3]])
        wts = np.array([0.5])
    elif degree <= 2:
        # 3 点规则 (精确到 2 次)
        pts = np.array([
            [1.0 / 6, 1.0 / 6],
            [2.0 / 3, 1.0 / 6],
            [1.0 / 6, 2.0 / 3]
        ])
        wts = np.array([1.0 / 6, 1.0 / 6, 1.0 / 6])
    elif degree <= 4:
        # 6 点规则 (精确到 4 次, Dunavant)
        a1 = 0.445948490915965
        a2 = 0.091576213509771
        w1 = 0.111690794839005
        w2 = 0.054975871827661
        pts = np.array([
            [a1, a1], [1 - 2 * a1, a1], [a1, 1 - 2 * a1],
            [a2, a2], [1 - 2 * a2, a2], [a2, 1 - 2 * a2]
        ])
        wts = np.array([w1, w1, w1, w2, w2, w2])
    else:
        # 7 点规则 (精确到 5 次)
        pts = np.array([
            [1.0 / 3, 1.0 / 3],
            [0.059715871789770, 0.470142064105115],
            [0.470142064105115, 0.059715871789770],
            [0.470142064105115, 0.470142064105115],
            [0.797426985353087, 0.101286507323456],
            [0.101286507323456, 0.797426985353087],
            [0.101286507323456, 0.101286507323456],
        ])
        wts = np.array([
            0.1125,
            0.066197076394253, 0.066197076394253, 0.066197076394253,
            0.062969590272414, 0.062969590272414, 0.062969590272414,
        ])
    return pts, wts


# ---------------------------------------------------------------------------
# 6. 网格细化 (自适应)
# ---------------------------------------------------------------------------

def refine_mesh_uniform(vertices: np.ndarray, elements: np.ndarray,
                        n_refine: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """均匀网格细化: 每个三角形分成 4 个 (中点细分).
    源自红-绿细化策略的最简形式.

    每条边取中点, 形成 4 个子三角形.
    """
    for _ in range(n_refine):
        edge_midpoints = {}
        new_vertices = list(vertices)
        new_elements = []

        def get_midpoint(i1, i2):
            edge = (min(i1, i2), max(i1, i2))
            if edge not in edge_midpoints:
                mid = 0.5 * (vertices[i1] + vertices[i2])
                idx = len(new_vertices)
                new_vertices.append(mid)
                edge_midpoints[edge] = idx
            return edge_midpoints[edge]

        for tri in elements:
            v0, v1, v2 = tri
            m01 = get_midpoint(v0, v1)
            m12 = get_midpoint(v1, v2)
            m20 = get_midpoint(v2, v0)
            new_elements.append([v0, m01, m20])
            new_elements.append([m01, v1, m12])
            new_elements.append([m20, m12, v2])
            new_elements.append([m01, m12, m20])

        vertices = np.array(new_vertices)
        elements = np.array(new_elements, dtype=int)

    return vertices, elements
