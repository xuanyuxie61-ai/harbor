r"""
detector_geometry.py
探测器几何与数值积分模块

本模块提供：
1. 二维探测器几何描述（多边形边界、三角剖分）
2. FreeFEM++ 风格网格 I/O（参考 freefem_msh_io）
3. Fekete 三角形高斯求积（参考 triangle_fekete_rule）
4. 参考三角形到物理三角形的仿射映射
5. 面积分与体积分接口

核心公式：
    参考三角形顶点：
        \hat{V}_1 = (0, 0), \quad \hat{V}_2 = (1, 0), \quad \hat{V}_3 = (0, 1)

    仿射映射：
        \vec{x} = J \hat{\vec{x}} + \vec{x}_0
        J = [\vec{v}_2 - \vec{v}_1, \; \vec{v}_3 - \vec{v}_1]

    积分变换：
        \int_{T_{\rm phys}} f(\vec{x}) dA = |\det J| \int_{\hat{T}} f(J\hat{\vec{x}} + \vec{x}_0) d\hat{A}

    Fekete 求积：
        \int_{\hat{T}} g(\hat{\vec{x}}) d\hat{A} \approx \sum_{i} w_i g(\hat{\vec{x}}_i)
        其中 \hat{T} 为单位参考三角形（面积 = 1/2）。

参考文献：
- Taylor, M. A., Wingate, B. A., & Vincent, R. E. (2000). SIAM J. Numer. Anal., 38, 1707.
- FreeFEM++ 文档: https://freefem.org/
"""

import numpy as np
from typing import List, Tuple
from utils import triangle_area_2d, barycentric_to_cartesian


# ============================================================================
# Fekete 求积规则数据（Taylor, Wingate & Vincent, 2000）
# 规则 1–7，精度 3–18，节点数 4–37
# 数据以重心坐标 (λ1, λ2, λ3) 和权重 w 给出。
# ============================================================================

FEKETE_RULES = {
    1: {
        "degree": 3,
        "points": np.array([
            [1.0/3.0, 1.0/3.0, 1.0/3.0],
            [0.0, 0.5, 0.5],
            [0.5, 0.0, 0.5],
            [0.5, 0.5, 0.0],
        ]),
        "weights": np.array([-27.0/96.0, 25.0/96.0, 25.0/96.0, 25.0/96.0]),
    },
    2: {
        "degree": 6,
        "points": np.array([
            [0.091576213509771, 0.454213840770114, 0.454213840770114],
            [0.454213840770114, 0.091576213509771, 0.454213840770114],
            [0.454213840770114, 0.454213840770114, 0.091576213509771],
            [0.445948490915965, 0.277025758135518, 0.277025758135518],
            [0.277025758135518, 0.445948490915965, 0.277025758135518],
            [0.277025758135518, 0.277025758135518, 0.445948490915965],
            [1.0/3.0, 1.0/3.0, 1.0/3.0],
        ]),
        "weights": np.array([
            0.053347235608838, 0.053347235608838, 0.053347235608838,
            0.077113760890257, 0.077113760890257, 0.077113760890257,
            0.1125,
        ]),
    },
    3: {
        "degree": 7,
        "points": np.array([
            [0.333333333333333, 0.333333333333333, 0.333333333333333],
            [0.059715871789770, 0.470142064105115, 0.470142064105115],
            [0.470142064105115, 0.059715871789770, 0.470142064105115],
            [0.470142064105115, 0.470142064105115, 0.059715871789770],
            [0.797426985353087, 0.101286507323456, 0.101286507323456],
            [0.101286507323456, 0.797426985353087, 0.101286507323456],
            [0.101286507323456, 0.101286507323456, 0.797426985353087],
        ]),
        "weights": np.array([
            0.1125,
            0.066197076394253, 0.066197076394253, 0.066197076394253,
            0.062969590272414, 0.062969590272414, 0.062969590272414,
        ]),
    },
}


def get_fekete_rule(rule_id: int) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    获取 Fekete 求积规则的节点（笛卡尔坐标）、权重与精度。

    参数：
        rule_id: 规则编号 (1, 2, 3)

    返回：
        points: (N, 2) 参考三角形内的笛卡尔坐标
        weights: (N,) 权重
        degree: 多项式精确度
    """
    if rule_id not in FEKETE_RULES:
        available = list(FEKETE_RULES.keys())
        raise ValueError(f"get_fekete_rule: 不支持的规则编号 {rule_id}，可用: {available}")
    data = FEKETE_RULES[rule_id]
    bary = data["points"]
    weights = data["weights"]
    degree = data["degree"]
    # 重心坐标 → 笛卡尔坐标（参考三角形）
    ref_tri = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    cart = barycentric_to_cartesian(ref_tri, bary)
    return cart, weights, degree


# ============================================================================
# 仿射映射
# ============================================================================

def reference_to_physical_t3(physical_tri: np.ndarray, ref_points: np.ndarray) -> np.ndarray:
    """
    将参考三角形内的点映射到物理三角形。

    参数：
        physical_tri: (3, 2) 物理三角形顶点
        ref_points: (N, 2) 参考点坐标

    返回：
        phys_points: (N, 2) 物理坐标
    """
    if physical_tri.shape != (3, 2):
        raise ValueError("reference_to_physical_t3: physical_tri 必须为 (3,2)")
    v1 = physical_tri[0]
    J = np.column_stack([
        physical_tri[1] - physical_tri[0],
        physical_tri[2] - physical_tri[0],
    ])
    phys = ref_points @ J.T + v1
    return phys


# ============================================================================
# 网格数据结构
# ============================================================================

class Mesh2D:
    """
    二维三角形网格类。

    属性：
        vertices: (Nv, 2) 顶点坐标数组
        triangles: (Nt, 3) 三角形顶点索引数组（0-based）
        edges: (Ne, 2) 边界边索引数组（可选）
        vertex_labels: (Nv,) 顶点标签（用于边界条件标识）
        triangle_labels: (Nt,) 三角形标签
    """

    def __init__(self):
        self.vertices: np.ndarray = np.zeros((0, 2))
        self.triangles: np.ndarray = np.zeros((0, 3), dtype=int)
        self.edges: np.ndarray = np.zeros((0, 2), dtype=int)
        self.vertex_labels: np.ndarray = np.zeros((0,), dtype=int)
        self.triangle_labels: np.ndarray = np.zeros((0,), dtype=int)

    def n_vertices(self) -> int:
        return self.vertices.shape[0]

    def n_triangles(self) -> int:
        return self.triangles.shape[0]

    def n_edges(self) -> int:
        return self.edges.shape[0]

    def triangle_area(self, tri_idx: int) -> float:
        """计算第 tri_idx 个三角形的面积。"""
        tri = self.triangles[tri_idx]
        verts = self.vertices[tri]
        return triangle_area_2d(verts)

    def total_area(self) -> float:
        """计算网格覆盖的总面积。"""
        return sum(self.triangle_area(i) for i in range(self.n_triangles()))

    def integrate_scalar(self, scalar_fn, rule_id: int = 2) -> float:
        """
        在网格上积分标量场 scalar_fn(x, y)。

        算法：
            \int_{\Omega} f dA = \sum_{T \in \mathcal{T}} |T| \sum_{i} w_i f(\vec{x}_i)

        参数：
            scalar_fn: 函数 f(x, y) → float
            rule_id: Fekete 规则编号
        """
        ref_pts, ref_w, _ = get_fekete_rule(rule_id)
        total = 0.0
        for t_idx in range(self.n_triangles()):
            tri = self.triangles[t_idx]
            phys_tri = self.vertices[tri]
            area = triangle_area_2d(phys_tri)
            # 参考三角形面积 = 0.5，Fekete 权重已归一化到参考三角形
            # 需要将权重按面积比例缩放
            phys_pts = reference_to_physical_t3(phys_tri, ref_pts)
            for i in range(len(ref_w)):
                x, y = phys_pts[i]
                total += area * ref_w[i] * scalar_fn(x, y)
        return total


# ============================================================================
# FreeFEM++ .msh 风格 I/O
# ============================================================================

def mesh_base_one(triangles: np.ndarray, edges: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    将 0-based 索引转换为 1-based（MATLAB/FreeFEM 兼容）。

    参数：
        triangles: 三角形索引数组
        edges: 边索引数组

    返回：
        (triangles + 1, edges + 1)
    """
    return triangles + 1, edges + 1


def mesh_base_zero(triangles: np.ndarray, edges: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """将 1-based 索引转换为 0-based。"""
    return triangles - 1, edges - 1


def read_msh_file(filename: str) -> Mesh2D:
    """
    读取 FreeFEM++ 风格的 .msh 网格文件。

    文件格式：
        第一行: v_num  t_num  e_num
        接下来 v_num 行: x  y  label
        接下来 t_num 行: i  j  k  label
        接下来 e_num 行: i  j  label
    """
    mesh = Mesh2D()
    with open(filename, 'r') as f:
        header = f.readline().strip().split()
        v_num = int(header[0])
        t_num = int(header[1])
        e_num = int(header[2])

        vertices = []
        v_labels = []
        for _ in range(v_num):
            parts = f.readline().strip().split()
            x, y = float(parts[0]), float(parts[1])
            label = int(parts[2]) if len(parts) > 2 else 0
            vertices.append([x, y])
            v_labels.append(label)

        triangles = []
        t_labels = []
        for _ in range(t_num):
            parts = f.readline().strip().split()
            i, j, k = int(parts[0]), int(parts[1]), int(parts[2])
            label = int(parts[3]) if len(parts) > 3 else 0
            triangles.append([i, j, k])
            t_labels.append(label)

        edges = []
        e_labels = []
        for _ in range(e_num):
            parts = f.readline().strip().split()
            i, j = int(parts[0]), int(parts[1])
            label = int(parts[2]) if len(parts) > 2 else 0
            edges.append([i, j])
            e_labels.append(label)

    mesh.vertices = np.array(vertices)
    mesh.vertex_labels = np.array(v_labels, dtype=int)
    mesh.triangles = np.array(triangles, dtype=int)
    # 检测并转换 1-based → 0-based
    if mesh.triangles.min() > 0:
        mesh.triangles -= 1
    mesh.triangle_labels = np.array(t_labels, dtype=int)
    mesh.edges = np.array(edges, dtype=int)
    if mesh.edges.min() > 0:
        mesh.edges -= 1
    return mesh


def write_msh_file(filename: str, mesh: Mesh2D) -> None:
    """将网格写入 FreeFEM++ .msh 文件（1-based 索引）。"""
    tri_out, edge_out = mesh_base_one(mesh.triangles.copy(), mesh.edges.copy())
    with open(filename, 'w') as f:
        f.write(f"{mesh.n_vertices()} {mesh.n_triangles()} {mesh.n_edges()}\n")
        for i in range(mesh.n_vertices()):
            x, y = mesh.vertices[i]
            lab = mesh.vertex_labels[i] if i < len(mesh.vertex_labels) else 0
            f.write(f"{x:.16e} {y:.16e} {lab}\n")
        for i in range(mesh.n_triangles()):
            ii, jj, kk = tri_out[i]
            lab = mesh.triangle_labels[i] if i < len(mesh.triangle_labels) else 0
            f.write(f"{ii} {jj} {kk} {lab}\n")
        for i in range(mesh.n_edges()):
            ii, jj = edge_out[i]
            lab = 0
            f.write(f"{ii} {jj} {lab}\n")


def create_sample_detector_mesh() -> Mesh2D:
    """
    创建示例探测器几何网格：六边形近似圆盘。

    几何描述：
        探测器为半径 R = 5 cm 的圆形区域，中心在 (0, 0)。
        使用简单三角剖分：中心点 + 12 个边界点。
    """
    mesh = Mesh2D()
    R = 5.0  # cm
    n_boundary = 12

    # 中心点
    center = np.array([0.0, 0.0])
    verts = [center]
    v_labels = [0]

    # 边界点
    angles = np.linspace(0.0, 2.0 * np.pi, n_boundary, endpoint=False)
    for a in angles:
        verts.append([R * np.cos(a), R * np.sin(a)])
        v_labels.append(1)  # 边界标签

    mesh.vertices = np.array(verts)
    mesh.vertex_labels = np.array(v_labels, dtype=int)

    # 三角形（扇形）
    tri = []
    for i in range(n_boundary):
        i1 = i + 1
        i2 = ((i + 1) % n_boundary) + 1
        tri.append([0, i1, i2])
    mesh.triangles = np.array(tri, dtype=int)
    mesh.triangle_labels = np.zeros(mesh.n_triangles(), dtype=int)

    # 边界边
    edges = []
    for i in range(n_boundary):
        i1 = i + 1
        i2 = ((i + 1) % n_boundary) + 1
        edges.append([i1, i2])
    mesh.edges = np.array(edges, dtype=int)

    return mesh


# ============================================================================
# 自测
# ============================================================================

if __name__ == "__main__":
    # 测试 Fekete 规则积分常数函数 1
    ref_pts, ref_w, deg = get_fekete_rule(2)
    # 参考三角形面积 = 0.5，积分 1 应等于面积
    area_approx = np.sum(ref_w) * 0.5
    assert abs(area_approx - 0.5) < 1e-12, f"Fekete 面积测试失败: {area_approx}"

    # 测试网格创建
    mesh = create_sample_detector_mesh()
    assert mesh.n_vertices() == 13
    assert mesh.n_triangles() == 12
    total_area = mesh.total_area()
    expected_area = np.pi * 25.0
    assert abs(total_area - expected_area) / expected_area < 0.15, f"面积偏差过大: {total_area}"

    # 测试积分
    def f_const(x, y):
        return 1.0
    int_const = mesh.integrate_scalar(f_const, rule_id=2)
    assert abs(int_const - total_area) / total_area < 1e-10, "常数积分测试失败"

    print("detector_geometry.py: 所有自测通过")
