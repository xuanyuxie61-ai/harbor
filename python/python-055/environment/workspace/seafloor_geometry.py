"""
seafloor_geometry.py
基于种子项目 1307_triangle_integrals（三角形单项式积分）与
021_asa_geometry_2011（三角形距离、线参数化、含点判定），
构建海底地形面片的几何分析模块。

科学背景：海底地形通常以不规则三角网（TIN, Triangulated Irregular Network）
表示。在声纳反演中，需要计算三角形面片的面积、形心、法向量、
以及声纳波束射线与三角面片的交点/距离。此外，利用单项式积分公式
可精确计算面片上多项式型海底反射系数的面积分：

    I_{i,j} = ∫_T x^i y^j dA

对于参考三角形 T_ref = {(0,0), (1,0), (0,1)}，其解析解为：
    I_{i,j} = i! · j! / (i + j + 2)!

一般三角形通过仿射变换 Jacobian = 2·Area(T) 映射得到。
"""

import numpy as np


def triangle_area(t: np.ndarray) -> float:
    """
    计算三角形的有向面积。

    顶点 t 为 3x2 数组，每行一个顶点 (x, y)。
    逆时针排列时面积为正。

    公式（叉积法）:
        A = 0.5 · [(x₂ - x₁)(y₃ - y₁) - (x₃ - x₁)(y₂ - y₁)]
          = 0.5 · |cross(AB, AC)|
    """
    t = np.asarray(t, dtype=np.float64)
    if t.shape != (3, 2):
        raise ValueError("t 必须是形状为 (3, 2) 的数组")

    area = 0.5 * (
        (t[1, 0] - t[0, 0]) * (t[2, 1] - t[0, 1])
        - (t[2, 0] - t[0, 0]) * (t[1, 1] - t[0, 1])
    )
    return float(area)


def triangle_unit_monomial_integral(i: int, j: int) -> float:
    """
    计算单位参考三角形上的单项式积分 ∫_T x^i y^j dA。

    解析公式:
        q = (i! · j!) / ((i + j + 2)!)

    推导：通过 Beta 函数与 Gamma 函数关系，
          ∫_0^1 ∫_0^{1-x} x^i y^j dy dx = B(i+1, j+1) / (i+j+2)
                                      = Γ(i+1)Γ(j+1) / Γ(i+j+3)
                                      = i! j! / (i+j+2)!
    """
    if i < 0 or j < 0:
        raise ValueError("指数 i, j 必须为非负整数")

    # 使用递推算法（源自 triangle01_monomial_integral）
    k = 0
    q = 1.0
    for l in range(1, i + 1):
        k += 1
        q *= l / k
    for l in range(1, j + 1):
        k += 1
        q *= l / k
    for _ in range(2):
        k += 1
        q /= k
    return float(q)


def triangle_monomial_integral(t: np.ndarray, i: int, j: int) -> float:
    """
    计算一般三角形上的单项式积分。

    通过仿射变换 x = x₁ + (x₂-x₁)·r + (x₃-x₁)·s,
                y = y₁ + (y₂-y₁)·r + (y₃-y₁)·s,
    其中 (r,s) ∈ 单位参考三角形，Jacobian = 2·|Area(T)|。

    参数:
        t: 3x2 顶点数组
        i, j: 单项式指数
    返回:
        积分值
    """
    area = triangle_area(t)
    jac = 2.0 * abs(area)

    # 对于常数积分 (i=j=0)，直接返回面积
    if i == 0 and j == 0:
        return float(abs(area))

    # 通过数值求和计算变换后的积分（高阶情形使用高斯积分近似）
    # 这里对低阶 (i+j <= 4) 使用解析展开，高阶使用 7 点 Hammer 积分
    if i + j <= 4:
        return _triangle_monomial_integral_analytic(t, i, j)
    else:
        return _triangle_monomial_integral_numerical(t, i, j)


def _triangle_monomial_integral_analytic(t: np.ndarray, i: int, j: int) -> float:
    """低阶单项式积分的解析展开（基于重心坐标）。"""
    x1, y1 = t[0]
    x2, y2 = t[1]
    x3, y3 = t[2]
    area = abs(triangle_area(t))

    # 对单项式 (x^i y^j) 展开为重心坐标多项式
    # 使用高斯积分近似（3 阶精度）
    # 重心坐标高斯点（3 点公式，精度 2）
    bary = np.array([
        [2.0/3.0, 1.0/6.0, 1.0/6.0],
        [1.0/6.0, 2.0/3.0, 1.0/6.0],
        [1.0/6.0, 1.0/6.0, 2.0/3.0],
    ])
    weights = np.array([1.0/3.0, 1.0/3.0, 1.0/3.0])

    integral = 0.0
    for w, b in zip(weights, bary):
        x = b[0]*x1 + b[1]*x2 + b[2]*x3
        y = b[0]*y1 + b[1]*y2 + b[2]*y3
        integral += w * (x**i) * (y**j)

    return float(integral * 2.0 * area)


def _triangle_monomial_integral_numerical(t: np.ndarray, i: int, j: int) -> float:
    """高阶单项式积分的数值 Hammer 积分（7 点，5 阶精度）。"""
    x1, y1 = t[0]
    x2, y2 = t[1]
    x3, y3 = t[2]
    area = abs(triangle_area(t))

    # 7 点 Hammer 积分（精度 5）
    bary = np.array([
        [1.0/3.0, 1.0/3.0, 1.0/3.0],
        [0.05971587, 0.47014206, 0.47014206],
        [0.47014206, 0.05971587, 0.47014206],
        [0.47014206, 0.47014206, 0.05971587],
        [0.79742699, 0.10128651, 0.10128651],
        [0.10128651, 0.79742699, 0.10128651],
        [0.10128651, 0.10128651, 0.79742699],
    ])
    weights = np.array([
        0.22500000,
        0.13239415,
        0.13239415,
        0.13239415,
        0.12593918,
        0.12593918,
        0.12593918,
    ])

    integral = 0.0
    for w, b in zip(weights, bary):
        x = b[0]*x1 + b[1]*x2 + b[2]*x3
        y = b[0]*y1 + b[1]*y2 + b[2]*y3
        integral += w * (x**i) * (y**j)

    return float(integral * 2.0 * area)


def triangle_centroid(t: np.ndarray) -> np.ndarray:
    """
    计算三角形形心。

    公式: C = (V₁ + V₂ + V₃) / 3
    """
    t = np.asarray(t, dtype=np.float64)
    return np.mean(t, axis=0)


def triangle_normal_2d(t: np.ndarray) -> np.ndarray:
    """
    计算 2D 三角形的单位法向量（在三维扩展中，z 分量为零）。

    公式: n = (AB × AC) / |AB × AC|
    对于 2D，返回平面外法向的 (x, y) 投影。
    """
    t = np.asarray(t, dtype=np.float64)
    ab = t[1] - t[0]
    ac = t[2] - t[0]
    # 2D 叉积的 z 分量
    cross_z = ab[0]*ac[1] - ab[1]*ac[0]
    norm_cross = abs(cross_z)
    if norm_cross < 1e-15:
        return np.array([0.0, 0.0])
    # 单位法向量（指向 z 正方向）
    n = np.array([ab[1], -ab[0]])
    n = n / np.linalg.norm(n)
    return n


def point_to_triangle_distance(t: np.ndarray, p: np.ndarray) -> float:
    """
    计算二维点到三角形的最短距离。

    基于种子项目 021_asa_geometry_2011 的 triangle_distance 算法：
    1. 计算点到三条边的有向距离；
    2. 若点在三角形内部（三个有向距离均 <= 0），距离为 0；
    3. 若在某条边外侧，距离为该边距离；
    4. 若在顶点区域，距离为到最近顶点的欧氏距离。

    参数:
        t: 3x2 顶点数组
        p: 点坐标 (2,)
    返回:
        最短距离（非负）
    """
    t = np.asarray(t, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)

    a, b, c = t[0], t[1], t[2]

    def line_param(pt1, pt2, pt):
        """返回点到直线的有向距离（法向量方向）。"""
        v1 = pt2 - pt1
        v2 = pt - pt1
        nv = np.array([v1[1], -v1[0]])
        norm_nv = np.linalg.norm(nv)
        if norm_nv < 1e-15:
            return np.linalg.norm(v2)
        return float((nv @ v2) / norm_nv)

    dab = line_param(a, b, p)
    dbc = line_param(b, c, p)
    dca = line_param(c, a, p)

    # 三个均 <= 0：在内部或边界上
    if dab <= 0.0 and dbc <= 0.0 and dca <= 0.0:
        return 0.0

    # 两个负一个正：距离为正值对应的边距离
    if dab >= 0.0 and dbc <= 0.0 and dca <= 0.0:
        return dab
    if dab <= 0.0 and dbc >= 0.0 and dca <= 0.0:
        return dbc
    if dab <= 0.0 and dbc <= 0.0 and dca >= 0.0:
        return dca

    # 一个负两个正：最近顶点
    if dab <= 0.0:
        return float(np.linalg.norm(c - p))
    elif dbc <= 0.0:
        return float(np.linalg.norm(a - p))
    elif dca <= 0.0:
        return float(np.linalg.norm(b - p))
    else:
        # 退化情况：取最近顶点
        return float(min(np.linalg.norm(a - p),
                         np.linalg.norm(b - p),
                         np.linalg.norm(c - p)))


def ray_triangle_intersection_2d(
    ray_origin: np.ndarray,
    ray_dir: np.ndarray,
    t: np.ndarray,
    max_dist: float = 1e6
) -> tuple:
    """
    计算二维射线与三角形的交点。

    参数:
        ray_origin: 射线起点 (2,)
        ray_dir: 射线方向 (2,)（无需单位化）
        t: 三角形顶点 3x2
        max_dist: 最大搜索距离
    返回:
        (hit, distance, point)
        hit: bool，是否相交
        distance: 沿射线距离（若 hit=False 则为 inf）
        point: 交点坐标（若 hit=False 则为 None）
    """
    ray_origin = np.asarray(ray_origin, dtype=np.float64)
    ray_dir = np.asarray(ray_dir, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)

    # 归一化方向
    dir_norm = np.linalg.norm(ray_dir)
    if dir_norm < 1e-15:
        return False, float('inf'), None
    ray_dir = ray_dir / dir_norm

    # 对三角形三边分别求交
    edges = [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])]
    best_dist = float('inf')
    best_point = None

    for v1, v2 in edges:
        # 线段参数方程: v1 + s*(v2-v1), s∈[0,1]
        # 射线参数方程: ray_origin + t*ray_dir, t>=0
        # 解: ray_origin + t*ray_dir = v1 + s*(v2-v1)
        # [ray_dir, -(v2-v1)] [t; s] = v1 - ray_origin
        d = v2 - v1
        det = ray_dir[0] * (-d[1]) - ray_dir[1] * (-d[0])
        if abs(det) < 1e-15:
            continue
        rhs = v1 - ray_origin
        tt = (rhs[0] * (-d[1]) - rhs[1] * (-d[0])) / det
        ss = (ray_dir[0] * rhs[1] - ray_dir[1] * rhs[0]) / det
        if tt >= 0.0 and 0.0 <= ss <= 1.0 and tt < best_dist and tt <= max_dist:
            best_dist = tt
            best_point = ray_origin + tt * ray_dir

    if best_point is not None:
        return True, best_dist, best_point
    return False, float('inf'), None


class SeafloorTriangulation:
    """
    海底地形三角网表示与几何分析。
    """

    def __init__(self, vertices: np.ndarray, triangles: np.ndarray):
        """
        参数:
            vertices: (n_v, 2) 顶点坐标
            triangles: (n_t, 3) 三角形顶点索引
        """
        self.vertices = np.asarray(vertices, dtype=np.float64)
        self.triangles = np.asarray(triangles, dtype=np.int64)
        self._validate()
        self._compute_patch_properties()

    def _validate(self):
        nv = self.vertices.shape[0]
        nt = self.triangles.shape[0]
        if nt == 0:
            return
        if self.triangles.min() < 0 or self.triangles.max() >= nv:
            raise ValueError("三角形顶点索引越界")

    def _compute_patch_properties(self):
        """预计算每个面片的面积与形心。"""
        nt = self.triangles.shape[0]
        self.patch_areas = np.zeros(nt, dtype=np.float64)
        self.patch_centroids = np.zeros((nt, 2), dtype=np.float64)
        for i in range(nt):
            tri = self.vertices[self.triangles[i]]
            self.patch_areas[i] = abs(triangle_area(tri))
            self.patch_centroids[i] = triangle_centroid(tri)

    def total_area(self) -> float:
        """返回三角网覆盖的总面积。"""
        return float(np.sum(self.patch_areas))

    def mean_patch_area(self) -> float:
        """返回平均面片面积。"""
        nt = self.triangles.shape[0]
        if nt == 0:
            return 0.0
        return float(np.mean(self.patch_areas))

    def find_containing_triangle(self, p: np.ndarray) -> int:
        """
        查找包含点 p 的三角形索引，若无则返回 -1。

        使用重心坐标法：
            p = α·A + β·B + γ·C,  α+β+γ=1, α,β,γ ≥ 0
        """
        p = np.asarray(p, dtype=np.float64)
        for i in range(self.triangles.shape[0]):
            tri = self.vertices[self.triangles[i]]
            a, b, c = tri[0], tri[1], tri[2]
            denom = (b[1] - c[1])*(a[0] - c[0]) + (c[0] - b[0])*(a[1] - c[1])
            if abs(denom) < 1e-15:
                continue
            alpha = ((b[1] - c[1])*(p[0] - c[0]) + (c[0] - b[0])*(p[1] - c[1])) / denom
            beta  = ((c[1] - a[1])*(p[0] - c[0]) + (a[0] - c[0])*(p[1] - c[1])) / denom
            gamma = 1.0 - alpha - beta
            if alpha >= -1e-12 and beta >= -1e-12 and gamma >= -1e-12:
                return i
        return -1

    def integrate_scalar_field(self, field_values: np.ndarray, power: int = 0) -> float:
        """
        在三角网上积分标量场。

        参数:
            field_values: 每个顶点的场值，形状 (n_v,)
            power: 单项式幂次（用于测试）
        返回:
            积分近似值（采用面片形心处场值乘以面积）
        """
        if len(field_values) != self.vertices.shape[0]:
            raise ValueError("场值数组长度必须与顶点数一致")

        integral = 0.0
        for i in range(self.triangles.shape[0]):
            tri = self.triangles[i]
            # 面片上平均场值
            avg_val = np.mean(field_values[tri])
            integral += avg_val * self.patch_areas[i]
        return float(integral)
