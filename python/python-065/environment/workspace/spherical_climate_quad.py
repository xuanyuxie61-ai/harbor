"""
spherical_climate_quad.py

基于 1126_sphere_quad 核心算法的球面气候求积模块。

原项目提供了多种单位球面上的数值积分方法，包括：
- 基于二十面体细分的重心求积规则（icos1c）
- 蒙特卡洛求积（mc）

在本气候归因框架中，球面求积用于计算：
1. 全球平均辐射强迫（W/m²）
2. 全球能量收支平衡
3. 极端事件对全球气候系统的影响积分

核心公式：
- 单位球面面积：A = 4π
- 球面三角形面积（L'Huilier 定理）：
    tan(E/4) = sqrt( tan(s/2) * tan((s-a)/2) * tan((s-b)/2) * tan((s-c)/2) )
    其中 a,b,c 为球面三角形的边长（大圆弧），s = (a+b+c)/2，E 为球面过剩。
- 球面重心求积：
    ∫_{S^2} f(Ω) dΩ ≈ Σ_i A_i * f(c_i)
    其中 A_i 为球面三角形面积，c_i 为其重心投影到球面上的点。
- 球面蒙特卡洛：
    ∫_{S^2} f(Ω) dΩ ≈ 4π * (1/N) * Σ_{k=1}^N f(x_k)
"""

import numpy as np


def icosahedron_shape():
    """
    生成单位球内接二十面体的顶点和面。

    二十面体有 12 个顶点、30 条边、20 个三角形面。
    黄金比例 φ = (1 + sqrt(5)) / 2。
    """
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    # 12 个顶点（归一化到单位球）
    verts = np.array([
        [-1,  phi,  0], [ 1,  phi,  0], [-1, -phi,  0], [ 1, -phi,  0],
        [ 0, -1,  phi], [ 0,  1,  phi], [ 0, -1, -phi], [ 0,  1, -phi],
        [ phi,  0, -1], [ phi,  0,  1], [-phi,  0, -1], [-phi,  0,  1],
    ], dtype=np.float64)
    verts /= np.linalg.norm(verts, axis=1, keepdims=True)

    faces = np.array([
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
    ], dtype=np.int64)
    return verts, faces


def sphere01_distance_xyz(xyz1, xyz2):
    """球面上两点的大圆弧距离（基于 1126 的 sphere01_distance_xyz）。"""
    # 使用稳定公式：d = 2 * arcsin( |v1 - v2| / 2 )
    diff = xyz1 - xyz2
    chord = np.sqrt(np.sum(diff ** 2))
    return 2.0 * np.arcsin(min(chord / 2.0, 1.0))


def sphere01_triangle_vertices_to_area(a_xyz, b_xyz, c_xyz):
    """
    计算球面三角形面积（基于 1126 的 sphere01_triangle_vertices_to_area）。

    使用 L'Huilier 定理计算球面过剩 E = A + B + C - π，
    球面面积 = E * R^2（R=1 时面积=E）。
    """
    a = sphere01_distance_xyz(b_xyz, c_xyz)
    b = sphere01_distance_xyz(a_xyz, c_xyz)
    c = sphere01_distance_xyz(a_xyz, b_xyz)

    s = 0.5 * (a + b + c)
    # 边界处理
    if s <= 0.0 or s >= np.pi:
        return 0.0
    # L'Huilier 公式
    try:
        tan_e4 = np.sqrt(
            max(0.0,
                np.tan(s / 2.0)
                * np.tan((s - a) / 2.0)
                * np.tan((s - b) / 2.0)
                * np.tan((s - c) / 2.0))
        )
    except ValueError:
        return 0.0
    e = 4.0 * np.arctan(tan_e4)
    return max(e, 0.0)


def sphere01_triangle_project(a_xyz, b_xyz, c_xyz, f1, f2, f3):
    """
    将重心坐标 (f1,f2,f3) 投影到球面上。

    基于 1126 的 sphere01_triangle_project：
        v = f1*a + f2*b + f3*c，然后归一化到单位球。
    """
    v = f1 * np.array(a_xyz) + f2 * np.array(b_xyz) + f3 * np.array(c_xyz)
    norm = np.linalg.norm(v)
    if norm < 1e-14:
        return np.array(a_xyz)
    return v / norm


def sphere01_quad_icos1c(factor, fun):
    """
    二十面体细分重心求积（基于 1126 的 sphere01_quad_icos1c）。

    Parameters
    ----------
    factor : int
        细分因子（>=1），每条边分成 factor 段。
    fun : callable
        被积函数 fun(xyz)，xyz  shape (3,) 或 (3, N)。

    Returns
    -------
    result : float
        积分近似值。
    node_num : int
        求积节点数。
    """
    if factor < 1:
        factor = 1
    verts, faces = icosahedron_shape()
    result = 0.0
    area_total = 0.0
    node_num = 0

    for face in faces:
        a = verts[face[0]]
        b = verts[face[1]]
        c = verts[face[2]]

        # 同向子三角形
        for f3 in range(1, 3 * factor - 1, 3):
            for f2 in range(1, 3 * factor - f3 - 1, 3):
                f1 = 3 * factor - f3 - f2
                node_xyz = sphere01_triangle_project(a, b, c, f1, f2, f3)
                a2 = sphere01_triangle_project(a, b, c, f1 + 2, f2 - 1, f3 - 1)
                b2 = sphere01_triangle_project(a, b, c, f1 - 1, f2 + 2, f3 - 1)
                c2 = sphere01_triangle_project(a, b, c, f1 - 1, f2 - 1, f3 + 2)
                area = sphere01_triangle_vertices_to_area(a2, b2, c2)
                v = fun(node_xyz)
                node_num += 1
                result += area * v
                area_total += area

        # 反向子三角形
        for f3 in range(2, 3 * factor - 3, 3):
            for f2 in range(2, 3 * factor - f3 - 2, 3):
                f1 = 3 * factor - f3 - f2
                node_xyz = sphere01_triangle_project(a, b, c, f1, f2, f3)
                a2 = sphere01_triangle_project(a, b, c, f1 - 2, f2 + 1, f3 + 1)
                b2 = sphere01_triangle_project(a, b, c, f1 + 1, f2 - 2, f3 + 1)
                c2 = sphere01_triangle_project(a, b, c, f1 + 1, f2 + 1, f3 - 2)
                area = sphere01_triangle_vertices_to_area(a2, b2, c2)
                v = fun(node_xyz)
                node_num += 1
                result += area * v
                area_total += area

    return result, node_num


def sphere01_sample_3d(n, seed=None):
    """
    在单位球面上均匀随机采样（基于 1126 的 sphere01_sample_3d）。

    使用球坐标：θ ∈ [0, 2π)，φ = arccos(2u - 1)，其中 u ~ U(0,1)。
    """
    rng = np.random.default_rng(seed)
    phi = 2.0 * np.pi * rng.random(n)
    z = 2.0 * rng.random(n) - 1.0
    r = np.sqrt(1.0 - z ** 2)
    x = r * np.cos(phi)
    y = r * np.sin(phi)
    return np.vstack([x, y, z]).T


def sphere01_quad_mc(fun, n, seed=None):
    """
    球面蒙特卡洛求积（基于 1126 的 sphere01_quad_mc）。

    公式：∫_{S^2} f dΩ ≈ 4π * (1/N) Σ f(x_k)
    """
    x = sphere01_sample_3d(n, seed)
    v = np.array([fun(x[k]) for k in range(n)])
    result = 4.0 * np.pi * np.mean(v)
    return result


def global_radiation_forcing_integral(forcing_field_latlon, factor=2):
    """
    使用球面求积计算全球平均辐射强迫。

    Parameters
    ----------
    forcing_field_latlon : callable
        函数 f(lat, lon) -> float，接受弧度值。
    factor : int
        二十面体细分因子。

    Returns
    -------
    mean_forcing : float
        全球平均辐射强迫（W/m²）。
    """
    def fun(xyz):
        # xyz -> lat, lon
        lat = np.arcsin(np.clip(xyz[2], -1.0, 1.0))
        lon = np.arctan2(xyz[1], xyz[0])
        return forcing_field_latlon(lat, lon)

    total, node_num = sphere01_quad_icos1c(factor, fun)
    # 球面总面积 = 4π，因此平均值 = total / (4π)
    mean_forcing = total / (4.0 * np.pi)
    return mean_forcing, node_num


def test_spherical():
    # 测试：积分常数 1 应等于 4π
    def fun1(xyz):
        return 1.0
    val, nn = sphere01_quad_icos1c(2, fun1)
    assert abs(val - 4.0 * np.pi) < 0.1
    print("spherical_climate_quad 自测试通过")


if __name__ == "__main__":
    test_spherical()
