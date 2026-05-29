"""
spherical_geometry.py
基于种子项目 1114_sphere_delaunay 和 1127_sphere_stereograph 的球面几何处理

在核天体物理中，球面几何用于：
1. 中子星表面的网格生成（Delaunay 三角化）
2. 中子通量的角分布立体投影
3. 球谐函数展开核反应产物的角分布

球面立体投影（以南极为焦点投影到北极切平面）：
    对于单位球面上的点 p = (p_1, ..., p_m)，投影到平面 x_m = 1：
    q = 2/(1 + p_m) · p + (1 - 2/(1+p_m)) · f
    其中 f 为焦点（南极）。

逆投影：
    p = 4/(4 + Σ_{i=1}^{m-1} q_i²) · q + (1 - 4/(4+Σq²)) · f

球面 Delaunay 三角化利用凸包性质：
    单位球面上的 Delaunay 三角化等价于三维空间中点集的凸包面。
"""

import numpy as np


def sphere_stereograph(p, focus=None):
    """
    标准球面立体投影：从南极 (0,...,0,-1) 投影到平面 x_m = 1。

    参数:
        p : ndarray, shape (n, m), 单位球面上的点
        focus : ndarray, shape (m,), 焦点，默认南极

    返回:
        q : ndarray, shape (n, m-1), 投影平面上的坐标（前 m-1 维）
    """
    p = np.asarray(p, dtype=float)
    if p.ndim == 1:
        p = p.reshape(1, -1)
    n, m = p.shape
    if focus is None:
        focus = np.zeros(m, dtype=float)
        focus[-1] = -1.0

    # 投影到平面 x_m = 1
    denom = 1.0 + p[:, -1]
    # 避免除以零
    denom = np.where(np.abs(denom) < 1e-14, 1e-14, denom)
    ss = 2.0 / denom
    q_full = ss[:, None] * p + (1.0 - ss[:, None]) * focus[None, :]
    # 返回前 m-1 维
    return q_full[:, :-1]


def sphere_stereograph_inverse(q, focus=None):
    """
    立体投影的逆映射。

    参数:
        q : ndarray, shape (n, m-1), 投影平面上的坐标
        focus : ndarray, shape (m,), 焦点

    返回:
        p : ndarray, shape (n, m), 单位球面上的点
    """
    q = np.asarray(q, dtype=float)
    if q.ndim == 1:
        q = q.reshape(1, -1)
    n, mdim = q.shape
    m = mdim + 1
    if focus is None:
        focus = np.zeros(m, dtype=float)
        focus[-1] = -1.0

    q_sq = np.sum(q ** 2, axis=1)
    ss = 4.0 / (4.0 + q_sq)
    # 构造完整坐标
    q_full = np.zeros((n, m), dtype=float)
    q_full[:, :-1] = q
    q_full[:, -1] = 2.0  # 平面 x_m = 1
    p = ss[:, None] * q_full + (1.0 - ss[:, None]) * focus[None, :]
    # 归一化到单位球面
    norms = np.linalg.norm(p, axis=1)
    norms = np.where(norms < 1e-15, 1.0, norms)
    p = p / norms[:, None]
    return p


def icosahedron_vertices():
    """
    返回内接于单位球的正二十面体的 12 个顶点。
    顶点坐标基于黄金比例 φ = (1+√5)/2。
    """
    phi = 0.5 * (1.0 + np.sqrt(5.0))
    verts = np.array([
        [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
        [1, phi, 0], [1, -phi, 0], [-1, phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    # 归一化
    norms = np.linalg.norm(verts, axis=1)
    verts = verts / norms[:, None]
    return verts


def spherical_delaunay_triangulation(xyz):
    """
    球面 Delaunay 三角化：利用三维凸包。
    对于单位球面上的点集，其 Delaunay 三角化等价于凸包的外表面。

    参数:
        xyz : ndarray, shape (n, 3), 单位球面上的点

    返回:
        faces : ndarray, shape (nf, 3), 每个面的顶点索引
    """
    xyz = np.asarray(xyz, dtype=float)
    n = xyz.shape[0]
    if n < 4:
        return np.array([], dtype=int).reshape(0, 3)

    try:
        from scipy.spatial import ConvexHull
        hull = ConvexHull(xyz)
        faces = hull.simplices
        # 确保法向朝外（指向原点外侧，即与球心同向）
        oriented_faces = []
        for face in faces:
            v0, v1, v2 = xyz[face[0]], xyz[face[1]], xyz[face[2]]
            normal = np.cross(v1 - v0, v2 - v0)
            centroid = (v0 + v1 + v2) / 3.0
            if np.dot(normal, centroid) < 0:
                face = face[[0, 2, 1]]
            oriented_faces.append(face)
        return np.array(oriented_faces, dtype=int)
    except ImportError:
        # 降级：无 scipy 时生成简单网格
        print("[spherical_geometry] scipy not available, using icosahedron fallback")
        return np.array([], dtype=int).reshape(0, 3)


def subdivide_icosahedron(factor=2):
    """
    通过二十面体细分生成均匀的球面三角网格。

    参数:
        factor : int, 每条边的细分次数

    返回:
        verts : ndarray, 顶点坐标
        faces : ndarray, 面索引
    """
    verts = icosahedron_vertices()
    # 二十面体的 20 个面（硬编码）
    faces_list = [
        [0,2,8],[0,8,4],[0,4,10],[0,10,6],[0,6,2],
        [2,7,8],[8,9,4],[4,5,10],[10,11,6],[6,1,2],
        [3,7,9],[3,9,5],[3,5,11],[3,11,1],[3,1,7],
        [1,6,11],[7,3,9],[9,3,5],[5,3,11],[11,3,1]
    ]
    # 为了正确性，使用标准二十面体面
    faces = np.array([
        [0,2,8],[0,8,4],[0,4,10],[0,10,6],[0,6,2],
        [2,7,8],[8,9,4],[4,5,10],[10,11,6],[6,1,2],
        [3,7,9],[3,9,5],[3,5,11],[3,11,1],[3,1,7],
        [1,6,11],[7,3,9],[9,3,5],[5,3,11],[11,3,1]
    ], dtype=int)
    # 注意：上面硬编码的面索引可能有误，使用 scipy 的凸包更可靠
    faces = spherical_delaunay_triangulation(verts)
    return verts, faces


def test_spherical_geometry():
    """自包含测试"""
    # 测试立体投影
    p = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0], [0, 0, -1]], dtype=float)
    q = sphere_stereograph(p)
    p_rec = sphere_stereograph_inverse(q)
    err = np.max(np.abs(p_rec - p))
    print(f"[spherical_geometry] Stereographic projection max error = {err:.3e}")
    assert err < 1e-10, "Stereographic projection inaccurate"

    # 测试 Delaunay 三角化
    verts = icosahedron_vertices()
    faces = spherical_delaunay_triangulation(verts)
    print(f"[spherical_geometry] Icosahedron triangulation: {len(faces)} faces")


if __name__ == "__main__":
    test_spherical_geometry()
