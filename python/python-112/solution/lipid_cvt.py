"""
lipid_cvt.py
=============
基于 Centroidal Voronoi Tessellation (CVT) 的膜脂双层与周围水分子
空间分布优化模块。

核心数学内容：
  - Lloyd 算法：迭代更新生成元位置至 Voronoi 区域的质心
      $g_i^{(k+1)} = \frac{\int_{V_i} x \rho(x) \, dx}{\int_{V_i} \rho(x) \, dx}$
  - 2D 三角形区域内的 CVT（膜脂头基平面分布）
  - 3D 盒域内的密度加权 CVT（水分子/脂酰链尾部的三维排布）
  - 能量泛函收敛性监测

种子项目映射：
  - 262_cvt_triangle_uniform  →  2D 三角形 CVT
  - 249_cvt_3d_lumping        →  3D 密度加权 CVT
"""

import numpy as np
from typing import Tuple, Callable, Optional


# ---------------------------------------------------------------------------
# 2D 三角形 CVT（种子项目 262_cvt_triangle_uniform）
# ---------------------------------------------------------------------------
def cvt_triangle_uniform(
    triangle: np.ndarray,
    n: int,
    sample_num: int,
    it_num: int,
    density: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    在三角形区域内计算 CVT 生成元位置与 Delaunay 三角剖分。

    参数：
        triangle    : shape (3, 2) 的顶点数组
        n           : 生成元数量，>= 3
        sample_num  : 每次迭代的采样点数
        it_num      : Lloyd 迭代次数
        density     : 密度函数 $\rho(x)$，接受 shape (m, 2) 返回 shape (m,)

    返回：
        p           : shape (n, 2) 的生成元位置
        tri         : Delaunay 三角剖分索引 (m, 3)

    算法流程：
        1. 在三角形内均匀随机采样 sample_num 个点
        2. 对每个采样点，找到最近的生成元（Voronoi 区域归属）
        3. 按密度加权更新生成元为各区域的质心
        4. 重复 it_num 次
    """
    if triangle.shape != (3, 2):
        raise ValueError("cvt_triangle_uniform: triangle must have shape (3, 2).")
    if n < 3:
        raise ValueError("cvt_triangle_uniform: n must be >= 3.")
    if sample_num < n:
        raise ValueError("cvt_triangle_uniform: sample_num must be >= n.")
    if it_num < 1:
        raise ValueError("cvt_triangle_uniform: it_num must be >= 1.")

    # 初始化：在三角形内均匀随机放置生成元
    p = _sample_triangle_uniform(triangle, n)

    for it in range(it_num):
        # 在三角形内采样
        s = _sample_triangle_uniform(triangle, sample_num)

        # 找到每个采样点最近的生成元
        # 使用向量化距离计算
        dists = np.linalg.norm(s[:, None, :] - p[None, :, :], axis=2)  # (sample_num, n)
        nearest = np.argmin(dists, axis=1)  # (sample_num,)

        # 按密度加权计算新质心
        if density is not None:
            rho_s = density(s)
            rho_s = np.clip(rho_s, 1.0e-12, None)
        else:
            rho_s = np.ones(sample_num, dtype=float)

        p_new = np.zeros_like(p)
        mass = np.zeros(n, dtype=float)
        for i in range(n):
            mask = nearest == i
            count = np.count_nonzero(mask)
            if count > 0:
                mass[i] = np.sum(rho_s[mask])
                p_new[i] = np.sum(s[mask] * rho_s[mask][:, None], axis=0) / mass[i]
            else:
                # 空区域：重新随机放置
                p_new[i] = _sample_triangle_uniform(triangle, 1)[0]

        p = p_new

    # 计算 Delaunay 三角剖分（使用 scipy）
    from scipy.spatial import Delaunay
    tri = Delaunay(p)

    return p, tri.simplices


def _sample_triangle_uniform(triangle: np.ndarray, n: int) -> np.ndarray:
    """
    在三角形内均匀随机采样 n 个点。

    数学方法：
        取 $\alpha = \sqrt{U_1}$, $\beta = U_2$，其中 $U_1, U_2 \sim \text{Uniform}(0,1)$。
        则点 $P = (1-\alpha) V_1 + \alpha((1-\beta) V_2 + \beta V_3)$ 在三角形内均匀分布。
    """
    alpha = np.sqrt(np.random.rand(n))
    beta = np.random.rand(n)

    p12 = (1.0 - alpha)[:, None] * triangle[0] + alpha[:, None] * triangle[1]
    p13 = (1.0 - alpha)[:, None] * triangle[0] + alpha[:, None] * triangle[2]

    p = (1.0 - beta)[:, None] * p12 + beta[:, None] * p13
    return p


# ---------------------------------------------------------------------------
# 3D 密度加权 CVT（种子项目 249_cvt_3d_lumping）
# ---------------------------------------------------------------------------
def cvt_3d_lumping(
    n: int,
    it_num: int,
    s_num: int,
    mu_fun: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
    box: Tuple[float, float, float, float, float, float] = (-1.0, 1.0, -1.0, 1.0, -1.0, 1.0),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    在三维盒域内进行密度加权的 Lloyd CVT 迭代。

    参数：
        n      : 生成元数量，>= 4
        it_num : 迭代次数
        s_num  : 每个维度的采样网格数（总采样点数 = s_num^3）
        mu_fun : 密度函数 $\mu(x,y,z)$，接受三维网格数组返回同形数组
        box    : (xmin, xmax, ymin, ymax, zmin, zmax)

    返回：
        g      : shape (n, 3) 的生成元位置
        energy : shape (it_num,) 的每次迭代能量
        motion : shape (it_num,) 的生成元平均位移

    数学内容：
        能量泛函：
            $E = \sum_{i} \int_{V_i} \rho(x) \|x - g_i\|^2 \, dx$
        其中 $V_i$ 为第 $i$ 个 Voronoi 单元。
    """
    if n < 4:
        raise ValueError("cvt_3d_lumping: n must be >= 4.")
    if it_num < 1:
        raise ValueError("cvt_3d_lumping: it_num must be >= 1.")
    if s_num < 2:
        raise ValueError("cvt_3d_lumping: s_num must be >= 2.")

    xmin, xmax, ymin, ymax, zmin, zmax = box

    # 初始化生成元
    g = np.zeros((n, 3), dtype=float)
    g[:, 0] = np.random.uniform(xmin, xmax, n)
    g[:, 1] = np.random.uniform(ymin, ymax, n)
    g[:, 2] = np.random.uniform(zmin, zmax, n)

    # 构造均匀采样网格（避开边界 epsilon）
    eps = 1.0e-12
    s_1d_x = np.linspace(xmin + eps, xmax - eps, s_num)
    s_1d_y = np.linspace(ymin + eps, ymax - eps, s_num)
    s_1d_z = np.linspace(zmin + eps, zmax - eps, s_num)

    sx, sy, sz = np.meshgrid(s_1d_x, s_1d_y, s_1d_z, indexing='ij')
    sx_vec = sx.ravel()
    sy_vec = sy.ravel()
    sz_vec = sz.ravel()
    s = np.column_stack((sx_vec, sy_vec, sz_vec))

    # 计算密度（截断避免奇点）
    mu_mat = mu_fun(sx, sy, sz)
    mu_mat = np.clip(mu_mat, 1.0e-12, 10.0)
    r_vec = mu_mat.ravel() ** 5  # 3D 中密度与生成元分布的关系指数

    energy = np.full(it_num, np.nan, dtype=float)
    motion = np.full(it_num, np.nan, dtype=float)

    g_new = np.zeros_like(g)

    for it in range(it_num):
        # 使用 scipy.spatial.cKDTree 进行快速最近邻搜索
        from scipy.spatial import cKDTree
        tree = cKDTree(g)
        k = tree.query(s, k=1)[1]  # 最近生成元索引

        # 质量加权平均
        m = np.zeros(n, dtype=float)
        g_new[:, 0] = 0.0
        g_new[:, 1] = 0.0
        g_new[:, 2] = 0.0

        for idx in range(n):
            mask = k == idx
            m[idx] = np.sum(r_vec[mask])
            if m[idx] > 0:
                g_new[idx, 0] = np.sum(r_vec[mask] * s[mask, 0]) / m[idx]
                g_new[idx, 1] = np.sum(r_vec[mask] * s[mask, 1]) / m[idx]
                g_new[idx, 2] = np.sum(r_vec[mask] * s[mask, 2]) / m[idx]
            else:
                # 空区域重新随机化
                g_new[idx] = [
                    np.random.uniform(xmin, xmax),
                    np.random.uniform(ymin, ymax),
                    np.random.uniform(zmin, zmax),
                ]

        # 能量 = 质量加权距离平方和 / s_num
        diff = s - g[k]
        energy[it] = np.sum(r_vec * np.sum(diff ** 2, axis=1)) / s_num

        # 生成元平均位移
        motion[it] = np.mean(np.sum((g_new - g) ** 2, axis=1))

        g = g_new.copy()

    return g, energy, motion


# ---------------------------------------------------------------------------
# 膜脂双层 CVT 布置器
# ---------------------------------------------------------------------------
def place_lipid_bilayer(
    n_lipids_per_leaflet: int = 50,
    protein_radius: float = 15.0,  # Å
    box_xy: float = 60.0,          # Å
    exclusion_radius: float = 18.0,  # Å
    it_num: int = 30,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    使用 CVT 在蛋白周围布置脂双层（上下两个 leaflet）。

    返回：
        upper_leaflet : shape (n, 2) 上 leaflet 脂质头基坐标 (x, y)
        lower_leaflet : shape (n, 2) 下 leaflet 脂质头基坐标 (x, y)
        upper_z       : 上 leaflet z 坐标（通常 +15 Å）
        lower_z       : 下 leaflet z 坐标（通常 -15 Å）
    """
    if n_lipids_per_leaflet < 3:
        raise ValueError("place_lipid_bilayer: n_lipids_per_leaflet must be >= 3.")
    if protein_radius < 0 or exclusion_radius < protein_radius:
        raise ValueError("place_lipid_bilayer: exclusion_radius must be > protein_radius >= 0.")

    half = box_xy / 2.0

    # 定义正方形域（除去中心圆形 exclusion zone）
    # 使用近似：在四个角上分别做 CVT，然后合并
    # 为简化，我们使用整个正方形做 CVT，但在密度函数中把中心区域设为零

    def lipid_density(pts: np.ndarray) -> np.ndarray:
        """中心排除区域的密度函数。"""
        d = np.linalg.norm(pts, axis=1)
        rho = np.ones(pts.shape[0], dtype=float)
        rho[d < exclusion_radius] = 0.0
        return rho

    # 四个象限的三角形（近似正方形域）
    tri1 = np.array([[0.0, 0.0], [half, 0.0], [half, half]], dtype=float)
    tri2 = np.array([[0.0, 0.0], [half, half], [0.0, half]], dtype=float)
    tri3 = np.array([[0.0, 0.0], [0.0, -half], [half, -half]], dtype=float)
    tri4 = np.array([[0.0, 0.0], [half, -half], [half, 0.0]], dtype=float)

    def run_cvt_for_triangle(tri, n_sub):
        p, _ = cvt_triangle_uniform(
            tri, n_sub, sample_num=2000 * n_sub, it_num=it_num, density=lipid_density
        )
        return p

    n_sub = max(3, n_lipids_per_leaflet // 4)
    p1 = run_cvt_for_triangle(tri1, n_sub)
    p2 = run_cvt_for_triangle(tri2, n_sub)
    p3 = run_cvt_for_triangle(tri3, n_sub)
    p4 = run_cvt_for_triangle(tri4, n_sub)

    # 合并并镜像到四个象限
    upper_all = np.vstack([p1, p2, p3, p4])
    # 过滤掉 exclusion zone 内的点
    mask = np.linalg.norm(upper_all, axis=1) >= exclusion_radius
    upper_all = upper_all[mask]

    # 随机选取指定数量的脂质
    if upper_all.shape[0] > n_lipids_per_leaflet:
        idx = np.random.choice(upper_all.shape[0], n_lipids_per_leaflet, replace=False)
        upper_leaflet = upper_all[idx]
    else:
        upper_leaflet = upper_all

    # 下 leaflet 对称
    lower_leaflet = upper_leaflet.copy()

    upper_z = 15.0
    lower_z = -15.0

    return upper_leaflet, lower_leaflet, upper_z, lower_z
