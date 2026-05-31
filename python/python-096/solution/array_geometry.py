"""
array_geometry.py
=================
天线阵列几何生成与网格质量评估模块

核心算法来源：
  - 1123_sphere_llt_grid：球面经纬度网格生成
  - 308_distmesh：基于距离函数的二维网格生成
  - 1236_tet_mesh_quality：四面体网格质量度量

在电磁学波束赋形中的角色：
  1. 使用球面经纬网格生成三维共形阵列（conformal array）单元坐标
  2. 基于距离函数思想在口径面生成自适应非均匀网格
  3. 引入四面体质量指标评估阵列单元分布的均匀性与病态程度
"""

import numpy as np
from typing import Tuple, Optional, Callable


def sphere_llt_grid_points(r: float, pc: np.ndarray,
                           lat_num: int, long_num: int) -> np.ndarray:
    """
    生成球面 LLT（Latitude-Longitude Triangle）网格点。

    来源：1123_sphere_llt_grid

    数学模型：
      球坐标 (r, \theta, \phi) 转笛卡尔坐标：
        x = x_c + r \sin\phi \cos\theta
        y = y_c + r \sin\phi \sin\theta
        z = z_c + r \cos\phi

      纬度环 \phi_k = \pi * k / (lat_num + 1),  k = 1, ..., lat_num
      经度点 \theta_m = 2\pi * m / long_num,    m = 0, ..., long_num-1

    参数：
        r:      球半径（米）
        pc:     球心坐标，形状 (3,)
        lat_num: 纬度圈数（不含两极）
        long_num: 每圈经度点数

    返回：
        p: 网格点坐标，形状 (point_num, 3)
    """
    pc = np.asarray(pc, dtype=float).flatten()
    if pc.size != 3:
        raise ValueError("pc 必须为三维坐标")
    if lat_num < 0 or long_num < 1:
        raise ValueError("lat_num >= 0 且 long_num >= 1")

    point_num = 2 + lat_num * long_num
    p = np.zeros((point_num, 3), dtype=float)
    n = 0

    # 北极
    p[n, :] = pc + np.array([0.0, 0.0, r])
    n += 1

    # 中间纬度环
    for lat in range(1, lat_num + 1):
        phi = np.pi * lat / (lat_num + 1)
        sin_phi = np.sin(phi)
        cos_phi = np.cos(phi)
        for long_idx in range(long_num):
            theta = 2.0 * np.pi * long_idx / long_num
            p[n, 0] = pc[0] + r * sin_phi * np.cos(theta)
            p[n, 1] = pc[1] + r * sin_phi * np.sin(theta)
            p[n, 2] = pc[2] + r * cos_phi
            n += 1

    # 南极
    p[n, :] = pc + np.array([0.0, 0.0, -r])
    n += 1

    return p


def sphere_llt_grid_line_count(lat_num: int, long_num: int) -> int:
    """
    计算 LLT 网格的线数（边数）。

    公式：
      L = long_num * (lat_num + 1)          # 经线
        + long_num * lat_num                # 纬线
        + long_num * (lat_num - 1)          # 对角线
    """
    if lat_num < 0 or long_num < 1:
        return 0
    return long_num * (lat_num + 1) + long_num * lat_num + long_num * max(lat_num - 1, 0)


def distmesh_2d_simple(fd: Callable[[np.ndarray], np.ndarray],
                       fh: Callable[[np.ndarray], np.ndarray],
                       h0: float,
                       box: np.ndarray,
                       iteration_max: int = 100,
                       pfix: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    简化版 DistMesh 二维网格生成器。

    来源：308_distmesh（Persson & Strang, SIAM Review 2004）

    物理背景：
      在天线阵列设计中，口径面（aperture）上单元并非总是均匀分布。
      例如为了降低旁瓣，常采用非均匀间距（密度函数 h(x,y) 控制）。
      DistMesh 通过力平衡方程：
          \mathbf{F}_i = \sum_{j \in \mathcal{N}(i)} (L_{0,ij} - L_{ij}) \frac{\mathbf{p}_i - \mathbf{p}_j}{L_{ij}}
      迭代移动节点位置，使边长趋近理想值 L_0 = h(\mathbf{p}) \cdot F_{scale} \cdot \bar{L}。

    参数：
        fd:   距离函数 d(x,y) < 0 表示区域内
        fh:   尺度函数 h(x,y) 控制局部边长
        h0:   目标边长
        box:  包围盒 [[xmin, ymin], [xmax, ymax]]
        iteration_max: 最大迭代次数
        pfix: 固定节点坐标 (Nfix, 2)

    返回：
        p: 节点坐标 (N, 2)
        t: 三角形单元索引 (Nt, 3)
    """
    if pfix is None:
        pfix = np.zeros((0, 2), dtype=float)

    dptol = 0.001
    ttol = 0.1
    Fscale = 1.2
    deltat = 0.2
    geps = 0.001 * h0
    deps = np.sqrt(np.finfo(float).eps) * h0

    # 1. 在包围盒内生成矩形初始网格
    x_grid = np.arange(box[0, 0], box[1, 0] + h0, h0)
    y_grid = np.arange(box[0, 1], box[1, 1] + h0 * np.sqrt(3.0) / 2.0,
                       h0 * np.sqrt(3.0) / 2.0)
    x_mesh, y_mesh = np.meshgrid(x_grid, y_grid)
    # 偶数行偏移
    x_mesh[1::2, :] += h0 / 2.0
    p = np.vstack([x_mesh.ravel(), y_mesh.ravel()]).T

    # 2. 保留区域内点并按密度函数筛选
    d_val = fd(p)
    p = p[d_val < geps, :]
    if p.shape[0] == 0:
        return pfix.copy(), np.zeros((0, 3), dtype=int)

    r0 = 1.0 / (fh(p) ** 2)
    r0_max = np.max(r0) if r0.size > 0 else 1.0
    keep = np.random.rand(p.shape[0]) < (r0 / r0_max)
    p = np.vstack([pfix, p[keep, :]])

    # 去重（保留固定点在前）
    p_unique, idx = np.unique(p, axis=0, return_index=True)
    # 按原始顺序排序以保持固定点在前
    order = np.argsort(idx)
    p = p_unique[order, :]
    N = p.shape[0]

    if iteration_max <= 0:
        # 仅做 Delaunay 剖分
        try:
            from scipy.spatial import Delaunay
            tri = Delaunay(p)
            t = tri.simplices
            pmid = (p[t[:, 0]] + p[t[:, 1]] + p[t[:, 2]]) / 3.0
            t = t[fd(pmid) < -geps, :]
            return p, t
        except Exception:
            return p, np.zeros((0, 3), dtype=int)

    pold = np.full_like(p, np.inf)
    iteration = 0
    triangulation_count = 0

    try:
        from scipy.spatial import Delaunay
    except Exception:
        return p, np.zeros((0, 3), dtype=int)

    t = None
    while iteration < iteration_max:
        iteration += 1
        displacement = np.sqrt(np.sum((p - pold) ** 2, axis=1)) / h0
        if np.max(displacement) > ttol or t is None:
            pold = p.copy()
            tri = Delaunay(p)
            triangulation_count += 1
            t = tri.simplices
            pmid = (p[t[:, 0]] + p[t[:, 1]] + p[t[:, 2]]) / 3.0
            t = t[fd(pmid) < -geps, :]
            if t.size == 0:
                break
            bars = np.vstack([t[:, [0, 1]], t[:, [0, 2]], t[:, [1, 2]]])
            bars = np.unique(np.sort(bars, axis=1), axis=0)
        else:
            bars = np.vstack([t[:, [0, 1]], t[:, [0, 2]], t[:, [1, 2]]])
            bars = np.unique(np.sort(bars, axis=1), axis=1)

        # 6. 力平衡驱动节点移动
        barvec = p[bars[:, 0], :] - p[bars[:, 1], :]
        L = np.sqrt(np.sum(barvec ** 2, axis=1))
        L = np.maximum(L, 1e-12)
        hbars = fh((p[bars[:, 0], :] + p[bars[:, 1], :]) / 2.0)
        L0 = hbars * Fscale * np.sqrt(np.sum(L ** 2) / np.sum(hbars ** 2))
        F = np.maximum(L0 - L, 0.0)
        Fvec = (F / L)[:, None] * barvec

        Ftot = np.zeros((N, 2), dtype=float)
        np.add.at(Ftot, bars[:, 0], Fvec)
        np.add.at(Ftot, bars[:, 1], -Fvec)
        if pfix.shape[0] > 0:
            Ftot[:pfix.shape[0], :] = 0.0
        p = p + deltat * Ftot

        # 7. 将越界点拉回边界
        d_val = fd(p)
        ix = d_val > 0
        if np.any(ix):
            px = p[ix, :].copy()
            dgradx = (fd(px + np.array([deps, 0.0])[None, :]) - d_val[ix]) / deps
            dgrady = (fd(px + np.array([0.0, deps])[None, :]) - d_val[ix]) / deps
            p[ix, 0] -= d_val[ix] * dgradx
            p[ix, 1] -= d_val[ix] * dgrady

        # 8. 终止准则
        interior = d_val < -geps
        if not np.any(interior):
            break
        max_move = np.max(np.sqrt(np.sum((deltat * Ftot[interior, :]) ** 2, axis=1))) / h0
        if max_move < dptol:
            break

    return p, t


def tet_mesh_quality_metrics(node_xyz: np.ndarray,
                             tetra_node: np.ndarray) -> dict:
    """
    计算四面体网格质量指标（基于 1236_tet_mesh_quality）。

    在阵列几何中，我们将四面体指标用于评估三维阵列单元分布的各向异性：
      - 体积与边长比
      - 外接球半径与内切球半径比
      - 奇异值条件数

    参数：
        node_xyz:   节点坐标 (N_node, 3)
        tetra_node: 四面体顶点索引 (N_tet, 4)，0-based

    返回：
        dict 包含 quality1~quality5 的统计量
    """
    node_xyz = np.asarray(node_xyz, dtype=float)
    tetra_node = np.asarray(tetra_node, dtype=int)
    nt = tetra_node.shape[0]
    if nt == 0:
        return {}

    q1 = np.zeros(nt, dtype=float)
    q2 = np.zeros(nt, dtype=float)
    q3 = np.zeros(nt, dtype=float)
    q4 = np.zeros(nt, dtype=float)
    q5 = np.zeros(nt, dtype=float)

    for e in range(nt):
        idx = tetra_node[e, :]
        v0 = node_xyz[idx[0], :]
        v1 = node_xyz[idx[1], :]
        v2 = node_xyz[idx[2], :]
        v3 = node_xyz[idx[3], :]

        # 边向量
        e1 = v1 - v0
        e2 = v2 - v0
        e3 = v3 - v0

        # 体积 = |det([e1, e2, e3])| / 6
        vol = abs(np.linalg.det(np.vstack([e1, e2, e3]))) / 6.0
        vol = max(vol, 1e-18)

        # 六条边长平方
        edges = [
            np.sum((v0 - v1) ** 2),
            np.sum((v0 - v2) ** 2),
            np.sum((v0 - v3) ** 2),
            np.sum((v1 - v2) ** 2),
            np.sum((v1 - v3) ** 2),
            np.sum((v2 - v3) ** 2),
        ]
        l_sum = sum(edges)
        l_max = max(edges)

        # Quality 1: 体积 / (边长平方和)^(3/2) 归一化
        q1[e] = 216.0 * np.sqrt(3.0) * vol / (l_sum ** 1.5)
        # Quality 2: 体积 / (l_max)^(3/2) 归一化
        q2[e] = 12.0 * np.sqrt(6.0) * vol / (l_max ** 1.5)
        # Quality 3: 奇异值条件数相关
        mat = np.vstack([e1, e2, e3])
        s = np.linalg.svd(mat, compute_uv=False)
        cond = s[0] / max(s[-1], 1e-18)
        q3[e] = 1.0 / cond
        # Quality 4: 体积与最小角相关
        q4[e] = 3.0 * vol / (np.sqrt(l_max) * l_sum)
        # Quality 5: 外接球/内切球半径比倒数
        # 外接球半径 R = ||a|| ||b|| ||c|| / (12 V)
        a_len = np.sqrt(np.sum(e1 ** 2))
        b_len = np.sqrt(np.sum(e2 ** 2))
        c_len = np.sqrt(np.sum(e3 ** 2))
        R = a_len * b_len * c_len / (12.0 * vol)
        # 内切球半径 r = 3V / A_sum
        face_areas = []
        for (a, b, c) in [(v0, v1, v2), (v0, v1, v3), (v0, v2, v3), (v1, v2, v3)]:
            u = b - a
            w = c - a
            cross = np.cross(u, w)
            face_areas.append(0.5 * np.linalg.norm(cross))
        A_sum = sum(face_areas)
        r_in = 3.0 * vol / max(A_sum, 1e-18)
        q5[e] = r_in / max(R, 1e-18)

    def stats(arr):
        return {
            'min': float(np.min(arr)),
            'mean': float(np.mean(arr)),
            'max': float(np.max(arr)),
            'var': float(np.var(arr))
        }

    return {
        'quality1': stats(q1),
        'quality2': stats(q2),
        'quality3': stats(q3),
        'quality4': stats(q4),
        'quality5': stats(q5),
    }


def generate_planar_array(nx: int, ny: int, dx: float, dy: float,
                          aperture_type: str = 'rectangular') -> np.ndarray:
    """
    生成平面阵列单元坐标。

    参数：
        nx, ny: x, y 方向单元数
        dx, dy: 单元间距（米）
        aperture_type: 'rectangular' 或 'circular'
    返回：
        pos: (N, 3) 单元位置
    """
    x = (np.arange(nx) - (nx - 1) / 2.0) * dx
    y = (np.arange(ny) - (ny - 1) / 2.0) * dy
    xv, yv = np.meshgrid(x, y)
    pos = np.vstack([xv.ravel(), yv.ravel(), np.zeros(xv.size)]).T
    if aperture_type == 'circular':
        r_max = min(nx * dx, ny * dy) / 2.0
        r = np.sqrt(pos[:, 0] ** 2 + pos[:, 1] ** 2)
        pos = pos[r <= r_max, :]
    return pos


def generate_conformal_array(r: float, lat_num: int, long_num: int) -> np.ndarray:
    """
    生成球面共形阵列单元坐标（基于 sphere_llt_grid）。
    """
    return sphere_llt_grid_points(r, np.array([0.0, 0.0, 0.0]), lat_num, long_num)
