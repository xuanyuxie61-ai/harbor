"""
lattice_generator.py
====================
光子晶体晶格结构生成器

融合原项目:
  - 330_ellipse_grid  : 二维椭圆内网格生成
  - 333_ellipsoid_grid: 三维椭球内网格生成  
  - 708_magic_matrix  : 幻方矩阵（用于准周期结构生成）

本模块生成二维/三维光子晶体的周期性及准周期性介电结构，
包括正方晶格、三角晶格、蜂窝晶格以及准晶格。
"""

import numpy as np


def ellipse_grid(n, rx, ry, cx, cy):
    """
    生成椭圆内部均匀网格点 —— 基于 330_ellipse_grid 核心算法
    
    椭圆方程:
        ((x-cx)/rx)² + ((y-cy)/ry)² = 1
    
    采用最短轴方向 N+1 个格点，通过镜像对称生成全部点。
    
    Parameters
    ----------
    n : int
        最短轴方向子区间数
    rx, ry : float
        椭圆半轴长度 [m]
    cx, cy : float
        椭圆中心坐标 [m]
    
    Returns
    -------
    xy : ndarray, shape (N, 2)
        椭圆内部网格点坐标
    """
    if n < 1:
        raise ValueError("n 必须 >= 1")
    if rx <= 0 or ry <= 0:
        raise ValueError("半轴长度必须为正")
    
    if rx < ry:
        h = 2.0 * rx / (2.0 * n + 1.0)
        ni = n
        nj = int(np.ceil(ry / rx) * n)
    else:
        h = 2.0 * ry / (2.0 * n + 1.0)
        nj = n
        ni = int(np.ceil(rx / ry) * n)
    
    points = []
    for j in range(nj + 1):
        i = 0
        x = cx
        y = cy + j * h
        points.append([x, y])
        if j > 0:
            points.append([x, 2 * cy - y])
        while True:
            i += 1
            x = cx + i * h
            if ((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2 > 1.0:
                break
            points.append([x, y])
            points.append([2 * cx - x, y])
            if j > 0:
                points.append([x, 2 * cy - y])
                points.append([2 * cx - x, 2 * cy - y])
    
    xy = np.array(points)
    # 去重并排序，保证数值稳定性
    xy = np.unique(np.round(xy, 12), axis=0)
    return xy


def ellipsoid_grid(n, rx, ry, rz, cx, cy, cz):
    """
    生成椭球内部均匀网格点 —— 基于 333_ellipsoid_grid 核心算法
    
    椭球方程:
        ((x-cx)/rx)² + ((y-cy)/ry)² + ((z-cz)/rz)² = 1
    
    Parameters
    ----------
    n : int
        最短轴方向子区间数
    rx, ry, rz : float
        椭球半轴长度 [m]
    cx, cy, cz : float
        椭球中心坐标 [m]
    
    Returns
    -------
    xyz : ndarray, shape (N, 3)
        椭球内部网格点坐标
    """
    if n < 1:
        raise ValueError("n 必须 >= 1")
    if rx <= 0 or ry <= 0 or rz <= 0:
        raise ValueError("半轴长度必须为正")
    
    r = np.array([rx, ry, rz])
    c = np.array([cx, cy, cz])
    rmin = np.min(r)
    h = 2.0 * rmin / (2.0 * n + 1.0)
    ni = int(np.ceil(rx / rmin) * n)
    nj = int(np.ceil(ry / rmin) * n)
    nk = int(np.ceil(rz / rmin) * n)
    
    points = []
    for k in range(nk + 1):
        z = cz + k * h
        for j in range(nj + 1):
            y = cy + j * h
            for i in range(ni + 1):
                x = cx + i * h
                if ((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2 + ((z - cz) / rz) ** 2 > 1.0:
                    break
                p = np.array([[x, y, z]])
                np_ = 1
                if i > 0:
                    q = p.copy()
                    q[:, 0] = 2 * cx - q[:, 0]
                    p = np.vstack([p, q])
                    np_ *= 2
                if j > 0:
                    q = p.copy()
                    q[:, 1] = 2 * cy - q[:, 1]
                    p = np.vstack([p, q])
                    np_ *= 2
                if k > 0:
                    q = p.copy()
                    q[:, 2] = 2 * cz - q[:, 2]
                    p = np.vstack([p, q])
                    np_ *= 2
                points.extend(p.tolist())
    
    xyz = np.array(points)
    xyz = np.unique(np.round(xyz, 12), axis=0)
    return xyz


def magic_matrix(n):
    """
    生成奇数阶幻方矩阵 —— 基于 708_magic_matrix 核心算法
    
    幻方矩阵每行、每列、对角线元素和相同，可用于构造
    准周期性光子晶体中的非周期但确定性的介电结构。
    
    算法（Siamese 方法）:
        a) 从顶行中间开始，k=1
        b) 填入 k
        c) 若 k=n², 结束
        d) k 增加 1
        e) 向右上移动一格（循环边界）
        f) 若该格已被占据，从当前位置向下移动一格
        g) 回到步骤 (b)
    
    Parameters
    ----------
    n : int
        矩阵阶数，必须为奇数
    
    Returns
    -------
    A : ndarray, shape (n, n)
        幻方矩阵
    """
    if n % 2 != 1 or n < 1:
        raise ValueError("n 必须为正奇数")
    
    A = np.zeros((n, n), dtype=int)
    k = 1
    i = 0
    j = n // 2
    A[i, j] = k
    
    while k < n * n:
        k += 1
        im1 = (i - 1) % n
        jp1 = (j + 1) % n
        if A[im1, jp1] != 0:
            im1 = (i + 1) % n
            jp1 = j
        A[im1, jp1] = k
        i, j = im1, jp1
    
    return A


def square_photonic_crystal(nx, ny, a, r_hole, eps_bg, eps_hole):
    """
    二维正方晶格光子晶体介电分布
    
    晶格常数 a，空气孔半径 r_hole，背景介电常数 eps_bg，
    孔内介电常数 eps_hole。
    
    Parameters
    ----------
    nx, ny : int
        每个原胞的网格点数
    a : float
        晶格常数 [m]
    r_hole : float
        空气孔半径 [m]
    eps_bg : float
        背景材料相对介电常数
    eps_hole : float
        孔内材料相对介电常数
    
    Returns
    -------
    eps_r : ndarray, shape (nx, ny)
        相对介电常数分布
    x, y : ndarray
        网格坐标 [m]
    """
    if nx < 3 or ny < 3:
        raise ValueError("网格分辨率至少为 3×3")
    if a <= 0 or r_hole < 0:
        raise ValueError("晶格常数必须为正，孔半径必须非负")
    if r_hole > a / 2.0:
        raise ValueError("孔半径不能超过 a/2（避免孔重叠）")
    
    dx = a / nx
    dy = a / ny
    x = np.linspace(0, a - dx, nx)
    y = np.linspace(0, a - dy, ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    # 计算到最近晶格点的距离（周期性边界）
    dx_p = np.minimum(np.mod(X, a), a - np.mod(X, a))
    dy_p = np.minimum(np.mod(Y, a), a - np.mod(Y, a))
    dist = np.sqrt(dx_p ** 2 + dy_p ** 2)
    
    eps_r = np.where(dist < r_hole, eps_hole, eps_bg)
    return eps_r, x, y


def triangular_photonic_crystal(nx, ny, a, r_hole, eps_bg, eps_hole):
    """
    二维三角晶格光子晶体介电分布
    
    三角晶格基矢:
        a₁ = a(1, 0)
        a₂ = a(1/2, √3/2)
    
    倒格矢:
        b₁ = (2π/a)(1, -1/√3)
        b₂ = (2π/a)(0, 2/√3)
    
    Parameters
    ----------
    nx, ny : int
        沿 a₁, a₂ 方向的网格点数
    a : float
        晶格常数 [m]
    r_hole : float
        空气孔半径 [m]
    eps_bg : float
        背景介电常数
    eps_hole : float
        孔内介电常数
    
    Returns
    -------
    eps_r : ndarray, shape (nx, ny)
        相对介电常数分布（在斜坐标系中）
    x, y : ndarray, shape (nx, ny)
        笛卡尔坐标网格 [m]
    """
    if nx < 3 or ny < 3:
        raise ValueError("网格分辨率至少为 3×3")
    if a <= 0 or r_hole < 0 or r_hole > a / np.sqrt(3):
        raise ValueError("参数超出物理允许范围")
    
    a1 = np.array([a, 0.0])
    a2 = np.array([a * 0.5, a * np.sqrt(3.0) / 2.0])
    
    # 生成斜坐标系对应的笛卡尔坐标
    i_idx = np.arange(nx)
    j_idx = np.arange(ny)
    I, J = np.meshgrid(i_idx, j_idx, indexing='ij')
    
    X = (I / nx) * a1[0] + (J / ny) * a2[0]
    Y = (I / nx) * a1[1] + (J / ny) * a2[1]
    
    # 三角晶格最近邻距离: a/√3
    # 用 Wigner-Seitz 原胞判断
    eps_r = np.zeros((nx, ny))
    for i in range(nx):
        for j in range(ny):
            # 计算该点到所有格点 (m*a1 + n*a2) 的最小距离
            min_dist = float('inf')
            for m in range(-1, 2):
                for n in range(-1, 2):
                    px = m * a1[0] + n * a2[0]
                    py = m * a1[1] + n * a2[1]
                    dx_ = X[i, j] - px
                    dy_ = Y[i, j] - py
                    d = np.sqrt(dx_ ** 2 + dy_ ** 2)
                    if d < min_dist:
                        min_dist = d
            eps_r[i, j] = eps_hole if min_dist < r_hole else eps_bg
    
    return eps_r, X, Y


def quasiperiodic_photonic_crystal(n, a_avg, r_hole, eps_bg, eps_hole, magic_order=5):
    """
    基于幻方矩阵的准周期性光子晶体结构
    
    利用幻方矩阵的非周期但确定性特征，构造介电常数的
    准周期调制，用于研究光子准晶中的局域态与带隙。
    
    调制公式:
        ε(r) = ε_bg + (ε_hole - ε_bg) · Θ(r_hole - |r - r_i|)
    
    其中格点位置 r_i 由幻方矩阵元素值调制:
        r_i = (i + α·M_ij) · a_avg
    
    Parameters
    ----------
    n : int
        网格点数 (n×n)
    a_avg : float
        平均晶格常数 [m]
    r_hole : float
        孔半径 [m]
    eps_bg, eps_hole : float
        背景与孔内介电常数
    magic_order : int
        幻方阶数 (奇数)
    
    Returns
    -------
    eps_r : ndarray, shape (n, n)
        介电常数分布
    x, y : ndarray
        坐标网格 [m]
    """
    if n < 3:
        raise ValueError("网格点数至少为 3")
    if magic_order % 2 != 1:
        raise ValueError("幻方阶数必须为奇数")
    
    M = magic_matrix(magic_order)
    M_norm = (M - np.min(M)) / (np.max(M) - np.min(M) + 1e-12)
    
    L = n * a_avg
    x = np.linspace(0, L, n)
    y = np.linspace(0, L, n)
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    eps_r = np.full((n, n), eps_bg)
    
    # 在幻方调制的位置放置孔
    n_holes = magic_order * magic_order
    for idx in range(1, n_holes + 1):
        pos = np.argwhere(M == idx)[0]
        i, j = pos
        cx = (i + 0.5 * M_norm[i, j]) * (L / magic_order)
        cy = (j + 0.5 * M_norm[j, i]) * (L / magic_order)
        dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
        mask = dist < r_hole
        eps_r[mask] = eps_hole
    
    return eps_r, x, y


def woodpile_photonic_crystal(nx, ny, nz, a, r_rod, eps_bg, eps_rod):
    """
    三维木堆结构光子晶体 (Woodpile)
    
    木堆结构由正交堆叠的介质棒层组成，相邻层之间旋转 90°，
    每隔一层平移 a/2，具有面心四方 (FCT) 对称性。
    
    带隙公式近似 (Joannopoulos 经验公式):
        Δω/ω ≈ 0.15 × (ε_rod - ε_bg) / (ε_rod + ε_bg) × (2r_rod/a)
    
    Parameters
    ----------
    nx, ny, nz : int
        每个方向的网格点数
    a : float
        面内晶格常数 [m]
    r_rod : float
        介质棒半径 [m]
    eps_bg : float
        背景介电常数
    eps_rod : float
        介质棒介电常数
    
    Returns
    -------
    eps_r : ndarray, shape (nx, ny, nz)
        三维介电常数分布
    x, y, z : ndarray
        坐标轴 [m]
    """
    if nx < 3 or ny < 3 or nz < 3:
        raise ValueError("网格分辨率至少为 3×3×3")
    if a <= 0 or r_rod < 0 or r_rod > a / 2:
        raise ValueError("参数超出物理允许范围")
    
    dx = a / nx
    dy = a / ny
    dz = a / nz
    x = np.linspace(0, a - dx, nx)
    y = np.linspace(0, a - dy, ny)
    z = np.linspace(0, a - dz, nz)
    
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    eps_r = np.full((nx, ny, nz), eps_bg)
    
    for layer in range(nz):
        z_layer = z[layer]
        layer_idx = int(np.round(z_layer / (a / 4))) % 4
        
        if layer_idx in [0, 2]:
            # x 方向棒
            y_center = a / 4 if layer_idx == 0 else 3 * a / 4
            for rod in range(2):
                yc = y_center + rod * a / 2
                dist = np.sqrt((Y[:, :, layer] - yc) ** 2)
                mask = dist < r_rod
                eps_r[:, :, layer][mask] = eps_rod
        else:
            # y 方向棒 (旋转 90°)
            x_center = a / 4 if layer_idx == 1 else 3 * a / 4
            for rod in range(2):
                xc = x_center + rod * a / 2
                dist = np.sqrt((X[:, :, layer] - xc) ** 2)
                mask = dist < r_rod
                eps_r[:, :, layer][mask] = eps_rod
    
    return eps_r, x, y, z


def inverse_opal_structure(n, a, r_sphere, eps_bg, eps_sphere):
    """
    三维反蛋白石结构光子晶体
    
    由密堆球体空隙中填充高折射率材料构成，具有面心立方 (FCC)
    对称性，产生完整的三维光子带隙。
    
    FCC 格点位置:
        R = (i+j/2+k/2, √3/2·j+√3/6·k, √(2/3)·k) · a
    
    Parameters
    ----------
    n : int
        每个方向网格点数
    a : float
        晶格常数 [m]
    r_sphere : float
        球体半径 [m] (通常 ≈ 0.25a√2 对应密堆)
    eps_bg : float
        背景介电常数
    eps_sphere : float
        球体介电常数
    
    Returns
    -------
    eps_r : ndarray, shape (n, n, n)
        三维介电常数分布
    """
    if n < 3:
        raise ValueError("网格点数至少为 3")
    if a <= 0 or r_sphere < 0:
        raise ValueError("参数必须为正")
    
    dx = a / n
    x = np.linspace(0, a - dx, n)
    y = np.linspace(0, a - dx, n)
    z = np.linspace(0, a - dx, n)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    eps_r = np.full((n, n, n), eps_bg)
    
    # FCC 原胞基矢
    a1 = np.array([a, 0, 0])
    a2 = np.array([a / 2, a * np.sqrt(3) / 2, 0])
    a3 = np.array([a / 2, a * np.sqrt(3) / 6, a * np.sqrt(6) / 3])
    
    # 检查每个网格点到最近 FCC 格点的距离
    for i in range(n):
        for j in range(n):
            for k in range(n):
                min_dist = float('inf')
                for ii in range(-1, 2):
                    for jj in range(-1, 2):
                        for kk in range(-1, 2):
                            px = ii * a1[0] + jj * a2[0] + kk * a3[0]
                            py = ii * a1[1] + jj * a2[1] + kk * a3[1]
                            pz = ii * a1[2] + jj * a2[2] + kk * a3[2]
                            d = np.sqrt((X[i, j, k] - px) ** 2 +
                                        (Y[i, j, k] - py) ** 2 +
                                        (Z[i, j, k] - pz) ** 2)
                            if d < min_dist:
                                min_dist = d
                if min_dist < r_sphere:
                    eps_r[i, j, k] = eps_sphere
    
    return eps_r, x, y, z
