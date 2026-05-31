"""
脑血流动力学 — 脑血管网格生成模块
基于 distmesh 2D/3D 核心算法，构建二维脑血管切片与三维脑血管网络的几何模型。

科学背景:
- 脑血管网络的网格质量直接影响血流模拟的精度。
- 采用 signed distance function (SDF) 描述血管腔边界，结合力平衡迭代优化节点分布。
- 在二维切片中模拟 Willis 环截面，在三维中模拟脑实质血管树。
"""

import numpy as np
from scipy.spatial import Delaunay


def dcircle(p, xc, yc, r):
    """二维圆的有向距离函数: d = sqrt((x-xc)^2 + (y-yc)^2) - r"""
    return np.sqrt((p[:, 0] - xc) ** 2 + (p[:, 1] - yc) ** 2) - r


def drectangle(p, xmin, xmax, ymin, ymax):
    """二维矩形的有向距离函数。"""
    return -np.minimum(
        np.minimum(np.minimum(-xmin + p[:, 0], xmax - p[:, 0]),
                   -ymin + p[:, 1]),
        ymax - p[:, 1]
    )


def ddiff(d1, d2):
    """两个有向距离函数的差集（孔洞）。"""
    return np.maximum(d1, -d2)


def dunion(d1, d2):
    """两个有向距离函数的并集。"""
    return np.minimum(d1, d2)


def dsphere(p, xc, yc, zc, r):
    """三维球的有向距离函数。"""
    return np.sqrt((p[:, 0] - xc) ** 2 + (p[:, 1] - yc) ** 2 + (p[:, 2] - zc) ** 2) - r


def huniform(p):
    """均匀尺寸场。"""
    return np.ones(p.shape[0])


def distmesh_2d(fd, fh, h0, box, iteration_max, pfix):
    """
    二维 DistMesh 算法实现。
    参考文献: Persson & Strang, SIAM Review 46(2):329-345, 2004.
    """
    dptol = 0.001
    ttol = 0.1
    Fscale = 1.2
    deltat = 0.2
    geps = 0.001 * h0
    deps = np.sqrt(np.finfo(float).eps) * h0
    iteration = 0
    triangulation_count = 0

    # 1. 初始点分布
    x, y = np.meshgrid(
        np.arange(box[0, 0], box[1, 0] + h0, h0),
        np.arange(box[0, 1], box[1, 1] + h0 * np.sqrt(3) / 2, h0 * np.sqrt(3) / 2)
    )
    x[1::2, :] += h0 / 2
    p = np.column_stack((x.ravel(), y.ravel()))

    # 2. 保留区域内的点，按密度函数稀疏化
    p = p[fd(p) < geps, :]
    r0 = 1.0 / fh(p) ** 2
    if pfix.size > 0:
        pfix_arr = np.atleast_2d(pfix)
        p = np.vstack((pfix_arr, p[np.random.rand(p.shape[0]) < r0 / np.max(r0), :]))
    else:
        p = p[np.random.rand(p.shape[0]) < r0 / np.max(r0), :]

    # 去重
    p = np.unique(p, axis=0)
    N = p.shape[0]

    if iteration_max <= 0:
        t = Delaunay(p).simplices
        return p, t

    pold = np.inf * np.ones_like(p)

    while iteration < iteration_max:
        iteration += 1

        # 3. 若节点移动足够大则重新三角化
        if ttol < np.max(np.sqrt(np.sum((p - pold) ** 2, axis=1)) / h0):
            N = p.shape[0]
            pold = p.copy()
            tri = Delaunay(p)
            t = tri.simplices
            triangulation_count += 1
            pmid = (p[t[:, 0], :] + p[t[:, 1], :] + p[t[:, 2], :]) / 3.0
            t = t[fd(pmid) < -geps, :]

            # 4. 提取边
            bars = np.vstack((t[:, [0, 1]], t[:, [0, 2]], t[:, [1, 2]]))
            bars = np.unique(np.sort(bars, axis=1), axis=0)

        # 6. 基于边长的力平衡
        barvec = p[bars[:, 0], :] - p[bars[:, 1], :]
        L = np.sqrt(np.sum(barvec ** 2, axis=1))
        hbars = fh((p[bars[:, 0], :] + p[bars[:, 1], :]) / 2.0)
        L0 = hbars * Fscale * np.sqrt(np.sum(L ** 2) / np.sum(hbars ** 2))
        F = np.maximum(L0 - L, 0.0)
        Fvec = F[:, None] / L[:, None] * barvec

        rows = np.repeat(bars[:, 0], 2)
        cols = np.tile(np.arange(2), bars.shape[0])
        vals = Fvec.ravel()
        rows2 = np.repeat(bars[:, 1], 2)
        vals2 = -Fvec.ravel()

        Ftot = np.zeros((N, 2))
        for i in range(len(rows)):
            Ftot[rows[i], cols[i]] += vals[i]
        for i in range(len(rows2)):
            Ftot[rows2[i], cols[i]] += vals2[i]

        if pfix.size > 0:
            nfix = np.atleast_2d(pfix).shape[0]
            Ftot[:nfix, :] = 0.0

        p = p + deltat * Ftot

        # 7. 将外部点拉回边界
        d = fd(p)
        ix = d > 0
        if np.any(ix):
            dgradx = (fd(np.column_stack((p[ix, 0] + deps, p[ix, 1]))) - d[ix]) / deps
            dgrady = (fd(np.column_stack((p[ix, 0], p[ix, 1] + deps))) - d[ix]) / deps
            p[ix, 0] -= d[ix] * dgradx
            p[ix, 1] -= d[ix] * dgrady

        # 8. 终止条件
        interior = d < -geps
        if np.any(interior):
            max_move = np.max(np.sqrt(np.sum((deltat * Ftot[interior, :]) ** 2, axis=1)) / h0)
            if max_move < dptol:
                break

    return p, t


def distmesh_3d(fd, fh, h0, box, iteration_max, pfix):
    """
    三维 DistMesh 算法实现。
    参考文献: Persson & Strang, SIAM Review 46(2):329-345, 2004.
    """
    dim = 3
    ptol = 0.001
    ttol = 0.1
    L0mult = 1 + 0.4 / 2 ** (dim - 1)
    deltat = 0.1
    geps = 0.1 * h0
    deps = np.sqrt(np.finfo(float).eps) * h0
    iteration = 0

    # 1. 初始点分布
    grids = [np.arange(box[0, i], box[1, i] + h0, h0) for i in range(dim)]
    mesh = np.meshgrid(*grids, indexing='ij')
    p = np.column_stack([m.ravel() for m in mesh])

    # 2. 保留区域内的点
    p = p[fd(p) < geps, :]
    r0 = fh(p)
    if pfix.size > 0:
        pfix_arr = np.atleast_2d(pfix)
        prob = np.min(r0) ** dim / r0 ** dim
        prob = np.clip(prob, 0.0, 1.0)
        p = np.vstack((pfix_arr, p[np.random.rand(p.shape[0]) < prob, :]))
    else:
        prob = np.min(r0) ** dim / r0 ** dim
        prob = np.clip(prob, 0.0, 1.0)
        p = p[np.random.rand(p.shape[0]) < prob, :]
    p = np.unique(p, axis=0)

    if iteration_max <= 0:
        t = Delaunay(p).simplices
        return p, t

    N = p.shape[0]
    count = 0
    p0 = np.inf * np.ones_like(p)

    while iteration < iteration_max:
        iteration += 1

        # 3. 重新三角化
        if ttol * h0 < np.max(np.sqrt(np.sum((p - p0) ** 2, axis=1))):
            p0 = p.copy()
            tri = Delaunay(p)
            t = tri.simplices
            pmid = np.zeros((t.shape[0], dim))
            for ii in range(dim + 1):
                pmid += p[t[:, ii], :] / (dim + 1)
            t = t[fd(pmid) < -geps, :]

            # 4. 提取边
            localpairs = np.array([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]])
            pair = np.vstack([t[:, localpairs[i, :]] for i in range(localpairs.shape[0])])
            pair = np.unique(np.sort(pair, axis=1), axis=0)
            count += 1

        # 6. 力平衡
        bars = p[pair[:, 0], :] - p[pair[:, 1], :]
        L = np.sqrt(np.sum(bars ** 2, axis=1))
        L0 = fh((p[pair[:, 0], :] + p[pair[:, 1], :]) / 2.0)
        L0 = L0 * L0mult * (np.sum(L ** dim) / np.sum(L0 ** dim)) ** (1.0 / dim)
        F = np.maximum(L0 - L, 0.0)
        Fbar = np.column_stack((bars, -bars)) * np.repeat(F / L, 2 * dim).reshape(-1, 2 * dim)

        dp = np.zeros((N, dim))
        for i in range(pair.shape[0]):
            for d in range(dim):
                dp[pair[i, 0], d] += Fbar[i, d]
                dp[pair[i, 1], d] += Fbar[i, d + dim]

        if pfix.size > 0:
            nfix = np.atleast_2d(pfix).shape[0]
            dp[:nfix, :] = 0.0
        p = p + deltat * dp

        # 7. 拉回边界（两次梯度步）
        for _ in range(2):
            d = fd(p)
            ix = d > 0
            if not np.any(ix):
                continue
            gradd = np.zeros((np.sum(ix), dim))
            for ii in range(dim):
                a = np.zeros(dim)
                a[ii] = deps
                pshift = p[ix, :] + np.ones((np.sum(ix), 1)) * a
                d1x = fd(pshift)
                gradd[:, ii] = (d1x - d[ix]) / deps
            p[ix, :] -= d[ix][:, None] * gradd

        # 8. 终止条件
        d = fd(p)
        interior = d < -geps
        if np.any(interior):
            maxdp = np.max(deltat * np.sqrt(np.sum(dp[interior, :] ** 2, axis=1)))
            if maxdp < ptol * h0:
                break

    return p, t


def generate_willis_ring_mesh(h0=0.15, iteration_max=50):
    """
    生成 Willis 环二维截面网格。
    Willis 环由前后交通动脉与左右颈内动脉、椎动脉组成，近似为多个圆盘交集。
    """
    # 简化模型：以 (0,0) 为中心的主圆，加上四个分支圆
    def fd(p):
        d_main = dcircle(p, 0.0, 0.0, 1.0)
        d_left = dcircle(p, -1.2, 0.3, 0.4)
        d_right = dcircle(p, 1.2, 0.3, 0.4)
        d_bottom = dcircle(p, 0.0, -1.0, 0.35)
        return dunion(dunion(dunion(d_main, d_left), d_right), d_bottom)

    box = np.array([[-2.0, -2.0], [2.0, 2.0]])
    pfix = np.array([[-1.2, 0.3], [1.2, 0.3], [0.0, -1.0], [0.0, 1.0]])
    p, t = distmesh_2d(fd, huniform, h0, box, iteration_max, pfix)
    return p, t


def generate_cerebral_vessel_3d(h0=0.25, iteration_max=30):
    """
    生成简化的三维脑血管网格（球体内血管树区域）。
    """
    def fd(p):
        return dsphere(p, 0.0, 0.0, 0.0, 1.5)

    box = np.array([[-1.6, -1.6, -1.6], [1.6, 1.6, 1.6]])
    pfix = np.zeros((0, 3))
    p, t = distmesh_3d(fd, huniform, h0, box, iteration_max, pfix)
    return p, t
