# -*- coding: utf-8 -*-
"""
plasma_mesh.py
==============
Fast Ignition 靶区二维三角形网格生成.

核心算法 (来自 1329_triangulate_rectangle):
-------------------------------------------
将矩形区域 [xl, xr] × [yb, yt] 划分为 xn × yn 个四边形,
每个四边形再对角切分为 2 个三角形, 共 2 · xn · yn 个三角单元.

节点编号:
    (i, j) → k = j · (xn+1) + i    (0-indexed)
    0 ≤ i ≤ xn,  0 ≤ j ≤ yn

三角单元 (逆时针):
    下三角: (sw, se, nw)
    上三角: (ne, nw, se)
    其中 sw = j(xn+1)+i,  se = sw+1,  ne = sw+(xn+2),  nw = sw+(xn+1)

物理应用:
---------
Fast ignition 靶具有明显的分层结构:
- 低密度日冕 (corona)
- 临界面 (critical surface)
- 过稠密区 (overdense core)

三角形网格允许在日冕-芯部界面处局部加密, 提高能量沉积分辨率.
"""

import math


# ============================================================
# 矩形三角剖分 (来自 1329_triangulate_rectangle)
# ============================================================
def triangulate_rectangle(xl, xr, xn, yb, yt, yn, base=0):
    """
    矩形区域三角形剖分 (来自 1329_triangulate_rectangle).

    参数:
        xl, xr: x 方向左右边界
        xn    : x 方向单元数
        yb, yt: y 方向上下边界
        yn    : y 方向单元数
        base  : 节点编号基数 (0 或 1)
    返回:
        xy  : 节点坐标, shape (2, nn), nn = (xn+1)(yn+1)
        elem: 单元连接, shape (3, ne), ne = 2·xn·yn
    """
    # 一维网格
    if xn <= 0 or yn <= 0:
        raise ValueError("xn, yn 必须为正整数")
    x1d = [xl + (xr - xl) * i / xn for i in range(xn + 1)]
    y1d = [yb + (yt - yb) * j / yn for j in range(yn + 1)]

    # 节点坐标
    nn = (xn + 1) * (yn + 1)
    xy = [[0.0] * nn for _ in range(2)]
    k = 0
    for j in range(yn + 1):
        for i in range(xn + 1):
            xy[0][k] = x1d[i]
            xy[1][k] = y1d[j]
            k += 1

    # 单元连接
    ne = 2 * xn * yn
    elem = [[0] * ne for _ in range(3)]
    e = 0
    for j in range(1, yn + 1):
        for i in range(1, xn + 1):
            sw = (j - 1) * (xn + 1) + (i - 1) + base
            se = (j - 1) * (xn + 1) + i + base
            ne_node = j * (xn + 1) + i + base
            nw = j * (xn + 1) + (i - 1) + base
            # 下三角
            elem[0][e] = sw
            elem[1][e] = se
            elem[2][e] = nw
            e += 1
            # 上三角
            elem[0][e] = ne_node
            elem[1][e] = nw
            elem[2][e] = se
            e += 1

    return xy, elem


# ============================================================
# 等离子体分层网格 (双曲正切拉伸)
# ============================================================
def stretched_grid_1d(xmin, xmax, n, interface_frac=0.5,
                      stretch_factor=3.0):
    """
    一维拉伸网格: 界面附近加密.

    使用双曲正切拉伸函数:
        x(ξ) = x_min + (L/2)(1 + tanh(a(ξ - ξ_0)) / tanh(a ξ_0))
    其中 ξ ∈ [0, 1] 为计算坐标, ξ_0 为界面位置.

    参数:
        xmin, xmax    : 物理边界
        n             : 网格点数
        interface_frac: 界面位置 (归一化)
        stretch_factor: 拉伸强度
    返回:
        list[float]: 递增网格坐标
    """
    if n < 3:
        return [xmin + (xmax - xmin) * i / max(n - 1, 1) for i in range(n)]
    L = xmax - xmin
    a = stretch_factor
    xi0 = interface_frac
    th0 = math.tanh(a * xi0)
    grid = []
    for i in range(n):
        xi = i / (n - 1)
        x = xmin + 0.5 * L * (1.0 + math.tanh(a * (xi - xi0)) / th0)
        grid.append(x)
    # 确保端点精确
    grid[0] = xmin
    grid[-1] = xmax
    return grid


def plasma_target_mesh(Lx, Ly, nx, ny, interface_x_frac=0.6):
    """
    生成 fast ignition 靶区结构化三角形网格.

    靶结构:
        [0, 0.4 Lx]    : 日冕 (corona), n_e ~ 10 n_c
        [0.4 Lx, 0.6 Lx]: 临界面过渡区 (gradient)
        [0.6 Lx, Lx]   : 稠密芯部 (overdense), n_e ~ 1000 n_c

    在 x 方向使用拉伸网格 (界面加密), y 方向均匀.

    参数:
        Lx, Ly         : 域尺寸 [m]
        nx, ny         : x, y 方向单元数
        interface_x_frac: 日冕-芯部界面位置
    返回:
        dict: 包含 'xy', 'elem', 'nx', 'ny', 'ne', 'nn',
              'x_1d', 'y_1d', 'areas'
    """
    # x 方向: 拉伸网格
    x_1d = stretched_grid_1d(0.0, Lx, nx + 1,
                             interface_frac=interface_x_frac,
                             stretch_factor=3.0)
    # y 方向: 均匀网格
    y_1d = [Ly * j / ny for j in range(ny + 1)]

    # 使用自定义网格进行三角剖分
    nn = (nx + 1) * (ny + 1)
    xy = [[0.0] * nn for _ in range(2)]
    k = 0
    for j in range(ny + 1):
        for i in range(nx + 1):
            xy[0][k] = x_1d[i]
            xy[1][k] = y_1d[j]
            k += 1

    # 三角剖分
    ne = 2 * nx * ny
    elem = [[0] * ne for _ in range(3)]
    e = 0
    for j in range(1, ny + 1):
        for i in range(1, nx + 1):
            sw = (j - 1) * (nx + 1) + (i - 1)
            se = (j - 1) * (nx + 1) + i
            ne_node = j * (nx + 1) + i
            nw = j * (nx + 1) + (i - 1)
            elem[0][e] = sw
            elem[1][e] = se
            elem[2][e] = nw
            e += 1
            elem[0][e] = ne_node
            elem[1][e] = nw
            elem[2][e] = se
            e += 1

    # 计算单元面积
    areas = [0.0] * ne
    for e in range(ne):
        n0 = elem[0][e]
        n1 = elem[1][e]
        n2 = elem[2][e]
        x0, y0 = xy[0][n0], xy[1][n0]
        x1, y1 = xy[0][n1], xy[1][n1]
        x2, y2 = xy[0][n2], xy[1][n2]
        # 叉积面积公式: A = 0.5 |det([x1-x0, y1-y0; x2-x0, y2-y0])|
        areas[e] = 0.5 * abs((x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0))

    # 计算最小网格间距 (用于 CFL 条件)
    dx_min = float('inf')
    for i in range(1, len(x_1d)):
        dx_min = min(dx_min, x_1d[i] - x_1d[i - 1])
    dy = Ly / ny
    dy_min = dy

    return {
        "xy": xy,
        "elem": elem,
        "nx": nx,
        "ny": ny,
        "ne": ne,
        "nn": nn,
        "x_1d": x_1d,
        "y_1d": y_1d,
        "areas": areas,
        "dx_min": dx_min,
        "dy_min": dy_min,
        "Lx": Lx,
        "Ly": Ly,
        "interface_x_frac": interface_x_frac,
    }


def plasma_density_profile(x, Lx, n_corona, n_core, interface_frac=0.6,
                           gradient_scale=0.05):
    """
    等离子体密度剖面 (指数渐变):

    n_e(x) = n_corona + (n_core - n_corona) / (1 + exp(-(x - x_int) / L_g))

    参数:
        x              : 位置 [m]
        Lx             : 域总长
        n_corona       : 日冕密度
        n_core         : 芯部密度
        interface_frac : 界面位置
        gradient_scale : 渐变尺度 (归一化)
    返回:
        n_e [1/m^3]
    """
    x_int = interface_frac * Lx
    L_g = gradient_scale * Lx
    if L_g <= 0:
        return n_core if x >= x_int else n_corona
    xi = (x - x_int) / L_g
    # 防止 exp 溢出
    if xi > 20:
        return n_core
    if xi < -20:
        return n_corona
    sigmoid = 1.0 / (1.0 + math.exp(-xi))
    return n_corona + (n_core - n_corona) * sigmoid


def mesh_quality_check(mesh):
    """
    检查网格质量: 最小/最大面积比, 最小角度.

    返回质量报告字典.
    """
    areas = mesh["areas"]
    ne = mesh["ne"]
    if ne == 0:
        return {"valid": False, "reason": "no elements"}

    a_min = min(areas)
    a_max = max(areas)
    a_mean = sum(areas) / ne

    # 面积比
    ratio = a_max / max(a_min, 1.0e-300)

    # 最小角估计 (使用最简方法)
    xy = mesh["xy"]
    elem = mesh["elem"]
    min_angle_deg = 180.0
    for e in range(min(ne, 50)):  # 抽样检查
        n0, n1, n2 = elem[0][e], elem[1][e], elem[2][e]
        x0, y0 = xy[0][n0], xy[1][n0]
        x1, y1 = xy[0][n1], xy[1][n1]
        x2, y2 = xy[0][n2], xy[1][n2]
        # 边长
        l01 = math.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)
        l12 = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        l20 = math.sqrt((x0 - x2) ** 2 + (y0 - y2) ** 2)
        # 余弦定理
        if l01 > 0 and l20 > 0:
            cos0 = ((x1 - x0) * (x2 - x0) + (y1 - y0) * (y2 - y0)) / (l01 * l20)
            cos0 = max(-1.0, min(1.0, cos0))
            angle0 = math.degrees(math.acos(cos0))
            min_angle_deg = min(min_angle_deg, angle0)

    return {
        "valid": True,
        "n_elements": ne,
        "n_nodes": mesh["nn"],
        "area_min": a_min,
        "area_max": a_max,
        "area_mean": a_mean,
        "area_ratio": ratio,
        "min_angle_deg": min_angle_deg,
        "dx_min": mesh["dx_min"],
        "dy_min": mesh["dy_min"],
    }


def print_mesh_summary(mesh):
    """打印网格摘要."""
    quality = mesh_quality_check(mesh)
    print("\n" + "=" * 72)
    print("Fast Ignition 靶区三角形网格")
    print("=" * 72)
    print("  域尺寸: {:.2e} × {:.2e} m".format(mesh["Lx"], mesh["Ly"]))
    print("  单元数: {}".format(quality["n_elements"]))
    print("  节点数: {}".format(quality["n_nodes"]))
    print("  最小面积: {:.4e} m^2".format(quality["area_min"]))
    print("  最大面积: {:.4e} m^2".format(quality["area_max"]))
    print("  面积比  : {:.2f}".format(quality["area_ratio"]))
    print("  最小角度: {:.2f} deg".format(quality["min_angle_deg"]))
    print("  最小 Δx : {:.4e} m".format(quality["dx_min"]))
    print("  最小 Δy : {:.4e} m".format(quality["dy_min"]))
    print("  界面位置: {:.2f} × Lx".format(mesh["interface_x_frac"]))

    # 密度剖面示例
    print("  密度剖面 (x 方向中线):")
    x_vals = [0.0, 0.3, 0.5, 0.6, 0.7, 0.9, 1.0]
    from plasma_parameters import CORONA_ELECTRON_DENSITY, CORE_ELECTRON_DENSITY
    for xf in x_vals:
        ne = plasma_density_profile(
            xf * mesh["Lx"], mesh["Lx"],
            CORONA_ELECTRON_DENSITY, CORE_ELECTRON_DENSITY,
            mesh["interface_x_frac"])
        print("    x/Lx={:.2f} : n_e={:.3e}".format(xf, ne))
    print("=" * 72)
