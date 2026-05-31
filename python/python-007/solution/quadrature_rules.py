"""
三维数值积分模块
整合自：231_cube_exactness（3D高斯-勒让德求积）

在吸积盘模拟中用于：
  1. 计算吸积盘体积内的物理量积分（质量、角动量、能量）
  2. 构造高斯-勒让德节点和权重用于谱方法投影
  3. 验证数值积分的精确度
"""
import numpy as np


# 预计算的1D高斯-勒让德节点和权重（N=1..10）
_LEGENDRE_ABSCISSAS = {
    1: np.array([0.0]),
    2: np.array([-0.57735026918962576451, 0.57735026918962576451]),
    3: np.array([-0.77459666924148337704, 0.0, 0.77459666924148337704]),
    4: np.array([-0.86113631159405257522, -0.33998104358485626480,
                 0.33998104358485626480, 0.86113631159405257522]),
    5: np.array([-0.90617984593866399280, -0.53846931010568309104, 0.0,
                 0.53846931010568309104, 0.90617984593866399280]),
    6: np.array([-0.93246951420315202781, -0.66120938646626451366,
                 -0.23861918608319690863, 0.23861918608319690863,
                 0.66120938646626451366, 0.93246951420315202781]),
    7: np.array([-0.94910791234275852453, -0.74153118559939443986,
                 -0.40584515137739716691, 0.0,
                 0.40584515137739716691, 0.74153118559939443986,
                 0.94910791234275852453]),
    8: np.array([-0.96028985649753623168, -0.79666647741362673959,
                 -0.52553240991632898582, -0.18343464249564980494,
                 0.18343464249564980494, 0.52553240991632898582,
                 0.79666647741362673959, 0.96028985649753623168]),
}

_LEGENDRE_WEIGHTS = {
    1: np.array([2.0]),
    2: np.array([1.0, 1.0]),
    3: np.array([0.55555555555555555556, 0.88888888888888888889, 0.55555555555555555556]),
    4: np.array([0.34785484513745385737, 0.65214515486254614263,
                 0.65214515486254614263, 0.34785484513745385737]),
    5: np.array([0.23692688505618908751, 0.47862867049936646804, 0.56888888888888888889,
                 0.47862867049936646804, 0.23692688505618908751]),
    6: np.array([0.17132449237917034504, 0.36076157304813860757,
                 0.46791393457269104739, 0.46791393457269104739,
                 0.36076157304813860757, 0.17132449237917034504]),
    7: np.array([0.12948496616886969327, 0.27970539148927666790,
                 0.38183005050511894495, 0.41795918367346938776,
                 0.38183005050511894495, 0.27970539148927666790,
                 0.12948496616886969327]),
    8: np.array([0.10122853629037625915, 0.22238103445337447054,
                 0.31370664587788728734, 0.36268378337836198297,
                 0.36268378337836198297, 0.31370664587788728734,
                 0.22238103445337447054, 0.10122853629037625915]),
}


def legendre_set(n):
    """
    获取 n 点高斯-勒让德求积的节点和权重。
    1D积分公式：
        ∫_{-1}^{1} f(x) dx ≈ Σ_{i=1}^{n} w_i · f(x_i)

    该公式对次数 ≤ 2n-1 的多项式精确成立。

    参数:
        n: 节点数 (1..8)

    返回:
        x, w: 节点和权重数组
    """
    if n not in _LEGENDRE_ABSCISSAS:
        raise ValueError(f"Only n=1..8 supported, got {n}")
    return _LEGENDRE_ABSCISSAS[n].copy(), _LEGENDRE_WEIGHTS[n].copy()


def legendre_3d_set(nx, ny, nz, box):
    """
    构造3D张量积高斯-勒让德求积规则。

    对于盒子 [a_x, b_x] × [a_y, b_y] × [a_z, b_z]：
        ∫∫∫ f(x,y,z) dV ≈ Σ_{i,j,k} W_{ijk} · f(X_{ijk}, Y_{ijk}, Z_{ijk})

    其中通过仿射变换将 [-1,1] 映射到 [a,b]：
        x = (b-a)/2 · ξ + (a+b)/2
        dx = (b-a)/2 · dξ

    参数:
        nx, ny, nz: 各方向的节点数
        box: [xmin, xmax, ymin, ymax, zmin, zmax]

    返回:
        points: (n_points, 3) 求积节点
        weights: (n_points,) 求积权重
    """
    box = np.asarray(box, dtype=np.float64)
    if len(box) != 6:
        raise ValueError("box must have 6 elements: [xmin,xmax,ymin,ymax,zmin,zmax]")

    x_1d, wx = legendre_set(nx)
    y_1d, wy = legendre_set(ny)
    z_1d, wz = legendre_set(nz)

    # 仿射变换
    ax, bx = box[0], box[1]
    ay, by = box[2], box[3]
    az, bz = box[4], box[5]

    x_1d = 0.5 * (bx - ax) * x_1d + 0.5 * (bx + ax)
    y_1d = 0.5 * (by - ay) * y_1d + 0.5 * (by + ay)
    z_1d = 0.5 * (bz - az) * z_1d + 0.5 * (bz + az)

    wx = 0.5 * (bx - ax) * wx
    wy = 0.5 * (by - ay) * wy
    wz = 0.5 * (bz - az) * wz

    # 张量积
    n_total = nx * ny * nz
    points = np.zeros((n_total, 3), dtype=np.float64)
    weights = np.zeros(n_total, dtype=np.float64)

    idx = 0
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                points[idx] = [x_1d[i], y_1d[j], z_1d[k]]
                weights[idx] = wx[i] * wy[j] * wz[k]
                idx += 1

    return points, weights


def monomial_integral_3d(exponents, box):
    """
    计算单项式 x^a · y^b · z^c 在盒子上的精确积分。

    解析公式：
        ∫_{a_x}^{b_x} x^a dx = (b_x^{a+1} - a_x^{a+1}) / (a+1)
        总积分 = 三个方向积分之积

    参数:
        exponents: [a, b, c]
        box: [xmin, xmax, ymin, ymax, zmin, zmax]

    返回:
        积分值
    """
    exponents = np.asarray(exponents, dtype=np.int64)
    box = np.asarray(box, dtype=np.float64)

    if len(exponents) != 3 or len(box) != 6:
        raise ValueError("Invalid dimensions")

    result = 1.0
    for dim in range(3):
        a = exponents[dim]
        lo, hi = box[2 * dim], box[2 * dim + 1]
        if a < 0:
            raise ValueError("Exponents must be non-negative")
        if a == 0:
            val = hi - lo
        else:
            val = (hi ** (a + 1) - lo ** (a + 1)) / (a + 1)
        result *= val

    return result


def test_quadrature_exactness(nx, ny, nz, box, max_total_degree):
    """
    测试3D求积规则对单项式的精确度。
    对于 n 点高斯-勒让德，1D精确度为 2n-1，
    因此3D总次数精确度也为 2n-1。

    参数:
        nx, ny, nz: 各方向节点数
        box: 积分区域
        max_total_degree: 测试的最高总次数

    返回:
        errors: 字典，键为 (a,b,c)，值为相对误差
    """
    points, weights = legendre_3d_set(nx, ny, nz, box)

    errors = {}
    for a in range(max_total_degree + 1):
        for b in range(max_total_degree + 1 - a):
            for c in range(max_total_degree + 1 - a - b):
                # 求积近似
                vals = (points[:, 0] ** a) * (points[:, 1] ** b) * (points[:, 2] ** c)
                q_approx = np.dot(weights, vals)

                # 精确值
                q_exact = monomial_integral_3d([a, b, c], box)

                if abs(q_exact) > 1e-15:
                    rel_err = abs(q_approx - q_exact) / abs(q_exact)
                else:
                    rel_err = abs(q_approx)

                errors[(a, b, c)] = rel_err

    return errors


def cylindrical_quadrature(n_r, n_phi, n_z, r_in, r_out, z_min, z_max):
    """
    构造柱坐标系下的3D求积规则。

    柱坐标体积元：dV = r · dr · dφ · dz
    积分公式：
        ∫∫∫ f(r,φ,z) · r dr dφ dz

    参数:
        n_r, n_phi, n_z: 各方向节点数
        r_in, r_out: 径向范围
        z_min, z_max: 垂直范围

    返回:
        points_cyl: (n, 3) [r, φ, z]
        weights: (n,) 包含 Jacobian r 的权重
    """
    # 径向：高斯-勒让德在 [r_in, r_out] 上映射
    r_pts, r_w = legendre_set(n_r)
    r_pts = 0.5 * (r_out - r_in) * r_pts + 0.5 * (r_out + r_in)
    r_w = 0.5 * (r_out - r_in) * r_w

    # 角向：等间距（复合梯形法则，对周期函数指数收敛）
    phi_pts = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    phi_w = np.full(n_phi, 2 * np.pi / n_phi)

    # 垂直：高斯-勒让德
    z_pts, z_w = legendre_set(n_z)
    z_pts = 0.5 * (z_max - z_min) * z_pts + 0.5 * (z_max + z_min)
    z_w = 0.5 * (z_max - z_min) * z_w

    # 张量积
    n_total = n_r * n_phi * n_z
    points = np.zeros((n_total, 3), dtype=np.float64)
    weights = np.zeros(n_total, dtype=np.float64)

    idx = 0
    for i in range(n_r):
        for j in range(n_phi):
            for k in range(n_z):
                points[idx] = [r_pts[i], phi_pts[j], z_pts[k]]
                # Jacobian = r
                weights[idx] = r_w[i] * phi_w[j] * z_w[k] * r_pts[i]
                idx += 1

    return points, weights
