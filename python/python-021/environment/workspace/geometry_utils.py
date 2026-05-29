"""
geometry_utils.py
磁面几何判定与谱元插值工具。

核心物理模型：
  1. 点在等离子体边界内判定（基于原 toms112）：
     利用射线法（ray casting）判断测试粒子是否位于
     由 Miller 参数化定义的 last closed flux surface (LCFS) 内部。

     算法：
        从测试点出发作水平射线，计算其与多边形（LCFS 离散化）
        边界的交点个数。奇数 -> 内部；偶数 -> 外部。

  2. Fekete 点插值（基于原 line_fekete_rule）：
     在磁面（一维闭合曲线）上选取最优插值节点，
     用于高精度谱元离散 MHD 特征值问题。

     对于闭合曲线 Γ: θ → (R(θ), Z(θ))，在 θ ∈ [0, 2π] 上
     选取 m 个 Fekete 点 {θ_j}，使得插值矩阵条件数最小化。

  3. 磁面面积与体积计算：
     利用 Green 定理计算极向截面面积：
        A = 0.5 ∮ (R dZ - Z dR)
     环向体积：
        V = 2π ∮ R dA = 2π ∫_0^a ∮ R(r,θ) dθ dr
"""

import numpy as np
from parameters import R0, a_minor, KAPPA, DELTA, N_FEKETE


def point_in_flux_surface(R_test, Z_test, theta_poly=None, n_theta=128):
    """
    判定测试点 (R_test, Z_test) 是否在 LCFS 内部。

    基于原 point_in_polygon 的射线法算法（Shimrat ACM TOMS 112）。

    参数
    ------
    R_test, Z_test : float
        测试点坐标 [m]。
    theta_poly : ndarray or None
        LCFS 离散化角度。若为 None，则使用 Miller 参数化生成。
    n_theta : int
        边界离散点数。

    返回
    ------
    inside : bool
        True 如果在内部。
    R_poly, Z_poly : ndarray
        边界多边形顶点。
    """
    if theta_poly is None:
        theta_poly = np.linspace(0, 2.0 * np.pi, n_theta, endpoint=False)

    # Miller 参数化
    R_poly = R0 + a_minor * np.cos(theta_poly + DELTA * np.sin(theta_poly))
    Z_poly = KAPPA * a_minor * np.sin(theta_poly)

    n = len(R_poly)
    inside = False

    for i in range(n):
        ip1 = (i + 1) % n
        y_i = Z_poly[i]
        y_ip1 = Z_poly[ip1]

        # 检查边是否跨越测试点的水平射线
        cond = (y_ip1 < Z_test) == (Z_test <= y_i)
        if cond:
            x_i = R_poly[i]
            x_ip1 = R_poly[ip1]
            t = R_test - x_i - (Z_test - y_i) * (x_ip1 - x_i) / (y_ip1 - y_i + 1e-20)
            if t < 0.0:
                inside = not inside

    return inside, R_poly, Z_poly


def compute_poloidal_area(theta_poly=None, n_theta=256):
    """
    利用 Green 定理计算极向截面面积。

    公式
    ----
        A = 0.5 ∮ (R dZ - Z dR)
          = 0.5 Σ_i (R_i Z_{i+1} - R_{i+1} Z_i)

    参数
    ------
    theta_poly, n_theta

    返回
    ------
    area : float
        极向截面面积 [m²]。
    R_poly, Z_poly : ndarray
        边界顶点。
    """
    if theta_poly is None:
        theta_poly = np.linspace(0, 2.0 * np.pi, n_theta, endpoint=False)

    R_poly = R0 + a_minor * np.cos(theta_poly + DELTA * np.sin(theta_poly))
    Z_poly = KAPPA * a_minor * np.sin(theta_poly)

    n = len(R_poly)
    area = 0.0
    for i in range(n):
        ip1 = (i + 1) % n
        area += R_poly[i] * Z_poly[ip1] - R_poly[ip1] * Z_poly[i]

    return 0.5 * abs(area), R_poly, Z_poly


def compute_toroidal_volume(theta_poly=None, n_theta=256, n_radial=64):
    """
    计算环向等离子体体积。

    公式
    ----
        V = 2π ∫_0^a ∮ R(r,θ) dθ dr

    对于 Miller 几何：
        R(r,θ) = R_0 + r cos(θ + δ sin θ)
        V ≈ 2π² R_0 a² κ  (小 ε 近似)

    参数
    ------
    theta_poly, n_theta, n_radial

    返回
    ------
    volume : float
        环向体积 [m³]。
    volume_approx : float
        解析近似值。
    """
    # 数值积分
    r_nodes = np.linspace(0, a_minor, n_radial)
    dr = r_nodes[1] - r_nodes[0] if n_radial > 1 else 0
    theta = np.linspace(0, 2.0 * np.pi, n_theta)
    dtheta = theta[1] - theta[0] if n_theta > 1 else 0

    V = 0.0
    for r in r_nodes:
        R_loc = R0 + r * np.cos(theta + DELTA * np.sin(theta))
        V += np.sum(R_loc) * dtheta * r * dr

    volume = 2.0 * np.pi * V
    volume_approx = 2.0 * (np.pi ** 2) * R0 * (a_minor ** 2) * KAPPA

    return volume, volume_approx


def fekete_points_on_flux_surface(m=N_FEKETE, n_sample=400):
    """
    在 LCFS 上计算 Fekete 插值节点。

    基于原 line_fekete_rule 的核心思想，将一维区间 [0, 2π]
    映射到闭合磁面曲线 θ → (R(θ), Z(θ))。

    算法
    ----
    1. 在 θ ∈ [0, 2π] 上均匀采样 n_sample 个点。
    2. 构造基于弧长参数化的 Chebyshev-Vandermonde 矩阵。
    3. 求解矩问题，选取权重非零点作为 Fekete 节点。

    参数
    ------
    m : int
        插值阶数（基函数个数）。
    n_sample : int
        采样点数。

    返回
    ------
    theta_fekete : ndarray
        Fekete 节点的极向角 [rad]。
    R_fekete, Z_fekete : ndarray
        对应物理坐标。
    weights : ndarray
        求积权重。
    """
    from quadrature_engine import chebyshev_vandermonde

    theta_sample = np.linspace(0, 2.0 * np.pi, n_sample)
    R_sample = R0 + a_minor * np.cos(theta_sample + DELTA * np.sin(theta_sample))
    Z_sample = KAPPA * a_minor * np.sin(theta_sample)

    # 弧长参数化 s(θ)
    ds = np.sqrt(np.gradient(R_sample) ** 2 + np.gradient(Z_sample) ** 2)
    s = np.cumsum(ds)
    s = np.concatenate(([0.0], s[:-1]))  # 修正偏移
    s_total = s[-1] + ds[-1]
    if s_total < 1e-15:
        raise ValueError("弧长计算失败")

    # 归一化到 [0, 1]
    s_norm = s / s_total

    # 在弧长参数上构造 Chebyshev-Vandermonde
    a, b = 0.0, 1.0
    V = chebyshev_vandermonde(m, a, b, s_norm)

    # 矩向量
    mom = np.zeros(m)
    mom[0] = s_total  # 总弧长对应零阶矩
    for k in range(1, m):
        # 数值积分
        integrand = np.cos(k * np.arccos(np.clip(2.0 * s_norm - 1.0, -1.0, 1.0)))
        mom[k] = np.trapezoid(integrand, s)

    # 最小二乘求解权重
    w, _, _, _ = np.linalg.lstsq(V, mom, rcond=None)

    # 选取非零权重对应的点
    threshold = 1e-12 * np.max(np.abs(w))
    ind = np.where(np.abs(w) > threshold)[0]
    if len(ind) < m:
        ind = np.argsort(np.abs(w))[-m:]

    theta_fekete = theta_sample[ind]
    R_fekete = R0 + a_minor * np.cos(theta_fekete + DELTA * np.sin(theta_fekete))
    Z_fekete = KAPPA * a_minor * np.sin(theta_fekete)
    weights = w[ind]

    return theta_fekete, R_fekete, Z_fekete, weights


def compute_curvature_and_torsion(theta_points):
    """
    计算磁面曲线的曲率与挠率。

    公式
    ----
    对于参数曲线 r(θ) = (R(θ), Z(θ))：
        κ = |R' Z'' - R'' Z'| / (R'² + Z'²)^{3/2}
        挠率 τ = 0（平面曲线）

    曲率半径 ρ = 1/κ，与磁场曲率漂移直接相关：
        v_κ = (m v_∥² / qB²) (B × κ) / B

    参数
    ------
    theta_points : ndarray
        极向角采样点。

    返回
    ------
    kappa : ndarray
        曲率 [1/m]。
    radius_curvature : ndarray
        曲率半径 [m]。
    """
    theta = np.asarray(theta_points)
    dtheta = theta[1] - theta[0] if len(theta) > 1 else 1.0

    R = R0 + a_minor * np.cos(theta + DELTA * np.sin(theta))
    Z = KAPPA * a_minor * np.sin(theta)

    dR = np.gradient(R, dtheta)
    dZ = np.gradient(Z, dtheta)
    d2R = np.gradient(dR, dtheta)
    d2Z = np.gradient(dZ, dtheta)

    denom = (dR ** 2 + dZ ** 2) ** 1.5 + 1e-20
    kappa = np.abs(dR * d2Z - d2R * dZ) / denom
    radius_curvature = 1.0 / (kappa + 1e-20)

    return kappa, radius_curvature


def generate_triangular_mesh(n_r=8, n_theta=16):
    """
    生成极向截面的粗三角形网格（用于有限元刚度矩阵测试）。

    参数
    ------
    n_r, n_theta : int
        径向与极向分层数。

    返回
    ------
    vertices : ndarray, shape (n_vertices, 2)
        顶点坐标。
    triangles : ndarray, shape (n_triangles, 3)
        三角形顶点索引。
    """
    vertices = []
    for j in range(n_r + 1):
        r = a_minor * j / n_r
        for i in range(n_theta):
            theta = 2.0 * np.pi * i / n_theta
            R = R0 + r * np.cos(theta + DELTA * np.sin(theta))
            Z = KAPPA * r * np.sin(theta)
            vertices.append([R, Z])
    vertices = np.array(vertices)

    triangles = []
    for j in range(n_r):
        for i in range(n_theta):
            v0 = j * n_theta + i
            v1 = j * n_theta + ((i + 1) % n_theta)
            v2 = (j + 1) * n_theta + i
            v3 = (j + 1) * n_theta + ((i + 1) % n_theta)
            triangles.append([v0, v1, v2])
            triangles.append([v1, v3, v2])
    triangles = np.array(triangles)

    return vertices, triangles
