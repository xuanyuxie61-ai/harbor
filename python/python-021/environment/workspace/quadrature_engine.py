"""
quadrature_engine.py
高阶数值求积与有限元刚度矩阵组装。

核心物理模型：
  托卡马克等离子体分析中需要大量体积分与面积分，包括：
    - 聚变功率体积积分 P_fus = ∫ p_fus(ψ) dV
    - 磁面面积分 ∮ dl / B_p
    - 三角形有限元基函数的内积 ∫_Ω ∇φ_i · ∇φ_j dΩ

  本模块实现两类核心求积：

  1. Gauss-Legendre 求积（一维）：
     用于径向积分与磁面积分。

         ∫_a^b f(x) dx ≈ Σ_{i=1}^n w_i f(x_i)

     其中节点 x_i 为 n 阶 Legendre 多项式 P_n(x) 的零点，
     权重 w_i = 2 / [ (1 - x_i²) (P_n'(x_i))² ]

  2. 三角形对称求积（二维）：
     用于极向截面非结构化网格上的刚度矩阵组装。

         ∫_T f(x,y) dA ≈ Σ_{k=1}^N w_k f(x_k, y_k)

     采用 Stroud 对称求积法则，支持精度 p ≤ 20。

  3. Fekete 点插值（一维谱元）：
     在磁面上选取最优插值节点，最小化 Vandermonde 矩阵条件数。

         maximize det(V(x_1, ..., x_m))

     其中 V_{ij} = T_{j-1}(x_i) 为 Chebyshev-Vandermonde 矩阵。
"""

import numpy as np
from parameters import N_GAUSS


# ============================================================
# 1. Gauss-Legendre 求积
# ============================================================

def legendre_gauss_nodes_weights(n):
    """
    计算 [-1, 1] 上 n 点 Gauss-Legendre 节点与权重。

    算法：利用 Golub-Welsch 特征值算法。
    三对角 Jacobi 矩阵 J 的特征值为节点，
    特征向量第一分量平方和权重因子给出权重。

    参数
    ------
    n : int
        求积阶数 (n ≥ 1)。

    返回
    ------
    x : ndarray, shape (n,)
        节点。
    w : ndarray, shape (n,)
        权重。
    """
    if n < 1:
        raise ValueError("求积阶数 n 必须 ≥ 1")
    if n == 1:
        return np.array([0.0]), np.array([2.0])

    # Jacobi 矩阵（Legendre 多项式）
    i = np.arange(1.0, n, dtype=float)
    beta = i / np.sqrt(4.0 * i ** 2 - 1.0)
    J = np.diag(beta, 1) + np.diag(beta, -1)

    eigenvalues, eigenvectors = np.linalg.eigh(J)
    x = eigenvalues
    w = 2.0 * eigenvectors[0, :] ** 2

    return x, w


def gauss_quadrature(f, a, b, n=N_GAUSS):
    """
    Gauss-Legendre 求积计算 ∫_a^b f(x) dx。

    参数
    ------
    f : callable
        被积函数 f(x) -> float/ndarray。
    a, b : float
        积分上下限。
    n : int
        求积阶数。

    返回
    ------
    result : float
        积分值。
    """
    x, w = legendre_gauss_nodes_weights(n)
    # 线性变换到 [a, b]
    t = 0.5 * (b - a) * x + 0.5 * (b + a)
    jac = 0.5 * (b - a)
    ft = np.asarray([f(ti) for ti in t], dtype=float)
    return float(jac * np.sum(w * ft))


# ============================================================
# 2. 三角形对称求积（基于原 triangle_symq_rule）
# ============================================================

# 预定义的 Stroud 三角形求积法则（精度 p=3, 5, 7）
# 采用重心坐标 (λ1, λ2, λ3)，权重之和 = 0.5（标准参考三角形面积）
_TRIANGLE_RULES = {
    3: {
        "n": 1,
        "bary": np.array([[1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]]),
        "w": np.array([0.5]),
    },
    5: {
        "n": 4,
        "bary": np.array([
            [0.33333333333333, 0.33333333333333, 0.33333333333334],
            [0.6, 0.2, 0.2],
            [0.2, 0.6, 0.2],
            [0.2, 0.2, 0.6],
        ]),
        "w": np.array([-0.28125, 0.26041666666667, 0.26041666666667, 0.26041666666667]),
    },
    7: {
        "n": 7,
        "bary": np.array([
            [0.33333333333333, 0.33333333333333, 0.33333333333334],
            [0.79742698535309, 0.10128650732346, 0.10128650732345],
            [0.10128650732346, 0.79742698535309, 0.10128650732345],
            [0.10128650732346, 0.10128650732346, 0.79742698535308],
            [0.05971587178977, 0.47014206410512, 0.47014206410511],
            [0.47014206410512, 0.05971587178977, 0.47014206410511],
            [0.47014206410512, 0.47014206410512, 0.05971587178976],
        ]),
        "w": np.array([
            0.1125,
            0.06296959027241,
            0.06296959027241,
            0.06296959027242,
            0.06619707639425,
            0.06619707639425,
            0.06619707639426,
        ]),
    },
}


def triangle_quadrature(f, vert1, vert2, vert3, precision=7):
    """
    在三角形上计算积分 ∫_T f(x,y) dA。

    参数
    ------
    f : callable
        被积函数 f(x, y) -> float。
    vert1, vert2, vert3 : ndarray, shape (2,)
        三角形三个顶点坐标。
    precision : int
        求积精度（支持 3, 5, 7）。

    返回
    ------
    result : float
        积分值。
    """
    if precision not in _TRIANGLE_RULES:
        raise ValueError(f"不支持精度 {precision}，请选择 3, 5, 7")
    rule = _TRIANGLE_RULES[precision]
    bary = rule["bary"]
    w = rule["w"]

    # 重心坐标 -> 笛卡尔坐标
    # x = λ1 x1 + λ2 x2 + λ3 x3
    verts = np.array([vert1, vert2, vert3])
    xy = bary @ verts  # shape (n, 2)

    # Jacobian = 2 * Area(T)
    jac = abs(np.linalg.det(np.array([
        [vert2[0] - vert1[0], vert3[0] - vert1[0]],
        [vert2[1] - vert1[1], vert3[1] - vert1[1]],
    ])))

    vals = np.array([f(xy[k, 0], xy[k, 1]) for k in range(rule["n"])], dtype=float)
    return float(jac * np.sum(w * vals))


def assemble_stiffness_triangle(vert1, vert2, vert3):
    """
    组装线性有限元在单个三角形上的刚度矩阵。

    公式
    ----
    对于线性基函数 φ_i (i=1,2,3)：
        K_{ij} = ∫_T ∇φ_i · ∇φ_j dA

    解析表达式：
        K = (1 / (4 |T|)) · B^T B
        其中 B_{ki} = (v_{i+1} - v_{i+2})_k  (循环指标)

    参数
    ------
    vert1, vert2, vert3 : ndarray, shape (2,)

    返回
    ------
    K : ndarray, shape (3, 3)
        局部刚度矩阵。
    """
    v1, v2, v3 = np.asarray(vert1), np.asarray(vert2), np.asarray(vert3)
    area = 0.5 * abs((v2[0] - v1[0]) * (v3[1] - v1[1]) -
                     (v3[0] - v1[0]) * (v2[1] - v1[1]))
    if area < 1e-15:
        return np.zeros((3, 3))

    # 梯度常数
    b = np.array([v2[1] - v3[1], v3[1] - v1[1], v1[1] - v2[1]])
    c = np.array([v3[0] - v2[0], v1[0] - v3[0], v2[0] - v1[0]])

    K = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            K[i, j] = (b[i] * b[j] + c[i] * c[j]) / (4.0 * area)
    return K


# ============================================================
# 3. Fekete 点插值（一维谱元，基于原 line_fekete_rule）
# ============================================================

def chebyshev_vandermonde(m, a, b, x_nodes):
    """
    构造 [a, b] 上 m 阶 Chebyshev-Vandermonde 矩阵。

    递推关系：
        T_0(xi) = 1
        T_1(xi) = xi
        T_k(xi) = 2 xi T_{k-1}(xi) - T_{k-2}(xi)

    其中 xi = (-(b - x) + (x - a)) / (b - a) 为到 [-1,1] 的映射。

    参数
    ------
    m : int
        基函数个数（多项式次数 = m-1）。
    a, b : float
        区间端点。
    x_nodes : ndarray
        样本点坐标。

    返回
    ------
    V : ndarray, shape (m, len(x_nodes))
    """
    x_nodes = np.asarray(x_nodes, dtype=float).flatten()
    n = len(x_nodes)
    xi = (-(b - x_nodes) + (x_nodes - a)) / (b - a)

    V = np.zeros((m, n))
    V[0, :] = 1.0
    if m > 1:
        V[1, :] = xi
    for i in range(2, m):
        V[i, :] = 2.0 * xi * V[i - 1, :] - V[i - 2, :]
    return V


def line_fekete_points(m, a, b, n_sample=200):
    """
    计算区间 [a, b] 上近似 Fekete 插值点。

    算法（Bos-Levenberg 贪心算法）：
        1. 在 [a,b] 上取 n_sample 个均匀样本点。
        2. 构造 Chebyshev-Vandermonde 矩阵 V (m × n)。
        3. 求解矩问题 V w = mom，其中 mom 为前 m 个矩。
        4. 选取 w 中非零分量对应的样本点为 Fekete 点。

    参数
    ------
    m : int
        多项式基个数。
    a, b : float
        区间端点。
    n_sample : int
        样本点数。

    返回
    ------
    xf : ndarray
        Fekete 点坐标。
    wf : ndarray
        对应权重。
    Vf : ndarray
        非奇异子矩阵。
    """
    if n_sample < m:
        raise ValueError("样本点数必须不少于基函数数")

    x = np.linspace(a, b, n_sample)
    V = chebyshev_vandermonde(m, a, b, x)

    # 矩向量：∫_a^b T_k(xi) dx
    mom = np.zeros(m)
    mom[0] = np.pi * (b - a) / 2.0  # ∫ T_0 = 区间长度（缩放后近似）
    # 更高阶矩采用数值积分
    for k in range(1, m):
        mom[k] = gauss_quadrature(lambda xi: np.cos(k * np.arccos(
            np.clip((-(b - xi) + (xi - a)) / (b - a), -1.0, 1.0))),
            a, b, n=min(32, N_GAUSS))

    # 最小二乘求解 w
    w, _, _, _ = np.linalg.lstsq(V, mom, rcond=None)

    ind = np.where(np.abs(w) > 1e-12 * np.max(np.abs(w)))[0]
    if len(ind) < m:
        # 若不足 m 个，贪心选取前 m 个最大权重
        ind = np.argsort(np.abs(w))[-m:]

    xf = x[ind]
    wf = w[ind]
    Vf = V[:, ind]
    return xf, wf, Vf


# ============================================================
# 4. 托卡马克专用积分函数
# ============================================================

def toroidal_volume_integral(f_radial, R0, a, kappa, n_radial=64, n_theta=64):
    """
    环形体积积分：∫ f(r,θ) · R(r,θ) · r · dr dθ dφ。

    公式
    ----
        V = 2π ∫_0^{2π} ∫_0^a f(r,θ) · R(r,θ) · r dr dθ
        R(r,θ) = R_0 + r cos θ

    参数
    ------
    f_radial : callable
        f(r, θ) -> float，被积函数。
    R0, a, kappa : float
        托卡马克几何参数。
    n_radial, n_theta : int
        求积阶数。

    返回
    ------
    result : float
        积分值。
    """
    r_nodes, r_weights = legendre_gauss_nodes_weights(n_radial)
    r = 0.5 * a * (r_nodes + 1.0)  # [0, a]
    r_w = 0.5 * a * r_weights

    theta_nodes, theta_weights = legendre_gauss_nodes_weights(n_theta)
    theta = np.pi * (theta_nodes + 1.0)  # [0, 2π]
    theta_w = np.pi * theta_weights

    total = 0.0
    for i in range(n_radial):
        for j in range(n_theta):
            R_loc = R0 + r[i] * np.cos(theta[j])
            jac = R_loc * r[i]
            total += r_w[i] * theta_w[j] * jac * f_radial(r[i], theta[j])

    return 2.0 * np.pi * total


def magnetic_surface_average(B_p, R, Z, psi, psi_target):
    """
    计算某磁面上的 ⟨B_p⟩ 面积分平均。

    公式
    ----
        ⟨B_p⟩ = ∮ B_p dl / ∮ dl

    参数
    ------
    B_p : ndarray
        极向磁场强度。
    R, Z : ndarray
        坐标网格。
    psi : ndarray
        磁通函数。
    psi_target : float
        目标磁面值。

    返回
    ------
    avg : float
        面积分平均值。
    """
    mask = np.abs(psi - psi_target) < 0.05 * (psi.max() - psi.min())
    if not np.any(mask):
        return 0.0
    # 简化的线积分近似
    dl = np.sqrt(np.gradient(R[mask]) ** 2 + np.gradient(Z[mask]) ** 2)
    if np.sum(dl) < 1e-15:
        return 0.0
    return float(np.sum(B_p[mask] * dl) / np.sum(dl))
