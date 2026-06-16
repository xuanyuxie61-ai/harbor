"""
measurement_operator.py — 观测算子与稀疏采样模块
==================================================
来源项目映射:
  - 548_human_mesh2d           → 2D 三角网格采样
  - 891_polygonal_surface_display → 多边形曲面离散化
  - 681_line_integrals         → 线积分观测泛函

科学背景:
  稀疏优化中的观测算子 A ∈ R^{m×M} 由以下三部分构成:
    (1) 点观测: 在随机采样点 {x_i} 上评估 PCE 基 Ψ(x_i)
    (2) 线观测: 沿直线段 Γ 的积分 ∫_Γ Ψ(x) ds  (源自 line_integrals)
    (3) 面观测: 在多边形区域 Ω 上的加权积分

  本模块实现:
    - 在 d-维超立方体 / 单纯形上的采样
    - 线积分观测 (精确 + Gauss 求积)
    - 多边形面观测
    - 观测算子条件数诊断 (RIP 验证)

核心公式:
  线积分 (单变量换元):
    ∫_Γ Ψ(x) ds = |b-a| ∫_0^1 Ψ(a + t(b-a)) dt
  多边形积分 (三角剖分):
    ∫_Ω f(x) dx = sum_T ∫_T f(x) dx,  每个三角形用 3 点 Gauss 求积
"""
import numpy as np
from sparse_basis import pce_basis_eval


# ----------------------------------------------------------------------
# 采样策略
# ----------------------------------------------------------------------
def sample_hypercube(n, d, seed=0):
    """在 [-1,1]^d 上均匀采样 n 点 (准蒙特卡洛: Halton 序列近似)."""
    rng = np.random.RandomState(seed)
    return rng.uniform(-1.0, 1.0, size=(n, d))


def sample_simplex(n, d, seed=0):
    """在 d-维标准单纯形 {x ∈ R^d : x_j ≥ 0, sum x_j ≤ 1} 上采样."""
    rng = np.random.RandomState(seed)
    # 通过指数分布归一化
    E = rng.exponential(1.0, size=(n, d + 1))
    S = E.sum(axis=1, keepdims=True)
    return E[:, :d] / S


def sample_sphere_surface(n, d, seed=0):
    """在 S^{d-1} 球面上均匀采样."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n, d)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X


# ----------------------------------------------------------------------
# 线积分观测 (源自 681_line_integrals)
# ----------------------------------------------------------------------
def line_segment_points(a, b, n_quad=5):
    """线段 a-b 上的 Gauss-Legendre 求积节点.

    ∫_Γ f ds = |b-a| ∑_k w_k f(a + t_k (b-a))
    """
    t_ref, w_ref = np.polynomial.legendre.leggauss(n_quad)
    t = 0.5 * (t_ref + 1.0)  # 映射到 [0,1]
    w = 0.5 * w_ref
    pts = np.outer(1.0 - t, a) + np.outer(t, b)
    length = np.linalg.norm(b - a)
    return pts, w * length


def line_integral_observer(a, b, indices, n_quad=5, basis='legendre'):
    """构造单条线段的观测行: row_m = ∫_Γ Ψ_m ds.

    输入:
        a, b     : (d,) 线段端点
        indices  : (M, d) 多指标
    返回:
        row : (M,)
    """
    pts, w = line_segment_points(a, b, n_quad)
    Psi = pce_basis_eval(pts, indices, basis=basis)
    return w @ Psi


def line_integrals_batch(segments, indices, n_quad=5, basis='legendre'):
    """批量构造线积分观测矩阵.

    输入:
        segments : list of (a, b) 对
    返回:
        A : (len(segments), M)
    """
    rows = []
    for (a, b) in segments:
        rows.append(line_integral_observer(
            np.asarray(a), np.asarray(b), indices, n_quad, basis))
    return np.vstack(rows)


# ----------------------------------------------------------------------
# 多边形面观测 (源自 548, 891)
# ----------------------------------------------------------------------
def triangle_quadrature_2d(v1, v2, v3, order=2):
    """三角形上 Gauss 求积 (精度 order).

    顶点 v1, v2, v3 ∈ R^2.
    返回: (pts, w), pts 为 (K, 2), w 为权重 (面积缩放).
    """
    v1, v2, v3 = np.asarray(v1), np.asarray(v2), np.asarray(v3)
    area = 0.5 * abs((v2[0] - v1[0]) * (v3[1] - v1[1]) -
                     (v3[0] - v1[0]) * (v2[1] - v1[1]))
    if order <= 1:
        # 1 点规则 (精确至 1 阶)
        pts = ((v1 + v2 + v3) / 3.0).reshape(1, 2)
        w = np.array([area])
    elif order <= 2:
        # 3 点规则 (精确至 2 阶)
        bary = np.array([[1 / 6, 1 / 6, 2 / 3],
                         [2 / 3, 1 / 6, 1 / 6],
                         [1 / 6, 2 / 3, 1 / 6]])
        pts = bary @ np.stack([v1, v2, v3])
        w = np.full(3, area / 3.0)
    else:
        # 4 点规则 (精确至 3 阶)
        bary = np.array([[1 / 3, 1 / 3, 1 / 3],
                         [0.6, 0.2, 0.2],
                         [0.2, 0.6, 0.2],
                         [0.2, 0.2, 0.6]])
        w_int = np.array([-27 / 96, 25 / 96, 25 / 96, 25 / 96]) * 96 / 25
        # 归一化使得 sum(w) = 1
        w_int = np.array([-9 / 16, 25 / 48, 25 / 48, 25 / 48])
        w_int /= w_int.sum()
        w_int *= area
        pts = bary @ np.stack([v1, v2, v3])
        w = w_int
    return pts, w


def triangulate_polygon(vertices):
    """简单扇形三角剖分 (适用于凸多边形)."""
    v = np.asarray(vertices)
    n = v.shape[0]
    triangles = [(0, i, i + 1) for i in range(1, n - 1)]
    return triangles


def polygon_observer(vertices, indices, quad_order=2, basis='legendre'):
    """多边形面观测行: row_m = ∫_Ω Ψ_m dx."""
    d = np.asarray(vertices).shape[1]
    triangles = triangulate_polygon(vertices)
    all_rows = []
    for (i1, i2, i3) in triangles:
        pts, w = triangle_quadrature_2d(
            vertices[i1], vertices[i2], vertices[i3], quad_order)
        # 升维到 d (如果需要)
        if d > 2:
            pts_full = np.zeros((pts.shape[0], d))
            pts_full[:, :2] = pts
            pts = pts_full
        Psi = pce_basis_eval(pts, indices, basis=basis)
        all_rows.append(w @ Psi)
    return sum(all_rows)


# ----------------------------------------------------------------------
# 组合观测算子
# ----------------------------------------------------------------------
def build_observation_operator(config, indices, basis='legendre'):
    """按配置构造完整观测算子 A.

    配置 dict:
        - n_point   : 点观测数目
        - segments  : 线观测列表
        - polygons  : 多边形观测列表
        - seed      : 随机种子
        - d         : 维数
    返回:
        A : (m, M)
    """
    d = config['d']
    n_point = config.get('n_point', 0)
    segments = config.get('segments', [])
    polygons = config.get('polygons', [])
    seed = config.get('seed', 0)
    rows = []

    # 1. 点观测
    if n_point > 0:
        X = sample_hypercube(n_point, d, seed)
        Psi_pt = pce_basis_eval(X, indices, basis=basis)
        rows.append(Psi_pt)

    # 2. 线观测
    if segments:
        A_line = line_integrals_batch(segments, indices, basis=basis)
        rows.append(A_line)

    # 3. 面观测
    for poly in polygons:
        row = polygon_observer(np.asarray(poly), indices, basis=basis)
        rows.append(row.reshape(1, -1))

    if not rows:
        raise ValueError("观测算子为空")
    return np.vstack(rows)


def check_rip_constant(A, s, n_test=50, seed=42):
    """经验估计阶数为 s 的 RIP 常数 δ_s.

    δ_s = max_{|T|≤s} |σ_max(A_T)² - 1|, σ_min(A_T)² - 1|
    通过随机支撑集抽样近似.
    """
    rng = np.random.RandomState(seed)
    m, M = A.shape
    delta = 0.0
    for _ in range(n_test):
        T = rng.choice(M, size=s, replace=False)
        A_T = A[:, T]
        # 归一化列
        norms = np.linalg.norm(A_T, axis=0)
        norms = np.where(norms < 1e-14, 1.0, norms)
        A_Tn = A_T / norms
        G = A_Tn.T @ A_Tn / m * M
        eigs = np.linalg.eigvalsh(G)
        delta = max(delta, abs(eigs.max() - 1.0), abs(eigs.min() - 1.0))
    return float(delta)
