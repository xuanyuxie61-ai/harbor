"""
interpolation_background.py
双变量背景分布插值与估计

基于 1279_toms886 项目重构:
  Padua 点上的双变量 Chebyshev 多项式插值
  
物理应用:
  在 H->ZZ*->4l 分析中，双轻子不变质量对 (m_{Z1}, m_{Z2}) 平面上
  的背景分布通常用平滑函数建模。本模块提供:
    - Padua 点构造
    - 双变量 Chebyshev 基函数求值
    - 二维插值系数计算 (通过 BLAS 风格的 GEMM)
    - 背景分布估计
"""
import numpy as np
from constants import TINY, M_HIGGS, M_Z

# ============================================================
# 1. Chebyshev 多项式求值 (映射 1279_toms886 的 cheb)
# ============================================================
def chebyshev_norm(j, x):
    """
    归一化 Chebyshev 多项式:
      T_0(x) = 1
      T_j(x) = sqrt(2) * cos(j * arccos(x)), j >= 1
    
    在 [-1,1] 上关于权函数 (1-x^2)^{-1/2} 正交归一:
      int_{-1}^1 T_i(x) T_j(x) / sqrt(1-x^2) dx = delta_{ij}
    """
    x = np.clip(x, -1.0, 1.0)
    if j == 0:
        return np.ones_like(x)
    return np.sqrt(2.0) * np.cos(j * np.arccos(x))


def chebyshev_eval_matrix(degree, pts):
    """
    构造 Chebyshev 求值矩阵 P_{i,j} = T_j(x_i)
    
    参数:
        degree: 最高阶数
        pts: 求值点数组
    返回:
        矩阵 shape (len(pts), degree+1)
    """
    pts = np.asarray(pts, dtype=float)
    n = len(pts)
    m = degree + 1
    P = np.zeros((n, m))
    for j in range(m):
        P[:, j] = chebyshev_norm(j, pts)
    return P


# ============================================================
# 2. Padua 点构造 (映射 1279_toms886 的 pdpts)
# ============================================================
def padua_points(deg):
    """
    构造 degree 阶 Padua 点集
    
    Padua 点是正方形 [-1,1]^2 上唯一的最优插值点集，
    点数为 N = (deg+1)(deg+2)/2
    
    构造方法 (第一族 Padua 点):
      从 Chebyshev-Lobatto 网格
        x_i = cos((i-1)*pi/deg), i = 0,...,deg
      中选取满足 i+j 为偶数的点
    
    权重: 边界点权重减半，角点权重为 1/4
    
    参数:
        deg: 多项式阶数
    返回:
        pts: 点数组 shape (N, 2)
        weights: 求积权重 shape (N,)
    """
    if deg < 0:
        return np.zeros((0, 2)), np.array([])
    
    # Chebyshev-Lobatto 节点
    grid = np.cos(np.arange(deg + 1) * np.pi / deg) if deg > 0 else np.array([0.0])
    
    pts = []
    wts = []
    for i in range(deg + 1):
        for j in range(deg + 1):
            if (i + j) % 2 == 0:
                x = grid[i]
                y = grid[j]
                # 权重修正
                w = 1.0
                if i == 0 or i == deg:
                    w *= 0.5
                if j == 0 or j == deg:
                    w *= 0.5
                pts.append([x, y])
                wts.append(w)
    
    pts = np.array(pts)
    wts = np.array(wts)
    # 归一化权重使其和为 4 (单位正方形面积)
    if np.sum(wts) > TINY:
        wts = wts / np.sum(wts) * 4.0
    return pts, wts


# ============================================================
# 3. 双变量插值系数计算 (映射 1279_toms886 的 padua2)
# ============================================================
def bivariate_chebyshev_coeffs(deg, pts, fvals, weights):
    """
    计算双变量 Chebyshev 插值系数
    
    基函数: Phi_{j,k}(x,y) = T_j(x) * T_{k-j}(y), 0 <= j <= k <= deg
    
    系数计算 (离散正交投影):
      C_{j,k} = sum_{i=1}^N w_i * f(x_i, y_i) * T_j(x_i) * T_{k-j}(y_i)
    
    参数:
        deg: 阶数
        pts: Padua 点, shape (N, 2)
        fvals: 函数值, shape (N,)
        weights: 求积权重, shape (N,)
    返回:
        coeffs: 系数矩阵 C_{j,k}, shape (deg+1, deg+1)
        esterr: 误差估计 (基于最高阶系数)
    """
    N = len(pts)
    fvals = np.asarray(fvals, dtype=float)
    weights = np.asarray(weights, dtype=float)
    
    # 构造 P1, P2 矩阵
    P = chebyshev_eval_matrix(deg, pts[:, 0])  # (N, deg+1)
    Q = chebyshev_eval_matrix(deg, pts[:, 1])  # (N, deg+1)
    
    # 加权函数值
    G = np.diag(weights * fvals)  # (N, N)
    
    # 系数: C = P^T @ G @ Q
    # 但只保留三角部分 (j + l <= deg)
    C = np.zeros((deg + 1, deg + 1))
    for j in range(deg + 1):
        for l in range(deg + 1 - j):
            C[j, l] = np.sum(weights * fvals * P[:, j] * Q[:, l])
    
    # 误差估计: 最高阶系数范数
    esterr = 0.0
    for j in range(deg + 1):
        for l in range(deg + 1 - j):
            if j + l == deg:
                esterr += C[j, l] ** 2
    esterr = np.sqrt(esterr)
    
    return C, esterr


def bivariate_chebyshev_eval(deg, coeffs, x, y):
    """
    使用双变量 Chebyshev 系数求值插值函数
    
    f(x,y) = sum_{j=0}^{deg} sum_{l=0}^{deg-j} C_{j,l} * T_j(x) * T_l(y)
    
    参数:
        deg: 阶数
        coeffs: 系数矩阵
        x, y: 求值点 (标量)
    返回:
        f(x,y)
    """
    x = np.clip(float(x), -1.0, 1.0)
    y = np.clip(float(y), -1.0, 1.0)
    
    P = chebyshev_eval_matrix(deg, np.array([x]))[0, :]  # (deg+1,)
    Q = chebyshev_eval_matrix(deg, np.array([y]))[0, :]  # (deg+1,)
    
    val = 0.0
    for j in range(deg + 1):
        for l in range(deg + 1 - j):
            val += coeffs[j, l] * P[j] * Q[l]
    return val


# ============================================================
# 4. 物理背景分布模型
# ============================================================
def continuum_background_model(m1, m2, params=None):
    """
    ZZ 连续谱背景模型 (q qbar -> ZZ 过程的 SM 近似)
    
    双轻子不变质量分布的平滑背景:
      B(m1, m2) ~ (m1 * m2)^{-alpha} * exp(-beta*(m1+m2)/m_Z)
    
    参数:
        m1, m2: 双轻子不变质量 [GeV]
        params: 背景参数 {'alpha': 1.5, 'beta': 0.3}
    返回:
        背景密度
    """
    if params is None:
        params = {"alpha": 1.5, "beta": 0.3, "norm": 1.0e-3}
    
    alpha = params["alpha"]
    beta = params["beta"]
    norm = params["norm"]
    
    if m1 <= 0.0 or m2 <= 0.0 or m1 + m2 > M_HIGGS:
        return 0.0
    
    val = norm * (m1 * m2) ** (-alpha) * np.exp(-beta * (m1 + m2) / M_Z)
    return float(val)


def build_background_interpolant(deg=8, m_range=(10.0, 120.0)):
    """
    在 (m1, m2) 平面上构造背景分布的双变量 Chebyshev 插值
    
    步骤:
      1. 在 Padua 点上计算背景模型
      2. 计算 Chebyshev 插值系数
      3. 返回求值函数
    
    返回:
        eval_func(m1, m2): 可调用函数
        coeffs: 系数矩阵
        esterr: 估计误差
    """
    pts, wts = padua_points(deg)
    
    # 将 Padua 点从 [-1,1]^2 映射到 [m_min, m_max]^2
    m_min, m_max = m_range
    scale = (m_max - m_min) / 2.0
    shift = (m_max + m_min) / 2.0
    phys_pts = pts * scale + shift
    
    fvals = []
    for p in phys_pts:
        fvals.append(continuum_background_model(p[0], p[1]))
    fvals = np.array(fvals)
    
    coeffs, esterr = bivariate_chebyshev_coeffs(deg, pts, fvals, wts)
    
    def eval_func(m1, m2):
        # 映射回 [-1,1]
        x1 = 2.0 * (m1 - m_min) / (m_max - m_min) - 1.0
        x2 = 2.0 * (m2 - m_min) / (m_max - m_min) - 1.0
        return bivariate_chebyshev_eval(deg, coeffs, x1, x2)
    
    return eval_func, coeffs, esterr


# ============================================================
# 5. 信号-背景判别函数
# ============================================================
def s_b_ratio(m1, m2, signal_func, background_func):
    """
    信号背景比 S/B
    
    在 (m1, m2) 处的判别:
      R = signal(m1,m2) / (background(m1,m2) + epsilon)
    
    参数:
        m1, m2: 双轻子不变质量
        signal_func: 信号密度函数
        background_func: 背景密度函数
    返回:
        S/B 比值
    """
    s = float(signal_func(m1, m2))
    b = float(background_func(m1, m2))
    if b < TINY:
        if s < TINY:
            return 0.0
        return 100.0
    return s / b
