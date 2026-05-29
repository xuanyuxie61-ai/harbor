"""
pce_basis.py
============
多项式混沌展开(PCE)基函数模块（融合 854_pce_ode_hermite + 1381_vandermonde）

功能：
- 概率化Hermite多项式 He_n(x) 的递归计算
- 双积与三积积分（Galerkin投影的核心）
- Vandermonde矩阵构造（用于s-step Krylov的Newton基）
- 广义PCE展开系数的时间推进

数学公式：
- Hermite多项式（概率学家版本）:
  He_0(x) = 1
  He_1(x) = x
  He_{n+1}(x) = x He_n(x) - n He_{n-1}(x)
- 正交性: ∫ He_m(x) He_n(x) φ(x) dx = n! δ_{mn}
- 三积积分: C_{ijk} = E[He_i He_j He_k] / E[He_k²]
  仅当 i+j+k 为偶数且三角不等式满足时非零
- Vandermonde: V_{ij} = x_j^{i-1},  det(V) = ∏_{i<j} (x_j - x_i)
"""

import numpy as np


def hermite_he_prob(n, x):
    """
    计算概率学家Hermite多项式 He_n(x) 在x处的值。
    使用三项递推关系。
    """
    x = np.asarray(x, dtype=float)
    if n < 0:
        return np.zeros_like(x)
    if n == 0:
        return np.ones_like(x)
    if n == 1:
        return x
    
    he_prev2 = np.ones_like(x)
    he_prev1 = x
    
    for k in range(1, n):
        he_curr = x * he_prev1 - k * he_prev2
        he_prev2 = he_prev1
        he_prev1 = he_curr
    
    return he_prev1


def hermite_he_prob_matrix(degree, x):
    """
    计算所有阶数 0..degree 的He多项式值，返回 (len(x), degree+1) 矩阵。
    """
    x = np.asarray(x, dtype=float)
    n_points = x.size if x.ndim == 0 else x.shape[0]
    if x.ndim == 0:
        x = np.array([x])
        n_points = 1
    
    H = np.zeros((n_points, degree + 1))
    H[:, 0] = 1.0
    if degree >= 1:
        H[:, 1] = x
    for k in range(1, degree):
        H[:, k + 1] = x * H[:, k] - k * H[:, k - 1]
    return H


def he_double_product_integral(i, j):
    """
    E[He_i * He_j] = i! * δ_{ij}
    """
    if i != j:
        return 0.0
    # i! 对于小整数
    import math
    return float(math.factorial(i))


def he_triple_product_integral(i, j, k):
    """
    计算 E[He_i * He_j * He_k] / E[He_k²]。
    利用线性化公式:
    He_i He_j = Σ_{m=0}^{min(i,j)} C(i,m) C(j,m) m! He_{i+j-2m}
    因此与He_k的内积非零仅当 k = i+j-2m 对某个m成立。
    
    公式: C_{ijk} = (i! j! k!) / [ ((i+j-k)/2)! ((j+k-i)/2)! ((k+i-j)/2)! ]
    当 i+j+k 为奇数或违反三角不等式时为0。
    """
    if (i + j + k) % 2 == 1:
        return 0.0
    s = (i + j + k) // 2
    if s < i or s < j or s < k:
        return 0.0
    
    a = s - k
    b = s - i
    c = s - j
    
    # C_{ijk} = i! j! / (a! b! c!)
    # 注意要除以 k! = E[He_k²] 做归一化
    import math
    num = math.factorial(i) * math.factorial(j)
    den = math.factorial(a) * math.factorial(b) * math.factorial(c) * math.factorial(k)
    return float(num) / float(den)


def build_pce_galerkin_matrix(degree, alpha_mu, alpha_sigma):
    """
    为随机ODE du/dt = -α(ξ) u 构建PCE-Galerkin矩阵。
    α(ξ) = α_μ + α_σ ξ, ξ ~ N(0,1)
    
    Galerkin系统: dU_k/dt = -α_μ U_k - α_σ Σ_j C_{1jk} U_j
    返回 (degree+1, degree+1) 矩阵 A，使得 dU/dt = -A U。
    
    这是 pce_ode_hermite.m 核心思想的向量化实现。
    """
    # HOLE 1: 需要实现PCE-Galerkin矩阵的构建
    # 核心知识：
    #   - 使用 he_double_product_integral 计算正交模长
    #   - 使用 he_triple_product_integral 计算三积积分 C_{1jk}
    #   - 矩阵元素: A[k,k] += alpha_mu
    #   - 矩阵元素: A[k,j] += alpha_sigma * C_{1jk} / (k!)
    # 返回: (degree+1, degree+1) 的稠密矩阵
    raise NotImplementedError("HOLE 1: build_pce_galerkin_matrix 待修复")


def vandermonde_matrix(n, x):
    """
    构造Vandermonde矩阵 V_{ij} = x_j^{i-1}, i,j=1..n
    融合 1381_vandermonde/vand1.m 的核心思想。
    边界处理：当x_j=0且i=1时，定义0^0=1。
    """
    x = np.asarray(x, dtype=float)
    V = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == 0:
                V[i, j] = 1.0
            else:
                V[i, j] = x[j] ** i
    return V


def vandermonde_solve(V, b):
    """
    使用LU分解求解Vandermonde线性系统 V c = b。
    数值鲁棒性处理：检测条件数，必要时用最小二乘。
    """
    cond = np.linalg.cond(V)
    if cond > 1e14:
        # 病态情况下使用最小二乘求解
        c, residuals, rank, s = np.linalg.lstsq(V, b, rcond=1e-14)
        return c
    return np.linalg.solve(V, b)


def newton_basis_vandermonde(degree, nodes):
    """
    构造Newton基的广义Vandermonde矩阵，用于s-step Krylov方法。
    列j表示多项式 N_j(x) = ∏_{k=0}^{j-1} (x - s_k) 在采样点处的值。
    """
    nodes = np.asarray(nodes, dtype=float)
    m = len(nodes)
    n = degree + 1
    V = np.zeros((m, n))
    V[:, 0] = 1.0
    for j in range(1, n):
        V[:, j] = V[:, j - 1] * (nodes - nodes[j - 1])
    return V
