"""
sparse_basis.py — 稀疏多项式基构造模块
===========================================
来源项目映射:
  - 777_monomial_value      → 单项式求值 monomial_value()
  - 662_legendre_product    → Legendre 乘积与正交性
  - 605_jacobi_exactness    → Jacobi 多项式与求积精确性

科学背景:
  在多项式混沌展开(PCE)中, 稀疏基的选择直接决定 L1 正则化恢复的成功率.
  本模块构造三类基函数并计算其 Gram 矩阵:
    (1) 单项式基       φ_k(x) = x^k,              k = 0,1,...,p
    (2) Legendre 基    P_k(x),                    正交于 [-1,1]
    (3) Jacobi 基      P_k^{(α,β)}(x),            正交于 w(x)=(1-x)^α(1+x)^β
  对多变量情形采用张量积构造, 并按总阶 |α| = α_1+...+α_d ≤ p 进行截断.

核心公式:
  三项递推 (Legendre):
    (k+1) P_{k+1}(x) = (2k+1) x P_k(x) - k P_{k-1}(x)
  Jacobi 递推:
    P_0 = 1,  P_1 = 0.5[(α-β) + (α+β+2)x]
    2k(k+α+β)(2k+α+β-2) P_k = ... (标准递推)
  多指标 α ∈ N^d, 总阶截断:
    I_p^d = {α ∈ N^d : |α| ≤ p}
"""
import numpy as np


# ----------------------------------------------------------------------
# 一维单项式求值  (源自 777_monomial_value)
# ----------------------------------------------------------------------
def monomial_value_1d(x, p):
    """计算单项式向量 [1, x, x^2, ..., x^p] 于每个 x.

    输入:
        x : (n,) 数组
        p : 最高阶数
    返回:
        V : (n, p+1)  Vandermonde 矩阵
    """
    x = np.asarray(x, dtype=float).ravel()
    n = x.size
    V = np.zeros((n, p + 1))
    if p >= 0:
        V[:, 0] = 1.0
    for k in range(1, p + 1):
        V[:, k] = V[:, k - 1] * x
    return V


def monomial_value_nd(X, alpha):
    """多变量单项式 ∏_j x_j^{alpha_j}.

    输入:
        X     : (n, d)
        alpha : (d,) 非负整数向量
    返回:
        v : (n,)
    """
    X = np.asarray(X, dtype=float)
    alpha = np.asarray(alpha, dtype=int)
    v = np.ones(X.shape[0])
    for j in range(X.shape[1]):
        if alpha[j] > 0:
            v *= X[:, j] ** alpha[j]
    return v


# ----------------------------------------------------------------------
# Legendre 多项式 (源自 662_legendre_product)
# ----------------------------------------------------------------------
def legendre_eval(x, p):
    """Legendre 多项式 P_0, ..., P_p 在 x 处的值 (三项递推).

    递推公式:
        (k+1) P_{k+1}(x) = (2k+1) x P_k(x) - k P_{k-1}(x)
    """
    x = np.asarray(x, dtype=float).ravel()
    n = x.size
    L = np.zeros((n, p + 1))
    if p >= 0:
        L[:, 0] = 1.0
    if p >= 1:
        L[:, 1] = x
    for k in range(1, p):
        L[:, k + 1] = ((2 * k + 1) * x * L[:, k] - k * L[:, k - 1]) / (k + 1)
    return L


def legendre_product_integral(p, q, n_quad=64):
    """∫_{-1}^{1} P_p(x) P_q(x) dx  (Gauss-Legendre 求积).

    正交性: 当 p != q 时积分 = 0;  = 2/(2p+1) 当 p = q.
    本函数以 n_quad 点 Gauss 求积进行数值验证, 用于评估基条件数.
    """
    nodes, weights = np.polynomial.legendre.leggauss(n_quad)
    Lp = legendre_eval(nodes, max(p, q))
    return float(np.sum(weights * Lp[:, p] * Lp[:, q]))


# ----------------------------------------------------------------------
# Jacobi 多项式 (源自 605_jacobi_exactness)
# ----------------------------------------------------------------------
def jacobi_eval(x, alpha, beta, p):
    """Jacobi 多项式 P_k^{(α,β)}(x), k = 0,...,p.

    递推 (Abramowitz & Stegun 22.7.1):
        a_k = (2k+1+α+β)(2k+2+α+β) / (2(k+1)(k+1+α+β))
        b_k = (β²-α²)(2k+1+α+β) / (2(k+1)(k+1+α+β)(2k+α+β))
        c_k = (k+α)(k+β)(2k+2+α+β) / ((k+1)(k+1+α+β)(2k+α+β))
        P_{k+1} = (a_k x + b_k) P_k - c_k P_{k-1}
    """
    x = np.asarray(x, dtype=float).ravel()
    n = x.size
    J = np.zeros((n, p + 1))
    if p >= 0:
        J[:, 0] = 1.0
    if p >= 1:
        J[:, 1] = 0.5 * ((alpha - beta) + (alpha + beta + 2.0) * x)
    for k in range(1, p):
        ak = (2 * k + 1 + alpha + beta) * (2 * k + 2 + alpha + beta) / \
             (2.0 * (k + 1) * (k + 1 + alpha + beta))
        bk = (beta ** 2 - alpha ** 2) * (2 * k + 1 + alpha + beta) / \
             (2.0 * (k + 1) * (k + 1 + alpha + beta) * (2 * k + alpha + beta))
        ck = (k + alpha) * (k + beta) * (2 * k + 2 + alpha + beta) / \
             ((k + 1) * (k + 1 + alpha + beta) * (2 * k + alpha + beta))
        J[:, k + 1] = (ak * x + bk) * J[:, k] - ck * J[:, k - 1]
    return J


def jacobi_exactness_degree(alpha, beta, n_quad):
    """n_quad 点 Gauss-Jacobi 求积精确成立的最大多项式阶 (应为 2*n_quad-1).

    使用 Golub-Welsch 算法构造 Jacobi 求积节点/权重.
    """
    nodes, weights = _gauss_jacobi(n_quad, alpha, beta)
    max_exact = 0
    for deg in range(2 * n_quad + 2):
        exact = _jacobi_moment(alpha, beta, deg)
        numer = np.sum(weights * nodes ** deg)
        if abs(numer - exact) < 1e-9 * (abs(exact) + 1.0):
            max_exact = deg
        else:
            break
    return max_exact


def _gauss_jacobi(n, alpha, beta):
    """Golub-Welsch 算法求 Gauss-Jacobi 节点/权重.

    通过三对角 Jacobi 矩阵的特征分解获得.
    """
    if n == 1:
        node = (beta - alpha) / (alpha + beta + 2.0)
        from scipy.special import beta as beta_fn
        weight = beta_fn(alpha + 1, beta + 1)
        return np.array([node]), np.array([weight])
    i = np.arange(1, n)
    # 三对角矩阵元素
    a_diag = np.zeros(n)
    a_diag[0] = (beta - alpha) / (alpha + beta + 2.0)
    if n > 1:
        # 通用递推: a_k = (β²-α²)/((2k+α+β)(2k+α+β+2))
        for k in range(1, n):
            denom = (2 * k + alpha + beta) * (2 * k + alpha + beta + 2)
            a_diag[k] = (beta ** 2 - alpha ** 2) / denom if denom != 0 else 0.0
    # 次对角
    b_sub = np.zeros(n - 1)
    for k in range(n - 1):
        kk = k + 1
        num = 4.0 * kk * (kk + alpha) * (kk + beta) * (kk + alpha + beta)
        den = ((2 * kk + alpha + beta) ** 2) * ((2 * kk + alpha + beta) ** 2 - 1)
        b_sub[k] = np.sqrt(num / den) if den > 0 else 0.0
    # 构造对称三对角矩阵
    J = np.diag(a_diag) + np.diag(b_sub, 1) + np.diag(b_sub, -1)
    eigs, vecs = np.linalg.eigh(J)
    from scipy.special import beta as beta_fn
    mu0 = beta_fn(alpha + 1, beta + 1) * (2 ** (alpha + beta + 1))
    # 归一化: 总权重 = 2^{α+β+1} B(α+1, β+1)
    weights = mu0 * vecs[0, :] ** 2
    order = np.argsort(eigs)
    return eigs[order], weights[order]


def _jacobi_moment(alpha, beta, deg):
    """∫_{-1}^{1} (1-x)^α (1+x)^β x^deg dx  (Beta 函数闭式).

    展开 x^deg = ((1+x)-(1-x))/2 的二项式, 逐项 Beta 积分.
    """
    from math import comb
    val = 0.0
    for j in range(deg + 1):
        # x^deg = sum_j C(deg,j) ((1+x)/2)^j (-(1-x)/2)^(deg-j) ... 采用直接展开
        # 用 (1+x)^j (1-x)^(deg-j) 的 Beta 积分
        c = comb(deg, j) * ((-1) ** (deg - j)) / (2.0 ** deg)
        # ∫ (1+x)^{β+j} (1-x)^{α+deg-j} dx = 2^{α+β+deg+1} B(α+deg-j+1, β+j+1)
        from scipy.special import beta as beta_fn
        val += c * (2.0 ** (alpha + beta + deg + 1)) * \
               beta_fn(alpha + deg - j + 1, beta + j + 1)
    return val


# ----------------------------------------------------------------------
# 多变量稀疏多项式基
# ----------------------------------------------------------------------
def multi_index_set(d, p, hyperbolic_cross=False, q=0.5):
    """生成多指标集 {α ∈ N^d : |α|_q ≤ p}.

    参数:
        d              : 变量维数
        p              : 总阶上界
        hyperbolic_cross : 若为 True 则采用双曲交叉截断 (||α||_q ≤ p, q<1)
        q              : 准范数指数 (0<q≤1), q=1 为全阶截断
    返回:
        indices : (M, d) 整数数组, M = C(d+p, d) 或更小
    """
    if d == 1:
        return np.arange(p + 1).reshape(-1, 1)

    def _recurse(dim, remaining):
        if dim == 1:
            return [[r] for r in range(remaining + 1)]
        out = []
        for k in range(remaining + 1):
            for tail in _recurse(dim - 1, remaining - k):
                out.append([k] + tail)
        return out

    if hyperbolic_cross and q < 1.0:
        # 双曲交叉: 按 sum(α_j^{1/q}) ≤ p 筛选
        cand = _recurse(d, int(p / q) + 1)
        out = []
        for a in cand:
            s = sum((aj ** (1.0 / q)) for aj in a)
            if s <= p + 1e-12:
                out.append(a)
        indices = np.array(out, dtype=int)
    else:
        indices = np.array(_recurse(d, p), dtype=int)
    return indices


def pce_basis_eval(X, indices, basis='legendre'):
    """在采样点 X 上评估多维 PCE 基 Ψ_α(x).

    输入:
        X        : (n, d)
        indices  : (M, d) 多指标
        basis    : 'legendre' | 'jacobi' | 'monomial'
    返回:
        Psi : (n, M) 基矩阵
    """
    X = np.asarray(X, dtype=float)
    n, d = X.shape
    indices = np.asarray(indices, dtype=int)
    M = indices.shape[0]
    Psi = np.ones((n, M))

    for j in range(d):
        max_k = int(indices[:, j].max())
        if basis == 'legendre':
            Lj = legendre_eval(X[:, j], max_k)
        elif basis == 'jacobi':
            Lj = jacobi_eval(X[:, j], 0.5, 0.5, max_k)  # Chebyshev-I 权重
        elif basis == 'monomial':
            Lj = monomial_value_1d(X[:, j], max_k)
        else:
            raise ValueError(f"未知基类型: {basis}")
        for m in range(M):
            k = indices[m, j]
            Psi[:, m] *= Lj[:, k]
    return Psi


def gram_matrix(X, indices, basis='legendre', weights=None):
    """经验 Gram 矩阵 G = Ψ^T W Ψ,  W = diag(weights).

    条件数 cond(G) 直接决定 L1 恢复的 RIP 常数.
    """
    Psi = pce_basis_eval(X, indices, basis=basis)
    if weights is None:
        weights = np.ones(X.shape[0]) / X.shape[0]
    W = np.diag(weights)
    G = Psi.T @ W @ Psi
    return G
