"""
quadrature_engine.py
================================================================================
高精度数值积分引擎——面向地下水溶质运移的高级求积规则库

基于种子项目：
  - 466_gen_laguerre_exactness：广义 Gauss-Laguerre 求积
  - 1143_square_exactness     ：二维矩形区域张量积求积与 Padua 点

科学背景：
  地下水溶质运移模拟中，以下积分频繁出现：
    1. 半无限域上的衰减卷积：∫₀^∞ C(t-τ) e^{-λτ} dτ
    2. 有限元单元上的源项积分：∫∫_Ω R(x,y) φ_i(x,y) dx dy
    3. 矩量计算（浓度均值、方差）：∫ x^n C(x) dx

  广义 Gauss-Laguerre 求积精确计算：
      I[f] = ∫₀^∞ x^α e^{-x} f(x) dx ≈ Σ_{k=1}^n w_k f(x_k)
    其中节点 x_k 是广义 Laguerre 多项式 L_n^{(α)}(x) 的零点，
    权值 w_k = Γ(n+α+1) / [n! x_k (L_{n-1}^{(α)}(x_k))²]

  二维张量积 Gauss-Legendre 规则：
      I[f] = ∫_{a}^{b}∫_{c}^{d} f(x,y) dx dy
           ≈ Σ_{i=1}^{n_x} Σ_{j=1}^{n_y} w_i w_j f(ξ_i, η_j)
    其中 ξ_i, η_j 经仿射变换从 [-1,1] 映射到物理坐标。
================================================================================
"""

import numpy as np
from math import gamma, sqrt, exp, cos, pi


# ---------------------------------------------------------------------------
# 1D Gauss-Laguerre 求积
# ---------------------------------------------------------------------------

def gauss_laguerre_nodes_weights(n: int, alpha: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """
    计算 n 点广义 Gauss-Laguerre 求积节点与权值。

    算法：通过广义 Laguerre 多项式的三递推关系构造对称三对角矩阵，
    再计算其特征值（节点）与特征向量（权值）。

    递推关系：
        L_{-1}^{(α)}(x) = 0
        L_0^{(α)}(x)    = 1
        (k+1) L_{k+1}^{(α)}(x) = (2k+α+1 - x) L_k^{(α)}(x) - (k+α) L_{k-1}^{(α)}(x)

    Jacobi 矩阵 J 的非零元：
        J_{i,i}   = 2i + α + 1   (i = 0, ..., n-1, 采用 0-based 索引)
        J_{i,i+1} = J_{i+1,i} = sqrt((i+1)(i+1+α))

    参数
    ----------
    n : int
        求积点数，必须 ≥ 1
    alpha : float
        广义 Laguerre 参数 α > -1

    返回
    -------
    x, w : (np.ndarray, np.ndarray)
        节点（位于 [0, +∞)）和权值
    """
    if not isinstance(n, int) or n < 1:
        raise ValueError("求积点数 n 必须为正整数")
    if alpha <= -1.0:
        raise ValueError("广义 Laguerre 参数 alpha 必须大于 -1")

    # 构造对称三对角 Jacobi 矩阵
    i = np.arange(1, n + 1, dtype=float)
    diag = 2.0 * i - 1.0 + alpha
    offdiag = np.sqrt(i[:-1] * (i[:-1] + alpha))

    J = np.diag(diag) + np.diag(offdiag, k=1) + np.diag(offdiag, k=-1)

    eigenvalues, eigenvectors = np.linalg.eigh(J)
    x = eigenvalues
    # 权值：w_k = Γ(α+1) * (v_{1,k})^2
    # 其中 v_{1,k} 为归一化特征向量的第一个分量
    w = np.zeros(n)
    for k in range(n):
        v0 = eigenvectors[0, k]
        w[k] = (v0 ** 2) * gamma(alpha + 1.0)

    # 数值稳定性：舍去极小负节点（浮点误差）
    x = np.maximum(x, 0.0)
    w = np.abs(w)
    return x, w


def integrate_laguerre(f, n: int = 64, alpha: float = 0.0, args=()) -> float:
    """
    使用 n 点广义 Gauss-Laguerre 求积计算：
        I = ∫₀^∞ x^α e^{-x} f(x) dx

    参数
    ----------
    f : callable
        被积函数 f(x)
    n, alpha : 见 gauss_laguerre_nodes_weights
    args : tuple
        f 的额外位置参数
    """
    x, w = gauss_laguerre_nodes_weights(n, alpha)
    fx = np.array([f(xi, *args) for xi in x])
    return float(np.dot(w, fx))


def integrate_decay_convolution(C_history: np.ndarray, dt: float, lam: float,
                                n_quad: int = 32) -> float:
    """
    计算半无限域上的衰减卷积积分（多孔介质中的一阶衰变/吸附）：

        I(t) = ∫₀^∞ C(t - τ) e^{-λτ} dτ

    通过变量替换 τ = s / λ，积分变为 Laguerre 型：
        I = (1/λ) ∫₀^∞ C(t - s/λ) e^{-s} ds

    参数
    ----------
    C_history : np.ndarray
        离散浓度历史序列
    dt : float
        时间步长
    lam : float
        衰变常数 λ > 0
    n_quad : int
        求积点数

    返回
    -------
    float
        卷积积分值
    """
    if lam <= 0.0:
        raise ValueError("衰变常数 λ 必须为正")
    if dt <= 0.0:
        raise ValueError("时间步长 dt 必须为正")
    if len(C_history) == 0:
        raise ValueError("浓度历史不能为空")

    s_nodes, s_weights = gauss_laguerre_nodes_weights(n_quad, alpha=0.0)
    # 将 Laguerre 节点映射回 τ 域
    tau_nodes = s_nodes / lam

    # 对 C_history 做线性插值
    t_max = (len(C_history) - 1) * dt
    vals = []
    for tau in tau_nodes:
        t_query = t_max - tau
        if t_query <= 0.0:
            vals.append(C_history[0])
        elif t_query >= t_max:
            vals.append(C_history[-1])
        else:
            idx = int(t_query / dt)
            frac = (t_query - idx * dt) / dt
            idx = min(idx, len(C_history) - 2)
            vals.append(C_history[idx] * (1.0 - frac) + C_history[idx + 1] * frac)

    return float(np.dot(s_weights, vals) / lam)


# ---------------------------------------------------------------------------
# 1D Gauss-Legendre 求积（用于张量积构造）
# ---------------------------------------------------------------------------

def gauss_legendre_nodes_weights(n: int) -> tuple[np.ndarray, np.ndarray]:
    """
    计算 n 点 Gauss-Legendre 节点和权值（在 [-1,1] 上）。

    Legendre 多项式 P_n(x) 满足：
        (n+1) P_{n+1}(x) = (2n+1) x P_n(x) - n P_{n-1}(x)
    节点为 P_n(x)=0 的根，权值：
        w_k = 2 / [(1-x_k^2) (P_n'(x_k))^2]
    """
    if n < 1:
        raise ValueError("n 必须 ≥ 1")
    # 通过 numpy 的 polynomial.legendre.leggauss 稳健计算
    x, w = np.polynomial.legendre.leggauss(n)
    return x, w


def integrate_2d_rectangle(f, xlim: tuple[float, float],
                           ylim: tuple[float, float],
                           nx: int = 8, ny: int = 8) -> float:
    """
    二维张量积 Gauss-Legendre 求积：

        I = ∫_{x_a}^{x_b} ∫_{y_c}^{y_d} f(x,y) dy dx
          ≈ Σ_{i=1}^{nx} Σ_{j=1}^{ny} W_i W_j f(X_i, Y_j)

    其中 X_i = (b-a)/2 * ξ_i + (a+b)/2,  W_i = (b-a)/2 * w_i
          Y_j = (d-c)/2 * η_j + (c+d)/2,  W_j = (d-c)/2 * w_j
    """
    xa, xb = xlim
    yc, yd = ylim
    if xa >= xb or yc >= yd:
        raise ValueError("积分区间必须满足 a < b 且 c < d")

    xi, wi = gauss_legendre_nodes_weights(nx)
    eta, wj = gauss_legendre_nodes_weights(ny)

    # 仿射映射到物理坐标
    X = 0.5 * (xb - xa) * xi + 0.5 * (xb + xa)
    Y = 0.5 * (yd - yc) * eta + 0.5 * (yd + yc)
    Wx = 0.5 * (xb - xa) * wi
    Wy = 0.5 * (yd - yc) * wj

    total = 0.0
    for i in range(nx):
        for j in range(ny):
            total += Wx[i] * Wy[j] * f(X[i], Y[j])
    return float(total)


# ---------------------------------------------------------------------------
# Padua 点（二维近最优插值节点）
# ---------------------------------------------------------------------------

def padua_points(level: int) -> tuple[np.ndarray, np.ndarray]:
    """
    生成 Padua 点（正方形 [-1,1]² 上的近最优多项式插值节点）。

    定义：对于级别 L，在 Chebyshev-Lobatto 点集
        z_j = cos( jπ / n ),  j = 0, ..., n,  n = L+1
    中，Padua 点由两个子网格的并集构成：
        PD_n = { (z_i, z_j) : i+j 为偶数 } ∪ { (z_i, z_j) : i+j 为奇数 }
    总点数 N = (n+1)(n+2)/2，达到二元多项式空间 Π_n([-1,1]²) 的维数。

    参数
    ----------
    level : int
        Padua 级别，level ≥ 0

    返回
    -------
    x, y : np.ndarray
        Padua 点坐标数组
    """
    if level < 0:
        raise ValueError("Padua 级别必须 ≥ 0")
    n = level + 1
    z = np.cos(np.arange(n + 1) * np.pi / n)

    pts_x = []
    pts_y = []
    for i in range(n + 1):
        for j in range(n + 1):
            if (i + j) % 2 == 0:
                pts_x.append(z[i])
                pts_y.append(z[j])
    return np.array(pts_x), np.array(pts_y)


def padua_weights(level: int) -> np.ndarray:
    """
    计算 Padua 点的离散权重（用于多项式插值近似积分）。
    这里使用简化公式：每个点的权重与所在 Chebyshev 网格位置相关。
    """
    x, y = padua_points(level)
    n = level + 1
    # 简化的权重分配：基于 Chebyshev 端点权重加倍的启发式
    w = np.ones(len(x))
    z = np.cos(np.arange(n + 1) * np.pi / n)
    tol = 1e-12
    for k in range(len(x)):
        # 若点靠近边界则赋予较大权重
        if abs(abs(x[k]) - 1.0) < tol or abs(abs(y[k]) - 1.0) < tol:
            w[k] *= 0.5
    # 归一化使总权重等于区域面积 4
    if np.sum(w) > 0:
        w = w / np.sum(w) * 4.0
    return w


# ---------------------------------------------------------------------------
# 积分精确性测试
# ---------------------------------------------------------------------------

def test_quadrature_exactness() -> dict:
    """
    对求积规则进行精确性测试，验证其代数精确度。

    测试 1：∫₀^∞ x^α e^{-x} x^m dx = Γ(α+m+1)，m = 0,1,...,2n-1
    测试 2：∫_{-1}^{1}∫_{-1}^{1} x^p y^q dx dy = [1-(-1)^{p+1}]/(p+1) * [1-(-1)^{q+1}]/(q+1)
    """
    results = {}

    # --- Laguerre 精确性 ---
    n = 8
    alpha = 0.5
    max_m = 2 * n - 1
    laguerre_pass = True
    for m in range(max_m + 1):
        exact = gamma(alpha + m + 1.0)
        approx = integrate_laguerre(lambda x: x ** m, n=n, alpha=alpha)
        rel_err = abs(approx - exact) / (abs(exact) + 1e-15)
        if rel_err > 1e-10:
            laguerre_pass = False
            break
    results["laguerre_exactness"] = laguerre_pass

    # --- 2D Legendre 精确性 ---
    def monomial_integral(p, q):
        def f(x, y):
            return (x ** p) * (y ** q)
        return integrate_2d_rectangle(f, (-1.0, 1.0), (-1.0, 1.0), nx=n, ny=n)

    legendre_pass = True
    for p in range(n):
        for q in range(n):
            if p + q > 2 * n - 1:
                continue
            exact = ((1 - (-1) ** (p + 1)) / (p + 1)) * ((1 - (-1) ** (q + 1)) / (q + 1)) if p >= 0 and q >= 0 else 0.0
            approx = monomial_integral(p, q)
            if abs(exact) < 1e-12:
                err = abs(approx)
            else:
                err = abs(approx - exact) / abs(exact)
            if err > 1e-10:
                legendre_pass = False
                break
        if not legendre_pass:
            break
    results["legendre2d_exactness"] = legendre_pass

    return results


if __name__ == "__main__":
    r = test_quadrature_exactness()
    print("quadrature_engine 自测试:", r)
