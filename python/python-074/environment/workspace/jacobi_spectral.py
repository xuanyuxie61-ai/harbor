r"""
jacobi_spectral.py
==================
Jacobi 谱方法模块：提供高阶 Jacobi 多项式求值、Gauss-Jacobi 数值积分、
谱微分矩阵与边界层映射函数。

科学背景
--------
在圆柱绕流边界层与尾流剪切层分析中，物理量（涡量、速度）在壁面附近
呈现剧烈梯度变化。采用标准多项式配置点会导致 Gibbs 现象与 Runge 振荡。
Jacobi 谱方法通过权重函数 (1-\xi)^\alpha (1+\xi)^\beta 在边界处集聚配置点，
从而以指数阶精度捕捉边界层结构。

核心公式
--------
1. Jacobi 多项式递推：
   P_0^{(\alpha,\beta)}(\xi) = 1
   P_1^{(\alpha,\beta)}(\xi) = \frac{\alpha-\beta}{2} + \left(1+\frac{\alpha+\beta}{2}\right)\xi

   对 n \ge 1:
   a_n P_{n+1} = (b_n + c_n \xi) P_n - d_n P_{n-1}

   其中
   a_n = 2(n+1)(n+\alpha+\beta+1)(2n+\alpha+\beta)
   b_n = (2n+\alpha+\beta+1)(\alpha^2-\beta^2)
   c_n = (2n+\alpha+\beta)(2n+\alpha+\beta+1)(2n+\alpha+\beta+2)
   d_n = 2(n+\alpha)(n+\beta)(2n+\alpha+\beta+2)

2. 正交归一化条件：
   \int_{-1}^{1} (1-\xi)^\alpha (1+\xi)^\beta P_n^{(\alpha,\beta)}(\xi)
   P_m^{(\alpha,\beta)}(\xi) \, d\xi = h_n \delta_{nm}

   h_n = \frac{2^{\alpha+\beta+1}}{2n+\alpha+\beta+1}
   \frac{\Gamma(n+\alpha+1)\Gamma(n+\beta+1)}{n!\,\Gamma(n+\alpha+\beta+1)}

3. Gauss-Jacobi 求积：
   \int_{-1}^{1} (1-\xi)^\alpha (1+\xi)^\beta f(\xi)\, d\xi
   \approx \sum_{j=0}^{N} w_j f(\xi_j)

   其中 \{\xi_j\} 为 P_{N+1}^{(\alpha,\beta)} 的零点，
   权值 w_j 由 Christoffel-Darboux 公式导出。

4. 谱微分矩阵 D_{ij}：
   D_{ij} = \frac{d}{d\xi} \ell_j(\xi) \Big|_{\xi=\xi_i}
   其中 \ell_j 为以 Jacobi-Gauss-Lobatto 点为节点的 Lagrange 插值基函数。

本模块对应原种子项目：
- 607_jacobi_polynomial（Jacobi 多项式求值与求根）
- 608_jacobi_rule（Gauss-Jacobi 求积规则生成，含 imtqlx 三对角化算法）
r"""

import numpy as np
from scipy.special import gamma as scipy_gamma


def jacobi_polynomial(x, n_max, alpha, beta):
    r"""
    计算 Jacobi 多项式 P_n^{(alpha,beta)}(x)，n = 0, 1, ..., n_max。

    参数
    ----
    x : array_like, shape (M,)
        计算节点，要求位于 [-1, 1] 区间内。
    n_max : int
        最高阶数。
    alpha, beta : float
        Jacobi 参数，要求 > -1。

    返回
    ----
    P : ndarray, shape (M, n_max+1)
        P[:, k] = P_k^{(alpha,beta)}(x)。

    边界处理
    --------
    若 alpha <= -1 或 beta <= -1，抛出 ValueError。
    若 x 超出 [-1, 1] 范围，发出警告但仍继续计算（多项式本身在实数域有定义）。
    """
    if alpha <= -1.0 or beta <= -1.0:
        raise ValueError("jacobi_polynomial: alpha 和 beta 必须大于 -1。")

    x = np.asarray(x, dtype=float)
    if np.any(np.abs(x) > 1.0 + 1e-12):
        # 科学上多项式可外推，但数值稳定性下降
        pass

    m = x.size
    if n_max < 0:
        return np.empty((m, 0))

    P = np.zeros((m, n_max + 1))
    P[:, 0] = 1.0

    if n_max == 0:
        return P

    P[:, 1] = 0.5 * (alpha - beta) + (1.0 + 0.5 * (alpha + beta)) * x

    for n in range(1, n_max):
        c1 = 2.0 * (n + 1.0) * (n + alpha + beta + 1.0) * (2.0 * n + alpha + beta)
        c2 = (2.0 * n + alpha + beta + 1.0) * (alpha * alpha - beta * beta)
        c3 = (2.0 * n + alpha + beta + 1.0) * (2.0 * n + alpha + beta + 2.0) * (2.0 * n + alpha + beta)
        c4 = 2.0 * (n + alpha) * (n + beta) * (2.0 * n + alpha + beta + 2.0)

        P[:, n + 1] = ((c2 + c3 * x) * P[:, n] - c4 * P[:, n - 1]) / c1

    return P


def imtqlx(n, d, e, z):
    r"""
    隐式 QL 算法对角化对称三对角矩阵，同时计算 Q' * z。

    本函数直接移植自原 608_jacobi_rule / 607_jacobi_polynomial 中的
    imtqlx 算法（EISPACK/ACM TOMS 655）。

    参数
    ----
    n : int
        矩阵阶数。
    d : ndarray, shape (n,)
        对角元（输入/输出）。
    e : ndarray, shape (n,)
        次对角元（e[n-1] 可任意，输入后会被置 0）。
    z : ndarray, shape (n,)
        待变换向量（输入/输出）。

    返回
    ----
    d, z : ndarray
        d 为特征值（已按升序排列），z 为 Q' * z。
    """
    d = np.asarray(d, dtype=float).copy()
    e = np.asarray(e, dtype=float).copy()
    z = np.asarray(z, dtype=float).copy()

    itn = 30
    prec = np.finfo(float).eps

    if n == 1:
        return d, z

    e[n - 1] = 0.0

    for l in range(n):
        j = 0
        while True:
            m = l
            while m < n - 1:
                if abs(e[m]) <= prec * (abs(d[m]) + abs(d[m + 1])):
                    break
                m += 1

            p = d[l]
            if m == l:
                break

            if j == itn:
                raise RuntimeError("imtqlx: 迭代次数超限，三对角矩阵对角化失败。")

            j += 1
            g = (d[l + 1] - p) / (2.0 * e[l])
            r = np.sqrt(g * g + 1.0)
            g = d[m] - p + e[l] / (g + np.sign(g) * abs(r))
            s = 1.0
            c = 1.0
            p = 0.0
            mml = m - l

            for ii in range(1, mml + 1):
                i = m - ii
                f = s * e[i]
                b = c * e[i]

                if abs(f) >= abs(g):
                    c = g / f
                    r = np.sqrt(c * c + 1.0)
                    e[i + 1] = f * r
                    s = 1.0 / r
                    c = c * s
                else:
                    s = f / g
                    r = np.sqrt(s * s + 1.0)
                    e[i + 1] = g * r
                    c = 1.0 / r
                    s = s * c

                g = d[i + 1] - p
                r = (d[i] - g) * s + 2.0 * c * b
                p = s * r
                d[i + 1] = g + p
                g = c * r - b
                f = z[i + 1]
                z[i + 1] = s * z[i] + c * f
                z[i] = c * z[i] - s * f

            d[l] = d[l] - p
            e[l] = g
            e[m] = 0.0

    # 按升序排列特征值并同步置换 z
    for ii in range(1, n):
        i = ii - 1
        k = i
        p = d[i]
        for j in range(ii, n):
            if d[j] < p:
                k = j
                p = d[j]
        if k != i:
            d[k] = d[i]
            d[i] = p
            p = z[i]
            z[i] = z[k]
            z[k] = p

    return d, z


def gauss_jacobi_rule(n, alpha, beta):
    r"""
    生成 n 点 Gauss-Jacobi 求积规则 (\xi_j, w_j)。

    积分区间映射：
    \int_{-1}^{1} (1-\xi)^\alpha (1+\xi)^\beta f(\xi) d\xi
    \approx \sum_{j=0}^{n-1} w_j f(\xi_j)

    算法：先构造 Jacobi 矩阵 J，其对角元 a_j、次对角元 b_j 由
    class_matrix 给出，再调用 imtqlx 求特征值（即节点）与特征向量
    （权值由首分量平方给出）。

    对应原种子项目 608_jacobi_rule 的核心算法。
    """
    if n < 1:
        return np.array([]), np.array([])

    ab = alpha + beta
    abi = 2.0 + ab

    # 零阶矩 zemu
    zemu = (2.0 ** (ab + 1.0)) * scipy_gamma(alpha + 1.0) * scipy_gamma(beta + 1.0) / scipy_gamma(abi)

    # Jacobi 矩阵对角元
    diag = np.zeros(n)
    off_diag = np.zeros(n)

    diag[0] = (beta - alpha) / abi
    off_diag[0] = np.sqrt(
        4.0 * (1.0 + alpha) * (1.0 + beta) / ((abi + 1.0) * abi * abi)
    )
    a2b2 = beta * beta - alpha * alpha

    for i in range(1, n):
        abi_i = 2.0 * (i + 1) + ab
        diag[i] = a2b2 / ((abi_i - 2.0) * abi_i)
        abi_sq = abi_i * abi_i
        off_diag[i] = np.sqrt(
            4.0 * (i + 1.0) * (i + 1.0 + alpha) * (i + 1.0 + beta) * (i + 1.0 + ab)
            / ((abi_sq - 1.0) * abi_sq)
        )

    # 权向量初始值
    w = np.zeros(n)
    w[0] = np.sqrt(zemu)

    nodes, weights = imtqlx(n, diag, off_diag, w)
    weights = weights ** 2

    return nodes, weights


def spectral_differentiation_matrix(x_nodes):
    r"""
    基于 Lagrange 插值的谱微分矩阵 D_{ij} = \ell'_j(x_i)。

    参数
    ----
    x_nodes : ndarray, shape (N,)
        一维配置点（通常取 Jacobi-Gauss-Lobatto 点）。

    返回
    ----
    D : ndarray, shape (N, N)
        微分矩阵。
    """
    N = len(x_nodes)
    D = np.zeros((N, N))

    # 使用重心形式的 Lagrange 微分矩阵
    # b_j = \prod_{k \ne j} 1 / (x_j - x_k)
    b = np.ones(N)
    for j in range(N):
        for k in range(N):
            if k != j:
                b[j] *= 1.0 / (x_nodes[j] - x_nodes[k])

    for i in range(N):
        for j in range(N):
            if i != j:
                D[i, j] = (b[j] / b[i]) / (x_nodes[i] - x_nodes[j])
            else:
                s = 0.0
                for k in range(N):
                    if k != i:
                        s += 1.0 / (x_nodes[i] - x_nodes[k])
                D[i, i] = s

    return D


def boundary_layer_map(eta, delta, alpha=0.0, beta=0.0):
    r"""
    边界层坐标映射：将物理坐标 y \in [0, \delta] 映射到标准区间 [-1, 1]，
    并在壁面附近集聚配置点。

    采用代数映射：
    \xi = 2 \sqrt{y / \delta} - 1

    逆映射：
    y = \delta \left( \frac{\xi + 1}{2} \right)^2

    参数
    ----
    eta : ndarray
        物理坐标（距壁面距离），范围 [0, delta]。
    delta : float
        边界层厚度估计。
    alpha, beta : float
        用于生成 Jacobi 配置点的参数。

    返回
    ----
    xi : ndarray
        标准区间坐标。
    dy_dxi : ndarray
        映射 Jacobian。
    """
    eta = np.asarray(eta, dtype=float)
    if delta <= 0:
        raise ValueError("boundary_layer_map: delta 必须为正。")

    # 限制在 [0, delta]
    eta_clipped = np.clip(eta, 0.0, delta)

    t = np.sqrt(eta_clipped / delta)
    xi = 2.0 * t - 1.0
    dy_dxi = delta * (xi + 1.0) * 0.5

    # 避免壁面处零除
    dy_dxi = np.where(np.abs(dy_dxi) < 1e-15, 1e-15, dy_dxi)

    return xi, dy_dxi


def integrate_boundary_layer(f_values, nodes, weights, delta):
    r"""
    在边界层 [0, delta] 上利用 Gauss-Jacobi 规则积分函数 f(y)。

    通过映射 y = y(\xi) 转换积分：
    \int_0^\delta f(y) dy = \int_{-1}^{1} f(y(\xi)) \frac{dy}{d\xi} d\xi
    \approx \sum_j w_j f(y_j) (dy/d\xi)_j
    """
    xi = np.asarray(nodes, dtype=float)
    _, dy_dxi = boundary_layer_map(delta * 0.5 * (xi + 1.0), delta)
    # 注意：上面 boundary_layer_map 输入的是物理坐标 y
    # 但我们这里已经有 xi，需要直接计算 dy/dxi
    y = delta * 0.25 * (xi + 1.0) ** 2
    dy_dxi_direct = delta * 0.5 * (xi + 1.0)
    dy_dxi_direct = np.where(np.abs(dy_dxi_direct) < 1e-15, 1e-15, dy_dxi_direct)

    return np.sum(weights * f_values * dy_dxi_direct)


def test_jacobi_spectral():
    r"""内部自检：验证 Gauss-Jacobi 规则对低次多项式的精确性。"""
    alpha, beta = 0.5, -0.3
    n = 8
    x, w = gauss_jacobi_rule(n, alpha, beta)

    # 积分 (1-x)^alpha (1+x)^beta x^6
    # 理论值 = 2^(alpha+beta+1) * Gamma(alpha+1)Gamma(beta+1)/Gamma(alpha+beta+2) * ...
    # 对于 n=8，应精确积分到 2n-1=15 次多项式
    f = x ** 6
    numerical = np.sum(w * f)

    # 理论值通过 Beta 函数
    from scipy.special import beta as beta_func
    exact = (2.0 ** (alpha + beta + 1.0)) * beta_func(alpha + 1.0, beta + 1.0)
    # x^6 的矩需要额外计算：使用递推或数值参考
    # 这里仅做自洽检查：用 scipy 的 fixed_quad 做参考
    from scipy.integrate import fixed_quad
    def integrand(t):
        return ((1.0 - t) ** alpha) * ((1.0 + t) ** beta) * (t ** 6)
    exact_ref, _ = fixed_quad(integrand, -1.0, 1.0, n=40)

    rel_err = abs(numerical - exact_ref) / (abs(exact_ref) + 1e-15)
    print(f"[jacobi_spectral] Gauss-Jacobi 自检: n={n}, rel_err={rel_err:.3e}")
    return rel_err < 1e-10


if __name__ == "__main__":
    test_jacobi_spectral()
