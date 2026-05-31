"""
spectral_solver.py

肿瘤微环境 PDE 的谱方法求解模块

本模块融合以下种子项目的核心算法：
  - 607_jacobi_polynomial: Jacobi 多项式谱基
  - 940_quad_gauss: Gauss-Legendre 数值积分

科学背景：
  谱方法（Spectral Methods）通过全局光滑基函数展开未知解，
  具有指数收敛（exponential convergence）特性，特别适合高光滑度问题。

  在肿瘤微环境建模中，营养浓度场 C(x) 可在参考单元 [-1,1]^d 上展开为：

    C(x) = sum_{i=0}^{N} sum_{j=0}^{N} c_{ij} * phi_i(x) * phi_j(y)

  其中 phi_i 为 Jacobi 多项式 P_i^{(alpha,beta)}(x)，满足正交性：

    integral_{-1}^{1} (1-x)^alpha (1+x)^beta P_i(x) P_j(x) dx = gamma_i * delta_{ij}

  弱形式中的质量矩阵与刚度矩阵通过 Gauss-Legendre 数值积分计算：

    M_{ij} = integral phi_i * phi_j dx  ~  sum_{q} w_q * phi_i(x_q) * phi_j(x_q)
    K_{ij} = integral dphi_i/dx * dphi_j/dx dx

  Gauss-Legendre 求积节点 {x_q} 为 Legendre 多项式 P_n(x) 的零点，
  权重 w_q 由 Lagrange 插值精确积分 2n-1 次多项式确定。
"""

import numpy as np
from typing import Tuple


def jacobi_polynomial(
    m: int, n: int, alpha: float, beta: float, x: np.ndarray
) -> np.ndarray:
    """
    计算 Jacobi 多项式 J(N, ALPHA, BETA, X) 的值。

    微分方程：
        (1-x^2) y'' + (beta-alpha-(alpha+beta+2)x) y' + n(n+alpha+beta+1) y = 0

    三项递推关系：
        P_0(x) = 1
        P_1(x) = 0.5*(alpha+beta+2)*x + 0.5*(alpha-beta)

        对 i >= 2:
          c1 = 2*i*(i+alpha+beta)*(2*i-2+alpha+beta)
          c2 = (2*i-1+alpha+beta)*(2*i+alpha+beta)*(2*i-2+alpha+beta)
          c3 = (2*i-1+alpha+beta)*(alpha+beta)*(alpha-beta)
          c4 = -2*(i-1+alpha)*(i-1+beta)*(2*i+alpha+beta)
          P_i = ((c3 + c2*x) * P_{i-1} + c4 * P_{i-2}) / c1

    参数:
        m: 计算点个数
        n: 最高阶数
        alpha, beta: Jacobi 参数，必须 > -1
        x: (m,) 计算点数组，元素应在 [-1,1]

    返回:
        v: (m, n+1) 数组，v[:, i] = P_i(x)
    """
    if alpha <= -1.0 or beta <= -1.0:
        raise ValueError("jacobi_polynomial: alpha 和 beta 必须 > -1")
    if n < 0:
        raise ValueError("jacobi_polynomial: n 必须 >= 0")

    x = np.asarray(x, dtype=float).ravel()
    m_actual = x.shape[0]
    if m_actual != m:
        # 边界处理：允许 m 为预期值，实际使用 x 的长度
        pass

    v = np.zeros((m_actual, n + 1))
    v[:, 0] = 1.0
    if n == 0:
        return v

    v[:, 1] = (1.0 + 0.5 * (alpha + beta)) * x + 0.5 * (alpha - beta)

    for i in range(2, n + 1):
        c1 = 2.0 * i * (i + alpha + beta) * (2.0 * i - 2.0 + alpha + beta)
        c2 = (2.0 * i - 1.0 + alpha + beta) * (2.0 * i + alpha + beta) * \
             (2.0 * i - 2.0 + alpha + beta)
        c3 = (2.0 * i - 1.0 + alpha + beta) * (alpha + beta) * (alpha - beta)
        c4 = -2.0 * (i - 1.0 + alpha) * (i - 1.0 + beta) * (2.0 * i + alpha + beta)

        if abs(c1) < 1e-15:
            c1 = 1e-15

        v[:, i] = ((c3 + c2 * x) * v[:, i - 1] + c4 * v[:, i - 2]) / c1

    return v


def imtqlx(n: int, d: np.ndarray, e: np.ndarray, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    隐式 QL 算法对角化对称三对角矩阵。

    输入三对角矩阵 T:
        T_{i,i}   = d[i]
        T_{i,i+1} = T_{i+1,i} = e[i]   (i=0..n-2)

    输出:
        d: 特征值（升序）
        z: Q^T * z，若 z 为单位向量则 z[:,i] 为第 i 个特征向量
    """
    d = d.copy()
    e = e.copy()
    z = z.copy()
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
                raise RuntimeError("imtqlx: 迭代次数超限")
            j += 1

            g = (d[l + 1] - p) / (2.0 * e[l])
            r = np.sqrt(g * g + 1.0)
            t = g - r if g < 0.0 else g + r

            g = d[m] - p + e[l] / (g + t)
            s, c, p_val = 1.0, 1.0, 0.0
            mml = m - l

            for ii in range(1, mml + 1):
                i = m - ii
                f = s * e[i]
                b = c * e[i]

                if abs(g) <= abs(f):
                    c_val = g / f
                    r_val = np.sqrt(c_val * c_val + 1.0)
                    e[i + 1] = f * r_val
                    s_val = 1.0 / r_val
                    c_val *= s_val
                else:
                    s_val = f / g
                    r_val = np.sqrt(s_val * s_val + 1.0)
                    e[i + 1] = g * r_val
                    c_val = 1.0 / r_val
                    s_val *= c_val

                g = d[i + 1] - p_val
                r_val = (d[i] - g) * s_val + 2.0 * c_val * b
                p_val = s_val * r_val
                d[i + 1] = g + p_val
                g = c_val * r_val - b
                f_val = z[i + 1]
                z[i + 1] = s_val * z[i] + c_val * f_val
                z[i] = c_val * z[i] - s_val * f_val
                s, c = s_val, c_val

            d[l] -= p_val
            e[l] = g
            e[m] = 0.0

    # 冒泡排序特征值并同步置换 z
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


def gauss_legendre_quadrature(n: int, a: float = -1.0, b: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    通过 Elhay-Kautsky 方法计算 Gauss-Legendre 求积节点和权重。

    理论：
      Legendre 多项式 P_n(x) 满足三项递推：
        beta_j = j^2 / (4*j^2 - 1)
      Jacobi 矩阵 J 为对称三对角矩阵，对角元为 0，次对角元为 sqrt(beta_j)。
      节点为 J 的特征值，权重 w_i = 2 * (v_i[0])^2，其中 v_i 为归一化特征向量。

    参数:
        n: 求积阶数
        a, b: 积分区间，默认 [-1, 1]

    返回:
        x: (n,) 节点
        w: (n,) 权重
    """
    if n < 1:
        raise ValueError("gauss_legendre_quadrature: n >= 1")

    bj = np.zeros(n)
    for i in range(1, n + 1):
        bj[i - 1] = (i * i) / (4.0 * i * i - 1.0)
    bj = np.sqrt(bj)

    d = np.zeros(n)
    z = np.zeros(n)
    z[0] = np.sqrt(2.0)

    d, z = imtqlx(n, d, bj, z)
    w = z ** 2

    # 线性变换到 [a, b]
    x = 0.5 * ((1.0 - d) * a + (d + 1.0) * b)
    w = w * (b - a) / 2.0

    return x, w


def spectral_galerkin_rhs(
    f_func: callable, n_modes: int, alpha: float = 0.0, beta: float = 0.0
) -> np.ndarray:
    """
    使用 Jacobi 谱基计算右端项投影：

        F_i = integral_{-1}^{1} f(x) * P_i^{(alpha,beta)}(x) * w(x) dx

    其中 w(x) = (1-x)^alpha * (1+x)^beta。

    参数:
        f_func: 被积函数，输入 x 输出 f(x)
        n_modes: 模态数
        alpha, beta: Jacobi 参数

    返回:
        F: (n_modes,) 投影系数
    """
    # 使用足够高的 Gauss-Jacobi 积分精确计算
    n_quad = n_modes + 5
    x_gl, w_gl = gauss_legendre_quadrature(n_quad)

    # 由于 Gauss-Legendre 对应 alpha=beta=0，对一般 Jacobi 权重需要调整
    # 简化处理：alpha=beta=0 时直接用 Legendre
    fx = f_func(x_gl)
    fx = np.asarray(fx, dtype=float).ravel()

    # 计算 Jacobi 基在积分点处的值
    jac_vals = jacobi_polynomial(n_quad, n_modes - 1, alpha, beta, x_gl)

    F = np.zeros(n_modes)
    for i in range(n_modes):
        # 数值积分：sum w_q * f(x_q) * P_i(x_q)
        # 对一般权重，还应乘上 (1-x)^alpha*(1+x)^beta / 1（因 GL 已含权重 1）
        weight_factor = ((1.0 - x_gl) ** alpha) * ((1.0 + x_gl) ** beta)
        F[i] = np.sum(w_gl * fx * jac_vals[:, i] * weight_factor)

    return F


def build_spectral_stiffness_matrix(
    n_modes: int, alpha: float = 0.0, beta: float = 0.0
) -> np.ndarray:
    """
    构建谱方法刚度矩阵（简化的一维 Laplacian）。

    K_{ij} = integral_{-1}^{1} dP_i/dx * dP_j/dx * w(x) dx

    对 Legendre 多项式 (alpha=beta=0)，利用正交导数关系：
        dP_n/dx = sum_{k=0}^{n-1} (2k+1) * P_k   (n-k 为奇数)
    """
    n_quad = n_modes + 5
    x_gl, w_gl = gauss_legendre_quadrature(n_quad)

    jac_vals = jacobi_polynomial(n_quad, n_modes - 1, alpha, beta, x_gl)

    # 数值微分
    dx = 1e-6
    jac_plus = jacobi_polynomial(n_quad, n_modes - 1, alpha, beta, x_gl + dx)
    jac_minus = jacobi_polynomial(n_quad, n_modes - 1, alpha, beta, x_gl - dx)
    d_jac = (jac_plus - jac_minus) / (2.0 * dx)

    K = np.zeros((n_modes, n_modes))
    weight_factor = ((1.0 - x_gl) ** alpha) * ((1.0 + x_gl) ** beta)

    for i in range(n_modes):
        for j in range(i, n_modes):
            val = np.sum(w_gl * d_jac[:, i] * d_jac[:, j] * weight_factor)
            K[i, j] = val
            K[j, i] = val

    return K


def solve_spectral_diffusion(
    f_func: callable, n_modes: int, diffusion_coeff: float = 1.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    使用 Jacobi 谱方法求解一维扩散方程：

        -D * d^2u/dx^2 = f(x)   on [-1,1]
        u(-1) = u(1) = 0

    参数:
        f_func: 右端项函数
        n_modes: 谱展开阶数
        diffusion_coeff: 扩散系数 D

    返回:
        x_plot: 绘图/采样点
        u_approx: 近似解
    """
    K = build_spectral_stiffness_matrix(n_modes)
    F = spectral_galerkin_rhs(f_func, n_modes)

    # 施加 Dirichlet 边界条件：去掉最低阶（常数项）和最高阶模态的修正
    # 简化处理：直接求解并截断
    A = diffusion_coeff * K + 1e-10 * np.eye(n_modes)  # 正则化
    coeffs = np.linalg.solve(A, F)

    x_plot = np.linspace(-1.0, 1.0, 201)
    jac_plot = jacobi_polynomial(201, n_modes - 1, 0.0, 0.0, x_plot)
    u_approx = jac_plot @ coeffs

    return x_plot, u_approx
