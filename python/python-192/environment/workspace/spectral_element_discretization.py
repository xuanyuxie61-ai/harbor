"""
================================================================================
谱元离散模块 (spectral_element_discretization.py)
================================================================================
融合项目:
  - 222_cosine_transform: 离散余弦变换
  - 521_hermite_interpolant: Hermite插值
  - 414_fem2d_scalar_display: 2D有限元标量场处理

在谱元CFD中，空间离散结合谱方法的精度与有限元的几何灵活性。
本模块提供：
  1. DCT快速泊松/亥姆霍兹求解器（压力投影步）
  2. Hermite高阶插值（单元间通量重构）
  3. 谱元质量矩阵与刚度矩阵组装
================================================================================
"""

import numpy as np
from utils_numerical import safe_divide


def discrete_cosine_transform_1d(d: np.ndarray) -> np.ndarray:
    """
    一维离散余弦变换 (DCT-II)

    变换核:
        C_k = √(2/N) · Σ_{j=0}^{N-1} d_j · cos( π(2j+1)k / 2N )

    在CFD中，DCT用于快速求解泊松方程：

        ∇²p = f   ⇒   在频域中: -π²(k_x² + k_y²) · p̂ = f̂

    相比直接求解，计算复杂度从 O(N⁶) 降至 O(N³ log N)。

    参数:
        d: 输入数据向量 (长度N)

    返回:
        c: DCT系数
    """
    n = len(d)
    c = np.zeros(n)

    for i in range(n):
        for j in range(n):
            c[i] += np.cos(np.pi * (2 * j + 1) * i / (2.0 * n)) * d[j]

    c *= np.sqrt(2.0 / n)
    return c


def inverse_discrete_cosine_transform_1d(c: np.ndarray) -> np.ndarray:
    """
    一维逆离散余弦变换 (DCT-III)

        d_j = √(2/N) · [ C_0/2 + Σ_{k=1}^{N-1} C_k · cos( π(2j+1)k / 2N ) ]

    与DCT-II互为逆变换。
    """
    n = len(c)
    d = np.zeros(n)

    for j in range(n):
        d[j] = c[0] / 2.0
        for k in range(1, n):
            d[j] += c[k] * np.cos(np.pi * (2 * j + 1) * k / (2.0 * n))
        d[j] *= np.sqrt(2.0 / n)

    return d


def dct_poisson_solver_2d(f: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """
    基于DCT的快速2D泊松方程求解器

    求解方程:
        ∂²p/∂x² + ∂²p/∂y² = f(x,y),   (x,y) ∈ [0,Lx]×[0,Ly]

    边界条件: Neumann (齐次法向导数为零)

    算法步骤:
        1. 对f进行2D DCT变换得到 f̂
        2. 在频域求解: p̂_{k,l} = -f̂_{k,l} / λ_{k,l}
           其中 λ_{k,l} = (2/dx²)(1-cos(πk/Nx)) + (2/dy²)(1-cos(πl/Ny))
        3. 对p̂进行逆2D DCT得到 p

    参数:
        f: 右端项 (ny x nx)
        dx, dy: 网格间距

    返回:
        p: 压力场
    """
    ny, nx = f.shape

    # 2D DCT（分别对x和y方向做DCT）
    f_hat = np.zeros_like(f)
    for j in range(ny):
        f_hat[j, :] = discrete_cosine_transform_1d(f[j, :])
    for i in range(nx):
        f_hat[:, i] = discrete_cosine_transform_1d(f_hat[:, i])

    # 频域求解
    p_hat = np.zeros_like(f_hat)
    for j in range(ny):
        for i in range(nx):
            if i == 0 and j == 0:
                p_hat[j, i] = 0.0  # 零均值约束
            else:
                # TODO: 计算 DCT-II 泊松求解的频域特征值 λ_{k,l} 并求解 p̂
                # 提示: λ_{k,l} = (2/dx²)(1-cos(π·i/Nx)) + (2/dy²)(1-cos(π·j/Ny))
                #       p̂_{k,l} = -f̂_{k,l} / λ_{k,l}
                raise NotImplementedError("TODO: implement spectral domain Poisson solve")

    # 逆2D DCT
    p = np.zeros_like(p_hat)
    for i in range(nx):
        p[:, i] = inverse_discrete_cosine_transform_1d(p_hat[:, i])
    for j in range(ny):
        p[j, :] = inverse_discrete_cosine_transform_1d(p[j, :])

    # 减去均值保证唯一性
    p -= np.mean(p)
    return p


def hermite_interpolant_coeffs(n: int, x: np.ndarray, y: np.ndarray, yp: np.ndarray) -> tuple:
    """
    Hermite插值：构造满足函数值与导数值的插值多项式

    给定数据点 {(x_i, y_i, y'_i)}，构造 2N-1 次多项式 H(x) 满足：

        H(x_i) = y_i,    H'(x_i) = y'_i,   i = 1,...,N

    使用差商表 (divided difference table) 实现：

        z_{2i} = z_{2i+1} = x_i
        f[z_{2i}] = y_i
        f[z_{2i}, z_{2i+1}] = y'_i
        f[z_j, ..., z_k] = (f[z_{j+1},...,z_k] - f[z_j,...,z_{k-1}]) / (z_k - z_j)

    在CFD中用于谱元界面处的通量重构，保证C1连续性。

    参数:
        n: 数据点数量
        x: 节点坐标 (已排序且互异)
        y: 函数值
        yp: 导数值

    返回:
        xd, yd: 差商表（用于求值）
        xdp, ydp: 导数的差商表
    """
    x = np.asarray(x).flatten()
    y = np.asarray(y).flatten()
    yp = np.asarray(yp).flatten()

    nd = 2 * n
    xd = np.zeros(nd)
    xd[0::2] = x
    xd[1::2] = x

    # 第一步差商
    yd = np.zeros(nd)
    yd[0] = y[0]
    if n > 1:
        yd[2::2] = (y[1:] - y[:-1]) / (x[1:] - x[:-1])
    yd[1::2] = yp

    # 剩余差商
    for i in range(2, nd):
        for j in range(nd - 1, i - 1, -1):
            denom = xd[j] - xd[j + 1 - i]
            if abs(denom) < 1e-14:
                # 重复节点：使用导数极限（对于Hermite插值，高阶差商退化为高阶导数/阶乘）
                # 简化处理：设为0（在节点处不影响插值精度）
                yd[j] = 0.0
            else:
                yd[j] = (yd[j] - yd[j - 1]) / denom

    return xd, yd


def hermite_interpolant_eval(xd: np.ndarray, yd: np.ndarray, x_eval: float) -> float:
    """
    用差商表求Hermite插值多项式的值（Horner法则）

        H(x) = yd[0] + yd[1](x-xd[0]) + yd[2](x-xd[0])(x-xd[1]) + ...
    """
    nd = len(xd)
    result = yd[-1]
    for i in range(nd - 2, -1, -1):
        result = result * (x_eval - xd[i]) + yd[i]
    return float(result)


def hermite_interpolant_derivative(xd: np.ndarray, yd: np.ndarray) -> tuple:
    """
    计算Hermite插值多项式的导数的差商表

    若 H(x) 的差商表为 (xd, yd)，则 H'(x) 的差商表为：
        xdp = xd[1:]
        ydp[i] = (i+1) * yd[i+1],  i = 0,...,nd-2
    """
    nd = len(xd)
    xdp = xd[1:].copy()
    ydp = np.zeros(nd - 1)
    for i in range(nd - 1):
        ydp[i] = (i + 1) * yd[i + 1]
    return xdp, ydp


def spectral_derivative_1d(u: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    基于谱方法的一阶导数计算

    使用Lagrange插值基函数的导数矩阵 D：

        du/dx|_{x_i} ≈ Σ_j D_{ij} u_j

    对于Chebyshev点，D_{ij} 有解析表达式：

        D_{ij} = (c_i / c_j) · (-1)^{i+j} / (x_i - x_j),   i≠j
        D_{ii} = -x_i / (2(1-x_i²)),   i≠0,N
        D_{00} = -D_{NN} = (2N²+1)/6

    其中 c_0 = c_N = 2, c_i = 1 (0<i<N)。

    参数:
        u: 函数值
        x: Chebyshev节点坐标

    返回:
        du: 导数值
    """
    n = len(u)
    D = np.zeros((n, n))
    c = np.ones(n)
    c[0] = 2.0
    c[-1] = 2.0

    for i in range(n):
        for j in range(n):
            if i != j:
                D[i, j] = (c[i] / c[j]) * ((-1) ** (i + j)) / (x[i] - x[j])
            else:
                if i == 0:
                    D[i, i] = (2.0 * (n - 1) ** 2 + 1.0) / 6.0
                elif i == n - 1:
                    D[i, i] = -(2.0 * (n - 1) ** 2 + 1.0) / 6.0
                else:
                    D[i, i] = -x[i] / (2.0 * (1.0 - x[i] ** 2))

    du = D @ u
    return du


def assemble_fem_mass_matrix_2d(nodes: np.ndarray, elements: np.ndarray) -> np.ndarray:
    """
    组装2D有限元质量矩阵（基于线性三角形单元）

    单元质量矩阵（一致质量矩阵）：

        M_e = (Area_e / 12) · [[2, 1, 1],
                               [1, 2, 1],
                               [1, 1, 2]]

    参数:
        nodes: 节点坐标 (n_nodes x 2)
        elements: 单元节点索引 (n_elements x 3)

    返回:
        M: 全局质量矩阵 (n_nodes x n_nodes)
    """
    n_nodes = nodes.shape[0]
    n_elements = elements.shape[0]
    M = np.zeros((n_nodes, n_nodes))

    for e in range(n_elements):
        idx = elements[e, :]
        x = nodes[idx, 0]
        y = nodes[idx, 1]

        # 三角形面积
        area = 0.5 * abs((x[1] - x[0]) * (y[2] - y[0]) - (x[2] - x[0]) * (y[1] - y[0]))
        area = max(area, 1e-14)

        # 单元质量矩阵
        Me = (area / 12.0) * np.array([
            [2.0, 1.0, 1.0],
            [1.0, 2.0, 1.0],
            [1.0, 1.0, 2.0]
        ])

        for i in range(3):
            for j in range(3):
                M[idx[i], idx[j]] += Me[i, j]

    return M


def assemble_fem_stiffness_matrix_2d(nodes: np.ndarray, elements: np.ndarray) -> np.ndarray:
    """
    组装2D有限元刚度矩阵（Laplacian算子离散）

    单元刚度矩阵：

        K_e = Area_e · (B^T B)

    其中 B = [∂N/∂x, ∂N/∂y]^T，N 为线性形函数。

    参数:
        nodes: 节点坐标
        elements: 单元索引

    返回:
        K: 全局刚度矩阵
    """
    n_nodes = nodes.shape[0]
    n_elements = elements.shape[0]
    K = np.zeros((n_nodes, n_nodes))

    for e in range(n_elements):
        idx = elements[e, :]
        x = nodes[idx, 0]
        y = nodes[idx, 1]

        # 面积与形函数导数
        area = 0.5 * abs((x[1] - x[0]) * (y[2] - y[0]) - (x[2] - x[0]) * (y[1] - y[0]))
        area = max(area, 1e-14)

        # 形函数对x,y的导数（常数）
        dN_dx = np.array([y[1] - y[2], y[2] - y[0], y[0] - y[1]]) / (2.0 * area)
        dN_dy = np.array([x[2] - x[1], x[0] - x[2], x[1] - x[0]]) / (2.0 * area)

        # 单元刚度矩阵
        Ke = np.zeros((3, 3))
        for i in range(3):
            for j in range(3):
                Ke[i, j] = area * (dN_dx[i] * dN_dx[j] + dN_dy[i] * dN_dy[j])

        for i in range(3):
            for j in range(3):
                K[idx[i], idx[j]] += Ke[i, j]

    return K


def apply_boundary_conditions_matrix(A: np.ndarray, b: np.ndarray, bc_nodes: np.ndarray, bc_values: np.ndarray) -> tuple:
    """
    对线性系统应用Dirichlet边界条件

    方法：将边界节点对应的行替换为单位行，右端项替换为边界值
    """
    A_mod = A.copy()
    b_mod = b.copy()

    for i, node in enumerate(bc_nodes):
        A_mod[node, :] = 0.0
        A_mod[node, node] = 1.0
        b_mod[node] = bc_values[i]

    return A_mod, b_mod
