"""
fem_approximation.py
基于种子项目 385_fem1d_approximate 的一维有限元数据拟合

在 r 过程核合成中，温度依赖的反应率、中子通量径向分布等
都可以通过一维有限元（分段线性帽函数）基来拟合。

对于区间 [x_0, x_n] 上的网格 x_0 < x_1 < ... < x_n，
帽函数 φ_i(x) 满足：
    φ_i(x_{i-1}) = 0, φ_i(x_i) = 1, φ_i(x_{i+1}) = 0
    φ_i(x) = (x - x_{i-1})/(x_i - x_{i-1})  for x ∈ [x_{i-1}, x_i]
    φ_i(x) = (x_{i+1} - x)/(x_{i+1} - x_i)  for x ∈ [x_i, x_{i+1}]

拟合函数：f(x) = Σ_{i=0}^n c_i φ_i(x)
系数通过加权最小二乘确定：
    min_{c} [ w_a ||Ac - y||² + w_d ||Dc||² + w_b ||Bc - b||² ]
其中 A 为数据点处的基函数矩阵，D 为二阶导数正则化，B 为边界约束。
"""

import numpy as np


def hat_function(x, x_left, x_center, x_right):
    """
    分段线性帽函数。

    参数:
        x : float 或 ndarray
        x_left, x_center, x_right : float

    返回:
        val : 同类型
    """
    x = np.asarray(x, dtype=float)
    val = np.zeros_like(x)
    mask1 = (x >= x_left) & (x <= x_center)
    if x_center > x_left:
        val[mask1] = (x[mask1] - x_left) / (x_center - x_left)
    mask2 = (x > x_center) & (x <= x_right)
    if x_right > x_center:
        val[mask2] = (x_right - x[mask2]) / (x_right - x_center)
    return val


def data_bracket(mesh, x_data):
    """
    对每个数据点，找到其所在的网格区间索引。

    参数:
        mesh : ndarray, 网格节点
        x_data : ndarray, 数据点坐标

    返回:
        indices : ndarray, 每个数据点左端点索引
    """
    mesh = np.asarray(mesh, dtype=float)
    x_data = np.asarray(x_data, dtype=float)
    indices = np.searchsorted(mesh, x_data, side='right') - 1
    indices = np.clip(indices, 0, len(mesh) - 2)
    return indices


def fem1d_approximate(mesh, x_data, y_data, weight_approx=1.0,
                       weight_deriv=0.01, weight_boundary=1e6,
                       boundary_values=None):
    """
    一维有限元数据拟合。

    参数:
        mesh : ndarray, 网格节点
        x_data : ndarray, 数据点坐标
        y_data : ndarray, 数据值
        weight_approx : float, 拟合权重
        weight_deriv : float, 二阶导数正则化权重
        weight_boundary : float, 边界条件权重
        boundary_values : tuple (y_left, y_right), 边界值约束

    返回:
        coeffs : ndarray, FEM 系数 c_i
    """
    mesh = np.asarray(mesh, dtype=float)
    x_data = np.asarray(x_data, dtype=float)
    y_data = np.asarray(y_data, dtype=float)
    n_nodes = len(mesh)
    n_data = len(x_data)

    if n_data == 0:
        return np.zeros(n_nodes)

    # 构造基函数矩阵 A: A[j,i] = φ_i(x_j)
    A = np.zeros((n_data, n_nodes))
    for i in range(n_nodes):
        x_l = mesh[max(0, i - 1)]
        x_c = mesh[i]
        x_r = mesh[min(n_nodes - 1, i + 1)]
        A[:, i] = hat_function(x_data, x_l, x_c, x_r)

    # 正则化矩阵 D（二阶导数惩罚，简化为相邻节点差分）
    D = np.zeros((n_nodes - 2, n_nodes))
    for i in range(n_nodes - 2):
        h1 = mesh[i + 1] - mesh[i]
        h2 = mesh[i + 2] - mesh[i + 1]
        if h1 > 0 and h2 > 0:
            D[i, i] = 1.0 / h1
            D[i, i + 1] = -1.0 / h1 - 1.0 / h2
            D[i, i + 2] = 1.0 / h2

    # 组装最小二乘系统
    M = weight_approx * (A.T @ A) + weight_deriv * (D.T @ D)
    rhs = weight_approx * (A.T @ y_data)

    # 边界条件
    if boundary_values is not None:
        y_left, y_right = boundary_values
        B = np.zeros((2, n_nodes))
        B[0, 0] = 1.0
        B[1, -1] = 1.0
        b_vec = np.array([y_left, y_right])
        M += weight_boundary * (B.T @ B)
        rhs += weight_boundary * (B.T @ b_vec)

    # 求解
    try:
        coeffs = np.linalg.solve(M, rhs)
    except np.linalg.LinAlgError:
        coeffs = np.linalg.lstsq(M, rhs, rcond=None)[0]

    return coeffs


def fem1d_evaluate(x, mesh, coeffs):
    """
    在任意点处求值 FEM 拟合函数。

    参数:
        x : float 或 ndarray
        mesh : ndarray, 网格节点
        coeffs : ndarray, FEM 系数

    返回:
        y : 同类型
    """
    x = np.asarray(x, dtype=float)
    mesh = np.asarray(mesh, dtype=float)
    y = np.zeros_like(x)
    for i in range(len(mesh)):
        x_l = mesh[max(0, i - 1)]
        x_c = mesh[i]
        x_r = mesh[min(len(mesh) - 1, i + 1)]
        y += coeffs[i] * hat_function(x, x_l, x_c, x_r)
    return y


def test_fem_approximation():
    """自包含测试"""
    mesh = np.linspace(0, 1, 21)
    x_data = np.random.rand(100)
    y_data = np.sin(2 * np.pi * x_data) + 0.1 * np.random.randn(100)
    coeffs = fem1d_approximate(mesh, x_data, y_data,
                                weight_approx=1.0, weight_deriv=0.1,
                                weight_boundary=1e4,
                                boundary_values=(0.0, 0.0))
    x_test = np.linspace(0, 1, 200)
    y_fit = fem1d_evaluate(x_test, mesh, coeffs)
    y_exact = np.sin(2 * np.pi * x_test)
    err = np.mean((y_fit - y_exact) ** 2)
    print(f"[fem_approximation] FEM fit MSE = {err:.3e}")


if __name__ == "__main__":
    test_fem_approximation()
