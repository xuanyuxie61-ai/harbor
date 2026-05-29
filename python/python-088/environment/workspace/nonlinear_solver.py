"""
nonlinear_solver.py
非线性方程求解与特征值分析模块

融入种子项目:
  - 1404_wdk: Weierstrass-Durand-Kerner (WDK) 多项式求根算法
  - 710_mandelbrot: 复数迭代思想

功能:
  - WDK 多项式求根
  - 复数迭代方法
  - Newton-Raphson 非线性方程组求解
  - 多项式特征值问题
"""

import numpy as np
from typing import Optional, Tuple


def poly_eval(coeffs: np.ndarray, z: np.ndarray) -> np.ndarray:
    """
    计算多项式值 p(z) = c_0 + c_1 z + ... + c_d z^d。

    使用 Horner 法则:
        p(z) = c_0 + z(c_1 + z(c_2 + ... + z(c_d)...))

    参数:
        coeffs: 系数数组 [c_0, c_1, ..., c_d]
        z: 求值点（可为复数）

    返回:
        多项式值
    """
    result = np.zeros_like(z, dtype=complex)
    for c in reversed(coeffs):
        result = result * z + c
    return result


def wdk_roots(
    coeffs: np.ndarray, tol: float = 1e-12, max_iter: int = 1000
) -> np.ndarray:
    """
    Weierstrass-Durand-Kerner (WDK) 算法求多项式全部根。

    基于 1404_wdk 的核心算法。

    对于多项式:
        p(z) = c_d z^d + c_{d-1} z^{d-1} + ... + c_0

    WDK 迭代格式:
        z_i^{(k+1)} = z_i^{(k)} - \\frac{p(z_i^{(k)})}{\\\prod_{j \\ne i} (z_i^{(k)} - z_j^{(k)})}

    初始猜测取 Cauchy 界内的单位根:
        R = 1 + \\max_{0 \\le k \\le d} |c_k / c_d|
        z_i^{(0)} = R \\exp(2\\pi i (i-1)/d)

    WDK 具有二阶收敛性，且对简单根全局收敛（以概率1）。

    参数:
        coeffs: 多项式系数，从高次到低次 [c_d, c_{d-1}, ..., c_0]
        tol: 收敛容差
        max_iter: 最大迭代次数

    返回:
        根数组（复数）
    """
    d = len(coeffs) - 1
    if d < 1:
        return np.array([])

    # 标准化使最高次系数为 1
    leading = coeffs[0]
    if abs(leading) < 1e-15:
        raise ValueError("Leading coefficient is zero")
    coeffs = coeffs / leading

    # Cauchy 界
    R = 1.0 + np.max(np.abs(coeffs[1:]))

    # 初始猜测: 单位根
    theta = np.linspace(0, 2 * np.pi, d + 1)[:-1]
    roots = R * np.exp(1j * theta)

    for iteration in range(max_iter):
        roots_old = roots.copy()

        for i in range(d):
            zi = roots_old[i]
            # 计算分母: prod_{j != i} (zi - zj)
            denom = 1.0 + 0j
            for j in range(d):
                if i != j:
                    denom *= (zi - roots[j])
            if abs(denom) < 1e-30:
                denom = 1e-30
            roots[i] = zi - poly_eval(np.concatenate([[1.0], coeffs[1:]])[::-1], np.array([zi]))[0] / denom

        max_change = np.max(np.abs(roots - roots_old))
        if max_change < tol:
            break

    return roots


def newton_raphson_scalar(
    f, df, x0: float, tol: float = 1e-12, max_iter: int = 100
) -> float:
    """
    Newton-Raphson 方法求解标量非线性方程 f(x) = 0。

    迭代格式:
        x_{k+1} = x_k - f(x_k) / f'(x_k)

    局部二次收敛：|e_{k+1}| \\approx C |e_k|^2。

    参数:
        f: 目标函数
        df: 导数函数
        x0: 初始猜测
        tol: 收敛容差
        max_iter: 最大迭代次数

    返回:
        近似根
    """
    x = x0
    for _ in range(max_iter):
        fx = f(x)
        dfx = df(x)
        if abs(dfx) < 1e-15:
            break
        x_new = x - fx / dfx
        if abs(x_new - x) < tol:
            return x_new
        x = x_new
    return x


def newton_raphson_system(
    F, JF, x0: np.ndarray, tol: float = 1e-10, max_iter: int = 50
) -> np.ndarray:
    """
    Newton-Raphson 方法求解非线性方程组 F(x) = 0。

    迭代格式:
        J_F(x_k) \\delta x = -F(x_k)
        x_{k+1} = x_k + \\delta x

    其中 J_F 为 Jacobian 矩阵，元素 (J_F)_{ij} = \\partial F_i / \\partial x_j。

    参数:
        F: 向量值函数，返回 numpy 数组
        JF: Jacobian 函数，返回 numpy 矩阵
        x0: 初始猜测
        tol: 收敛容差
        max_iter: 最大迭代次数

    返回:
        近似解
    """
    x = x0.copy().astype(float)
    for _ in range(max_iter):
        Fx = F(x)
        if np.linalg.norm(Fx) < tol:
            break
        J = JF(x)
        try:
            dx = np.linalg.solve(J, -Fx)
        except np.linalg.LinAlgError:
            # 奇异时添加正则化
            dx = np.linalg.solve(J + np.eye(len(x)) * 1e-8, -Fx)
        x = x + dx
        if np.linalg.norm(dx) < tol:
            break
    return x


def fixed_point_iteration(
    g, x0: float, tol: float = 1e-10, max_iter: int = 1000
) -> float:
    """
    不动点迭代求解 x = g(x)。

    收敛条件: |g'(x^*)| < 1 在不动点附近。

    参数:
        g: 迭代函数
        x0: 初始猜测
        tol: 容差
        max_iter: 最大迭代次数

    返回:
        不动点
    """
    x = x0
    for _ in range(max_iter):
        x_new = g(x)
        if abs(x_new - x) < tol:
            return x_new
        x = x_new
    return x


def companion_matrix_eigenvalues(coeffs: np.ndarray) -> np.ndarray:
    """
    通过友矩阵特征值求多项式根。

    对于首一多项式:
        p(z) = z^d + a_{d-1} z^{d-1} + ... + a_1 z + a_0

    友矩阵 (companion matrix) 为:
        [ 0   0   ...  0  -a_0   ]
        [ 1   0   ...  0  -a_1   ]
        [ 0   1   ...  0  -a_2   ]
        [ ... ... ... ... ...    ]
        [ 0   0   ...  1  -a_{d-1}]

    p(z) 的根即为友矩阵的特征值。

    参数:
        coeffs: 系数 [c_d, c_{d-1}, ..., c_0]

    返回:
        特征值（根）
    """
    d = len(coeffs) - 1
    if d < 1:
        return np.array([])

    # 标准化
    leading = coeffs[0]
    if abs(leading) < 1e-15:
        raise ValueError("Leading coefficient is zero")
    a = coeffs[1:] / leading

    C = np.zeros((d, d))
    C[:-1, 1:] = np.eye(d - 1)
    C[:, 0] = -a[::-1]

    return np.linalg.eigvals(C)


def durand_kerner_step(
    coeffs: np.ndarray, roots: np.ndarray
) -> np.ndarray:
    """
    单步 Durand-Kerner 迭代。

    用于需要逐步控制迭代过程的场景。

    参数:
        coeffs: 多项式系数
        roots: 当前根近似

    返回:
        更新后的根近似
    """
    d = len(roots)
    new_roots = roots.copy()
    for i in range(d):
        denom = 1.0 + 0j
        for j in range(d):
            if i != j:
                denom *= (roots[i] - roots[j])
        if abs(denom) < 1e-30:
            denom = 1e-30
        pz = poly_eval(coeffs[::-1], np.array([roots[i]]))[0]
        new_roots[i] = roots[i] - pz / denom
    return new_roots


def polynomial_characteristic_values(
    A: np.ndarray, B: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    求解广义特征值问题 A v = \\lambda B v。

    对于粘弹性问题，特征值对应于松弛时间:
        \\tau_i = -1 / \\lambda_i

    参数:
        A: 矩阵
        B: 矩阵（None 时退化为标准特征值）

    返回:
        特征值数组
    """
    if B is None:
        return np.linalg.eigvals(A)
    else:
        return scipy_eigvals_generalized(A, B)


def scipy_eigvals_generalized(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    求解广义特征值问题 A v = \\lambda B v（不使用 scipy 的简化版本）。

    通过变换为标准特征值问题:
        B^{-1} A v = \\lambda v

    参数:
        A, B: 方阵

    返回:
        特征值
    """
    try:
        B_inv = np.linalg.inv(B)
        return np.linalg.eigvals(B_inv @ A)
    except np.linalg.LinAlgError:
        # 使用伪逆
        B_pinv = np.linalg.pinv(B)
        return np.linalg.eigvals(B_pinv @ A)


def complex_iterative_refine(
    f, z0: complex, tol: float = 1e-12, max_iter: int = 100
) -> complex:
    """
    复平面上的迭代精化，融入 Mandelbrot 集的迭代思想。

    对于映射 f: C -> C，迭代:
        z_{k+1} = f(z_k)

    若 |z_{k+1} - z_k| < tol，认为收敛到不动点。

    参数:
        f: 复映射函数
        z0: 初始复数
        tol: 容差
        max_iter: 最大迭代次数

    返回:
        收敛点或最后迭代值
    """
    z = z0
    for _ in range(max_iter):
        z_new = f(z)
        if abs(z_new - z) < tol:
            return z_new
        z = z_new
    return z
