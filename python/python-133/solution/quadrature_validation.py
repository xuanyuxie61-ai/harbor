"""
quadrature_validation.py
=========================
多维求积规则精确性验证与单项式对称化处理

基于种子项目 930_pyramid_exactness 与 776_monomial_symmetrize 融合重构。

科学背景：
---------
在聚合反应动力学的不确定性量化中，高维积分（如参数空间上的
期望计算）依赖于多维求积规则。本模块实现求积规则精确性验证
框架，确保所使用的 Gauss-Hermite 稀疏网格能够精确积分一定
阶数以下的多项式。

验证方法：
    对于 d 维求积规则 {x_j, w_j}_{j=1}^N，检验其对单项式
    x^α = x_1^{α_1} ... x_d^{α_d} 的积分是否精确：

        I_exact(α) = ∫_{R^d} x^α exp(-||x||²) dx
        I_quad(α) = Σ_j w_j x_j^α

    误差：err(α) = |I_quad(α) - I_exact(α)|

对于 Gauss-Hermite 积分，精确积分：
    ∫_{-∞}^{+∞} x^{2m} exp(-x²) dx = Γ(m+1/2) = (2m-1)!! sqrt(π)/2^m
    ∫_{-∞}^{+∞} x^{2m+1} exp(-x²) dx = 0

单项式对称化（基于 monomial_symmetrize.m）：
    在高维空间中，等价的单项式（通过指标置换得到）应具有
    相同的系数。本模块对测试单项式集合进行对称化，确保
    验证覆盖所有独立对称类。

参考：
  Arthur Stroud, Approximate Calculation of Multiple Integrals,
  Prentice Hall, 1971.
"""

import numpy as np
from typing import Tuple, List
from math import factorial


def hermite_exact_integral_1d(power: int) -> float:
    """
    一维 Gauss-Hermite 精确积分（ physicist's 形式）：

        I_n = ∫_{-∞}^{+∞} x^n exp(-x²) dx

    奇数 n：I_n = 0
    偶数 n=2m：I_n = (2m-1)!! * sqrt(π) / 2^m = Γ(m+1/2)
    """
    if power % 2 == 1:
        return 0.0
    m = power // 2
    # (2m-1)!! = (2m)! / (2^m m!)
    double_fact = factorial(2 * m) // (2 ** m * factorial(m))
    return double_fact * np.sqrt(np.pi) / (2.0 ** m)


def hermite_exact_integral_nd(exponents: np.ndarray) -> float:
    """
    d 维 Gauss-Hermite 精确积分：

        I(α) = π^{d/2} * Π_{i=1}^d H_{α_i}

    其中 H_{α_i} 为一维精确积分值除以 sqrt(π)。
    更简洁地：I(α) = Π_i I_1d(α_i)
    """
    exponents = np.asarray(exponents, dtype=int)
    result = 1.0
    for e in exponents:
        result *= hermite_exact_integral_1d(int(e))
    return result


def monomial_value(dim_num: int, point_num: int,
                   exponents: np.ndarray,
                   x: np.ndarray) -> np.ndarray:
    """
    计算单项式值：
        v_j = Π_{i=1}^{dim_num} x_{i,j}^{exponent_i}

    基于 monomial_value.m
    """
    exponents = np.asarray(exponents, dtype=int)
    x = np.asarray(x)
    if x.shape[0] != dim_num:
        x = x.T

    values = np.ones(point_num)
    for d in range(dim_num):
        if exponents[d] != 0:
            values *= x[d, :] ** exponents[d]
    return values


def comp_next_composition(n: int, k: int) -> List[np.ndarray]:
    """
    生成整数 n 的所有 k-部分组合。
    基于 comp_next.m 的迭代版本。
    """
    compositions = []
    a = np.zeros(k, dtype=int)
    a[0] = n
    more = (a[-1] != n)
    compositions.append(a.copy())
    h = 0
    t = n

    while more:
        if 1 < t:
            h = 0
        h += 1
        t = a[h - 1]
        a[h - 1] = 0
        a[0] = t - 1
        a[h] += 1
        more = (a[-1] != n)
        compositions.append(a.copy())

    return compositions


def vector_representative(dim: int, base: int, vec: np.ndarray) -> np.ndarray:
    """
    计算向量的代表元（排序后的向量）。
    基于 vector_representative_next.m 的思想。
    """
    return np.sort(vec)


def vector_equivalent_next(vec: np.ndarray) -> Tuple[np.ndarray, bool]:
    """
    生成向量的下一个等价排列（基于 vector_equivalent_next.m）。
    使用 numpy 的 next permutation 近似。
    """
    vec = np.asarray(vec)
    # 使用 np.argsort 找到下一个排列（简化版）
    # 这里直接返回 False，因为在验证中不需要遍历所有等价类
    return vec, False


def symmetrize_monomial_coeffs(dim: int, coeffs: np.ndarray,
                               exponents_list: List[np.ndarray]) -> np.ndarray:
    """
    对等价单项式的系数进行对称化平均。
    基于 monomial_symmetrize.m 的思想。

    对于每个等价类（指标排序相同的单项式），将其系数平均后
    重新分配回所有成员。
    """
    n = len(exponents_list)
    coeffs_new = coeffs.copy()

    # 按代表元分组
    groups = {}
    for i, exp in enumerate(exponents_list):
        rep = tuple(np.sort(exp))
        if rep not in groups:
            groups[rep] = []
        groups[rep].append(i)

    # 对称化
    for rep, indices in groups.items():
        avg_coeff = np.mean(coeffs[indices])
        for idx in indices:
            coeffs_new[idx] = avg_coeff

    return coeffs_new


def validate_quadrature_rule(grid_point: np.ndarray,
                             grid_weight: np.ndarray,
                             dim_num: int,
                             degree_max: int) -> dict:
    """
    验证求积规则对多项式的精确性。

    参数：
        grid_point   : (dim_num, n_points) 节点坐标
        grid_weight  : (n_points,) 权重
        dim_num      : 维度
        degree_max   : 最大检验阶数

    返回：
        包含各阶误差统计的字典
    """
    n_points = grid_point.shape[1]
    errors_by_degree = {}
    max_error = 0.0
    total_tests = 0

    for degree in range(degree_max + 1):
        compositions = comp_next_composition(degree, dim_num)
        degree_errors = []

        for comp in compositions:
            exponents = comp
            # 计算求积近似值
            v = monomial_value(dim_num, n_points, exponents, grid_point)
            quad_val = np.dot(grid_weight, v)

            # 精确值
            exact_val = hermite_exact_integral_nd(exponents)

            err = abs(quad_val - exact_val)
            degree_errors.append(err)
            max_error = max(max_error, err)
            total_tests += 1

        errors_by_degree[degree] = {
            'count': len(degree_errors),
            'max_error': max(degree_errors) if degree_errors else 0.0,
            'mean_error': np.mean(degree_errors) if degree_errors else 0.0,
        }

    return {
        'max_error': max_error,
        'total_tests': total_tests,
        'errors_by_degree': errors_by_degree,
    }


def convergence_order_estimate(errors: List[float], n_points: List[int]) -> float:
    """
    基于误差-节点数关系估计收敛阶：

        err ~ C * N^{-p}

    取对数后线性拟合估计 p。
    """
    log_n = np.log(n_points)
    log_err = np.log(errors)
    # 线性回归
    A = np.vstack([log_n, np.ones(len(log_n))]).T
    p, _ = np.linalg.lstsq(A, log_err, rcond=None)[0]
    return -float(p)
