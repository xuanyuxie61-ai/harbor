"""
polynomial_basis.py
===================
多元多项式基函数与分子描述符

融合种子项目:
  - 893_polynomial : 多元多项式的 graded lexicographic 排序、秩/逆秩、
                     多项式算术、微分、稀疏表示

科学背景:
  分子势能面 (PES) 常用多元多项式展开:
      E(q_1, ..., q_d) = Σ_{|α|≤p} c_α q^α
  其中 q_i 为内坐标（键长、键角、二面角），α 为多指标。
  多元多项式基的枚举与求值是构建高阶力场 (如 ReaxFF、MMFF)
  和机器学习势函数的核心步骤。

  本模块实现多元多项式的生成、求值与微分，用于从原子坐标
  构造高阶不变量描述符。
"""

import numpy as np
from typing import List, Tuple
from math import comb


# ------------------------------------------------------------------
# 1. 单项式枚举 (graded lexicographic order)
# ------------------------------------------------------------------

def mono_upto_enum(m: int, n: int) -> int:
    """
    m 个变量、总次数 ≤ n 的单项式个数:
        C(m + n, n)
    """
    return comb(m + n, n)


def mono_next_grlex(m: int, x: np.ndarray) -> np.ndarray:
    """
    给定 m 元单项式 x（指数向量），返回 grlex 序下的下一个单项式。
    算法 (D. E. Knuth / Burkardt):
      1. 找最右满足 x[j] > 0 的 j。
      2. 若 j=0（最左），则 t = x[0], x[0] = 0, x[1] = t + 1。
      3. 否则 x[j-1] += 1, t = x[j] - 1, x[j] = 0, x[m-1] = t。
    """
    x = np.asarray(x, dtype=np.int32).copy()
    j = m - 1
    while j >= 0 and x[j] == 0:
        j -= 1
    if j < 0:
        x[m - 1] = 1
        return x
    if j == 0:
        t = x[0]
        x[0] = 0
        x[1] = t + 1
    elif j > 0:
        x[j - 1] += 1
        t = x[j] - 1
        x[j] = 0
        x[m - 1] = t
    return x


def generate_monomials(m: int, degree: int) -> np.ndarray:
    """
    生成 m 个变量、总次数 ≤ degree 的所有单项式指数向量，按 grlex 排序。
    返回 shape (n_mono, m)。
    """
    monos = []
    for d in range(degree + 1):
        def backtrack(remaining, current, pos):
            if pos == m - 1:
                monos.append(current + [remaining])
                return
            for v in range(remaining + 1):
                backtrack(remaining - v, current + [v], pos + 1)
        backtrack(d, [], 0)
    return np.array(monos, dtype=np.int32)


# ------------------------------------------------------------------
# 2. 多项式求值与微分
# ------------------------------------------------------------------

def evaluate_monomial(exponents: np.ndarray, point: np.ndarray) -> float:
    """
    求单项式 x^α 在 point 处的值。
    """
    val = 1.0
    for i, e in enumerate(exponents):
        if e > 0:
            pi = float(point[i])
            # 边界处理：避免 0^0 歧义
            if abs(pi) < 1e-15 and e == 0:
                continue
            val *= pi ** e
    return val


def evaluate_polynomial(coeffs: np.ndarray, monomials: np.ndarray,
                        point: np.ndarray) -> float:
    """
    求多元多项式值: p(x) = Σ c_i * x^{α_i}。
    """
    total = 0.0
    for c, alpha in zip(coeffs, monomials):
        total += c * evaluate_monomial(alpha, point)
    return total


def polynomial_gradient(coeffs: np.ndarray, monomials: np.ndarray,
                        point: np.ndarray) -> np.ndarray:
    """
    计算多项式梯度 ∇p(x)。
    ∂/∂x_j [x^α] = α_j * x^{α - e_j} (若 α_j > 0)。
    """
    m = monomials.shape[1]
    grad = np.zeros(m, dtype=np.float64)
    for c, alpha in zip(coeffs, monomials):
        for j in range(m):
            if alpha[j] > 0:
                beta = alpha.copy()
                beta[j] -= 1
                grad[j] += c * (alpha[j]) * evaluate_monomial(beta, point)
    return grad


# ------------------------------------------------------------------
# 3. 分子描述符：基于距离的多项式不变量
# ------------------------------------------------------------------

def compute_polynomial_descriptors(atoms: np.ndarray, degree: int = 3) -> np.ndarray:
    """
    计算分子内坐标的多项式描述符。

    步骤:
      1. 构建成对距离矩阵 D（对称、对角为 0）。
      2. 对每一对 (i, j)，取内坐标 q = [D_ij, 1/D_ij, D_ij^2]。
      3. 构造总次数 ≤ degree 的多元多项式基，在该 q 上求值。
      4. 对所有原子对求和，得到分子级描述符向量。

    这些描述符具有置换不变性与平移/旋转不变性（基于距离）。
    """
    n = atoms.shape[0]
    if n < 2:
        return np.zeros(mono_upto_enum(3, degree), dtype=np.float64)

    # 成对距离
    dists = []
    for i in range(n):
        for j in range(i + 1, n):
            r = np.linalg.norm(atoms[i] - atoms[j])
            r = max(r, 0.5)  # 避免除零
            dists.append([r, 1.0 / r, r ** 2])
    dists = np.array(dists, dtype=np.float64)

    m = 3  # [r, 1/r, r^2]
    monos = generate_monomials(m, degree)
    n_mono = monos.shape[0]
    desc = np.zeros(n_mono, dtype=np.float64)
    coeffs = np.ones(n_mono, dtype=np.float64)  # 单位系数

    for q in dists:
        vals = np.array([evaluate_monomial(monos[k], q) for k in range(n_mono)], dtype=np.float64)
        desc += vals

    # 归一化
    norm = np.linalg.norm(desc)
    if norm > 1e-12:
        desc = desc / norm
    return desc


def polynomial_axpy(a: float, p_coeffs: np.ndarray, q_coeffs: np.ndarray) -> np.ndarray:
    """
    多项式线性组合: a * p + q（假设同构单项式基）。
    """
    return a * p_coeffs + q_coeffs


def falling_factorial(x: float, n: int) -> float:
    """
    下降阶乘 [x]_n = x (x-1) ... (x-n+1)。
    源自 i4_fall。
    """
    if n < 0:
        return 0.0
    if n == 0:
        return 1.0
    val = 1.0
    for k in range(n):
        val *= (x - k)
    return val
