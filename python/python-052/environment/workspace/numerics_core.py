"""
numerics_core.py
核心数值算法库

融合来源:
- 939_quad_fast_rule: Clenshaw-Curtis, Fejér, Gauss-Legendre 快速求积规则
- 454_gaussian: Hermite 正交多项式递推
- 911_prime_factors: 质因数分解(用于FFT最优尺寸)
- 026_asa007: Cholesky分解(稀疏/稠密矩阵)

科学功能:
- 高精度数值积分 (谱精度求积)
- 概率学家 Hermite 多项式及其导数
- 协方差矩阵的 Cholesky 分解与求逆
- FFT 友好网格尺寸的质因数分解优化
"""

import numpy as np
from typing import Tuple, Optional


# ============================================================
# 1. 快速求积规则 (from 939_quad_fast_rule)
# ============================================================

def clenshaw_curtis_nodes_weights(n: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Clenshaw-Curtis 求积规则: 在 [-1,1] 上基于 Chebyshev 节点的高精度积分.

    节点: x_j = cos(j*pi/n), j=0,...,n
    权重通过 Trefethen 的 clencurt 算法计算.

    对光滑函数可达到谱精度收敛: 误差 ~ O(n^{-s}) 对任意 s>0.
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    theta = np.pi * np.arange(n + 1) / n
    x = np.cos(theta)

    if n == 1:
        return x, np.array([1.0, 1.0])

    w = np.zeros(n + 1)
    interior = np.arange(1, n)
    v = np.ones(n - 1)

    if n % 2 == 0:
        w[0] = 1.0 / (n ** 2 - 1)
        w[n] = 1.0 / (n ** 2 - 1)
        for k in range(1, n // 2):
            v = v - 2.0 * np.cos(2 * k * theta[interior]) / (4 * k ** 2 - 1)
        v = v - np.cos(n * theta[interior]) / (n ** 2 - 1)
    else:
        w[0] = 1.0 / (n ** 2)
        w[n] = 1.0 / (n ** 2)
        for k in range(1, (n + 1) // 2):
            v = v - 2.0 * np.cos(2 * k * theta[interior]) / (4 * k ** 2 - 1)

    w[interior] = 2.0 * v / n
    return x, w


def fejer1_nodes_weights(n: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fejér Type-1 求积规则: 开区间 Chebyshev 节点 (不含端点).

    节点: x_j = cos((2j-1)*pi/(2n)), j=1,...,n
    权重通过 IFFT 快速计算.

    对周期性和光滑被积函数具有指数收敛性.
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    j = np.arange(1, n + 1)
    x = np.cos((2.0 * j - 1.0) * np.pi / (2.0 * n))

    # 权重计算
    N = n
    k = np.arange(1, N)
    # Fejér-1 权重: w_j = (2/N) * sum_{m=0}^{N-1} alpha_m cos(m*theta_j)
    # 其中 alpha_0 = 1, alpha_m = 2/(1 - 4m^2) for m>=1
    alpha = np.ones(N)
    alpha[1:] = 2.0 / (1.0 - 4.0 * k ** 2)

    theta = (2.0 * j - 1.0) * np.pi / (2.0 * N)
    w = np.zeros(N)
    for idx in range(N):
        w[idx] = np.sum(alpha * np.cos(np.arange(N) * theta[idx]))
    w *= (2.0 / N)
    return x, w


def gauss_legendre_nodes_weights(n: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Gauss-Legendre 求积规则: n 点达到 2n-1 次代数精度.

    通过 Jacobi 矩阵特征值分解求解节点和权重:
      J_{i,i} = 0
      J_{i+1,i} = J_{i,i+1} = i / sqrt(4*i^2 - 1)
    节点为 J 的特征值, 权重 = 2 * (v_1)^2 其中 v 是归一化特征向量.

    对多项式被积函数具有最高代数精度.
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    if n == 1:
        return np.array([0.0]), np.array([2.0])

    # 构造对称三对角 Jacobi 矩阵
    i = np.arange(1.0, n)
    beta = i / np.sqrt(4.0 * i ** 2 - 1.0)
    J = np.diag(beta, 1) + np.diag(beta, -1)

    # 特征值分解
    eigvals, eigvecs = np.linalg.eigh(J)
    x = eigvals
    # 权重: 2 * (第一个分量)^2
    w = 2.0 * (eigvecs[0, :] ** 2)
    return x, w


def composite_quad_rule(f, a: float, b: float, n_sub: int = 8, n_point: int = 8,
                        rule: str = "cc") -> float:
    """
    复合求积规则: 将 [a,b] 分为 n_sub 个子区间, 每个子区间用 n_point 点求积.

    Parameters
    ----------
    f : callable
        被积函数
    a, b : float
        积分区间
    n_sub : int
        子区间数
    n_point : int
        每子区间求积节点数
    rule : str
        "cc"=Clenshaw-Curtis, "fejer1"=Fejér-1, "gl"=Gauss-Legendre

    Returns
    -------
    float
        积分近似值
    """
    if a >= b:
        if a == b:
            return 0.0
        a, b = b, a
    if n_sub < 1 or n_point < 1:
        raise ValueError("n_sub and n_point must be >= 1")

    if rule == "cc":
        x_local, w_local = clenshaw_curtis_nodes_weights(n_point)
    elif rule == "fejer1":
        x_local, w_local = fejer1_nodes_weights(n_point)
    elif rule == "gl":
        x_local, w_local = gauss_legendre_nodes_weights(n_point)
    else:
        raise ValueError(f"Unknown rule: {rule}")

    # 将 [-1,1] 映射到每个子区间
    h = (b - a) / n_sub
    total = 0.0
    for s in range(n_sub):
        a_s = a + s * h
        b_s = a_s + h
        # 仿射变换: x_local ∈ [-1,1] → t ∈ [a_s, b_s]
        t = 0.5 * (b_s - a_s) * x_local + 0.5 * (b_s + a_s)
        jac = 0.5 * (b_s - a_s)
        ft = np.array([f(ti) for ti in t])
        total += np.sum(w_local * ft) * jac
    return float(total)


# ============================================================
# 2. Hermite 正交多项式 (from 454_gaussian)
# ============================================================

def hermite_polynomial_prob(n: int, x: np.ndarray) -> np.ndarray:
    """
    概率学家 Hermite 多项式 He_n(x) 的三项递推求值.

    递推关系:
      He_0(x) = 1
      He_1(x) = x
      He_n(x) = x * He_{n-1}(x) - (n-1) * He_{n-2}(x)

    正交性: ∫_{-∞}^{∞} He_m(x) He_n(x) exp(-x^2/2) dx = sqrt(2*pi) * n! * delta_{mn}

    Parameters
    ----------
    n : int
        最高阶数 (>=0)
    x : np.ndarray
        求值点

    Returns
    -------
    np.ndarray
        形状为 (n+1, len(x)) 的数组, [k,:] 为 He_k(x)
    """
    if n < 0:
        raise ValueError("n must be non-negative")
    x = np.asarray(x)
    if n == 0:
        return np.ones((1, x.size))
    if n == 1:
        return np.vstack([np.ones(x.shape), x])

    H = np.zeros((n + 1, x.size))
    H[0, :] = 1.0
    H[1, :] = x
    for k in range(2, n + 1):
        H[k, :] = x * H[k - 1, :] - (k - 1) * H[k - 2, :]
    return H


def hermite_polynomial_derivative(n: int, x: np.ndarray) -> np.ndarray:
    """
    He_n(x) 的导数: d/dx He_n(x) = n * He_{n-1}(x).
    """
    if n < 0:
        raise ValueError("n must be non-negative")
    if n == 0:
        return np.zeros_like(x)
    H = hermite_polynomial_prob(n - 1, x)
    return n * H[n - 1, :]


def gaussian_hermite_expand(coeffs: np.ndarray, x: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """
    高斯-Hermite 展开: f(x) = sum_{k=0}^{N-1} c_k * He_k(x/sigma) * exp(-x^2/(2*sigma^2)).

    用于描述海洋中尺度涡旋的速度/涡度场解析结构.
    """
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    N = len(coeffs)
    z = x / sigma
    H = hermite_polynomial_prob(N - 1, z)
    result = np.zeros_like(x, dtype=float)
    for k in range(N):
        result += coeffs[k] * H[k, :] * np.exp(-0.5 * z ** 2)
    return result


# ============================================================
# 3. Cholesky 分解与 SPD 矩阵求逆 (from 026_asa007)
# ============================================================

def cholesky_decompose(A: np.ndarray, tol: float = 1e-12) -> np.ndarray:
    """
    Cholesky 分解: A = U^T U (返回上三角矩阵 U).

    对对称正定 (SPD) 矩阵 A, 使用逐元素计算:
      U_{ii} = sqrt(A_{ii} - sum_{k=1}^{i-1} U_{ki}^2)
      U_{ij} = (A_{ij} - sum_{k=1}^{i-1} U_{ki} U_{kj}) / U_{ii},  j > i

    若遇到 U_{ii}^2 <= tol, 则判定矩阵非正定.

    Parameters
    ----------
    A : np.ndarray
        NxN 对称矩阵
    tol : float
        正定性容差

    Returns
    -------
    np.ndarray
        上三角矩阵 U
    """
    A = np.asarray(A, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be a square matrix")
    n = A.shape[0]
    U = np.zeros((n, n))
    for i in range(n):
        sum_sq = np.dot(U[:i, i], U[:i, i])
        diag = A[i, i] - sum_sq
        if diag <= tol:
            raise ValueError(f"Matrix is not positive definite (diag[{i}]={diag:.3e})")
        U[i, i] = np.sqrt(diag)
        for j in range(i + 1, n):
            sum_prod = np.dot(U[:i, i], U[:i, j])
            U[i, j] = (A[i, j] - sum_prod) / U[i, i]
    return U


def spd_inverse(A: np.ndarray, tol: float = 1e-12) -> np.ndarray:
    """
    对称正定矩阵求逆: 先 Cholesky 分解, 再求逆.

    利用三角矩阵求逆公式:
      (U^{-1})_{ii} = 1/U_{ii}
      (U^{-1})_{ij} = -(1/U_{ii}) * sum_{k=i+1}^j U_{ik} (U^{-1})_{kj}, i < j

    最终 A^{-1} = U^{-1} (U^{-1})^T.
    """
    U = cholesky_decompose(A, tol)
    n = U.shape[0]
    Uinv = np.zeros((n, n))
    for i in range(n - 1, -1, -1):
        Uinv[i, i] = 1.0 / U[i, i]
        for j in range(i + 1, n):
            s = np.dot(U[i, i + 1:j + 1], Uinv[i + 1:j + 1, j])
            Uinv[i, j] = -s / U[i, i]
    # A^{-1} = U^{-1} (U^{-1})^T
    return Uinv @ Uinv.T


# ============================================================
# 4. 质因数分解 (from 911_prime_factors)
# ============================================================

def prime_factors(n: int) -> list:
    """
    整数质因数分解: 返回 n 的所有质因子列表 (含重数).

    算法: 试除法, 从 2 开始依次试除.
    时间复杂度 O(sqrt(n)).
    """
    if n < 2:
        return []
    factors = []
    d = 2
    temp = n
    while d * d <= temp:
        while temp % d == 0:
            factors.append(d)
            temp //= d
        d += 1 if d == 2 else 2  # 2之后只试奇数
    if temp > 1:
        factors.append(temp)
    return factors


def next_fftfriendly_size(n: int) -> int:
    """
    寻找不小于 n 的最小 FFT 友好尺寸: 仅含质因子 2, 3, 5.

    FFT 在尺寸为 2^a * 3^b * 5^c 时效率最高.
    """
    if n < 1:
        return 1
    candidate = n
    while True:
        facs = prime_factors(candidate)
        # 检查是否只含 2,3,5
        if all(p in (2, 3, 5) for p in facs):
            return candidate
        candidate += 1


# ============================================================
# 5. 辅助数值工具
# ============================================================

def safe_sqrt(x: np.ndarray, eps: float = 1e-15) -> np.ndarray:
    """安全开方, 避免负数因数值误差导致 nan."""
    x = np.asarray(x)
    return np.sqrt(np.maximum(x, eps))


def safe_divide(a: np.ndarray, b: np.ndarray, eps: float = 1e-15) -> np.ndarray:
    """安全除法, 分母接近零时返回零."""
    a = np.asarray(a)
    b = np.asarray(b)
    result = np.zeros_like(a, dtype=float)
    mask = np.abs(b) > eps
    result[mask] = a[mask] / b[mask]
    return result


def givens_rotation(a: float, b: float) -> Tuple[float, float]:
    """
    Givens 平面旋转: 计算 c, s 使得
      [ c  s] [a]   [r]
      [-s  c] [b] = [0]
    """
    if b == 0.0:
        return 1.0, 0.0
    if abs(b) > abs(a):
        tau = -a / b
        s = 1.0 / np.sqrt(1.0 + tau ** 2)
        c = s * tau
    else:
        tau = -b / a
        c = 1.0 / np.sqrt(1.0 + tau ** 2)
        s = c * tau
    return c, s


# ============================================================
# 模块自检
# ============================================================
if __name__ == "__main__":
    # 测试求积规则
    f_test = lambda x: np.exp(x)
    exact = np.exp(1) - np.exp(-1)
    for rule_name in ["cc", "fejer1", "gl"]:
        val = composite_quad_rule(f_test, -1.0, 1.0, n_sub=2, n_point=8, rule=rule_name)
        print(f"{rule_name}: {val:.12f}, err={abs(val-exact):.2e}")

    # 测试 Hermite
    x = np.linspace(-3, 3, 7)
    H = hermite_polynomial_prob(4, x)
    print("He_4(0)=", H[4, 3])

    # 测试 Cholesky
    A = np.array([[4.0, 2.0, 1.0],
                  [2.0, 5.0, 3.0],
                  [1.0, 3.0, 6.0]])
    U = cholesky_decompose(A)
    print("Cholesky residual:", np.max(np.abs(A - U.T @ U)))

    # 测试质因数分解
    print("prime_factors(360)=", prime_factors(360))
    print("next_fftfriendly(127)=", next_fftfriendly_size(127))
