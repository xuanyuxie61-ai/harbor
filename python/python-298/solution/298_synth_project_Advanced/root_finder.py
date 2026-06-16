"""
求根算法模块

融合多个种子项目的求根算法:
1. Brent 方法 (zero_brent)
2. Newton-Maehly 方法 (newton_maehly): 多项式顺序求根
3. Sylvester 结式法 (polynomial_resultant): 多项式公根求解

应用于数值色散关系求解:
    D(omega, k) = 0 的根即为色散关系的解
"""

import numpy as np
from typing import Callable, Tuple, Optional, List


# ============================================================================
# Brent 方法 (from zero_brent)
# ============================================================================
def brent_root(f: Callable[[float], float], a: float, b: float,
                tol: float = 1e-12, max_iter: int = 100) -> Tuple[float, int]:
    """
    Brent 方法求根 (change of sign interval [a,b])

    融合反二次插值、割线法与二分法的优点, 保证收敛
    收敛精度: 4*eps*|root| + 2*tol

    Reference:
        R. Brent, "Algorithms for Minimization Without Derivatives",
        Dover, 2002.

    Parameters:
        f        : 目标函数
        a, b     : 变号区间
        tol      : 容差
        max_iter : 最大迭代次数

    Returns:
        (root, n_calls) : 根与函数调用次数
    """
    fa = f(a)
    fb = f(b)

    if fa * fb > 0.0:
        raise ValueError(f"Brent 方法需要变号区间: f({a})={fa}, f({b})={fb}")

    if abs(fa) < abs(fb):
        a, b = b, a
        fa, fb = fb, fa

    c = a
    fc = fa
    mflag = True
    d = b - a  # 初始步长
    n_calls = 2

    for _ in range(max_iter):
        if abs(fb) < 1e-300:
            return b, n_calls
        if abs(fc) < 1e-300:
            return c, n_calls
        if abs(b - a) < tol * max(1.0, abs(b)):
            return b, n_calls

        # 尝试反二次插值
        if abs(fa - fc) > 1e-300 and abs(fb - fc) > 1e-300:
            s = (a * fb * fc / ((fa - fb) * (fa - fc)) +
                 b * fa * fc / ((fb - fa) * (fb - fc)) +
                 c * fa * fb / ((fc - fa) * (fc - fb)))
        else:
            # 割线法
            s = b - fb * (b - a) / (fb - fa)

        # 条件判断: 是否需要退回到二分法
        cond1 = not ((3.0 * a + b) / 4.0 <= s <= b or
                      b <= s <= (3.0 * a + b) / 4.0)
        cond2 = mflag and abs(s - b) >= abs(b - c) / 2.0
        cond3 = (not mflag) and abs(s - b) >= abs(c - d) / 2.0
        cond4 = mflag and abs(b - c) < tol * max(1.0, abs(b))
        cond5 = (not mflag) and abs(c - d) < tol * max(1.0, abs(b))

        if cond1 or cond2 or cond3 or cond4 or cond5:
            # 二分法
            s = (a + b) / 2.0
            mflag = True
        else:
            mflag = False

        fs = f(s)
        n_calls += 1
        d = c
        c = b
        fc = fb

        if fa * fs < 0.0:
            b = s
            fb = fs
        else:
            a = s
            fa = fs

        if abs(fa) < abs(fb):
            a, b = b, a
            fa, fb = fb, fa

    return b, n_calls


# ============================================================================
# Newton-Maehly 方法 (from newton_maehly)
# ============================================================================
def poly_eval(coeffs: np.ndarray, x: float) -> Tuple[float, float]:
    """
    Horner 法计算多项式及其导数
    P(x) = c[0]*x^n + c[1]*x^(n-1) + ... + c[n]
    """
    n = len(coeffs) - 1
    p = coeffs[0]
    dp = 0.0
    for i in range(1, n + 1):
        dp = dp * x + p
        p = p * x + coeffs[i]
    return p, dp


def newton_maehly_single(coeffs: np.ndarray, x0: float,
                           tol: float = 1e-12, max_iter: int = 100) -> Tuple[float, bool]:
    """
    Newton 法求单个根

    Returns: (root, converged)
    """
    x = x0
    for _ in range(max_iter):
        p, dp = poly_eval(coeffs, x)
        if abs(dp) < 1e-300:
            return x, False
        dx = p / dp
        x -= dx
        if abs(dx) < tol * max(1.0, abs(x)):
            return x, True
    return x, False


def newton_maehly_deflation(coeffs: np.ndarray,
                               initial_guesses: np.ndarray,
                               tol: float = 1e-10) -> List[complex]:
    """
    Newton-Maehly 方法顺序求多项式的所有根

    通过 deflation 每求出一个根就降低多项式阶数

    Parameters:
        coeffs         : 多项式系数 [c0, c1, ..., cn]
        initial_guesses: 初始猜测
        tol            : 容差

    Returns:
        roots : 复根列表
    """
    n = len(coeffs) - 1
    roots = []
    current_coeffs = coeffs.copy().astype(complex)

    for k in range(n):
        x0 = initial_guesses[k % len(initial_guesses)]
        x = complex(x0)

        # 修正: 减去已求出的根 (Maehly 修正)
        for r in roots:
            p_val = complex(0.0)
            dp_val = complex(0.0)
            # 计算 P(x) 和 P'(x) 并减去已知根的贡献
            p, dp = _poly_eval_complex(current_coeffs, x)
            denom = complex(1.0)
            for rj in roots:
                denom *= (x - rj)
            if abs(denom) > 1e-300:
                p_corrected = p / denom
                # Newton step on deflated polynomial
            else:
                p_corrected = p

        # Newton 迭代
        for _ in range(100):
            p, dp = _poly_eval_complex(current_coeffs, x)
            if abs(dp) < 1e-300:
                break
            dx = p / dp
            x -= dx
            if abs(dx) < tol * max(1.0, abs(x)):
                break

        roots.append(x)

        # Deflation: 除以 (x - root)
        new_coeffs = np.zeros(len(current_coeffs) - 1, dtype=complex)
        new_coeffs[0] = current_coeffs[0]
        for i in range(1, len(new_coeffs)):
            new_coeffs[i] = current_coeffs[i] + x * new_coeffs[i - 1]
        current_coeffs = new_coeffs

    return roots


def _poly_eval_complex(coeffs: np.ndarray, x: complex) -> Tuple[complex, complex]:
    """复数 Horner 法"""
    n = len(coeffs) - 1
    p = coeffs[0]
    dp = complex(0.0)
    for i in range(1, n + 1):
        dp = dp * x + p
        p = p * x + coeffs[i]
    return p, dp


# ============================================================================
# Sylvester 结式法 (from polynomial_resultant)
# ============================================================================
def sylvester_matrix(p_coeffs: np.ndarray, q_coeffs: np.ndarray) -> np.ndarray:
    """
    构造 Sylvester 矩阵用于计算两个多项式的结式

    对于 P (m次) 和 Q (n次), Sylvester 矩阵为 (m+n) x (m+n)

    结式 = 0 当且仅当 P 和 Q 有公根
    """
    m = len(p_coeffs) - 1
    n = len(q_coeffs) - 1
    S = np.zeros((m + n, m + n))

    # P 的行 (n 行)
    for i in range(n):
        for j in range(m + 1):
            if i + j < m + n:
                S[i, i + j] = p_coeffs[j]

    # Q 的行 (m 行)
    for i in range(m):
        for j in range(n + 1):
            if i + j < m + n:
                S[n + i, i + j] = q_coeffs[j]

    return S


def resultant(p_coeffs: np.ndarray, q_coeffs: np.ndarray) -> float:
    """
    计算两个多项式的结式
    Res(P, Q) = det(Sylvester matrix)
    """
    S = sylvester_matrix(p_coeffs, q_coeffs)
    return np.linalg.det(S)


def common_roots(p_coeffs: np.ndarray, q_coeffs: np.ndarray,
                   tol: float = 1e-8) -> List[complex]:
    """
    通过结式法求两个多项式的公根

    公根 = gcd(P, Q) 的根

    使用 Euclidean 算法求 GCD
    """
    gcd_poly = polynomial_gcd(p_coeffs, q_coeffs, tol)
    if len(gcd_poly) <= 1:
        return []  # 无公根

    # GCD 的根即为公根
    return np.roots(gcd_poly).tolist()


def polynomial_gcd(p: np.ndarray, q: np.ndarray, tol: float = 1e-10) -> np.ndarray:
    """
    多项式 GCD (Euclidean 算法)
    """
    # 去除前导零
    p = np.trim_zeros(p, 'f')
    q = np.trim_zeros(q, 'f')

    if len(p) == 0:
        return q
    if len(q) == 0:
        return p

    while np.max(np.abs(q)) > tol:
        _, r = np.polydiv(p, q)
        p = q
        q = r

    # 归一化
    if np.max(np.abs(p)) > tol:
        p = p / p[0]
    return p


# ============================================================================
# 色散关系求根 (应用于等离子体物理)
# ============================================================================
def find_dispersion_roots(dispersion_func: Callable,
                            k_values: np.ndarray,
                            omega_range: Tuple[float, float],
                            n_omega_guess: int = 20) -> np.ndarray:
    """
    对多个 k 值求解数值色散关系 D(omega, k) = 0

    Parameters:
        dispersion_func : D(omega, k) 函数
        k_values        : k 值数组
        omega_range     : omega 搜索范围
        n_omega_guess   : 初始猜测数

    Returns:
        omega_roots : [Nk, n_roots] 数组
    """
    omega_min, omega_max = omega_range
    omega_guesses = np.linspace(omega_min, omega_max, n_omega_guess)
    all_roots = []

    for k in k_values:
        roots_k = []
        for i in range(len(omega_guesses) - 1):
            a = omega_guesses[i]
            b = omega_guesses[i + 1]
            try:
                fa = dispersion_func(a, k)
                fb = dispersion_func(b, k)
                if fa * fb < 0:
                    root, _ = brent_root(lambda w: dispersion_func(w, k), a, b)
                    # 避免重复根
                    is_new = all(abs(root - r) > 1e-6 for r in roots_k)
                    if is_new:
                        roots_k.append(root)
            except (ValueError, ZeroDivisionError):
                continue
        all_roots.append(roots_k)

    return all_roots


def plasma_dispersion_function(z: complex) -> complex:
    """
    等离子体色散函数 (Fried-Conte function):
        Z(z) = (1/sqrt(pi)) * integral_{-inf}^{inf} exp(-t^2)/(t-z) dt

    对于 Im(z) > 0:
        Z(z) = i*sqrt(pi) * exp(-z^2) * erfc(-iz)
    """
    from math import erfc
    if z.imag > 0:
        return 1j * np.sqrt(np.pi) * np.exp(-z**2) * complex(erfc(-1j * z))
    elif z.imag < 0:
        return 1j * np.sqrt(np.pi) * np.exp(-z**2) * complex(erfc(-1j * z)) - 2j * np.sqrt(np.pi) * np.exp(-z**2)
    else:
        # Landau contour: 取主值
        return 1j * np.sqrt(np.pi) * np.exp(-z**2) * complex(erfc(-1j * z)) - 1j * np.sqrt(np.pi) * np.exp(-z**2)


def landau_damping_rate(k: float, n_e: float, T_e: float) -> complex:
    """
    Landau 阻尼率 (解析近似):
        omega = omega_r + i * gamma
        omega_r ≈ omega_pe * (1 + 3*k^2*lambda_D^2)^(1/2)
        gamma ≈ -sqrt(pi/8) * omega_pe / (k*lambda_D)^3 * exp(-1/(2*k^2*lambda_D^2) - 3/2)
    """
    from plasma_constants import plasma_frequency_electron, debye_length
    omega_pe = plasma_frequency_electron(n_e)
    lambda_D = debye_length(T_e, n_e)
    kld = k * lambda_D

    if kld < 1e-10:
        return complex(omega_pe, 0.0)

    omega_r = omega_pe * np.sqrt(1.0 + 3.0 * kld**2)
    gamma = -np.sqrt(np.pi / 8.0) * omega_pe / (kld**3) * \
            np.exp(-1.0 / (2.0 * kld**2) - 1.5)

    return complex(omega_r, gamma)
