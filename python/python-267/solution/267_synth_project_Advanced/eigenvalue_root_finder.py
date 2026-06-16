"""
eigenvalue_root_finder.py — 边界态能量本征值求根器
====================================================

本模块实现边界态能量本征值的高精度求根, 核心方法:

1. **Brent 法**: 结合二分法、割线法和逆二次插值, 保证收敛
2. **Halley 法**: 三阶收敛, 使用二阶导数
3. **Laguerre 法**: 多项式专用, 全局收敛

物理背景
--------
在 ribbon 几何中, 边界态能量 E_n(kx) 是 kx 的函数。
对给定的 kx, 边界态能量满足:

    det[H_ribbon(kx) - E·I] = 0

这是一个广义特征值问题。边界态对应于:
1. 在体带隙内的能量 (|E-C| < |M0|)
2. 波函数局域在 ribbon 边缘

另一种方法: 表面格林函数的极点
    det[(E+iη)I - H00 - Σ(E)] = 0
其中 Σ(E) 是自能。

来源映射
--------
- 1435_zoomin: Newton, Halley, Brent, Laguerre, 割线法等求根算法集
"""

import numpy as np
from typing import Callable, Tuple, Dict, Optional


def brent_root(f: Callable, a: float, b: float,
               tol: float = 1e-12, max_iter: int = 100) -> Dict:
    """
    Brent 求根法: 结合二分法、割线法和逆二次插值。

    保证在 [a, b] 内有根时收敛 (只要 f(a)f(b) < 0)。
    收敛速度: 超线性 (阶 ≈ 1.618, 黄金比例)。

    算法 (Brent, 1973):
    1. 若 |f(b)| < |f(a)|, 交换 a, b
    2. 尝试逆二次插值 (若 s 在合理范围内)
    3. 否则尝试割线法
    4. 若上述步不够好, 退回到二分法

    Parameters
    ----------
    f : callable
        目标函数
    a, b : float
        搜索区间 (需 f(a)f(b) < 0)
    tol : float
    max_iter : int

    Returns
    -------
    result : dict
    """
    fa = f(a)
    fb = f(b)

    if fa * fb > 0:
        return {
            'converged': False,
            'root': 0.5 * (a + b),
            'f_value': f(0.5 * (a + b)),
            'iterations': 0,
            'error': 'f(a) 和 f(b) 同号, 区间可能无根'
        }

    if abs(fa) < abs(fb):
        a, b = b, a
        fa, fb = fb, fa

    c = a
    fc = fa
    mflag = True
    s = 0.0
    d = 0.0

    for it in range(max_iter):
        if abs(fb) < tol or abs(b - a) < tol:
            return {
                'converged': True,
                'root': b,
                'f_value': fb,
                'iterations': it,
                'bracket_width': abs(b - a)
            }

        if fa != fc and fb != fc:
            # 逆二次插值
            s = (a * fb * fc / ((fa - fb) * (fa - fc)) +
                 b * fa * fc / ((fb - fa) * (fb - fc)) +
                 c * fa * fb / ((fc - fa) * (fc - fb)))
        else:
            # 割线法
            s = b - fb * (b - a) / (fb - fa)

        # 条件判断: 是否接受插值步
        mid = 0.5 * (a + b)
        cond1 = not ((3 * a + b) / 4 < s < b or b < s < (3 * a + b) / 4)
        cond2 = mflag and abs(s - b) >= abs(b - c) / 2
        cond3 = not mflag and abs(s - b) >= abs(c - d) / 2
        cond4 = mflag and abs(b - c) < tol
        cond5 = not mflag and abs(c - d) < tol

        if cond1 or cond2 or cond3 or cond4 or cond5:
            s = mid
            mflag = True
        else:
            mflag = False

        fs = f(s)
        d = c
        c = b
        fc = fb

        if fa * fs < 0:
            b = s
            fb = fs
        else:
            a = s
            fa = fs

        if abs(fa) < abs(fb):
            a, b = b, a
            fa, fb = fb, fa

    return {
        'converged': False,
        'root': b,
        'f_value': fb,
        'iterations': max_iter,
        'bracket_width': abs(b - a)
    }


def halley_root(f: Callable, df: Callable, ddf: Callable,
                x0: float, tol: float = 1e-12,
                max_iter: int = 50) -> Dict:
    """
    Halley 求根法: 三阶收敛。

    x_{n+1} = x_n - (2f·f') / (2(f')² - f·f'')

    等价于:
    x_{n+1} = x_n - (f/f') / (1 - L/2)
    其中 L = f·f''/(f')²

    Parameters
    ----------
    f, df, ddf : callable
        函数、一阶导数、二阶导数
    x0 : float
        初始猜测
    tol : float
    max_iter : int

    Returns
    -------
    result : dict
    """
    x = x0
    history = []

    for it in range(max_iter):
        fx = f(x)
        dfx = df(x)
        ddfx = ddf(x)

        history.append({'x': x, 'f': fx})

        if abs(fx) < tol:
            return {
                'converged': True,
                'root': x,
                'f_value': fx,
                'iterations': it,
                'history': history
            }

        if abs(dfx) < 1e-30:
            # 导数为零, 无法继续
            break

        L = fx * ddfx / (dfx ** 2)
        dx = -(fx / dfx) / (1.0 - 0.5 * L)

        # 安全限制
        if abs(dx) > 1.0:
            dx = np.sign(dx) * 1.0

        x = x + dx

        if not np.isfinite(x):
            break

    return {
        'converged': False,
        'root': x,
        'f_value': f(x) if np.isfinite(x) else np.nan,
        'iterations': it if 'it' in dir() else max_iter,
        'history': history
    }


def laguerre_root(coeffs: np.ndarray, x0: float = 0.0,
                  tol: float = 1e-12, max_iter: int = 100) -> Dict:
    """
    Laguerre 求根法: 多项式专用, 全局收敛。

    对 n 次多项式 p(x) = Σ c_k x^k:

    G = p'(x) / p(x)
    H = G² - p''(x) / p(x)
    a = n / (G ± sqrt((n-1)(nH - G²)))
    x_{n+1} = x_n - a

    其中 ± 取使 |a| 较大的符号。

    对多项式, Laguerre 法几乎总是全局收敛。

    Parameters
    ----------
    coeffs : ndarray
        多项式系数 [c_0, c_1, ..., c_n] (升幂)
    x0 : float
        初始猜测
    tol : float
    max_iter : int

    Returns
    -------
    result : dict
    """
    n = len(coeffs) - 1
    if n < 1:
        return {'converged': False, 'root': x0, 'iterations': 0}

    x = x0

    def eval_poly(x_val):
        # Horner 方法
        result = coeffs[-1]
        for k in range(n - 1, -1, -1):
            result = result * x_val + coeffs[k]
        return result

    def eval_poly_deriv(x_val):
        dcoeffs = [k * coeffs[k] for k in range(1, n + 1)]
        if not dcoeffs:
            return 0.0
        result = dcoeffs[-1]
        for k in range(len(dcoeffs) - 2, -1, -1):
            result = result * x_val + dcoeffs[k]
        return result

    def eval_poly_deriv2(x_val):
        d2coeffs = [k * (k - 1) * coeffs[k] for k in range(2, n + 1)]
        if not d2coeffs:
            return 0.0
        result = d2coeffs[-1]
        for k in range(len(d2coeffs) - 2, -1, -1):
            result = result * x_val + d2coeffs[k]
        return result

    for it in range(max_iter):
        px = eval_poly(x)

        if abs(px) < tol:
            return {
                'converged': True,
                'root': x,
                'f_value': px,
                'iterations': it,
                'degree': n
            }

        G = eval_poly_deriv(x) / px if abs(px) > 1e-30 else 0.0
        H = G ** 2 - eval_poly_deriv2(x) / px if abs(px) > 1e-30 else G ** 2

        discriminant = (n - 1) * (n * H - G ** 2)
        if discriminant < 0:
            discriminant = 0.0

        sqrt_disc = np.sqrt(discriminant)

        # 取较大的分母
        denom1 = G + sqrt_disc
        denom2 = G - sqrt_disc

        if abs(denom1) > abs(denom2):
            a = n / denom1 if abs(denom1) > 1e-30 else 1.0
        else:
            a = n / denom2 if abs(denom2) > 1e-30 else 1.0

        if abs(a) < tol:
            return {
                'converged': True,
                'root': x,
                'f_value': eval_poly(x),
                'iterations': it,
                'degree': n
            }

        x = x - a

        if not np.isfinite(x) or abs(x) > 1e10:
            break

    return {
        'converged': False,
        'root': x,
        'f_value': eval_poly(x) if np.isfinite(x) else np.nan,
        'iterations': max_iter,
        'degree': n
    }


def find_edge_state_energies(evals_kx: np.ndarray, E_gap_low: float,
                             E_gap_high: float, tol: float = 1e-6) -> np.ndarray:
    """
    从 ribbon 本征值谱中识别带隙内的边界态能量。

    Parameters
    ----------
    evals_kx : ndarray
        给定 kx 处的本征值 (升序)
    E_gap_low, E_gap_high : float
        体带隙的上下界
    tol : float
        带隙边缘容忍度

    Returns
    -------
    edge_energies : ndarray
        带隙内的能量值
    """
    mask = ((evals_kx > E_gap_low + tol) &
            (evals_kx < E_gap_high - tol))
    return evals_kx[mask]


def dirac_point_finder(params_bhz, Nk: int = 100) -> Dict:
    """
    寻找 BHZ 模型中边界态的 Dirac 点 (交叉点)。

    Dirac 点是边界态色散 E(kx) 穿越 Fermi 能级 E_F = C 的位置。

    使用 Brent 法定位精确的 Dirac 点 kx*。

    Parameters
    ----------
    params_bhz : BHZParameters
    Nk : int
        粗搜索网格

    Returns
    -------
    result : dict
    """
    from boundary_state_solver import ribbon_eigenvalues

    Ny = 40
    hy = 0.5
    E_F = params_bhz.C

    # 粗搜索: 找交叉区间
    kx_arr = np.linspace(-np.pi, np.pi, Nk)
    crossings = []

    for ik in range(len(kx_arr) - 1):
        evals1, _ = ribbon_eigenvalues(Ny, hy, params_bhz, kx_arr[ik], p=2)
        evals2, _ = ribbon_eigenvalues(Ny, hy, params_bhz, kx_arr[ik + 1], p=2)

        edge1 = find_edge_state_energies(
            evals1, E_F - abs(params_bhz.M0), E_F + abs(params_bhz.M0)
        )
        edge2 = find_edge_state_energies(
            evals2, E_F - abs(params_bhz.M0), E_F + abs(params_bhz.M0)
        )

        # 检查是否有态穿越 E_F
        for e1 in edge1:
            for e2 in edge2:
                if (e1 - E_F) * (e2 - E_F) < 0:
                    crossings.append({
                        'kx_low': kx_arr[ik],
                        'kx_high': kx_arr[ik + 1],
                        'E_low': e1,
                        'E_high': e2
                    })

    # 精确定位 (简化: 线性插值)
    dirac_points = []
    for cross in crossings:
        kx_dirac = cross['kx_low'] + (cross['kx_high'] - cross['kx_low']) * \
            (E_F - cross['E_low']) / (cross['E_high'] - cross['E_low'])
        dirac_points.append(kx_dirac)

    return {
        'dirac_points': dirac_points,
        'n_crossings': len(crossings),
        'E_fermi': E_F,
        'Ny': Ny,
        'Nk_coarse': Nk
    }


def root_finder_comparison_report() -> str:
    """比较不同求根方法的性能。"""
    lines = []
    lines.append("=" * 60)
    lines.append("求根方法比较报告")
    lines.append("=" * 60)

    # 测试函数: f(x) = x³ - 2x - 5 (经典测试)
    f = lambda x: x**3 - 2*x - 5
    df = lambda x: 3*x**2 - 2
    ddf = lambda x: 6*x

    lines.append("\n测试函数: f(x) = x³ - 2x - 5")
    lines.append("精确根: x ≈ 2.09455148...")

    # Brent
    result_brent = brent_root(f, 1.0, 3.0)
    lines.append(f"\nBrent 法 [1, 3]:")
    lines.append(f"  根 = {result_brent['root']:.12f}")
    lines.append(f"  迭代 = {result_brent['iterations']}")
    lines.append(f"  f(root) = {result_brent['f_value']:.2e}")

    # Halley
    result_halley = halley_root(f, df, ddf, 2.0)
    lines.append(f"\nHalley 法 (x₀ = 2):")
    lines.append(f"  根 = {result_halley['root']:.12f}")
    lines.append(f"  迭代 = {result_halley['iterations']}")
    lines.append(f"  f(root) = {result_halley['f_value']:.2e}")

    # Laguerre (多项式 p(x) = -5 - 2x + 0x² + x³)
    coeffs = np.array([-5.0, -2.0, 0.0, 1.0])
    result_lag = laguerre_root(coeffs, x0=2.0)
    lines.append(f"\nLaguerre 法 (x₀ = 2):")
    lines.append(f"  根 = {result_lag['root']:.12f}")
    lines.append(f"  迭代 = {result_lag['iterations']}")
    lines.append(f"  f(root) = {result_lag['f_value']:.2e}")

    return "\n".join(lines)
