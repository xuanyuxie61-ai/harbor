"""
resonance_finder.py — 共振质量搜索
====================================
种子项目映射:
  095_bisection_integer (整数二分) → 连续二分法搜索共振质量
  1097_catniplab (吸引子) → 不动点搜索与收敛分析

物理: 通过二分法搜索使显著性最大化的BSM质量参数.
"""
import numpy as np


def bisection_search(f, a, b, tol=1e-6, max_iter=100):
    """
    二分法求根: 找 f(x) = 0 在 [a,b] 中的根.
    种子项目 095_bisection_integer 的连续版本.

    收敛条件: |b-a| < tol 或 |f(c)| < tol.
    """
    fa, fb = f(a), f(b)
    if fa * fb > 0:
        raise ValueError(f"f(a)={fa}, f(b)={fb}, 符号相同, 无根")

    history = []
    for i in range(max_iter):
        c = 0.5 * (a + b)
        fc = f(c)
        history.append((c, fc))

        if abs(fc) < tol or abs(b - a) < tol:
            return c, {'converged': True, 'iterations': i + 1, 'residual': abs(fc)}

        if fa * fc < 0:
            b, fb = c, fc
        else:
            a, fa = c, fc

    c = 0.5 * (a + b)
    return c, {'converged': False, 'iterations': max_iter, 'residual': abs(f(c))}


def brent_search(f, a, b, tol=1e-8, max_iter=100):
    """
    Brent法求根 (结合二分、割线、反二次插值).
    比纯二分法收敛更快.
    """
    fa, fb = f(a), f(b)
    if fa * fb > 0:
        raise ValueError("区间两端函数值同号")

    if abs(fa) < abs(fb):
        a, b = b, a
        fa, fb = fb, fa

    c, fc = a, fa
    mflag = True
    d = 0.0

    for i in range(max_iter):
        if abs(fb) < tol or abs(b - a) < tol:
            return b, {'converged': True, 'iterations': i + 1}

        # 尝试反二次插值
        if abs(fa - fc) > 1e-15 and abs(fb - fc) > 1e-15:
            s = (a * fb * fc / ((fa - fb) * (fa - fc))
                 + b * fa * fc / ((fb - fa) * (fb - fc))
                 + c * fa * fb / ((fc - fa) * (fc - fb)))
        else:
            # 割线法
            s = b - fb * (b - a) / (fb - fa) if abs(fb - fa) > 1e-15 else 0.5*(a+b)

        # 条件检查
        cond1 = not ((3*a+b)/4 < s < b or b < s < (3*a+b)/4)
        cond2 = mflag and abs(s - b) >= abs(b - c) / 2
        cond3 = not mflag and abs(s - b) >= abs(c - d) / 2
        cond4 = mflag and abs(b - c) < tol
        cond5 = not mflag and abs(c - d) < tol

        if cond1 or cond2 or cond3 or cond4 or cond5:
            s = 0.5 * (a + b)
            mflag = True
        else:
            mflag = False

        fs = f(s)
        d = c
        c, fc = b, fb

        if fa * fs < 0:
            b, fb = s, fs
        else:
            a, fa = s, fs

        if abs(fa) < abs(fb):
            a, b = b, a
            fa, fb = fb, fa

    return b, {'converged': False, 'iterations': max_iter}


def find_resonance_mass(compute_significance, target_Z=3.0, M_range=(100, 2000), tol=1.0):
    """
    搜索使显著性 = target_Z 的共振质量.

    f(M) = Z(M) - target_Z = 0
    """
    def f(M):
        return compute_significance(M) - target_Z

    try:
        M_res, info = bisection_search(f, M_range[0], M_range[1], tol=tol)
        return M_res, info
    except ValueError:
        # 无根, 返回最接近的点
        M_scan = np.linspace(M_range[0], M_range[1], 100)
        Z_scan = [compute_significance(M) for M in M_scan]
        idx = np.argmin([abs(Z - target_Z) for Z in Z_scan])
        return M_scan[idx], {'converged': False, 'note': 'no root in range'}
