"""
nonlinear_rootfinder.py — 非线性求根: Regula Falsi + 不动点迭代

科学背景
========
在置信带校准中, 需要求解非线性方程:
    g(c) = P(M_n ≤ c) - (1-α) = 0
其中 M_n = max_x |Z_n(x)| 为标准化残差场的极大值,
P(M_n ≤ c) 为从 Bootstrap/MC 估计的 CDF.

本模块提供两种求根方法:
1. Regula Falsi (假位法): 保证收敛但可能缓慢
2. 不动点迭代: 适合压缩映射

算法来源 (种子项目 807_nonlin_fixed_point, 809_nonlin_regula)
============================================================
Regula Falsi (种子 809):
    c = (a·f(b) - b·f(a)) / (f(b) - f(a))
    若 f(c)·f(a) < 0, 则 b ← c; 否则 a ← c

Fixed Point (种子 807):
    x_{n+1} = g(x_n)
    收敛条件: |g'(x*)| < 1 (压缩映射)

在本项目中的角色
================
1. 校准同时置信带临界值 c_α
2. 求解 Bootstrap CDF 的分位数
3. 优化预测带宽度

核心公式
========
1. Regula Falsi 步:
   c = (a·f(b) - b·f(a)) / (f(b) - f(a))
2. 不动点迭代:
   x_{k+1} = g(x_k),  收敛 ⟺ |g'(x*)| < 1
3. Illinois 变体 (加速 regula falsi):
   若连续两次 c 在同侧, 则将该侧 f 值减半
"""

import numpy as np


def regula_falsi(f, a, b, tol=1.0e-10, max_iter=300, variant='illinois'):
    """Regula Falsi 求根法.

    在区间 [a,b] 上求 f(x)=0 的根, 要求 f(a)·f(b) < 0.

    参数
    ----
    f : callable
        目标函数
    a, b : float
        变号区间端点
    tol : float
        区间宽度容差
    max_iter : int
        最大迭代次数
    variant : str
        'basic'    — 经典 Regula Falsi
        'illinois' — Illinois 变体 (防单侧停滞)

    返回
    ----
    result : dict
        'root'     : float  近似根
        'f_root'   : float  f(root)
        'a', 'b'   : float  最终缩小区间
        'n_iter'   : int    迭代次数
        'converged': bool

    算法 (种子 809)
    ===============
    c = (a·f(b) - b·f(a)) / (f(b) - f(a))
    若 f(c)·f(a) < 0: b = c
    否则:              a = c
    Illinois 变体: 若连续同侧, 将该侧 f 值减半
    """
    fa = f(a)
    fb = f(b)

    # 检查变号条件
    if fa * fb > 0:
        # 尝试微小扰动
        if abs(fa) < abs(fb):
            a *= (1.0 + 1.0e-6)
            fa = f(a)
        else:
            b *= (1.0 + 1.0e-6)
            fb = f(b)
        if fa * fb > 0:
            return {
                'root': 0.5 * (a + b), 'f_root': f(0.5 * (a + b)),
                'a': a, 'b': b, 'n_iter': 0, 'converged': False,
                'message': 'f(a) 和 f(b) 同号, 无法使用 Regula Falsi'
            }

    # 权重因子 (Illinois 变体)
    wa, wb = 1.0, 1.0
    last_side = 0  # +1 = a 侧被替换, -1 = b 侧被替换

    for it in range(1, max_iter + 1):
        # Regula Falsi 步
        denom = wb * fb - wa * fa
        if abs(denom) < 1.0e-30:
            break
        c = (a * wb * fb - b * wa * fa) / denom
        fc = f(c)

        if abs(fc) < 1.0e-15 or abs(b - a) < tol:
            return {
                'root': c, 'f_root': fc,
                'a': a, 'b': b, 'n_iter': it, 'converged': True,
                'message': '收敛'
            }

        if fc * fa < 0:
            b = c
            fb = fc
            if variant == 'illinois' and last_side == -1:
                wa *= 0.5  # a 侧连续未替换, 减半权重
            last_side = -1
        else:
            a = c
            fa = fc
            if variant == 'illinois' and last_side == 1:
                wb *= 0.5  # b 侧连续未替换, 减半权重
            last_side = 1

    c = 0.5 * (a + b)
    return {
        'root': c, 'f_root': f(c),
        'a': a, 'b': b, 'n_iter': max_iter, 'converged': False,
        'message': f'达到最大迭代 {max_iter}'
    }


def fixed_point_iteration(g, x0, max_iter=200, tol=1.0e-10, damping=1.0):
    """不动点迭代: x_{n+1} = g(x_n).

    参数
    ----
    g : callable
        迭代函数
    x0 : float or ndarray
        初始猜测
    max_iter : int
    tol : float
        收敛容差: |x_{n+1} - x_n| < tol
    damping : float ∈ (0, 1]
        阻尼因子: x_{n+1} = (1-d)·x_n + d·g(x_n)

    返回
    ----
    result : dict
        'x'        : 最终迭代值
        'history'  : 迭代历史
        'n_iter'   : 迭代次数
        'converged': bool

    算法 (种子 807)
    ===============
    for k = 1, 2, ...:
        x_new = g(x_k)
        x_{k+1} = (1-d)·x_k + d·x_new
        if |x_{k+1} - x_k| < tol: 收敛
    """
    x = np.atleast_1d(np.array(x0, dtype=float)).copy()
    history = [x.copy()]

    for it in range(1, max_iter + 1):
        x_new = np.atleast_1d(np.array(g(x), dtype=float))
        # 阻尼更新
        x_next = (1.0 - damping) * x + damping * x_new

        # 收敛检查
        diff = np.max(np.abs(x_next - x))
        x = x_next
        history.append(x.copy())

        if diff < tol:
            return {
                'x': x, 'history': history,
                'n_iter': it, 'converged': True,
                'message': f'收敛 (diff={diff:.2e})'
            }

        # 发散检查
        if np.any(np.isnan(x)) or np.any(np.abs(x) > 1.0e15):
            return {
                'x': x, 'history': history,
                'n_iter': it, 'converged': False,
                'message': '发散: 出现 NaN 或极大值'
            }

    return {
        'x': x, 'history': history,
        'n_iter': max_iter, 'converged': False,
        'message': f'达到最大迭代 {max_iter}'
    }


def newton_method(f, fp, x0, tol=1.0e-10, max_iter=50):
    """Newton 法求根 (种子 807 中附带).

    x_{n+1} = x_n - f(x_n) / f'(x_n)

    参数
    ----
    f, fp : callable
        函数及其导数
    x0 : float
    tol : float
    max_iter : int
    """
    x = float(x0)
    for it in range(1, max_iter + 1):
        fx = f(x)
        fpx = fp(x)
        if abs(fpx) < 1.0e-30:
            return {
                'root': x, 'f_root': fx, 'n_iter': it,
                'converged': False, 'message': '导数过小'
            }
        x_new = x - fx / fpx
        if abs(x_new - x) < tol:
            return {
                'root': x_new, 'f_root': f(x_new), 'n_iter': it,
                'converged': True, 'message': '收敛'
            }
        if abs(fx) > 1.0e8 * abs(f(x0)) and it > 5:
            return {
                'root': x_new, 'f_root': f(x_new), 'n_iter': it,
                'converged': False, 'message': '函数值增长过快'
            }
        x = x_new
    return {
        'root': x, 'f_root': f(x), 'n_iter': max_iter,
        'converged': False, 'message': f'达到最大迭代 {max_iter}'
    }


def calibrate_coverage_critical_value(bootstrap_max_stats, alpha,
                                       method='regula_falsi'):
    """校准同时置信带的临界值 c_α.

    求解:  P(M_n ≤ c_α) = 1 - α
    即:    F_M(c_α) - (1-α) = 0
    其中 F_M 为 Bootstrap 极大统计量的经验 CDF.

    参数
    ----
    bootstrap_max_stats : ndarray
        Bootstrap 极大统计量样本 {M_n^{(b)}}
    alpha : float
        显著性水平
    method : str
        'regula_falsi' 或 'fixed_point'

    返回
    ----
    c_alpha : float
        临界值
    result : dict
        求根器返回的详细信息
    """
    target = 1.0 - alpha
    stats_sorted = np.sort(bootstrap_max_stats)
    n_boot = len(stats_sorted)

    # 经验 CDF
    def empirical_cdf(c):
        return np.searchsorted(stats_sorted, c, side='right') / n_boot

    def objective(c):
        return empirical_cdf(c) - target

    # 确定搜索区间
    c_low = stats_sorted[0]
    c_high = stats_sorted[-1]

    # 确保变号
    f_low = objective(c_low)
    f_high = objective(c_high)

    if f_low > 0:
        return c_low, {'converged': True, 'message': '下界已满足'}
    if f_high < 0:
        return c_high, {'converged': True, 'message': '上界不足'}

    if method == 'regula_falsi':
        result = regula_falsi(objective, c_low, c_high,
                              tol=1.0 / n_boot, max_iter=300)
        return result['root'], result
    elif method == 'fixed_point':
        # 构造不动点映射: g(c) = c + γ·(F_M(c) - target)
        gamma = (c_high - c_low) * 0.5
        def g_map(c_arr):
            c_val = float(c_arr[0])
            return np.array([c_val + gamma * objective(c_val)])
        result = fixed_point_iteration(g_map, np.array([0.5 * (c_low + c_high)]),
                                       max_iter=300, tol=1.0 / n_boot,
                                       damping=0.5)
        return float(result['x'][0]), result
    else:
        # 直接使用分位数
        c_alpha = np.percentile(stats_sorted, target * 100)
        return c_alpha, {'converged': True, 'method': 'percentile'}
