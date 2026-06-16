"""
gaussian_spiral_scan.py
=======================
高斯整数螺旋扫描策略: 在成分-温度空间中系统性搜索
CALPHAD 相边界和不变点。

种子项目 456_gaussian_prime_spiral:
  在高斯整数平面上构造螺旋路径:
  - 从原点出发, 沿东/北/西/南步进
  - 遇到高斯素数时, 左转 90 度
  - 高斯素数: a^2 + b^2 为素数

映射到 CALPHAD:
  将 (成分指数, 温度指数) 映射为类似螺旋平面的格点。
  扫描策略:
  1. 从参考点 (x0, T0) 出发
  2. 沿螺旋路径遍历 (x, T) 空间
  3. 在满足 "素性条件" 的点 (特殊热力学条件) 转向

  素性条件 (类比):
  - d²G/dc² 改变符号 → spinodal 穿越
  - 两相 Gibbs 能相等 → 相界穿越
  - 化学势等式满足 → 平衡点

  螺旋扫描的优势:
  - 自适应聚焦: 在 "特殊" 点附近加密搜索
  - 全局覆盖: 不会遗漏远离初始点的特征
  - 系统性: 可复现的确定性路径
"""

import numpy as np
from calphad_fec_constants import X_C_MIN, X_C_MAX, T_MIN, T_MAX


def is_gaussian_prime(a, b):
    """
    判断高斯整数 a + bi 是否为高斯素数。

    高斯素数条件:
    1. 若 a=0: |b| 为素数且 |b| ≡ 3 (mod 4)
    2. 若 b=0: |a| 为素数且 |a| ≡ 3 (mod 4)
    3. 若 a≠0 且 b≠0: a² + b² 为素数

    Parameters
    ----------
    a, b : int
        高斯整数的实部和虚部

    Returns
    -------
    bool
    """
    a, b = abs(a), abs(b)

    if a == 0 and b == 0:
        return False
    if a == 0:
        return _is_prime(b) and (b % 4 == 3)
    if b == 0:
        return _is_prime(a) and (a % 4 == 3)
    return _is_prime(a * a + b * b)


def _is_prime(n):
    """简单的素性测试。"""
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True


def thermal_spiral_scan(T_range, x_range, n_spiral=500,
                        phase='FCC', scan_type='spinodal'):
    """
    热力学螺旋扫描: 在 (x_C, T) 空间中沿螺旋路径搜索特征点。

    路径生成:
    从中心 (x_mid, T_mid) 出发, 按照阿基米德螺旋:
      x(k) = x_mid + r(k) * cos(theta(k))
      T(k) = T_mid + r(k) * sin(theta(k))
    其中:
      r(k) = a * sqrt(k)  (等面积螺旋)
      theta(k) = 2*pi * sqrt(k) / scale

    在 "热力学素数" 点转向:
    - spinodal 穿越: d²G/dc² = 0
    - 相界穿越: G_alpha = G_beta

    Parameters
    ----------
    T_range : tuple
        (T_min, T_max) K
    x_range : tuple
        (x_min, x_max)
    n_spiral : int
        螺旋步数
    phase : str
        目标相
    scan_type : str
        'spinodal' 或 'phase_boundary'

    Returns
    -------
    dict
        {
            'spiral_path': np.ndarray,     # (n, 2) 的 (x, T) 路径
            'feature_points': list,        # 发现的特征点
            'turning_points': list,        # 转向点
            'gaussian_primes_hit': list,   # 命中高斯素数的点
        }
    """
    from gibbs_energy_calphad import second_derivative_G, gibbs_substitutional

    x_mid = (x_range[0] + x_range[1]) / 2
    T_mid = (T_range[0] + T_range[1]) / 2
    x_scale = (x_range[1] - x_range[0]) / 2
    T_scale = (T_range[1] - T_range[0]) / 2

    path = []
    feature_points = []
    turning_points = []
    gaussian_hits = []

    # 螺旋参数
    a_r = 0.05  # 径向步长系数

    for k in range(n_spiral):
        r = a_r * np.sqrt(k + 1)
        theta = 2 * np.pi * np.sqrt(k + 1) / 3.0  # 3 圈一个周期

        x = x_mid + x_scale * r * np.cos(theta)
        T = T_mid + T_scale * r * np.sin(theta)

        # 边界裁剪
        x = np.clip(x, x_range[0], x_range[1])
        T = np.clip(T, T_range[0], T_range[1])

        path.append([x, T])

        # 检查热力学条件
        try:
            d2G = second_derivative_G(x, T, phase)

            if scan_type == 'spinodal' and abs(d2G) < 500.0:
                # 接近 spinodal
                feature_points.append({
                    'x': float(x), 'T': float(T),
                    'd2G': float(d2G), 'type': 'near_spinodal',
                })
        except Exception:
            pass

        # 高斯素数检测 (在格点化的 (i,j) 空间)
        i_grid = int(round((x - x_mid) / max(x_scale / 20, 1e-10)))
        j_grid = int(round((T - T_mid) / max(T_scale / 20, 1e-10)))
        if is_gaussian_prime(i_grid, j_grid):
            gaussian_hits.append({
                'k': k, 'x': float(x), 'T': float(T),
                'i': i_grid, 'j': j_grid,
            })
            turning_points.append(k)

    path = np.array(path)

    return {
        'spiral_path': path,
        'feature_points': feature_points,
        'turning_points': turning_points,
        'gaussian_primes_hit': gaussian_hits,
        'n_features': len(feature_points),
        'n_prime_hits': len(gaussian_hits),
    }


def spiral_phase_boundary_trace(T_start, x_start, phase_a, phase_b,
                                max_steps=200, step_size=5.0):
    """
    螺旋追踪相边界: 从一个已知平衡点出发, 沿相界追踪。

    使用预测-校正策略:
    1. 预测: 沿切线方向前进
    2. 校正: 使用 Newton 迭代回到平衡条件

    Parameters
    ----------
    T_start : float
        起始温度
    x_start : float
        起始成分
    phase_a : str
        alpha 相
    phase_b : str
        beta 相
    max_steps : int
        最大追踪步数
    step_size : float
        步长 (K, 温度方向)

    Returns
    -------
    dict
        相边界追踪结果
    """
    from newton_maehly_equilibrium import newton_maehly_solve

    # 初始平衡
    result = newton_maehly_solve(T_start, phase_a, phase_b,
                                 x_init=(x_start, x_start * 3))
    if not result['converged']:
        return {'found': False, 'message': '初始平衡未收敛'}

    T_vals = [T_start]
    xa_vals = [result['x_C_alpha']]
    xb_vals = [result['x_C_beta']]

    T_curr = T_start
    x_init = (result['x_C_alpha'], result['x_C_beta'])

    # 向上追踪
    for step in range(max_steps):
        T_next = T_curr + step_size
        if T_next > T_MAX:
            break

        result = newton_maehly_solve(T_next, phase_a, phase_b,
                                     x_init=x_init)
        if result['converged']:
            T_vals.append(T_next)
            xa_vals.append(result['x_C_alpha'])
            xb_vals.append(result['x_C_beta'])
            x_init = (result['x_C_alpha'], result['x_C_beta'])
            T_curr = T_next
        else:
            break

    # 向下追踪
    T_curr = T_start
    x_init = (xa_vals[0], xb_vals[0])
    T_down = []
    xa_down = []
    xb_down = []

    for step in range(max_steps):
        T_next = T_curr - step_size
        if T_next < T_MIN:
            break

        result = newton_maehly_solve(T_next, phase_a, phase_b,
                                     x_init=x_init)
        if result['converged']:
            T_down.append(T_next)
            xa_down.append(result['x_C_alpha'])
            xb_down.append(result['x_C_beta'])
            x_init = (result['x_C_alpha'], result['x_C_beta'])
            T_curr = T_next
        else:
            break

    # 合并
    T_all = list(reversed(T_down)) + T_vals
    xa_all = list(reversed(xa_down)) + xa_vals
    xb_all = list(reversed(xb_down)) + xb_vals

    return {
        'found': True,
        'T_boundary': np.array(T_all),
        'x_alpha_boundary': np.array(xa_all),
        'x_beta_boundary': np.array(xb_all),
        'n_points': len(T_all),
    }
