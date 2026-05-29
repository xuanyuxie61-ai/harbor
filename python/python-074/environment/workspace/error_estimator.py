r"""
error_estimator.py
==================
数值解误差估计与收敛分析模块。

科学背景
--------
在涡激振动流固耦合数值模拟中，需定量评估数值解的精度。
误差度量采用函数空间范数：

1. L^1 范数（误差总量）：
   \|e\|_{L^1} = \int_\Omega |u_{exact}(x) - u_{num}(x)| \, dx

2. L^2 范数（能量型误差）：
   \|e\|_{L^2} = \left( \int_\Omega |u_{exact} - u_{num}|^2 \, dx \right)^{1/2}

3. L^\infty 范数（最大点误差）：
   \|e\|_{L^\infty} = \max_{x \in \Omega} |u_{exact}(x) - u_{num}(x)|

4. H^1 半范数（梯度误差，用于评估涡量精度）：
   |e|_{H^1} = \left( \int_\Omega |\nabla u_{exact} - \nabla u_{num}|^2 \, dx \right)^{1/2}

对于没有精确解的问题，采用 Richardson 外推或网格细化法估计数值误差：

    \|e_h\| \approx \frac{\|u_h - u_{2h}\|}{2^p - 1}

其中 p 为方法的形式精度阶（对流项 p=1，扩散项 p=2）。

收敛阶计算
----------
对两组网格 h_1, h_2 的误差 e_1, e_2：

    p_{obs} = \frac{\log(e_1 / e_2)}{\log(h_1 / h_2)}

对应原种子项目：
- 812_norm_l1（函数 L^1 范数积分计算）
r"""

import numpy as np


def l1_norm_discrete(field, dx, dy=None):
    r"""
    离散 L^1 范数：
    \|f\|_1 = \sum_{i,j} |f_{ij}| \Delta x \Delta y

    参数
    ----
    field : ndarray
        场量。
    dx, dy : float
        网格间距。若 dy 为 None，视为一维问题。
    """
    if dy is None:
        vol = dx
    else:
        vol = dx * dy
    return np.sum(np.abs(field)) * vol


def l2_norm_discrete(field, dx, dy=None):
    r"""
    离散 L^2 范数：
    \|f\|_2 = \left( \sum_{i,j} |f_{ij}|^2 \Delta x \Delta y \right)^{1/2}
    """
    if dy is None:
        vol = dx
    else:
        vol = dx * dy
    return np.sqrt(np.sum(field ** 2) * vol)


def linfty_norm_discrete(field):
    r"""
    离散 L^\infty 范数：
    \|f\|_\infty = \max_{i,j} |f_{ij}|
    """
    return np.max(np.abs(field))


def h1_seminorm_discrete(field, dx, dy):
    r"""
    离散 H^1 半范数（梯度平方积分）：
    |f|_{H^1}^2 = \int |\nabla f|^2 d\Omega
                \approx \sum ( (\partial_x f)^2 + (\partial_y f)^2 ) \Delta x \Delta y
    """
    ny, nx = field.shape
    if nx < 3 or ny < 3:
        return 0.0

    # 内部点中心差分
    dfdx = np.zeros_like(field)
    dfdy = np.zeros_like(field)
    dfdx[:, 1:-1] = (field[:, 2:] - field[:, :-2]) / (2.0 * dx)
    dfdy[1:-1, :] = (field[2:, :] - field[:-2, :]) / (2.0 * dy)

    grad_sq = dfdx ** 2 + dfdy ** 2
    return np.sqrt(np.sum(grad_sq) * dx * dy)


def richardson_error_estimate(u_fine, u_coarse, ratio=2.0, order=2.0):
    r"""
    Richardson 外推误差估计。

    假设 u_coarse 为粗网格解（网格尺寸 h），u_fine 为细网格解（尺寸 h/ratio）。
    误差估计：
        e_{fine} \approx \frac{u_{fine} - u_{coarse}}{ratio^{order} - 1}

    参数
    ----
    u_fine, u_coarse : ndarray
        细网格与粗网格解。粗网格解需通过切片或插值映射到细网格。
    ratio : float
        网格细化比。
    order : float
        方法形式精度阶。

    返回
    ----
    error_est : ndarray
        逐点误差估计。
    """
    denom = ratio ** order - 1.0
    if denom < 1e-15:
        raise ValueError("ratio^order 过于接近 1，无法估计。")
    return (u_fine - u_coarse) / denom


def convergence_order(errors, grid_sizes):
    r"""
    由多组误差计算观测收敛阶。

    参数
    ----
    errors : list of float
        各网格误差。
    grid_sizes : list of float
        对应网格尺寸。

    返回
    ----
    orders : list of float
        相邻两组之间的收敛阶。
    """
    n = len(errors)
    if n != len(grid_sizes) or n < 2:
        return []

    orders = []
    for i in range(n - 1):
        e1, e2 = errors[i], errors[i + 1]
        h1, h2 = grid_sizes[i], grid_sizes[i + 1]
        if e1 <= 0 or e2 <= 0 or h1 <= 0 or h2 <= 0:
            orders.append(np.nan)
        else:
            p = np.log(e1 / e2) / np.log(h1 / h2)
            orders.append(p)
    return orders


def estimate_temporal_error(state_history, dt_values, order=2.0):
    r"""
    时间步长收敛分析：用不同 dt 计算结果估计时间离散误差。

    公式：
    e(dt) \approx C dt^p
    取对数后：\log e = \log C + p \log dt
    用最小二乘拟合求 p。
    """
    if len(dt_values) < 2 or len(state_history) < 2:
        return None, None

    # 以最后一时刻状态作为参考
    final_states = [s[-1] if hasattr(s, '__len__') else s for s in state_history]
    # 取相邻差异作为误差代理
    errors = []
    valid_dt = []
    for i in range(len(final_states) - 1):
        diff = np.abs(final_states[i] - final_states[-1])
        if hasattr(diff, '__len__'):
            err = np.max(diff)
        else:
            err = diff
        if err > 1e-15:
            errors.append(err)
            valid_dt.append(dt_values[i])

    if len(errors) < 2:
        return None, None

    log_dt = np.log(valid_dt)
    log_err = np.log(errors)

    # 线性拟合
    A = np.vstack([np.ones_like(log_dt), log_dt]).T
    coeffs, _, _, _ = np.linalg.lstsq(A, log_err, rcond=None)
    p_est = coeffs[1]
    C_est = np.exp(coeffs[0])

    return p_est, C_est


def compute_solution_quality_metrics(omega, psi, u, v, dx, dy,
                                     solid_mask=None):
    r"""
    综合计算流场解的质量指标。

    返回字典含：
    - omega_l2: 涡量 L2 范数
    - psi_l2: 流函数 L2 范数
    - kinetic_energy: 动能估计 = 0.5 * \int (u^2 + v^2) d\Omega
    - enstrophy: 涡量拟能 = 0.5 * \int \omega^2 d\Omega
    - divergence_rms: 速度散度 RMS（不可压约束违反量）
    """
    if solid_mask is not None:
        # 排除固体区域
        omega_f = np.where(solid_mask, 0.0, omega)
        u_f = np.where(solid_mask, 0.0, u)
        v_f = np.where(solid_mask, 0.0, v)
    else:
        omega_f = omega
        u_f = u
        v_f = v

    omega_l2 = l2_norm_discrete(omega_f, dx, dy)
    psi_l2 = l2_norm_discrete(psi, dx, dy)

    vol = dx * dy
    ke = 0.5 * np.sum(u_f ** 2 + v_f ** 2) * vol
    ens = 0.5 * np.sum(omega_f ** 2) * vol

    # 散度：\partial u/\partial x + \partial v/\partial y
    div = np.zeros_like(u)
    if u.shape[0] > 2 and u.shape[1] > 2:
        div[1:-1, 1:-1] = (
            (u_f[1:-1, 2:] - u_f[1:-1, :-2]) / (2.0 * dx)
            + (v_f[2:, 1:-1] - v_f[:-2, 1:-1]) / (2.0 * dy)
        )
    div_rms = np.sqrt(np.mean(div ** 2))

    return {
        'omega_l2': omega_l2,
        'psi_l2': psi_l2,
        'kinetic_energy': ke,
        'enstrophy': ens,
        'divergence_rms': div_rms,
    }
