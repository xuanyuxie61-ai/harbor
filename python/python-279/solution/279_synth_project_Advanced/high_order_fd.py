"""
high_order_fd.py
================
高阶有限差分算子构造模块。

科学背景:
  在材料基因组高通量筛选中，需要在不同材料参数下快速求解偏微分方程。
  高阶有限差分法 (High-Order Finite Difference, HOFD) 以少量网格点
  达到高精度，非常适合参数扫描。

  本模块构造:
    1. 中心差分: O(h^2), O(h^4), O(h^6), O(h^8) 精度
    2. 迎风差分: 1阶-3阶 迎风 (对流占优问题)
    3. 紧致差分 (Pade): 三对角隐式差分, 谱分辨率极高
    4. 非均匀网格修正: 基于泰勒展开的非等距差分系数
    5. 混合差分: 中心+人工粘性的组合

数值公式:
  一阶导数 O(h^2):
    f'(x) ≈ (-f(x+h) + f(x-h)) / (2h)

  一阶导数 O(h^4):
    f'(x) ≈ (f(x-2h) - 8f(x-h) + 8f(x+h) - f(x+2h)) / (12h)

  一阶导数 O(h^6):
    f'(x) ≈ (-f(x-3h) + 9f(x-2h) - 45f(x-h) + 45f(x+h) - 9f(x+2h) + f(x+3h)) / (60h)

  二阶导数 O(h^2):
    f''(x) ≈ (f(x-h) - 2f(x) + f(x+h)) / h^2

  二阶导数 O(h^4):
    f''(x) ≈ (-f(x-2h) + 16f(x-h) - 30f(x) + 16f(x+h) - f(x+2h)) / (12h^2)

  紧致 (Pade) 一阶导数:
    alpha*f'_{i-1} + f'_i + alpha*f'_{i+1}
      = a*(f_{i+1} - f_{i-1})/(2h) + b*(f_{i+2} - f_{i-2})/(4h)
    取 alpha=1/4, a=3/2, b=0 得 O(h^4) 紧致格式

  非均匀网格 (Lele 1992):
    对节点 x_{i-1}, x_i, x_{i+1} 间距 h_-, h_+:
    f'(x_i) ≈ (h_-^2 * f_{i+1} + (h_+^2 - h_-^2)*f_i - h_+^2 * f_{i-1})
              / (h_- * h_+ * (h_- + h_+))
"""

import numpy as np
from scipy import sparse
from material_constants import SMALL_NUMBER


# ============================================================================
# 均匀网格高阶差分系数 (Fornberg 1988 算法)
# ============================================================================
def fornberg_weights(x_sample, x_nodes, deriv_order=1):
    """
    Fornberg 算法计算任意节点分布上任意阶导数的差分权重。

    给定节点 x_nodes = [x_0, ..., x_{n-1}] 和求导点 x_sample,
    返回权重 w 使得 f^(m)(x_sample) ≈ sum_j w_j * f(x_j)

    算法基于 Fornberg (1988, Math. Comp.) 的递推公式 (MATLAB 版本翻译)。
    对 5 点中心差分一阶导数 (x = -2h,-h,0,h,2h; 求导点 x=0):
      w = [1/(12h), -8/(12h), 0, 8/(12h), -1/(12h)]
    """
    x_sample = float(x_sample)
    x_nodes = np.asarray(x_nodes, dtype=np.float64)
    n1 = len(x_nodes)
    M = deriv_order

    c1 = 1.0
    c4 = x_nodes[0] - x_sample
    C = np.zeros((n1, M + 1))
    C[0, 0] = 1.0

    for i in range(1, n1):
        mn = min(i, M)
        c2 = 1.0
        c5 = c4
        c4 = x_nodes[i] - x_sample
        for j in range(i):
            c3 = x_nodes[i] - x_nodes[j]
            c2 = c2 * c3
            for k in range(mn, 0, -1):
                C[j, k] = ((x_nodes[i] - x_sample) * C[j, k] - k * C[j, k-1]) / c3
            C[j, 0] = (x_nodes[i] - x_sample) * C[j, 0] / c3
        for k in range(mn, 0, -1):
            C[i, k] = c1 / c2 * (k * C[i-1, k-1] - c5 * C[i-1, k])
        C[i, 0] = -c1 / c2 * c5 * C[i-1, 0]
        c1 = c2

    return C[:, M]


def central_fd_1d_uniform(N, h, order=2, deriv=1):
    """
    均匀网格中心差分矩阵 (1D, 周期性边界条件).

    参数:
        N: 网格点数
        h: 网格间距
        order: 精度阶数 (2, 4, 6, 8)
        deriv: 导数阶数 (1 或 2)

    返回:
        D: [N, N] 稀疏差分矩阵

    各阶格式的截断误差:
      O(h^2), d=1:  [-1, 0, 1] / (2h)
      O(h^4), d=1:  [1, -8, 0, 8, -1] / (12h)
      O(h^6), d=1:  [-1, 9, -45, 0, 45, -9, 1] / (60h)
      O(h^8), d=1:  [3, -32, 168, -672, 0, 672, -168, 32, -3] / (840h)

      O(h^2), d=2:  [1, -2, 1] / h^2
      O(h^4), d=2:  [-1, 16, -30, 16, -1] / (12h^2)
      O(h^6), d=2:  [2, -27, 270, -490, 270, -27, 2] / (180h^2)
    """
    stencil_dict = {
        (1, 2): ([-1, 0, 1], np.array([-1, 0, 1]) / (2*h)),
        (1, 4): ([-2, -1, 0, 1, 2], np.array([1, -8, 0, 8, -1]) / (12*h)),
        (1, 6): ([-3, -2, -1, 0, 1, 2, 3], np.array([-1, 9, -45, 0, 45, -9, 1]) / (60*h)),
        (1, 8): ([-4, -3, -2, -1, 0, 1, 2, 3, 4],
                 np.array([3, -32, 168, -672, 0, 672, -168, 32, -3]) / (840*h)),
        (2, 2): ([-1, 0, 1], np.array([1, -2, 1]) / h**2),
        (2, 4): ([-2, -1, 0, 1, 2], np.array([-1, 16, -30, 16, -1]) / (12*h**2)),
        (2, 6): ([-3, -2, -1, 0, 1, 2, 3], np.array([2, -27, 270, -490, 270, -27, 2]) / (180*h**2)),
    }
    key = (deriv, order)
    if key not in stencil_dict:
        raise ValueError(f"Unsupported stencil: deriv={deriv}, order={order}")
    offsets, weights = stencil_dict[key]

    rows, cols, vals = [], [], []
    for i in range(N):
        for off, w in zip(offsets, weights):
            j = (i + off) % N  # 周期性
            rows.append(i)
            cols.append(j)
            vals.append(w)
    return sparse.csr_matrix((vals, (rows, cols)), shape=(N, N))


# ============================================================================
# 非均匀网格差分
# ============================================================================
def nonuniform_fd_matrix(x_nodes, deriv=1, bc_type='dirichlet'):
    """
    非均匀网格上的一阶/二阶导数差分矩阵。

    对内部节点 i, 使用三点非均匀公式:
      f'(x_i) ≈ w_- * f_{i-1} + w_0 * f_i + w_+ * f_{i+1}
    其中:
      h_- = x_i - x_{i-1}, h_+ = x_{i+1} - x_i
      w_- = -h_+ / (h_- * (h_- + h_+))
      w_0 = (h_+ - h_-) / (h_- * h_+)
      w_+ = h_- / (h_+ * (h_- + h_+))

    二阶导数:
      w_- = 2 / (h_- * (h_- + h_+))
      w_0 = -2 / (h_- * h_+)
      w_+ = 2 / (h_+ * (h_- + h_+))
    """
    x = np.asarray(x_nodes, dtype=np.float64)
    N = len(x)
    rows, cols, vals = [], [], []

    if deriv == 1:
        for i in range(1, N - 1):
            hm = x[i] - x[i-1]
            hp = x[i+1] - x[i]
            denom1 = hm * (hm + hp)
            denom0 = hm * hp
            denom2 = hp * (hm + hp)
            denom1 = denom1 if abs(denom1) > SMALL_NUMBER else SMALL_NUMBER
            denom0 = denom0 if abs(denom0) > SMALL_NUMBER else SMALL_NUMBER
            denom2 = denom2 if abs(denom2) > SMALL_NUMBER else SMALL_NUMBER
            rows.extend([i, i, i])
            cols.extend([i-1, i, i+1])
            vals.extend([-hp / denom1, (hp - hm) / denom0, hm / denom2])
    elif deriv == 2:
        for i in range(1, N - 1):
            hm = x[i] - x[i-1]
            hp = x[i+1] - x[i]
            denom1 = hm * (hm + hp)
            denom0 = hm * hp
            denom2 = hp * (hm + hp)
            denom1 = denom1 if abs(denom1) > SMALL_NUMBER else SMALL_NUMBER
            denom0 = denom0 if abs(denom0) > SMALL_NUMBER else SMALL_NUMBER
            denom2 = denom2 if abs(denom2) > SMALL_NUMBER else SMALL_NUMBER
            rows.extend([i, i, i])
            cols.extend([i-1, i, i+1])
            vals.extend([2.0 / denom1, -2.0 / denom0, 2.0 / denom2])
    else:
        raise ValueError(f"deriv={deriv} not supported for nonuniform FD")

    # 边界处理
    if bc_type == 'dirichlet':
        rows.extend([0, N-1])
        cols.extend([0, N-1])
        vals.extend([1.0, 1.0])
    elif bc_type == 'neumann':
        # 一阶单侧差分近似零法向导数
        rows.extend([0, 0, N-1, N-1])
        cols.extend([0, 1, N-2, N-1])
        vals.extend([-1.0, 1.0, -1.0, 1.0])

    return sparse.csr_matrix((vals, (rows, cols)), shape=(N, N))


# ============================================================================
# 紧致 (Pade) 差分
# ============================================================================
def compact_pade_1d(N, h, alpha=0.25, bc='periodic'):
    """
    紧致 Pade 差分格式 (Lele 1992).

    隐式关系:
      alpha * f'_{i-1} + f'_i + alpha * f'_{i+1}
        = a * (f_{i+1} - f_{i-1}) / (2h)

    取 alpha = 1/4, a = 3/2 得 O(h^4) 精度。
    取 alpha = 9/20, a = 15/16 搭配 b = 3/2 得 O(h^6).

    需要求解三对角系统 A * f' = b, 其中:
      A 为三对角 (alpha, 1, alpha)
      b 为显式差分结果

    返回:
        A_lhs: [N, N] 左侧三对角矩阵
        B_rhs: [N, N] 右侧显式差分矩阵
        D_effective: A^{-1} B (有效差分算子, 仅用于小规模验证)
    """
    a_coeff = 1.5  # O(h^4) with alpha=0.25

    # 左矩阵 A
    diag_main = np.ones(N)
    diag_upper = np.full(N - 1, alpha)
    diag_lower = np.full(N - 1, alpha)
    A = sparse.diags([diag_lower, diag_main, diag_upper], [-1, 0, 1], shape=(N, N), format='csr')
    if bc == 'periodic':
        A = A + sparse.csr_matrix(([alpha, alpha], ([0, N-1], [N-1, 0])), shape=(N, N))

    # 右矩阵 B
    B = central_fd_1d_uniform(N, h, order=2, deriv=1) * (2*h) * a_coeff / 2.0
    # 调整: 实际上 B * f 应等于 a * (f_{i+1} - f_{i-1}) / (2h)
    # 重新构造
    rows, cols, vals = [], [], []
    for i in range(N):
        rows.extend([i, i])
        cols.extend([(i-1) % N, (i+1) % N])
        vals.extend([-a_coeff / (2*h), a_coeff / (2*h)])
    B = sparse.csr_matrix((vals, (rows, cols)), shape=(N, N))

    # 有效算子
    A_dense = A.toarray()
    D_eff = np.linalg.solve(A_dense, B.toarray())

    return A, B, D_eff


# ============================================================================
# 混合差分 (中心 + 人工粘性)
# ============================================================================
def hybrid_fd_with_artificial_viscosity(N, h, order=4, epsilon_av=0.01):
    """
    混合差分: 中心差分 + 人工粘性项。

    用于对流占优的离子输运问题 (Pe = v*L/D >> 1):

    离散 PDE:
      -D * d^2c/dx^2 + v * dc/dx = f
    => 中心差分会产生非物理振荡

    人工粘性稳定化:
      epsilon_av * h * |v| * d^2c/dx^2

    有效扩散系数:
      D_eff = D + epsilon_av * h * |v|

    总算子:
      L_h = -D_eff * D2_h + v * D1_h

    参数:
        epsilon_av: 人工粘性系数 (典型值 0.01 - 0.1)
    """
    D1 = central_fd_1d_uniform(N, h, order=order, deriv=1)
    D2 = central_fd_1d_uniform(N, h, order=order, deriv=2)
    return D1, D2, epsilon_av


# ============================================================================
# 误差分析工具
# ============================================================================
def truncation_error_analysis(func, x_range, h_values, order_expected, deriv=1):
    """
    通过网格收敛验证差分格式的截断误差阶数。

    方法:
      对一系列 h, 用高阶差分 (O(h^8)) 作为参考真值,
      比较目标格式与参考的差, 拟合 log(err) = p * log(h) + C,
      验证 p ≈ order_expected.

    返回:
        errors: list of max errors
        observed_order: 观测到的收敛阶
    """
    errors = []
    for h in h_values:
        N = int((x_range[1] - x_range[0]) / h) + 1
        if N < 10:
            N = 10
        x = np.linspace(x_range[0], x_range[1], N)
        f_vals = func(x)

        # 使用更精细网格上的解析导数作为参考
        N_fine = 8 * N
        x_fine = np.linspace(x_range[0], x_range[1], N_fine)
        f_fine = func(x_fine)
        h_fine = (x_range[1] - x_range[0]) / (N_fine - 1)
        D_ref = central_fd_1d_uniform(N_fine, h_fine, order=8, deriv=deriv)
        df_fine = D_ref.dot(f_fine)

        # 在粗网格点上的参考值 (通过插值)
        df_ref_coarse = np.interp(x, x_fine, df_fine)

        # 目标格式
        h_actual = (x_range[1] - x_range[0]) / (N - 1)
        target_order = min(order_expected, 8)
        if target_order == 2:
            target_order_use = 2
        elif target_order <= 4:
            target_order_use = 4
        elif target_order <= 6:
            target_order_use = 6
        else:
            target_order_use = 8
        D_target = central_fd_1d_uniform(N, h_actual, order=target_order_use, deriv=deriv)
        df_approx = D_target.dot(f_vals)

        # 只在内部点比较 (避免边界效应)
        margin = max(4, target_order_use // 2 + 1)
        if N > 2 * margin:
            err = np.max(np.abs(df_approx[margin:-margin] - df_ref_coarse[margin:-margin]))
        else:
            err = np.max(np.abs(df_approx - df_ref_coarse))
        errors.append(max(err, SMALL_NUMBER))

    errors = np.array(errors)
    h_values = np.array(h_values)
    # 最小二乘拟合 log(err) = p * log(h) + C
    log_h = np.log(h_values)
    log_e = np.log(errors)
    coeffs = np.polyfit(log_h, log_e, 1)
    observed_order = coeffs[0]

    return errors, observed_order
