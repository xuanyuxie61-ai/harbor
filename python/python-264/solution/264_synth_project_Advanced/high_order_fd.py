# -*- coding: utf-8 -*-
"""
high_order_fd.py
================

高阶有限差分算子模块.

本模块实现磁层粒子输运方程求解所需的高阶空间离散化格式:
  - 中心差分 (2阶, 4阶, 6阶)
  - WENO5 (五阶加权本质无振荡格式)
  - 紧致有限差分 (4阶 Padé 格式)
  - 通量重构方法

核心数值方法:

1. WENO5 (Jiang & Shu 1996):
   对于通量 f 在 i+1/2 处的重构:
     f_{i+1/2} = w_0 * f^{(0)} + w_1 * f^{(1)} + w_2 * f^{(2)}

   其中三个候选模板上的重构为:
     f^{(0)} = (2*f_{i-2} - 7*f_{i-1} + 11*f_i) / 6
     f^{(1)} = (-f_{i-1} + 5*f_i + 2*f_{i+1}) / 6
     f^{(2)} = (2*f_i + 5*f_{i+1} - f_{i+2}) / 6

   理想权重: d_0 = 1/10, d_1 = 6/10, d_2 = 3/10
   光滑度指标:
     beta_0 = (13/12)*(f_{i-2} - 2*f_{i-1} + f_i)^2 + (1/4)*(f_{i-2} - 4*f_{i-1} + 3*f_i)^2
     beta_1 = (13/12)*(f_{i-1} - 2*f_i + f_{i+1})^2 + (1/4)*(f_{i-1} - f_{i+1})^2
     beta_2 = (13/12)*(f_i - 2*f_{i+1} + f_{i+2})^2 + (1/4)*(3*f_i - 4*f_{i+1} + f_{i+2})^2

   非线性权重:
     alpha_k = d_k / (epsilon + beta_k)^2
     w_k = alpha_k / (alpha_0 + alpha_1 + alpha_2)

2. 4阶紧致差分 (Lele 1992):
   对于一阶导数:
     (1/4)*f'_{i-1} + f'_i + (1/4)*f'_{i+1} = (3/(2*h))*(f_{i+1} - f_{i-1})/2

   对于二阶导数:
     (1/10)*f''_{i-1} + f''_i + (1/10)*f''_{i+1} = (6/5)*(f_{i+1} - 2*f_i + f_{i-1})/h^2

3. 物理背景:
   磁层辐射带粒子的相空间密度 f(L, E, t) 满足 Fokker-Planck 方程:
     df/dt = -v * df/dL + D * d^2f/dL^2 + ...

   其中对流项 v * df/dL 由磁漂移引起, 扩散项 D * d^2f/dL^2 由
   波-粒子散射引起. 高阶格式对于准确捕获相空间结构至关重要.

参考文献:
  [1] Jiang, G.-S. & Shu, C.-W., "Efficient Implementation of Weighted ENO
      Schemes", JCP 126, 202-228 (1996)
  [2] Lele, S.K., "Compact Finite Difference Schemes with Spectral-like
      Resolution", JCP 103, 16-42 (1992)
  [3] Shu, C.-W., "High Order Finite Difference Methods", Lect. Notes (2009)
"""

import numpy as np
import physical_constants as pc


# =============================================================================
#  WENO5 格式
# =============================================================================

def weno5_reconstruct(f, axis=0, direction='right'):
    """
    WENO5 重构, 计算半节点处的通量.

    对于数组 f = [f_0, f_1, ..., f_{N-1}], 计算半节点通量:
      f_{i+1/2} = w_0 * f^{(0)}_{i+1/2} + w_1 * f^{(1)}_{i+1/2} + w_2 * f^{(2)}_{i+1/2}

    其中 i = 2, 3, ..., N-3 (留出边界).

    参数
    ----
    f : ndarray
        节点值数组, 至少需要 6 个点
    axis : int
        沿哪个轴进行重构
    direction : str
        'right' 或 'left', 对应于正/负通量分裂

    返回
    -------
    f_half : ndarray, shape (N-5,)
        半节点通量值
    """
    f = np.asarray(f, dtype=np.float64)
    eps = pc.WENO_EPSILON

    # 理想权重
    d = np.array([1.0/10.0, 6.0/10.0, 3.0/10.0])

    # 移动 f 到合适的轴
    f = np.moveaxis(f, axis, 0)
    N = f.shape[0]

    if N < 6:
        raise ValueError(f"WENO5 至少需要 6 个点, 当前 N = {N}")

    # 内部节点: i = 2, 3, ..., N-3
    # 计算半节点通量 f_{i+1/2}, i = 2, ..., N-4
    n_half = N - 5
    f_half = np.zeros((n_half,) + f.shape[1:], dtype=f.dtype)

    for j in range(n_half):
        i = j + 2  # 内部节点索引

        # 三个模板上的候选重构
        # 模板 0: {i-2, i-1, i}
        q0 = (2.0*f[i-2] - 7.0*f[i-1] + 11.0*f[i]) / 6.0
        # 模板 1: {i-1, i, i+1}
        q1 = (-f[i-1] + 5.0*f[i] + 2.0*f[i+1]) / 6.0
        # 模板 2: {i, i+1, i+2}
        q2 = (2.0*f[i] + 5.0*f[i+1] - f[i+2]) / 6.0

        # 光滑度指标 (smoothness indicators)
        beta0 = (13.0/12.0) * (f[i-2] - 2.0*f[i-1] + f[i])**2 + \
                0.25 * (f[i-2] - 4.0*f[i-1] + 3.0*f[i])**2
        beta1 = (13.0/12.0) * (f[i-1] - 2.0*f[i] + f[i+1])**2 + \
                0.25 * (f[i-1] - f[i+1])**2
        beta2 = (13.0/12.0) * (f[i] - 2.0*f[i+1] + f[i+2])**2 + \
                0.25 * (3.0*f[i] - 4.0*f[i+1] + f[i+2])**2

        # 非线性权重
        alpha0 = d[0] / (eps + beta0)**2
        alpha1 = d[1] / (eps + beta1)**2
        alpha2 = d[2] / (eps + beta2)**2
        alpha_sum = alpha0 + alpha1 + alpha2

        # 防止除零
        alpha_sum = np.maximum(alpha_sum, pc.EPSILON_NUM)
        w0 = alpha0 / alpha_sum
        w1 = alpha1 / alpha_sum
        w2 = alpha2 / alpha_sum

        f_half[j] = w0 * q0 + w1 * q1 + w2 * q2

    f_half = np.moveaxis(f_half, 0, axis)
    return f_half


def weno5_flux_divergence(f, h):
    """
    使用 WENO5 计算通量散度 df/dx.

    采用 Lax-Friedrichs 通量分裂:
      f^+ = 0.5*(f + alpha*u), f^- = 0.5*(f - alpha*u)
    其中 alpha 为最大波速.

    对于纯标量输运方程 du/dt + a*du/dx = 0,
    通量 F = a*u, 分裂为 F^+ 和 F^-.

    参数
    ----
    f : ndarray, shape (N,)
        节点值
    h : float
        网格间距

    返回
    -------
    dfdx : ndarray, shape (N,)
        导数近似 (内部节点), 边界外推
    """
    N = len(f)
    if N < 6:
        raise ValueError(f"WENO5 至少需要 6 个点, 当前 N = {N}")

    dfdx = np.zeros_like(f)

    # 使用局部 Lax-Friedrichs 分裂
    # 估计最大波速 (简单差分)
    df_approx = np.diff(f) / h
    a_max = np.max(np.abs(df_approx)) + pc.EPSILON_NUM

    # 通量分裂: F = a*f, F^+ = 0.5*(F + a_max*f), F^- = 0.5*(F - a_max*f)
    # 对于单位速度对流: F = f
    F_plus = 0.5 * (f + a_max * f)
    F_minus = 0.5 * (f - a_max * f)

    # WENO 重构
    Fp_half = weno5_reconstruct(F_plus)
    Fm_half = weno5_reconstruct(F_minus[::-1])[::-1]  # 反向重构

    # 总通量在半节点
    F_half = Fp_half + Fm_half

    # 散度: (F_{i+1/2} - F_{i-1/2}) / h
    # 内部节点 i = 3, 4, ..., N-4
    for i in range(3, N - 3):
        j_right = i - 2  # F_half 的索引 (对应 i+1/2)
        j_left = i - 3   # 对应 i-1/2
        if 0 <= j_right < len(F_half) and 0 <= j_left < len(F_half):
            dfdx[i] = (F_half[j_right] - F_half[j_left]) / h

    # 边界: 使用低阶差分外推
    dfdx[0] = (f[1] - f[0]) / h
    dfdx[1] = (f[2] - f[0]) / (2.0 * h)
    dfdx[-1] = (f[-1] - f[-2]) / h
    dfdx[-2] = (f[-1] - f[-3]) / (2.0 * h)

    return dfdx


# =============================================================================
#  中心差分格式
# =============================================================================

def central_diff_2nd(f, h):
    """
    2阶中心差分: f'(x_i) ~ (f_{i+1} - f_{i-1}) / (2h)

    截断误差: O(h^2)
    """
    N = len(f)
    df = np.zeros_like(f)
    df[1:-1] = (f[2:] - f[:-2]) / (2.0 * h)
    # 边界: 单侧差分
    df[0] = (f[1] - f[0]) / h
    df[-1] = (f[-1] - f[-2]) / h
    return df


def central_diff_4th(f, h):
    """
    4阶中心差分: f'(x_i) ~ (-f_{i+2} + 8*f_{i+1} - 8*f_{i-1} + f_{i-2}) / (12h)

    截断误差: O(h^4)

    推导: 通过 Taylor 展开:
      f(x+h) = f + h*f' + h^2/2*f'' + h^3/6*f''' + h^4/24*f''''
      f(x-h) = f - h*f' + h^2/2*f'' - h^3/6*f''' + h^4/24*f''''
      f(x+2h) = f + 2h*f' + 2h^2*f'' + 4h^3/3*f''' + 2h^4/3*f''''

    组合: -f(x+2h) + 8*f(x+h) - 8*f(x-h) + f(x-2h) = 12h*f' + O(h^5)
    """
    N = len(f)
    df = np.zeros_like(f)
    if N < 5:
        # 退化到 2 阶
        return central_diff_2nd(f, h)
    df[2:-2] = (-f[4:] + 8.0*f[3:-1] - 8.0*f[1:-3] + f[:-4]) / (12.0 * h)
    # 边界处理
    df[0] = (f[1] - f[0]) / h
    df[1] = (f[2] - f[0]) / (2.0 * h)
    df[-2] = (f[-1] - f[-3]) / (2.0 * h)
    df[-1] = (f[-1] - f[-2]) / h
    return df


def central_diff_6th(f, h):
    """
    6阶中心差分: 截断误差 O(h^6).

    系数: f'(x_i) ~ (f_{i-3} - 9*f_{i-2} + 45*f_{i-1} - 45*f_{i+1} + 9*f_{i+2} - f_{i+3}) / (60h)
    """
    N = len(f)
    df = np.zeros_like(f)
    if N < 7:
        return central_diff_4th(f, h)
    df[3:-3] = (f[:-6] - 9.0*f[1:-5] + 45.0*f[2:-4]
                - 45.0*f[4:-2] + 9.0*f[5:-1] - f[6:]) / (60.0 * h)
    # 边界退化
    df[0] = (f[1] - f[0]) / h
    df[1] = (f[2] - f[0]) / (2.0 * h)
    df[2] = central_diff_4th(f, h)[2]
    df[-3] = central_diff_4th(f, h)[-3]
    df[-2] = (f[-1] - f[-3]) / (2.0 * h)
    df[-1] = (f[-1] - f[-2]) / h
    return df


def central_diff2_2nd(f, h):
    """
    2阶中心差分二阶导数: f''(x_i) ~ (f_{i+1} - 2*f_i + f_{i-1}) / h^2

    截断误差: O(h^2)
    """
    N = len(f)
    d2f = np.zeros_like(f)
    d2f[1:-1] = (f[2:] - 2.0*f[1:-1] + f[:-2]) / (h**2)
    d2f[0] = d2f[1]
    d2f[-1] = d2f[-2]
    return d2f


def central_diff2_4th(f, h):
    """
    4阶中心差分二阶导数:
      f''(x_i) ~ (-f_{i+2} + 16*f_{i+1} - 30*f_i + 16*f_{i-1} - f_{i-2}) / (12*h^2)

    截断误差: O(h^4)
    """
    N = len(f)
    d2f = np.zeros_like(f)
    if N < 5:
        return central_diff2_2nd(f, h)
    d2f[2:-2] = (-f[4:] + 16.0*f[3:-1] - 30.0*f[2:-2] + 16.0*f[1:-3] - f[:-4]) / (12.0 * h**2)
    d2f[0] = (f[2] - 2.0*f[1] + f[0]) / h**2
    d2f[1] = (f[3] - 2.0*f[2] + f[1]) / h**2
    d2f[-2] = (f[-1] - 2.0*f[-2] + f[-3]) / h**2
    d2f[-1] = (f[-1] - 2.0*f[-2] + f[-3]) / h**2
    return d2f


# =============================================================================
#  紧致有限差分 (Compact / Padé 格式)
# =============================================================================

def compact_first_derivative(f, h, alpha=1.0/3.0):
    """
    4阶紧致 (Padé) 一阶导数.

    隐式格式:
      alpha * f'_{i-1} + f'_i + alpha * f'_{i+1} = a * (f_{i+1} - f_{i-1}) / (2h)

    其中 alpha = 1/3, a = 4/3 给出 4 阶精度.

    推导 (Taylor 展开):
      f'_{i+1} = f'_i + h*f''_i + h^2/2*f'''_i + ...
      f'_{i-1} = f'_i - h*f''_i + h^2/2*f'''_i - ...

    代入:
      (2*alpha + 1)*f'_i + alpha*h^2*f'''_i + ... = a*h*f'_i + a*h^3/6*f'''_i + ...

    匹配阶数:
      2*alpha + 1 = a  (O(1) 项)
      alpha = a/6      (O(h^2) 项消去 -> 4阶)

    解: alpha = 1/3, a = 4/3

    参数
    ----
    f : ndarray
        函数值
    h : float
        网格间距
    alpha : float
        紧致参数, 默认 1/3 (4阶)

    返回
    -------
    df : ndarray
        一阶导数
    """
    N = len(f)
    a_coeff = (1.0 + 2.0 * alpha) / 2.0  # 保证至少 2 阶

    # 构建三对角系统 A * df = b
    # A: 主对角线为 1, 上下次对角线为 alpha
    A = np.zeros((N, N))
    for i in range(N):
        A[i, i] = 1.0
        if i > 0:
            A[i, i-1] = alpha
        if i < N - 1:
            A[i, i+1] = alpha

    # 右端项
    b = np.zeros(N)
    b[1:-1] = a_coeff * (f[2:] - f[:-2]) / (2.0 * h)
    b[0] = (f[1] - f[0]) / h
    b[-1] = (f[-1] - f[-2]) / h

    # 求解三对角系统 (Thomas 算法)
    df = thomas_solver(A, b)
    return df


def compact_second_derivative(f, h, alpha=1.0/10.0):
    """
    4阶紧致二阶导数.

    隐式格式:
      alpha * f''_{i-1} + f''_i + alpha * f''_{i+1} = a * (f_{i+1} - 2*f_i + f_{i-1}) / h^2
                                                   + b * (f_{i+2} - 2*f_i + f_{i-2}) / (4*h^2)

    对于 alpha = 1/10, a = 6/5, b = 0 给出 4 阶精度.

    参数
    ----
    f : ndarray
        函数值
    h : float
        网格间距
    alpha : float
        紧致参数, 默认 1/10

    返回
    -------
    d2f : ndarray
        二阶导数
    """
    N = len(f)
    a_coeff = 6.0 / 5.0  # (1 + 2*alpha) * 12 / (12 - ...)

    A = np.zeros((N, N))
    for i in range(N):
        A[i, i] = 1.0
        if i > 0:
            A[i, i-1] = alpha
        if i < N - 1:
            A[i, i+1] = alpha

    b = np.zeros(N)
    b[2:-2] = a_coeff * (f[3:-1] - 2.0*f[2:-2] + f[1:-3]) / h**2
    # 边界
    b[0] = (f[2] - 2.0*f[1] + f[0]) / h**2
    b[1] = (f[3] - 2.0*f[2] + f[1]) / h**2
    b[-2] = (f[-1] - 2.0*f[-2] + f[-3]) / h**2
    b[-1] = (f[-1] - 2.0*f[-2] + f[-3]) / h**2

    d2f = thomas_solver(A, b)
    return d2f


# =============================================================================
#  Thomas 算法 (三对角系统求解)
# =============================================================================

def thomas_solver(A, b):
    """
    Thomas 算法求解三对角系统 A*x = b.

    对于三对角矩阵:
      |b_0  c_0                   |
      |a_1  b_1  c_1              |
      |     a_2  b_2  c_2         |
      |         ...   ...   ...   |
      |               a_{n-1}  b_{n-1}|

    算法:
      前消:
        c'_0 = c_0 / b_0
        b'_i = b_i - a_i * c'_{i-1}
        c'_i = c_i / b'_i
        d'_i = (d_i - a_i * d'_{i-1}) / b'_i
      回代:
        x_{n-1} = d'_{n-1}
        x_i = d'_i - c'_i * x_{i+1}

    参数
    ----
    A : ndarray, shape (N, N)
        三对角矩阵 (作为稠密矩阵传入, 但内部提取三对角)
    b : ndarray, shape (N,)
        右端项

    返回
    -------
    x : ndarray, shape (N,)
        解向量
    """
    N = len(b)
    if N < 2:
        return b / A[0, 0] if N == 1 else b

    # 提取三对角
    a_diag = np.zeros(N)  # 下次对角线
    b_diag = np.zeros(N)  # 主对角线
    c_diag = np.zeros(N)  # 上次对角线

    for i in range(N):
        b_diag[i] = A[i, i]
        if i > 0:
            a_diag[i] = A[i, i-1]
        if i < N - 1:
            c_diag[i] = A[i, i+1]

    # Thomas 算法
    c_prime = np.zeros(N)
    d_prime = np.zeros(N)

    # 第一步
    if np.abs(b_diag[0]) < pc.EPSILON_NUM:
        # 主元为零, 添加小扰动
        b_diag[0] = pc.EPSILON_NUM
    c_prime[0] = c_diag[0] / b_diag[0]
    d_prime[0] = b[0] / b_diag[0]

    # 前消
    for i in range(1, N):
        denom = b_diag[i] - a_diag[i] * c_prime[i-1]
        if np.abs(denom) < pc.EPSILON_NUM:
            denom = np.sign(denom) * pc.EPSILON_NUM if denom != 0 else pc.EPSILON_NUM
        if i < N - 1:
            c_prime[i] = c_diag[i] / denom
        d_prime[i] = (b[i] - a_diag[i] * d_prime[i-1]) / denom

    # 回代
    x = np.zeros(N)
    x[-1] = d_prime[-1]
    for i in range(N - 2, -1, -1):
        x[i] = d_prime[i] - c_prime[i] * x[i+1]

    return x


# =============================================================================
#  高精度差分格式调度器
# =============================================================================

def compute_derivative(f, h, order=4, scheme='central'):
    """
    统一的一阶导数计算接口.

    参数
    ----
    f : ndarray
        函数值
    h : float
        网格间距
    order : int
        精度阶数: 2, 4, 6
    scheme : str
        格式类型: 'central', 'weno5', 'compact'

    返回
    -------
    df : ndarray
        一阶导数
    """
    if scheme == 'central':
        if order == 2:
            return central_diff_2nd(f, h)
        elif order == 4:
            return central_diff_4th(f, h)
        elif order == 6:
            return central_diff_6th(f, h)
        else:
            raise ValueError(f"不支持的精度阶数: {order}")
    elif scheme == 'weno5':
        return weno5_flux_divergence(f, h)
    elif scheme == 'compact':
        return compact_first_derivative(f, h)
    else:
        raise ValueError(f"未知格式: {scheme}")


def compute_second_derivative(f, h, order=4, scheme='central'):
    """
    统一的二阶导数计算接口.

    参数
    ----
    f : ndarray
        函数值
    h : float
        网格间距
    order : int
        精度阶数: 2, 4
    scheme : str
        格式类型: 'central', 'compact'

    返回
    -------
    d2f : ndarray
        二阶导数
    """
    if scheme == 'central':
        if order == 2:
            return central_diff2_2nd(f, h)
        elif order == 4:
            return central_diff2_4th(f, h)
        else:
            raise ValueError(f"不支持的二阶精度阶数: {order}")
    elif scheme == 'compact':
        return compact_second_derivative(f, h)
    else:
        raise ValueError(f"未知格式: {scheme}")


# =============================================================================
#  自检验证
# =============================================================================

def convergence_test():
    """
    对已知函数进行收敛性测试, 验证各阶差分格式的精度.

    测试函数: f(x) = sin(x)
    精确导数: f'(x) = cos(x)
    精确二阶导数: f''(x) = -sin(x)

    对 h = pi/10, pi/20, pi/40, ..., 计算 L_inf 误差,
    期望收敛阶: O(h^p), p 为格式阶数.
    """
    print("=" * 60)
    print("差分格式收敛性测试: f(x) = sin(x)")
    print("=" * 60)

    N_list = [20, 40, 80, 160, 320]
    # 实际 x 间距: h = 2*pi/N
    h_list = [2.0 * np.pi / n for n in N_list]

    # 2阶中心差分
    print("\n--- 2阶中心差分 (一阶导数) ---")
    errors_2 = []
    for N, h in zip(N_list, h_list):
        x = np.linspace(0, 2*np.pi, N, endpoint=False)
        f = np.sin(x)
        df_num = central_diff_2nd(f, h)
        df_exact = np.cos(x)
        err = np.max(np.abs(df_num - df_exact))
        errors_2.append(err)
        print(f"  N = {N:4d}, h = {h:.5f}, error = {err:.4e}")

    # 4阶中心差分
    print("\n--- 4阶中心差分 (一阶导数) ---")
    errors_4 = []
    for N, h in zip(N_list, h_list):
        x = np.linspace(0, 2*np.pi, N, endpoint=False)
        f = np.sin(x)
        df_num = central_diff_4th(f, h)
        df_exact = np.cos(x)
        err = np.max(np.abs(df_num - df_exact))
        errors_4.append(err)
        print(f"  N = {N:4d}, h = {h:.5f}, error = {err:.4e}")

    # 计算收敛阶
    print("\n--- 收敛阶 (相邻网格) ---")
    for i in range(1, len(h_list)):
        if errors_2[i-1] > 0 and errors_2[i] > 0:
            order_2 = np.log(errors_2[i-1] / errors_2[i]) / np.log(h_list[i-1] / h_list[i])
        else:
            order_2 = 0.0
        if errors_4[i-1] > 0 and errors_4[i] > 0:
            order_4 = np.log(errors_4[i-1] / errors_4[i]) / np.log(h_list[i-1] / h_list[i])
        else:
            order_4 = 0.0
        print(f"  h ratio {h_list[i-1]/h_list[i]:.1f}: 2nd order = {order_2:.2f}, 4th order = {order_4:.2f}")

    return h_list, errors_2, errors_4


if __name__ == "__main__":
    convergence_test()
