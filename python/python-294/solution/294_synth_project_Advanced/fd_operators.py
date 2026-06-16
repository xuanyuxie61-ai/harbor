"""
fd_operators.py - 高阶有限差分算子

本模块实现激光等离子体相互作用仿真中使用的高阶有限差分 (FD) 算子。
在 Vlasov-Maxwell 方程的数值求解中，空间导数的精度直接影响
波的色散特性和数值稳定性。

实现的差分格式 (中心差分):

  2阶精度 (3点模板):
    f'(x_i)  ≈ (-f_{i-1} + f_{i+1}) / (2h)
    f''(x_i) ≈ (f_{i-1} - 2f_i + f_{i+1}) / h^2

  4阶精度 (5点模板):
    f'(x_i) ≈ (f_{i-2} - 8f_{i-1} + 8f_{i+1} - f_{i+2}) / (12h)
    f''(x_i) ≈ (-f_{i-2} + 16f_{i-1} - 30f_i + 16f_{i+1} - f_{i+2}) / (12h^2)

  6阶精度 (7点模板):
    f'(x_i) ≈ (-f_{i-3} + 9f_{i-2} - 45f_{i-1} + 45f_{i+1} - 9f_{i+2} + f_{i+3}) / (60h)

  8阶精度 (9点模板):
    f'(x_i) ≈ (f_{i-4} - 32/3 f_{i-3} + 56/3 f_{i-2} - 224/3 f_{i-1}
                + 224/3 f_{i+1} - 56/3 f_{i+2} + 32/3 f_{i+3} - f_{i+4}) / (280h)

  4阶精度的四阶导数:
    f''''(x_i) ≈ (f_{i-2} - 4f_{i-1} + 6f_i - 4f_{i+1} + f_{i+2}) / h^4

边界处理策略:
  - 周期边界: wrap padding
  - 零 Neumann: mirror padding
  - 吸收边界 (PML): 渐进衰减层

数值色散关系:
  对于 k 阶有限差分近似的空间导数，修正波数为:
    k_eff(k*h) = sum_{j} c_j * sin(j*k*h) / h
  其中 c_j 为差分系数。数值相速度 v_phi = omega / k_eff 与
  真实相速度的偏差导致数值色散误差。
"""

import numpy as np
from numpy.polynomial import polynomial as P


def fd_coefficients_first_deriv(order):
    """获取中心差分一阶导数的系数。

    对于 2p 阶精度的中心差分，使用 2p+1 个网格点。
    系数通过 Taylor 展开匹配确定。

    一般公式 (通过 Fourier 分析导出):
        sum_{j=1}^{p} a_j * sin(j*k*h) / h = k + O((kh)^{2p})

    Parameters
    ----------
    order : int
        精度阶数 (2, 4, 6, 8)

    Returns
    -------
    coeffs : ndarray
        半模板系数 [a_1, a_2, ..., a_p]
        完整模板为 [-a_p, ..., -a_1, 0, a_1, ..., a_p]
    stencil_width : int
        模板半宽 p
    """
    if order == 2:
        # 2阶: (f_{i+1} - f_{i-1}) / (2h)
        # 写作 c_1 * (f_{i+1} - f_{i-1}) / h, c_1 = 1/2
        coeffs = np.array([0.5])
    elif order == 4:
        # 4阶: (-f_{i+2} + 8f_{i+1} - 8f_{i-1} + f_{i-2}) / (12h)
        # c_1 = 2/3, c_2 = -1/12
        coeffs = np.array([2.0 / 3.0, -1.0 / 12.0])
    elif order == 6:
        # 6阶: c_1 = 3/4, c_2 = -3/20, c_3 = 1/60
        coeffs = np.array([3.0 / 4.0, -3.0 / 20.0, 1.0 / 60.0])
    elif order == 8:
        # 8阶: c_1 = 8/5, c_2 = -1/5, c_3 = 4/105, c_4 = -1/280
        coeffs = np.array([8.0 / 5.0, -1.0 / 5.0, 4.0 / 105.0, -1.0 / 280.0])
    else:
        raise ValueError(f"不支持的精度阶数: {order}，支持 2, 4, 6, 8")
    return coeffs, order // 2


def fd_coefficients_second_deriv(order):
    """获取中心差分二阶导数的系数。

    对于 2p 阶精度的中心差分二阶导数:
        f''(x_i) ≈ (1/h^2) * sum_{j=-p}^{p} b_j * f_{i+j}

    2阶: [1, -2, 1]
    4阶: [-1/12, 4/3, -5/2, 4/3, -1/12]
    6阶: [1/90, -3/20, 3/2, -49/18, 3/2, -3/20, 1/90]

    Parameters
    ----------
    order : int
        精度阶数 (2, 4, 6, 8)

    Returns
    -------
    coeffs : ndarray
        半模板系数 [b_0, b_1, ..., b_p] (对称, 只给非负部分)
    stencil_width : int
        模板半宽 p
    """
    if order == 2:
        # 2阶: (f_{i-1} - 2f_i + f_{i+1}) / h^2
        coeffs = np.array([-2.0, 1.0])  # b_0, b_1
    elif order == 4:
        # 4阶: (-f_{i-2} + 16f_{i-1} - 30f_i + 16f_{i+1} - f_{i+2}) / (12h^2)
        coeffs = np.array([-5.0 / 2.0, 4.0 / 3.0, -1.0 / 12.0])
    elif order == 6:
        # 6阶: (f_{i-3} - 9f_{i-2} + 45f_{i-1} - 90f_i + 45f_{i+1} - 9f_{i+2} + f_{i+3}) / (180h^2)
        # b_0 = -90/180 = -1/2, b_1 = 45/180 = 3/4, b_2 = -9/180 = -3/60=-1/20, b_3 = 1/180
        # 但标准的6阶二阶导数系数是:
        # b_0 = 2*1/90 - 2*9/180...
        # 采用标准结果: b_0 = -2*(1+1/4+1/9) = -49/18
        coeffs = np.array([-49.0 / 18.0, 3.0 / 2.0, -3.0 / 20.0, 1.0 / 90.0])
    elif order == 8:
        # 8阶二阶导数系数
        # b_0 = -205/72, b_1 = 8/5, b_2 = -1/5, b_3 = 8/315, b_4 = -1/560
        # 但符号约定: L_h*f = (b_0*f_i + sum b_j*(f_{i+j}+f_{i-j}))/h^2
        # 中心系数应为负:
        coeffs = np.array([
            -205.0 / 72.0,   # b_0 (负)
            8.0 / 5.0,       # b_1
            -1.0 / 5.0,      # b_2
            8.0 / 315.0,     # b_3
            -1.0 / 560.0,    # b_4
        ])
    else:
        raise ValueError(f"不支持的精度阶数: {order}")
    return coeffs, order // 2


def apply_first_derivative(f, dx, order=4, bc='periodic'):
    """应用高阶一阶导数有限差分算子。

    计算公式:
        (D_h f)_i = (1/h) * sum_{j=1}^{p} a_j * (f_{i+j} - f_{i-j})

    修正波数分析:
        k_eff * h = sum_{j=1}^{p} a_j * sin(j * k * h)
        对于 4 阶: k_eff*h = (4/3)*sin(kh) - (1/6)*sin(2kh)

    Parameters
    ----------
    f : ndarray
        输入场数据 (1D 数组)
    dx : float
        网格间距
    order : int
        精度阶数 (2, 4, 6, 8)
    bc : str
        边界条件类型 ('periodic', 'neumann', 'dirichlet')

    Returns
    -------
    df : ndarray
        一阶导数近似
    """
    N = len(f)
    df = np.zeros(N)
    coeffs, p = fd_coefficients_first_deriv(order)

    # 扩展数组用于边界处理
    f_ext = _extend_array(f, p, bc)

    for j in range(1, p + 1):
        df += coeffs[j - 1] * (f_ext[p + j: p + j + N] - f_ext[p - j: p - j + N])

    df /= dx
    return df


def apply_second_derivative(f, dx, order=4, bc='periodic'):
    """应用高阶二阶导数有限差分算子。

    计算公式:
        (D2_h f)_i = (1/h^2) * [b_0 * f_i + sum_{j=1}^{p} b_j * (f_{i+j} + f_{i-j})]

    Parameters
    ----------
    f : ndarray
        输入场数据
    dx : float
        网格间距
    order : int
        精度阶数
    bc : str
        边界条件

    Returns
    -------
    d2f : ndarray
        二阶导数近似
    """
    N = len(f)
    d2f = np.zeros(N)
    coeffs, p = fd_coefficients_second_deriv(order)

    f_ext = _extend_array(f, p, bc)

    # 中心项
    d2f += coeffs[0] * f

    # 对称项
    for j in range(1, p + 1):
        d2f += coeffs[j] * (f_ext[p + j: p + j + N] + f_ext[p - j: p - j + N])

    d2f /= dx ** 2
    return d2f


def apply_fourth_derivative(f, dx, bc='periodic'):
    """应用四阶导数有限差分算子。

    用于超扩散项 (hyperviscosity):
        f''''(x_i) ≈ (f_{i-2} - 4f_{i-1} + 6f_i - 4f_{i+1} + f_{i+2}) / h^4

    四阶导数出现在:
      - 束流不稳定性分析中的色散修正
      - 超扩散人工粘性项
      - 薄板弯曲方程 (等离子体鞘层模型)

    Parameters
    ----------
    f : ndarray
        输入场
    dx : float
        网格间距
    bc : str
        边界条件

    Returns
    -------
    d4f : ndarray
        四阶导数近似
    """
    N = len(f)
    f_ext = _extend_array(f, 2, bc)

    d4f = (
        f_ext[0:N]
        - 4.0 * f_ext[1:N + 1]
        + 6.0 * f
        - 4.0 * f_ext[3:N + 3]
        + f_ext[4:N + 4]
    ) / dx ** 4

    return d4f


def modified_wavenumber(kh, order=4):
    """计算有限差分离散化的修正波数。

    数值色散分析的核心: 有限差分近似将真实波数 k 映射为
    修正波数 k_eff, 导致数值相速度和群速度偏差。

    修正波数公式:
        k_eff * h = sum_{j=1}^{p} a_j * sin(j * k * h)

    其中 a_j 为一阶导数差分系数。

    数值相速度误差:
        v_phi_err = |k_eff - k| / k = |1 - k_eff/k|

    Parameters
    ----------
    kh : ndarray
        无量纲波数 k*h
    order : int
        差分精度阶数

    Returns
    -------
    keff_h : ndarray
        修正无量纲波数 k_eff * h
    """
    coeffs, p = fd_coefficients_first_deriv(order)
    keff_h = np.zeros_like(kh)
    for j in range(1, p + 1):
        keff_h += coeffs[j - 1] * np.sin(j * kh)
    return keff_h


def spectral_error_analysis(N_x, order_list=None):
    """分析不同精度阶数的谱误差。

    比较各阶有限差分格式在可分辨波数范围内的色散误差。

    色散误差定义为:
        E(kh) = k_eff(kh) - kh

    对于 2p 阶格式,  leading-order 误差为:
        E(kh) ~ C_{2p} * (kh)^{2p+1}

    Parameters
    ----------
    N_x : int
        网格点数
    order_list : list of int, optional
        要比较的精度阶数列表

    Returns
    -------
    results : dict
        包含波数、各阶修正波数和色散误差的字典
    """
    if order_list is None:
        order_list = [2, 4, 6, 8]

    kh = np.linspace(0, np.pi, N_x)
    results = {'kh': kh, 'exact': kh}

    for order in order_list:
        keff = modified_wavenumber(kh, order)
        results[f'keff_order{order}'] = keff
        results[f'error_order{order}'] = keff - kh

    return results


def _extend_array(f, pad_width, bc):
    """根据边界条件扩展数组。

    Parameters
    ----------
    f : ndarray
        原始数组
    pad_width : int
        每侧扩展的点数
    bc : str
        边界条件类型

    Returns
    -------
    f_ext : ndarray
        扩展后的数组
    """
    if bc == 'periodic':
        return np.concatenate([f[-pad_width:], f, f[:pad_width]])
    elif bc == 'neumann':
        # 零梯度: 镜像复制
        left = f[pad_width:0:-1] if pad_width <= len(f) else f[::-1]
        right = f[-2:-2 - pad_width:-1] if pad_width <= len(f) else f[::-1]
        # 安全处理
        if len(left) < pad_width:
            left = np.full(pad_width, f[0])
        if len(right) < pad_width:
            right = np.full(pad_width, f[-1])
        return np.concatenate([left, f, right])
    elif bc == 'dirichlet':
        left = np.zeros(pad_width)
        right = np.zeros(pad_width)
        return np.concatenate([left, f, right])
    else:
        raise ValueError(f"未知边界条件类型: {bc}")
