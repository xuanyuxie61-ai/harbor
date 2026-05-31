"""
utils.py
========
通用辅助工具模块。
提供边界检查、数值稳定性处理、以及科学计算中常用的辅助函数。
"""

import numpy as np


def safe_divide(a, b, fill_value=0.0):
    """
    安全除法，避免除零错误。
    
    Parameters
    ----------
    a, b : array_like or float
        被除数与除数。
    fill_value : float
        除零时填充值。
    
    Returns
    -------
    result : ndarray or float
        a / b，其中 b == 0 的位置替换为 fill_value。
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    result = np.empty_like(a, dtype=float)
    mask = np.abs(b) > 1e-15
    result[mask] = a[mask] / b[mask]
    result[~mask] = fill_value
    return result


def clip_with_warning(x, xmin, xmax, name="variable"):
    """
    将变量裁剪到合法区间，并在越界时打印警告。
    
    Parameters
    ----------
    x : array_like
        输入变量。
    xmin, xmax : float
        下界与上界。
    name : str
        变量名称，用于警告信息。
    
    Returns
    -------
    x_clipped : ndarray
        裁剪后的变量。
    """
    x = np.asarray(x, dtype=float)
    if np.any(x < xmin - 1e-12) or np.any(x > xmax + 1e-12):
        print(f"[WARN] {name} out of bounds [{xmin}, {xmax}], clipping applied.")
    return np.clip(x, xmin, xmax)


def ensure_positive(x, eps=1e-12, name="variable"):
    """
    确保变量为正，将非正元素提升到 eps。
    
    Parameters
    ----------
    x : array_like
        输入变量。
    eps : float
        最小正值。
    name : str
        变量名称。
    
    Returns
    -------
    x_pos : ndarray
        保证为正的变量。
    """
    x = np.asarray(x, dtype=float)
    if np.any(x <= 0):
        print(f"[WARN] {name} contains non-positive values, raised to {eps}.")
    return np.where(x > eps, x, eps)


def relative_change(new, old):
    """
    计算相对变化量，用于收敛判断。
    
    Parameters
    ----------
    new, old : array_like
        新值与旧值。
    
    Returns
    -------
    rel : float
        max(|new - old| / (|old| + 1e-15))。
    """
    new = np.asarray(new, dtype=float)
    old = np.asarray(old, dtype=float)
    denom = np.abs(old) + 1e-15
    return np.max(np.abs(new - old) / denom)


def thermo_factor_check(T, Tmin=200.0, Tmax=800.0, Pmin=1e3, Pmax=5e6):
    """
    热力学状态边界检查：温度与压力的合理区间。
    
    Parameters
    ----------
    T : float
        温度 [K]。
    Tmin, Tmax : float
        温度上下界 [K]。
    Pmin, Pmax : float
        压力上下界 [Pa]。
    
    Returns
    -------
    T_safe : float
        安全温度。
    """
    return float(clip_with_warning(T, Tmin, Tmax, "Temperature"))
