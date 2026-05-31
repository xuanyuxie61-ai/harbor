"""
utils.py
========
通用工具函数模块

功能:
  - 数值稳定性工具
  - 边界条件处理
  - 结果格式化输出
"""

import numpy as np
import sys


def safe_divide(a, b, fill_value=0.0):
    """安全除法，避免除零。"""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    result = np.where(np.abs(b) < 1e-15, fill_value, a / b)
    return result


def clip_and_warn(arr, vmin, vmax, name="array"):
    """裁剪数组并警告越界。"""
    arr = np.asarray(arr, dtype=float)
    clipped = np.clip(arr, vmin, vmax)
    n_clipped = np.sum((arr < vmin) | (arr > vmax))
    if n_clipped > 0:
        print(f"[WARN] {name}: {n_clipped} 个值被裁剪到 [{vmin}, {vmax}]",
              file=sys.stderr)
    return clipped


def check_finite(arr, name="array"):
    """检查数组是否包含非有限值。"""
    arr = np.asarray(arr)
    if not np.all(np.isfinite(arr)):
        n_bad = np.sum(~np.isfinite(arr))
        raise ValueError(f"{name} 包含 {n_bad} 个非有限值 (nan/inf)")
    return True


def print_section(title, width=70):
    """打印格式化分隔线。"""
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


def print_subsection(title, width=70):
    """打印子节分隔线。"""
    print("\n" + "-" * width)
    print(f"  {title}")
    print("-" * width)


def save_results_to_text(filename, results_dict):
    """
    将结果保存为文本文件。

    Parameters
    ----------
    filename : str
    results_dict : dict
        {key: value_or_array}
    """
    with open(filename, 'w', encoding='utf-8') as f:
        for key, value in results_dict.items():
            f.write(f"{key}:\n")
            if isinstance(value, np.ndarray):
                f.write(str(value))
            else:
                f.write(str(value))
            f.write("\n\n")


def compute_rmse(a, b):
    """计算均方根误差。"""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("a 和 b 形状必须相同")
    return np.sqrt(np.mean((a - b)**2))


def compute_relative_error(exact, approx):
    """计算相对误差。"""
    exact = np.asarray(exact, dtype=float)
    approx = np.asarray(approx, dtype=float)
    denom = np.maximum(np.abs(exact), 1e-15)
    return np.mean(np.abs(exact - approx) / denom)


def gaussian_2d(X, Y, cx, cy, sigma_x, sigma_y, amplitude=1.0):
    """
    二维高斯函数。

    f(x,y) = A * exp( -[(x-cx)²/(2σx²) + (y-cy)²/(2σy²)] )
    """
    return amplitude * np.exp(
        -((X - cx)**2) / (2.0 * sigma_x**2)
        - ((Y - cy)**2) / (2.0 * sigma_y**2)
    )


def finite_difference_gradient_2d(F, dx, dy):
    """
    计算二维标量场的中心差分梯度。

    Returns
    -------
    dFdx, dFdy : ndarray
    """
    F = np.asarray(F, dtype=float)
    dFdx = np.zeros_like(F)
    dFdy = np.zeros_like(F)
    dFdx[1:-1, :] = (F[2:, :] - F[:-2, :]) / (2.0 * dx)
    dFdy[:, 1:-1] = (F[:, 2:] - F[:, :-2]) / (2.0 * dy)
    return dFdx, dFdy
