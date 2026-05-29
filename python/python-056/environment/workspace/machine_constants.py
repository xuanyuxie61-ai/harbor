"""
machine_constants.py
================================================================================
机器精度常数模块 (来源于 706_machine 项目)
================================================================================
本模块提供浮点运算中的机器精度阈值，用于数值算法的收敛判断、
稳定性分析及边界条件判定。在潮汐能提取的流固耦合问题中，这些
常数决定了迭代收敛容差、矩阵条件数报警阈值以及时间步长的下限。

核心公式:
    - 单精度最小正数: B^(EMIN-1)
    - 单精度最大数:   B^EMAX * (1 - B^(-T))
    - 最小相对间距:   B^(-T)
    - 最大相对间距:   B^(1-T)
    - log10(B)
"""

import sys
import numpy as np


def r1mach(i: int) -> float:
    """
    返回单精度实数机器常数。

    参数:
        i: 1~5，选择返回的常数
          1 = 最小正数, 2 = 最大数, 3 = 最小相对间距,
          4 = 最大相对间距, 5 = log10(基数)

    返回:
        对应的机器常数值
    """
    if i < 1 or i > 5:
        raise ValueError(f"R1MACH: 参数 i={i} 越界，合法范围为 1~5")
    values = {
        1: np.float32(1.1754944e-38),
        2: np.float32(3.4028235e38),
        3: np.float32(5.9604645e-08),
        4: np.float32(1.1920929e-07),
        5: np.float32(0.3010300),
    }
    return float(values[i])


def d1mach(i: int) -> float:
    """
    返回双精度实数机器常数。

    参数:
        i: 1~5，选择返回的常数

    返回:
        对应的双精度机器常数值
    """
    if i < 1 or i > 5:
        raise ValueError(f"D1MACH: 参数 i={i} 越界，合法范围为 1~5")
    values = {
        1: np.finfo(float).tiny,
        2: np.finfo(float).max,
        3: np.finfo(float).eps * 0.5,
        4: np.finfo(float).eps,
        5: np.log10(2.0),
    }
    return float(values[i])


def i1mach(i: int) -> int:
    """
    返回整数机器常数。

    参数:
        i: 选择返回的常数
          1 = 标准输入单元, 2 = 标准输出单元,
          4 = 整数字长(bits), 5 = 整数字长(digits),
          6 = 浮点基数, 7-10 = 浮点指数范围,
          11-16 = 双精度范围

    返回:
        对应的整数机器常数
    """
    if i < 1 or i > 16:
        raise ValueError(f"I1MACH: 参数 i={i} 越界，合法范围为 1~16")
    values = {
        1: 5,   # stdin
        2: 6,   # stdout
        4: 32,  # bits per int
        5: 9,   # decimal digits
        6: 2,   # floating point base
        7: 31,  # single exp min (approx)
        8: 127, # single exp max (approx)
        11: 52, # double mantissa bits
        12: -1021, # double exp min
        13: 1024,  # double exp max
    }
    return values.get(i, 0)


def get_machine_epsilon() -> float:
    """返回当前机器的双精度 epsilon。"""
    return d1mach(4)


def get_safe_tol(scale: float = 1.0) -> float:
    """
    计算数值迭代的安全容差。

    公式:
        tol = scale * sqrt(eps_machine)

    参数:
        scale: 缩放因子，默认 1.0

    返回:
        安全容差值
    """
    eps = d1mach(4)
    return scale * np.sqrt(eps)
