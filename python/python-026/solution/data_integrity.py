# -*- coding: utf-8 -*-
"""
data_integrity.py

基于 luhn (Luhn checksum algorithm) 的数值数据完整性校验模块。

原项目 704_luhn 提供了字符串数字的校验和算法。
在科学计算中，该思想被扩展到数值数组的校验和生成，
用于在长时间模拟或分布式计算中检测数据损坏或传输错误。

核心算法:
    1. 将浮点数组量化为整数序列（保留指定位数精度）。
    2. 对整数序列应用 Luhn 算法的变体:
        - 从右至左，每隔一位将数字翻倍，若结果 >= 10 则减去 9。
        - 将所有数字求和后对 10 取模，余数应为 0。
    3. 结合 CRC-32 与 Luhn 校验，生成双重校验码。

物理意义:
    在 ICF 模拟中，等离子体状态数据（密度、温度、速度场）
    的完整性至关重要。本模块为这些数据生成轻量级校验指纹。
"""

import numpy as np
import zlib


def float_array_to_digit_sequence(arr, precision=6):
    """
    将浮点数组转换为数字字符串序列。

    方法:
        1. 将数组展平并按科学计数法格式化，保留 precision 位有效数字。
        2. 提取所有数字字符（包括符号和小数点后的数字）。
        3. 拼接为连续数字字符串。

    Parameters
    ----------
    arr : ndarray
        输入浮点数组。
    precision : int, optional
        有效数字位数。

    Returns
    -------
    digit_str : str
        仅包含数字字符的字符串。
    """
    arr = np.asarray(arr, dtype=float)
    # 展平并转为科学计数法字符串列表
    strs = [f"{x:.{precision}e}" for x in arr.flatten()]
    # 提取数字字符
    digit_str = ''.join(ch for s in strs for ch in s if ch.isdigit())
    return digit_str


def luhn_checksum(digit_str):
    """
    计算 Luhn 校验和。

    原 luhn_checksum 算法:
        1. 从右往左，每隔一位将数字翻倍。
        2. 若翻倍结果 >= 10，则将其各位相加（等价于减 9）。
        3. 求和后对 10 取模。

    Parameters
    ----------
    digit_str : str
        数字字符串。

    Returns
    -------
    checksum : int
        Luhn 校验和 (0-9)。
    """
    digits = [int(ch) for ch in digit_str if ch.isdigit()]
    if not digits:
        return 0

    total = 0
    n = len(digits)
    for i, d in enumerate(reversed(digits)):
        if i % 2 == 1:
            d2 = d * 2
            if d2 >= 10:
                d2 = d2 - 9
            total += d2
        else:
            total += d

    checksum = total % 10
    return checksum


def compute_data_fingerprint(arr, precision=6):
    """
    计算数值数组的综合数据指纹。

    结合 CRC-32 与 Luhn 校验，生成一个紧凑的校验字符串。

    Parameters
    ----------
    arr : ndarray
        输入数组。
    precision : int, optional
        量化精度。

    Returns
    -------
    fingerprint : str
        形如 "crc:XXXXXXXX/luhn:Y" 的指纹字符串。
    """
    # CRC-32
    arr_bytes = np.asarray(arr, dtype=float).tobytes()
    crc_val = zlib.crc32(arr_bytes) & 0xffffffff

    # Luhn
    digit_str = float_array_to_digit_sequence(arr, precision)
    luhn_val = luhn_checksum(digit_str)

    fingerprint = f"crc:{crc_val:08x}/luhn:{luhn_val}"
    return fingerprint


def verify_data_fingerprint(arr, fingerprint, precision=6):
    """
    验证数值数组的数据指纹是否匹配。

    Parameters
    ----------
    arr : ndarray
        输入数组。
    fingerprint : str
        预期指纹。
    precision : int, optional
        量化精度。

    Returns
    -------
    valid : bool
        是否匹配。
    """
    computed = compute_data_fingerprint(arr, precision)
    return computed == fingerprint


def checksum_plasma_state(ne, Te, phi, precision=6):
    """
    为等离子体状态量 (密度、温度、电势) 生成组合校验码。

    Parameters
    ----------
    ne, Te, phi : ndarray
        电子密度、温度、电势场。
    precision : int, optional
        精度。

    Returns
    -------
    combined_checksum : str
        组合校验码。
    """
    fp_ne = compute_data_fingerprint(ne, precision)
    fp_Te = compute_data_fingerprint(Te, precision)
    fp_phi = compute_data_fingerprint(phi, precision)
    combined = f"ne:{fp_ne}|Te:{fp_Te}|phi:{fp_phi}"
    return combined
