"""
validation_utils.py
矩阵验证与校验和工具

基于 luhn 的核心算法:
    - 数字串校验和计算
    - 矩阵元素校验和
    - PMNS 矩阵幺正性验证
    - 概率守恒验证

物理应用:
    1. 验证 PMNS 矩阵满足幺正性 U U† = I
    2. 验证振荡概率之和为 1 (概率守恒)
    3. 验证数值计算的稳定性
"""

import numpy as np


def digit_checksum(s):
    """
    计算数字字符串的 Luhn 校验和。
    (源自 luhn_checksum)

    算法:
        1. 从右往左, 每隔一位取一个数字, 直接相加
        2. 从右往左第二个数字开始, 每隔一位, 将数字乘以 2,
           如果结果 >= 10, 则拆分为个位和十位相加
        3. 总和 mod 10 = 0 则有效

    参数:
        s: 数字字符串

    返回:
        checksum: 校验和 (0 表示有效)
    """
    if not s:
        return 0

    digits = []
    for ch in s:
        if ch.isdigit():
            digits.append(int(ch))

    if not digits:
        return 0

    n = len(digits)
    total = sum(digits[n - 1::-2])  # 从右数第 1, 3, 5... 位

    for i in range(n - 2, -1, -2):
        d2 = 2 * digits[i]
        total += d2 // 10 + d2 % 10

    return total % 10


def is_valid_luhn(s):
    """
    检查数字字符串是否通过 Luhn 校验。

    参数:
        s: 数字字符串

    返回:
        bool: 是否有效
    """
    return digit_checksum(s) == 0


def matrix_checksum(A):
    """
    计算矩阵的校验和, 用于验证数值一致性。

    参数:
        A: 矩阵

    返回:
        checksum: 校验和值
    """
    A = np.asarray(A)
    # 将矩阵元素映射为数字串
    flat = A.flatten()
    # 使用加权和对浮点数进行校验
    checksum = 0
    for i, val in enumerate(flat):
        checksum += (i + 1) * int(abs(val) * 1e6) % 1000
    return checksum % 10


def validate_probability_conservation(P_matrix, tol=1e-8):
    """
    验证振荡概率矩阵满足守恒条件。

    对于幺正演化:
        Σ_β P(α → β) = 1   (对所有 α)

    参数:
        P_matrix: (3, 3) 概率矩阵, P[α, β] = P(ν_α → ν_β)
        tol:      容差

    返回:
        is_valid: bool
        max_error: float
    """
    P = np.asarray(P_matrix, dtype=np.float64)
    if P.shape != (3, 3):
        raise ValueError("P_matrix must be 3x3")

    row_sums = np.sum(P, axis=1)
    col_sums = np.sum(P, axis=0)

    err_row = np.max(np.abs(row_sums - 1.0))
    err_col = np.max(np.abs(col_sums - 1.0))

    # 概率必须在 [0, 1] 内
    err_range = max(np.max(P) - 1.0, 0.0 - np.min(P))

    max_error = max(err_row, err_col, err_range)
    return max_error < tol, max_error


def validate_hermitian(H, tol=1e-10):
    """
    验证矩阵是否为厄米矩阵: H = H†。

    参数:
        H: 矩阵
        tol: 容差

    返回:
        is_hermitian: bool
        max_error:    float
    """
    H = np.asarray(H, dtype=np.complex128)
    diff = H - H.conj().T
    max_error = np.max(np.abs(diff))
    return max_error < tol, max_error


def validate_eigenvalue_ordering(eigenvalues, tol=1e-10):
    """
    验证本征值排序 (升序)。

    参数:
        eigenvalues: 本征值数组
        tol:         容差

    返回:
        is_ordered: bool
    """
    ev = np.asarray(eigenvalues, dtype=np.float64)
    for i in range(len(ev) - 1):
        if ev[i] > ev[i + 1] + tol:
            return False
    return True


def validate_pmns_completeness(U, tol=1e-10):
    """
    验证 PMNS 矩阵的完整性条件。

    条件:
        1. U U† = I
        2. Σ_i U_{αi} U*_{βi} = δ_{αβ}
        3. |U_{e1}|² + |U_{e2}|² + |U_{e3}|² = 1
        4. |U_{μ1}|² + |U_{μ2}|² + |U_{μ3}|² = 1
        5. |U_{τ1}|² + |U_{τ2}|² + |U_{τ3}|² = 1
    """
    U = np.asarray(U, dtype=np.complex128)
    if U.shape != (3, 3):
        raise ValueError("U must be 3x3")

    identity = np.eye(3, dtype=np.complex128)
    err1 = np.max(np.abs(U @ U.conj().T - identity))
    err2 = np.max(np.abs(U.conj().T @ U - identity))

    # 行和列的模方和
    row_sums = np.sum(np.abs(U) ** 2, axis=1)
    col_sums = np.sum(np.abs(U) ** 2, axis=0)
    err3 = np.max(np.abs(row_sums - 1.0))
    err4 = np.max(np.abs(col_sums - 1.0))

    max_err = max(err1, err2, err3, err4)
    return max_err < tol, max_err


def validate_oscillation_unitarity(U_prop, tol=1e-10):
    """
    验证演化矩阵的幺正性。

    对于幺正演化算符:
        U_prop · U_prop† = I

    参数:
        U_prop: (3, 3) 演化矩阵
        tol:    容差

    返回:
        is_unitary: bool
        max_error:  float
    """
    U = np.asarray(U_prop, dtype=np.complex128)
    identity = np.eye(U.shape[0], dtype=np.complex128)
    err1 = np.max(np.abs(U @ U.conj().T - identity))
    err2 = np.max(np.abs(U.conj().T @ U - identity))
    max_err = max(err1, err2)
    return max_err < tol, max_err
