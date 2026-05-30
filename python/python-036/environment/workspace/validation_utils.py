
import numpy as np


def digit_checksum(s):
    if not s:
        return 0

    digits = []
    for ch in s:
        if ch.isdigit():
            digits.append(int(ch))

    if not digits:
        return 0

    n = len(digits)
    total = sum(digits[n - 1::-2])

    for i in range(n - 2, -1, -2):
        d2 = 2 * digits[i]
        total += d2 // 10 + d2 % 10

    return total % 10


def is_valid_luhn(s):
    return digit_checksum(s) == 0


def matrix_checksum(A):
    A = np.asarray(A)

    flat = A.flatten()

    checksum = 0
    for i, val in enumerate(flat):
        checksum += (i + 1) * int(abs(val) * 1e6) % 1000
    return checksum % 10


def validate_probability_conservation(P_matrix, tol=1e-8):
    P = np.asarray(P_matrix, dtype=np.float64)
    if P.shape != (3, 3):
        raise ValueError("P_matrix must be 3x3")

    row_sums = np.sum(P, axis=1)
    col_sums = np.sum(P, axis=0)

    err_row = np.max(np.abs(row_sums - 1.0))
    err_col = np.max(np.abs(col_sums - 1.0))


    err_range = max(np.max(P) - 1.0, 0.0 - np.min(P))

    max_error = max(err_row, err_col, err_range)
    return max_error < tol, max_error


def validate_hermitian(H, tol=1e-10):
    H = np.asarray(H, dtype=np.complex128)
    diff = H - H.conj().T
    max_error = np.max(np.abs(diff))
    return max_error < tol, max_error


def validate_eigenvalue_ordering(eigenvalues, tol=1e-10):
    ev = np.asarray(eigenvalues, dtype=np.float64)
    for i in range(len(ev) - 1):
        if ev[i] > ev[i + 1] + tol:
            return False
    return True


def validate_pmns_completeness(U, tol=1e-10):
    U = np.asarray(U, dtype=np.complex128)
    if U.shape != (3, 3):
        raise ValueError("U must be 3x3")

    identity = np.eye(3, dtype=np.complex128)
    err1 = np.max(np.abs(U @ U.conj().T - identity))
    err2 = np.max(np.abs(U.conj().T @ U - identity))


    row_sums = np.sum(np.abs(U) ** 2, axis=1)
    col_sums = np.sum(np.abs(U) ** 2, axis=0)
    err3 = np.max(np.abs(row_sums - 1.0))
    err4 = np.max(np.abs(col_sums - 1.0))

    max_err = max(err1, err2, err3, err4)
    return max_err < tol, max_err


def validate_oscillation_unitarity(U_prop, tol=1e-10):
    U = np.asarray(U_prop, dtype=np.complex128)
    identity = np.eye(U.shape[0], dtype=np.complex128)
    err1 = np.max(np.abs(U @ U.conj().T - identity))
    err2 = np.max(np.abs(U.conj().T @ U - identity))
    max_err = max(err1, err2)
    return max_err < tol, max_err
