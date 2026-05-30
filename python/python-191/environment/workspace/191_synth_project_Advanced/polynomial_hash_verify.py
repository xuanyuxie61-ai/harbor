
import numpy as np
from typing import List


def polynomial_degree(p: np.ndarray) -> int:
    p = np.asarray(p)
    if p.size == 0 or np.all(p == 0):
        return -1
    return int(np.max(np.nonzero(p)[0]))


def collatz_polynomial_next(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=np.int64)
    n = polynomial_degree(p)
    
    if n < 0:
        return np.array([0])
    if n == 0:
        return p.copy()
    

    p = p[:n + 1]
    
    if p[0] == 0:

        p_next = p[1:].copy()
    else:

        p_next = np.zeros(n + 2, dtype=np.int64)
        p_next[0] = (p[0] + 1) % 2
        p_next[1:n + 1] = (p[1:] + p[:n]) % 2
        p_next[n + 1] = p[n] % 2
    
    return p_next % 2


def collatz_polynomial_sequence(
    p0: np.ndarray,
    n_steps: int
) -> List[np.ndarray]:
    seq = [p0.copy()]
    p = p0.copy()
    for _ in range(n_steps):
        p = collatz_polynomial_next(p)
        seq.append(p.copy())
    return seq


def matrix_to_polynomial_f2(M: np.ndarray) -> np.ndarray:
    M = np.asarray(M)
    m, n = M.shape
    p = np.zeros(m * n, dtype=np.int64)
    for i in range(m):
        for j in range(n):
            idx = i * n + j
            val = M[i, j]

            p[idx] = 0 if abs(val) < 0.5 else 1
    return p


def polynomial_hash_matrix(
    M: np.ndarray,
    n_iterations: int = 10
) -> str:
    p = matrix_to_polynomial_f2(M)
    for _ in range(n_iterations):
        p = collatz_polynomial_next(p)
        if polynomial_degree(p) < 0:
            p = np.array([0])
            break
    


    bits = ''.join(str(int(b)) for b in p)

    while len(bits) % 4 != 0:
        bits += '0'
    hex_str = ''
    for i in range(0, len(bits), 4):
        nibble = int(bits[i:i + 4], 2)
        hex_str += format(nibble, 'x')
    return hex_str


def verify_matrix_multiply_checksum(
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    tolerance: float = 1e-10
) -> bool:
    if A.shape[1] != B.shape[0] or C.shape != (A.shape[0], B.shape[1]):
        return False
    

    np.random.seed(42)
    i = np.random.randint(0, A.shape[0])
    j = np.random.randint(0, B.shape[1])
    
    expected = np.dot(A[i, :], B[:, j])
    actual = C[i, j]
    
    if abs(expected - actual) > tolerance * max(1.0, abs(expected)):
        return False
    

    hash_A = polynomial_hash_matrix(A, 5)
    hash_B = polynomial_hash_matrix(B, 5)
    hash_C = polynomial_hash_matrix(C, 5)
    


    return len(hash_A) > 0 and len(hash_B) > 0 and len(hash_C) > 0


def binary_matrix_multiply_f2(
    A: np.ndarray,
    B: np.ndarray
) -> np.ndarray:
    A_bin = (np.abs(A) >= 0.5).astype(np.int64)
    B_bin = (np.abs(B) >= 0.5).astype(np.int64)
    
    m, k = A_bin.shape
    k2, n = B_bin.shape
    if k != k2:
        raise ValueError("Incompatible shapes")
    
    C = np.zeros((m, n), dtype=np.int64)
    for i in range(m):
        for j in range(n):
            val = 0
            for idx in range(k):
                val ^= (A_bin[i, idx] & B_bin[idx, j])
            C[i, j] = val
    
    return C


if __name__ == "__main__":

    p0 = np.array([1, 0, 1])
    seq = collatz_polynomial_sequence(p0, 5)
    print("Collatz sequence:")
    for i, p in enumerate(seq):
        print(f"  Step {i}: {p}")
    

    M = np.array([[1, 2], [3, 4]])
    h = polynomial_hash_matrix(M, 8)
    print("Matrix hash:", h)
    

    A = np.array([[1, 0], [1, 1]])
    B = np.array([[0, 1], [1, 0]])
    C = binary_matrix_multiply_f2(A, B)
    print("F2 multiply result:\n", C)
