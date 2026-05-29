"""
mpi_cannon_multiply.py

MPI-Style Parallel Dense Matrix Multiplication via Cannon's Algorithm.

Scientific Background:
----------------------
Given matrices A, B in R^{n x n}, the standard matrix product is:

    C_{ij} = sum_{k=1}^{n} A_{ik} * B_{kj}     for i,j = 1,...,n

Cannon's algorithm distributes A and B over a sqrt(p) x sqrt(p) process grid.
Each process holds sub-blocks A_{ij}, B_{ij} of size (n/sqrt(p)) x (n/sqrt(p)).

Algorithm steps (q = sqrt(p)):
1. Initial alignment: A_{ij} shifted left by i, B_{ij} shifted up by j.
2. For step s = 0, ..., q-1:
   a. Local multiplication: C_{ij} += A_{ij} * B_{ij}
   b. Circular shift: A_{ij} left by 1, B_{ij} up by 1.

Communication complexity: O(n^2 / sqrt(p)) data volume per process.
Computation complexity: O(n^3 / p) flops per process.

We simulate MPI on a single node using Python's multiprocessing module.
"""

import numpy as np
import multiprocessing as mp
from typing import Tuple, List
import math


def _is_prime(n: int) -> bool:
    """Check if n is a prime number (for process-grid sizing)."""
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    r = int(math.sqrt(n))
    for i in range(3, r + 1, 2):
        if n % i == 0:
            return False
    return True


def _nearest_perfect_square_processes(max_procs: int) -> int:
    """
    Find the largest perfect-square process count <= max_procs.
    Cannon's algorithm requires a square process grid q x q.
    """
    q = int(math.sqrt(max_procs))
    while q > 1:
        if q * q <= max_procs:
            return q * q
        q -= 1
    return 1


def _block_multiply(args):
    """
    Worker function for local block multiplication.
    
    Args:
        A_block: local sub-block of A
        B_block: local sub-block of B
    
    Returns:
        C_block = A_block @ B_block
    """
    A_block, B_block = args
    if A_block.shape[1] != B_block.shape[0]:
        raise ValueError(
            f"Incompatible block shapes: {A_block.shape} vs {B_block.shape}"
        )
    return np.dot(A_block, B_block)


def cannon_multiply(
    A: np.ndarray,
    B: np.ndarray,
    num_processes: int = None
) -> np.ndarray:
    """
    Cannon's algorithm for parallel matrix multiplication.
    
    Mathematical guarantee:
        For matrices A, B with compatible shapes, returns C = A @ B.
        The floating-point error satisfies:
        ||C - fl(C)||_F <= gamma_n * ||A||_F * ||B||_F
        where gamma_n = n * eps / (1 - n * eps), eps is machine epsilon.
    
    Args:
        A: matrix of shape (n, n)
        B: matrix of shape (n, n)
        num_processes: number of parallel processes (default: CPU count)
    
    Returns:
        C: matrix of shape (n, n)
    """
    if A.ndim != 2 or B.ndim != 2:
        raise ValueError("A and B must be 2D arrays")
    n = A.shape[0]
    if A.shape[1] != n or B.shape[0] != n or B.shape[1] != n:
        raise ValueError(f"A and B must be square matrices of same size, got {A.shape}, {B.shape}")
    if n == 0:
        return np.zeros((0, 0), dtype=A.dtype)
    
    if num_processes is None:
        num_processes = mp.cpu_count()
    
    p = _nearest_perfect_square_processes(num_processes)
    q = int(math.sqrt(p))
    
    # If matrix is too small for parallel overhead, use sequential BLAS
    if n < q * 4 or p == 1:
        return np.dot(A, B)
    
    block_size = n // q
    if block_size * q != n:
        # Pad matrices to multiple of q
        pad = block_size * q + q - n
        A_pad = np.pad(A, ((0, pad), (0, pad)), mode='constant')
        B_pad = np.pad(B, ((0, pad), (0, pad)), mode='constant')
        C_pad = cannon_multiply(A_pad, B_pad, num_processes)
        return C_pad[:n, :n]
    
    # Split into q x q blocks
    A_blocks = [[None for _ in range(q)] for _ in range(q)]
    B_blocks = [[None for _ in range(q)] for _ in range(q)]
    C_blocks = [[np.zeros((block_size, block_size), dtype=np.float64) for _ in range(q)] for _ in range(q)]
    
    for i in range(q):
        for j in range(q):
            A_blocks[i][j] = A[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size].copy()
            B_blocks[i][j] = B[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size].copy()
    
    # Initial alignment
    for i in range(q):
        for j in range(q):
            # A shifted left by i
            src_j = (j + i) % q
            A_blocks[i][j] = A[i*block_size:(i+1)*block_size, src_j*block_size:(src_j+1)*block_size].copy()
            # B shifted up by j
            src_i = (i + j) % q
            B_blocks[i][j] = B[src_i*block_size:(src_i+1)*block_size, j*block_size:(j+1)*block_size].copy()
    
    pool = mp.Pool(processes=p)
    
    try:
        for step in range(q):
            # Prepare args for parallel local multiply
            tasks = []
            for i in range(q):
                for j in range(q):
                    tasks.append((A_blocks[i][j], B_blocks[i][j]))
            
            results = pool.map(_block_multiply, tasks)
            
            idx = 0
            for i in range(q):
                for j in range(q):
                    C_blocks[i][j] += results[idx]
                    idx += 1
            
            # Circular shift: A left by 1, B up by 1
            new_A = [[None for _ in range(q)] for _ in range(q)]
            new_B = [[None for _ in range(q)] for _ in range(q)]
            for i in range(q):
                for j in range(q):
                    new_A[i][j] = A_blocks[i][(j + 1) % q]
                    new_B[i][j] = B_blocks[(i + 1) % q][j]
            A_blocks = new_A
            B_blocks = new_B
    finally:
        pool.close()
        pool.join()
    
    # Assemble C from blocks
    C = np.zeros((n, n), dtype=np.float64)
    for i in range(q):
        for j in range(q):
            C[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size] = C_blocks[i][j]
    
    return C


def mpi_summa_multiply(
    A: np.ndarray,
    B: np.ndarray,
    num_processes: int = None
) -> np.ndarray:
    """
    SUMMA (Scalable Universal Matrix Multiplication Algorithm).
    
    Alternative to Cannon with better adaptivity to rectangular grids.
    In SUMMA, A is broadcast row-wise and B column-wise across the process grid.
    
    For each k = 0, ..., n-1 (in blocks):
        Broadcast A(:, k) across rows of process grid
        Broadcast B(k, :) across columns of process grid
        Local rank-1 update: C += A_col * B_row
    
    Args:
        A, B: square matrices
        num_processes: parallel process count
    
    Returns:
        C = A @ B
    """
    if A.ndim != 2 or B.ndim != 2:
        raise ValueError("A and B must be 2D arrays")
    n = A.shape[0]
    if A.shape[1] != n or B.shape[0] != n or B.shape[1] != n:
        raise ValueError("A and B must be compatible square matrices")
    if n == 0:
        return np.zeros((0, 0), dtype=A.dtype)
    
    if num_processes is None:
        num_processes = mp.cpu_count()
    
    p = _nearest_perfect_square_processes(num_processes)
    q = int(math.sqrt(p))
    
    if n < q * 4 or p == 1:
        return np.dot(A, B)
    
    block_size = n // q
    if block_size * q != n:
        pad = block_size * q + q - n
        A_pad = np.pad(A, ((0, pad), (0, pad)), mode='constant')
        B_pad = np.pad(B, ((0, pad), (0, pad)), mode='constant')
        C_pad = mpi_summa_multiply(A_pad, B_pad, num_processes)
        return C_pad[:n, :n]
    
    C = np.zeros((n, n), dtype=np.float64)
    
    pool = mp.Pool(processes=p)
    
    try:
        for k_block in range(q):
            tasks = []
            for i in range(q):
                for j in range(q):
                    A_col = A[i*block_size:(i+1)*block_size, k_block*block_size:(k_block+1)*block_size]
                    B_row = B[k_block*block_size:(k_block+1)*block_size, j*block_size:(j+1)*block_size]
                    tasks.append((A_col.copy(), B_row.copy()))
            
            results = pool.map(_block_multiply, tasks)
            
            idx = 0
            for i in range(q):
                for j in range(q):
                    C[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size] += results[idx]
                    idx += 1
    finally:
        pool.close()
        pool.join()
    
    return C


def frobenius_error(C_parallel: np.ndarray, C_reference: np.ndarray) -> float:
    """
    Compute relative Frobenius norm error:
    
        err = ||C_parallel - C_reference||_F / ||C_reference||_F
    
    Returns:
        relative error (float)
    """
    denom = np.linalg.norm(C_reference, 'fro')
    if denom == 0.0:
        return np.linalg.norm(C_parallel, 'fro')
    return np.linalg.norm(C_parallel - C_reference, 'fro') / denom


def benchmark_parallel_multiply(sizes: List[int] = None, num_processes: int = None):
    """
    Benchmark Cannon and SUMMA algorithms against numpy.dot.
    """
    if sizes is None:
        sizes = [64, 128, 256]
    
    results = []
    for n in sizes:
        A = np.random.randn(n, n)
        B = np.random.randn(n, n)
        
        C_ref = np.dot(A, B)
        C_cannon = cannon_multiply(A, B, num_processes)
        C_summa = mpi_summa_multiply(A, B, num_processes)
        
        err_cannon = frobenius_error(C_cannon, C_ref)
        err_summa = frobenius_error(C_summa, C_ref)
        
        results.append({
            'n': n,
            'cannon_error': err_cannon,
            'summa_error': err_summa,
        })
    
    return results


if __name__ == "__main__":
    # Self-test
    n = 32
    A = np.random.randn(n, n)
    B = np.random.randn(n, n)
    C_ref = np.dot(A, B)
    C_c = cannon_multiply(A, B, 4)
    print("Cannon error:", frobenius_error(C_c, C_ref))
    C_s = mpi_summa_multiply(A, B, 4)
    print("SUMMA error:", frobenius_error(C_s, C_ref))
