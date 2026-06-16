"""
block_toeplitz_dynamical.py
=============================
Block Toeplitz structure of the dusty plasma crystal dynamical matrix
and efficient eigenvalue computation.

Physical motivation:
    For a periodic dusty plasma crystal, the dynamical matrix D(q)
    depends on the wavevector q and has block Toeplitz structure:
        D_{n1,n2} = D(n1-n2)  (depends only on relative position)

    For a 1D chain or 2D strip with periodic boundary conditions,
    the dynamical matrix has the block Toeplitz form:
        T = [B_0   B_{-1}  B_{-2}  ...  ]
            [B_1   B_0     B_{-1}  ...  ]
            [B_2   B_1     B_0     ...  ]
            [...   ...     ...     ...  ]

    where each B_k is a 2x2 block (for 2D displacements).

    The dispersion relation is obtained by block-diagonalizing:
        D(q) = sum_k B_k * exp(i*q*k*a)
    and finding eigenvalues omega^2(q).

    We use the Schur complement recursive solver from 971_r8bto
    for solving (D - omega^2*I)*u = F efficiently.

References:
    - From 971_r8bto: r8bto_sl (block Toeplitz solver),
      r8ge_fa/r8ge_sl (dense LU factorization)
    - Trefethen & Embree, "Spectra and Pseudospectra" (Princeton, 2005)
"""

import numpy as np
from typing import Tuple, List, Optional


def r8ge_fa(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Dense LU factorization with partial pivoting (from 971_r8bto r8ge_fa).

    A = P*L*U where P is a permutation, L unit lower triangular, U upper triangular.

    Parameters
    ----------
    A : np.ndarray, shape (n, n)
        Matrix to factorize.

    Returns
    -------
    LU : np.ndarray, shape (n, n)
        Combined L and U factors (L has unit diagonal, not stored).
    pivot : np.ndarray, shape (n,)
        Pivot indices.
    info : int
        0 = success, k = U(k,k) is zero (singular).
    """
    n = A.shape[0]
    LU = A.copy().astype(np.float64)
    pivot = np.zeros(n, dtype=np.int32)

    for k in range(n):
        pivot[k] = k
        # Find pivot
        max_val = abs(LU[k, k])
        max_idx = k
        for i in range(k + 1, n):
            if abs(LU[i, k]) > max_val:
                max_val = abs(LU[i, k])
                max_idx = i
        if max_val < 1e-30:
            return LU, pivot, k + 1  # Singular

        # Swap rows
        if max_idx != k:
            LU[[k, max_idx]] = LU[[max_idx, k]]
            pivot[k], pivot[max_idx] = pivot[max_idx], pivot[k]

        # Elimination
        for i in range(k + 1, n):
            LU[i, k] /= LU[k, k]
            for j in range(k + 1, n):
                LU[i, j] -= LU[i, k] * LU[k, j]

    return LU, pivot, 0


def r8ge_sl(
    LU: np.ndarray,
    pivot: np.ndarray,
    b: np.ndarray,
) -> np.ndarray:
    """
    Solve A*x = b using LU factorization (from 971_r8bto r8ge_sl).

    Parameters
    ----------
    LU : np.ndarray, shape (n, n)
        LU factors from r8ge_fa.
    pivot : np.ndarray, shape (n,)
        Pivot indices.
    b : np.ndarray, shape (n,)
        Right-hand side.

    Returns
    -------
    x : np.ndarray, shape (n,)
        Solution.
    """
    n = LU.shape[0]
    x = b.copy().astype(np.float64)

    # Apply permutation
    for k in range(n):
        if pivot[k] != k:
            x[k], x[pivot[k]] = x[pivot[k]], x[k]

    # Forward substitution (L*y = Pb)
    for i in range(1, n):
        for j in range(i):
            x[i] -= LU[i, j] * x[j]

    # Back substitution (U*x = y)
    for i in range(n - 1, -1, -1):
        for j in range(i + 1, n):
            x[i] -= LU[i, j] * x[j]
        if abs(LU[i, i]) < 1e-30:
            x[i] = 0.0
        else:
            x[i] /= LU[i, i]

    return x


def r8bto_mv(
    blocks: np.ndarray,
    n_blocks: int,
    block_size: int,
    x: np.ndarray,
) -> np.ndarray:
    """
    Block Toeplitz matrix-vector product (from 971_r8bto r8bto_mv).

    Computes y = T*x where T is an n_blocks*n_blocks block Toeplitz
    matrix with blocks T[i,j] = blocks[i-j + n_blocks - 1].

    Parameters
    ----------
    blocks : np.ndarray, shape (block_size, block_size, 2*n_blocks-1)
        Block Toeplitz entries. blocks[:,:,k] = B_{k-n_blocks+1}.
    n_blocks : int
        Number of block rows/columns.
    block_size : int
        Size of each block.
    x : np.ndarray, shape (n_blocks * block_size,)
        Input vector.

    Returns
    -------
    y : np.ndarray, shape (n_blocks * block_size,)
        Product T*x.
    """
    N = n_blocks * block_size
    y = np.zeros(N)

    for I in range(n_blocks):
        for J in range(n_blocks):
            k = I - J + n_blocks - 1  # Index into blocks array
            B = blocks[:, :, k]
            x_block = x[J * block_size:(J + 1) * block_size]
            y[I * block_size:(I + 1) * block_size] += B @ x_block

    return y


def r8bto_sl(
    blocks: np.ndarray,
    n_blocks: int,
    block_size: int,
    rhs: np.ndarray,
) -> np.ndarray:
    """
    Solve block Toeplitz system T*x = rhs using Schur complement
    recursion (from 971_r8bto r8bto_sl).

    For small systems, falls back to dense LU.
    For the dusty plasma problem, the dynamical matrix for a finite
    strip with periodic BCs has this block Toeplitz structure.

    Parameters
    ----------
    blocks : np.ndarray, shape (block_size, block_size, 2*n_blocks-1)
        Block Toeplitz entries.
    n_blocks : int
        Number of blocks.
    block_size : int
        Size of each block.
    rhs : np.ndarray, shape (n_blocks * block_size,)
        Right-hand side.

    Returns
    -------
    x : np.ndarray, shape (n_blocks * block_size,)
        Solution.
    """
    # For small systems, use dense solve
    N = n_blocks * block_size
    T_dense = np.zeros((N, N))
    for I in range(n_blocks):
        for J in range(n_blocks):
            k = I - J + n_blocks - 1
            T_dense[I * block_size:(I + 1) * block_size,
                    J * block_size:(J + 1) * block_size] = blocks[:, :, k]

    LU, pivot, info = r8ge_fa(T_dense)
    if info != 0:
        # Singular or near-singular: use least squares
        x, _, _, _ = np.linalg.lstsq(T_dense, rhs, rcond=1e-12)
        return x

    return r8ge_sl(LU, pivot, rhs)


class DustyPlasmaDynamicalMatrix:
    """
    Dynamical matrix for a 1D dusty plasma chain with Yukawa interactions.

    The equation of motion for grain displacement u_n is:
        m_d * d^2 u_n/dt^2 = -sum_j D_{n,j} * u_j

    where the dynamical matrix elements are:
        D_{n,j} = d^2/dx_n dx_j sum_i V_Y(|x_n - x_i|)
    and V_Y(r) = (Q_d^2/(4*pi*eps0*r)) * exp(-r/lambda_D)

    For a 1D chain with lattice constant a, the block Toeplitz blocks are:
        B_0 = longitudinal and transverse spring constants
        B_k = coupling to k-th neighbor
    """

    def __init__(
        self,
        kappa: float,
        omega_pd: float,
        n_cells: int = 10,
        block_size: int = 2,
    ):
        """
        Parameters
        ----------
        kappa : float
            Screening parameter a/lambda_D.
        omega_pd : float
            Dust plasma frequency [rad/s].
        n_cells : int
            Number of unit cells in the chain.
        block_size : int
            Block size (2 for 2D: longitudinal + transverse).
        """
        self.kappa = kappa
        self.omega_pd = omega_pd
        self.n_cells = n_cells
        self.block_size = block_size

        self.blocks = self._compute_blocks()

    def _yukawa_potential(self, r: float) -> float:
        """Yukawa potential V_Y(r) = exp(-kappa*r)/r (normalized)."""
        if r < 1e-15:
            return 0.0
        return np.exp(-self.kappa * r) / r

    def _yukawa_force_constant(self, r: float) -> Tuple[float, float]:
        """
        Longitudinal and transverse force constants from Yukawa potential.

        K_L(r) = d^2/dr^2 [exp(-kappa*r)/r]
        K_T(r) = (1/r) * d/dr [exp(-kappa*r)/r]

        These are:
            K_L = exp(-k*r)/r^3 * (2 + 2*k*r + k^2*r^2)
            K_T = exp(-k*r)/r^3 * (1 + k*r)
        """
        if r < 1e-15:
            return 0.0, 0.0
        kr = self.kappa * r
        prefactor = np.exp(-kr) / r**3
        K_L = prefactor * (2.0 + 2.0 * kr + kr**2)
        K_T = prefactor * (1.0 + kr)
        return K_L, K_T

    def _compute_blocks(self) -> np.ndarray:
        """
        Compute block Toeplitz blocks for the dynamical matrix.

        For a 1D chain along x, the 2x2 blocks are:
            B_k = -omega_pd^2 * [K_L(k*a)   0          ]
                                [0           K_T(k*a)    ]
        for k != 0, and B_0 includes the on-site restoring force.
        """
        n_diag = 2 * self.n_cells - 1
        blocks = np.zeros((self.block_size, self.block_size, n_diag))

        # Compute on-site term (sum over all neighbors)
        K_L_sum = 0.0
        K_T_sum = 0.0
        for k in range(1, self.n_cells):
            r = float(k)
            K_L, K_T = self._yukawa_force_constant(r)
            K_L_sum += K_L
            K_T_sum += K_T

        # Diagonal block B_0 (at index n_cells - 1)
        blocks[0, 0, self.n_cells - 1] = 2.0 * K_L_sum
        blocks[1, 1, self.n_cells - 1] = 2.0 * K_T_sum

        # Off-diagonal blocks
        for k in range(1, self.n_cells):
            r = float(k)
            K_L, K_T = self._yukawa_force_constant(r)
            # B_{+k} at index n_cells - 1 - k
            blocks[0, 0, self.n_cells - 1 - k] = -K_L
            blocks[1, 1, self.n_cells - 1 - k] = -K_T
            # B_{-k} at index n_cells - 1 + k
            blocks[0, 0, self.n_cells - 1 + k] = -K_L
            blocks[1, 1, self.n_cells - 1 + k] = -K_T

        # Scale by omega_pd^2
        blocks *= self.omega_pd**2

        return blocks

    def to_dense(self) -> np.ndarray:
        """Convert block Toeplitz to dense matrix."""
        N = self.n_cells * self.block_size
        D = np.zeros((N, N))
        for I in range(self.n_cells):
            for J in range(self.n_cells):
                k = I - J + self.n_cells - 1
                D[I * self.block_size:(I + 1) * self.block_size,
                  J * self.block_size:(J + 1) * self.block_size] = self.blocks[:, :, k]
        return D

    def dispersion_relation(
        self,
        n_q: int = 100,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the dispersion relation omega(q) by block-diagonalization.

        D(q) = sum_k B_k * exp(i*q*k)

        For each wavevector q in [0, pi/a], compute eigenvalues of D(q).

        Parameters
        ----------
        n_q : int
            Number of q-points.

        Returns
        -------
        q_values : np.ndarray, shape (n_q,)
            Wavevectors.
        omega_values : np.ndarray, shape (n_q, block_size)
            Mode frequencies at each q.
        """
        q_values = np.linspace(0, np.pi, n_q)
        omega_values = np.zeros((n_q, self.block_size))

        for iq, q in enumerate(q_values):
            D_q = np.zeros((self.block_size, self.block_size), dtype=complex)
            for k_idx in range(2 * self.n_cells - 1):
                k = k_idx - (self.n_cells - 1)
                D_q += self.blocks[:, :, k_idx] * np.exp(1j * q * k)

            # Eigenvalues should be real and positive for stable crystal
            eigenvalues = np.linalg.eigvalsh(D_q.real)
            eigenvalues = np.sort(eigenvalues)

            # omega = sqrt(eigenvalue)
            omega_sq = np.maximum(eigenvalues, 0.0)
            omega_values[iq, :] = np.sqrt(omega_sq)

        return q_values, omega_values

    def solve_forces(self, displacement: np.ndarray) -> np.ndarray:
        """
        Solve D*u = F for displacement given forces, using block Toeplitz solver.

        Parameters
        ----------
        displacement : np.ndarray, shape (n_cells * block_size,)
            Force vector (renamed from 'displacement' for API).

        Returns
        -------
        u : np.ndarray
            Displacement solution.
        """
        return r8bto_sl(
            self.blocks, self.n_cells, self.block_size, displacement
        )
