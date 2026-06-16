"""
domain_decomposition.py — CUDA-style domain decomposition for the 1+1 wave grid.

Maps the conceptual structure of a CUDA kernel launch
    kernel <<< gridDim, blockDim >>> (args)
onto a 1D spatial grid used by the Regge-Wheeler solver.

In a full GPU implementation each thread would update one grid point;
here we *simulate* the block-thread assignment to:
  * partition the grid into overlapping subdomains (halo width = FD stencil radius),
  * compute the process linear index  K = threadIdx.x + blockDim.x blockIdx.x,
  * assign each subdomain its local task range,
  * support multi-dimensional block/thread layouts (bx, by, bz), (tx, ty, tz),
  * measure load balance and idle-thread fraction.

This is the natural framework for a future port to CuPy or Numba-CUDA while
keeping the small-scale reproducible experiment CPU-only.
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import List, Tuple, Dict


@dataclass(frozen=True)
class CudaLayout:
    """CUDA block / thread geometry and total task count."""
    blocks: Tuple[int, int, int]    # (gridDim.x, gridDim.y, gridDim.z)
    threads: Tuple[int, int, int]   # (blockDim.x, blockDim.y, blockDim.z)
    n_tasks: int                    # total grid points to cover

    @property
    def n_blocks(self) -> int:
        return self.blocks[0] * self.blocks[1] * self.blocks[2]

    @property
    def n_threads_per_block(self) -> int:
        return self.threads[0] * self.threads[1] * self.threads[2]

    @property
    def total_threads(self) -> int:
        return self.n_blocks * self.n_threads_per_block

    @property
    def chunk(self) -> int:
        """Stride between successive task indices for a given process."""
        return self.total_threads

    def validate(self) -> bool:
        return (all(b > 0 for b in self.blocks)
                and all(t > 0 for t in self.threads)
                and self.n_tasks > 0)


def process_linear_index(blockIdx: Tuple[int, int, int],
                         threadIdx: Tuple[int, int, int],
                         blockDim: Tuple[int, int, int],
                         gridDim: Tuple[int, int, int]) -> int:
    """CUDA process linear index
        K = threadIdx.x
          + blockDim.x * threadIdx.y
          + blockDim.x * blockDim.y * threadIdx.z
          + blockDim.x * blockDim.y * blockDim.z * blockIdx.x
          + blockDim.x * blockDim.y * blockDim.z * gridDim.x * blockIdx.y
          + blockDim.x * blockDim.y * blockDim.z * gridDim.x * gridDim.y * blockIdx.z.
    """
    bdx, bdy, bdz = blockDim
    gdx, gdy, gdz = gridDim
    tx, ty, tz = threadIdx
    bix, biy, biz = blockIdx
    return (tx
            + bdx * ty
            + bdx * bdy * tz
            + bdx * bdy * bdz * bix
            + bdx * bdy * bdz * gdx * biy
            + bdx * bdy * bdz * gdx * gdy * biz)


def task_sequence(K: int, layout: CudaLayout) -> List[int]:
    """Return the list of task indices executed by process K.

    Task T = K;  while T < N: T += chunk.
    """
    tasks = []
    T = K
    while T < layout.n_tasks:
        tasks.append(T)
        T += layout.chunk
    return tasks


# ---------------------------------------------------------------------------
#  1D domain partitioning with halo
# ---------------------------------------------------------------------------
def partition_grid_1d(N: int, n_subdomains: int,
                      halo: int = 2) -> List[Tuple[int, int]]:
    """Partition a 1D grid of N points into n_subdomains overlapping strips.

    Each subdomain has
        interior : [i_start, i_end)
        halo     : halo points on each side (clipped to [0, N-1]).

    Returns a list of (i_start, i_end) pairs for the interior only.
    """
    base = N // n_subdomains
    rem = N % n_subdomains
    parts = []
    start = 0
    for k in range(n_subdomains):
        size = base + (1 if k < rem else 0)
        end = start + size
        parts.append((start, end))
        start = end
    return parts


def partition_with_halo(N: int, n_subdomains: int,
                        halo: int = 2) -> List[Dict[str, int]]:
    """Return per-subdomain descriptors: interior and halo bounds."""
    interior = partition_grid_1d(N, n_subdomains, halo)
    descriptors = []
    for k, (i0, i1) in enumerate(interior):
        halo_lo = max(0, i0 - halo)
        halo_hi = min(N, i1 + halo)
        descriptors.append({
            "id": k,
            "i_start": i0,
            "i_end": i1,
            "halo_lo": halo_lo,
            "halo_hi": halo_hi,
            "interior_size": i1 - i0,
            "total_size": halo_hi - halo_lo,
        })
    return descriptors


# ---------------------------------------------------------------------------
#  Load-balance diagnostics
# ---------------------------------------------------------------------------
def load_balance_stats(layout: CudaLayout) -> Dict[str, float]:
    """Compute min/max/mean tasks per process and the idle fraction."""
    counts = [len(task_sequence(K, layout)) for K in range(layout.total_threads)]
    active = sum(1 for c in counts if c > 0)
    idle = layout.total_threads - active
    return {
        "total_threads": layout.total_threads,
        "active_threads": active,
        "idle_threads": idle,
        "idle_fraction": idle / max(layout.total_threads, 1),
        "min_tasks": min(counts) if counts else 0,
        "max_tasks": max(counts) if counts else 0,
        "mean_tasks": sum(counts) / max(len(counts), 1),
        "n_tasks": layout.n_tasks,
    }


# ---------------------------------------------------------------------------
#  Multi-dimensional grid mapping (for future 2+1 or 3+1 extensions)
# ---------------------------------------------------------------------------
def map_task_to_3d(task: int, Nx: int, Ny: int, Nz: int) -> Tuple[int, int, int]:
    """Convert a linear task index to (ix, iy, iz) on a 3D grid."""
    iz = task // (Nx * Ny)
    rem = task % (Nx * Ny)
    iy = rem // Nx
    ix = rem % Nx
    return ix, iy, iz


# ---------------------------------------------------------------------------
#  Parallel matrix assembly helper (Hilbert-like FD operator blocks)
# ---------------------------------------------------------------------------
def assemble_fd_block(rows: List[int], cols: List[int],
                      stencil_coeffs: List[float],
                      N: int) -> List[Tuple[int, int, float]]:
    """Build the non-zero entries of one block of the FD operator matrix.

    Returns a list of (i, j, v) triples in COO format.
    """
    entries = []
    s = len(stencil_coeffs) // 2
    for i in rows:
        for k, c in enumerate(stencil_coeffs):
            j = i + (k - s)
            if 0 <= j < N:
                entries.append((i, j, c))
    return entries


def assemble_fd_operator(N: int, stencil: List[float],
                         n_subdomains: int = 1) -> List[Tuple[int, int, float]]:
    """Assemble the full sparse FD operator matrix in parallel blocks.

    The work is partitioned across n_subdomains using the CUDA-style layout.
    Each subdomain assembles its rows independently (mirroring parfor).
    """
    parts = partition_grid_1d(N, n_subdomains)
    all_entries = []
    for i0, i1 in parts:
        rows = list(range(i0, i1))
        block = assemble_fd_block(rows, list(range(N)), stencil, N)
        all_entries.extend(block)
    return all_entries
