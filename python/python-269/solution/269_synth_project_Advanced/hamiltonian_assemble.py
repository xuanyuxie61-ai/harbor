"""
hamiltonian_assemble.py — 磁场薛定谔方程的哈密顿量组装
============================================================

本模块构建二维磁场薛定谔方程的离散哈密顿量矩阵:

    H = (1/2m)(p - A)² + V(r)

在 Landau 规范 A = (0, Bx, 0) 下展开:

    H = -(1/2m)[∂²_x + ∂²_y] - (iBx/m)∂_y + (B²x²/2m) + V(r)

各项的物理含义:
    -(1/2m)∇²       — 动能项
    -(iBx/m)∂_y     — 轨道耦合项 (来自 p·A + A·p)
    B²x²/(2m)       — 反磁性项 (diamagnetic term)
    V(r)             — 外加势 (约束势、杂质势等)

离散化方法:
    - 动能项: 高阶有限差分 (2/4/6阶精度)
    - 耦合项: 高阶有限差分 + 位置依赖系数
    - 势能项: 对角矩阵
    - 总矩阵: 稀疏 CSR 格式 (scipy.sparse)

并行组装策略 (源自 matrix_assemble_parfor):
    哈密顿量按列分块组装, 每列对应一个格点的方程.
    在并行环境中, 各列可独立计算后合并.

矩阵维度: N_total = (Nx+1) × (Ny+1)
内存估计: 对于 4阶模板, 每行最多 13 个非零元.
    对于 N=30×30=900: ~11700 个非零元, 矩阵大小 900×900

参考文献:
    [1] Kogut, J. & Susskind, L. Phys. Rev. D 11, 395 (1975)
    [2] Hofstadter, D. R. Phys. Rev. B 14, 2239 (1976)
"""

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as splinalg
from typing import Optional, Tuple, List
from high_order_fd import (build_1d_laplacian, build_1d_first_derivative)


def assemble_hamiltonian_landau(Nx: int, Ny: int, Lx: float, Ly: float,
                                 B: float, fd_order: int = 4,
                                 V_ext: Optional[np.ndarray] = None,
                                 mass: float = 1.0) -> sparse.csr_matrix:
    """组装 Landau 规范下的磁场薛定谔哈密顿量

    H = -(1/2m)∇² - (iBx/m)∂_y + (B²x²/2m) + V(r)

    使用张量积结构:
        ∇² = L_x ⊗ I_y + I_x ⊗ L_y
        ∂_y = I_x ⊗ D_y
        x² = X² ⊗ I_y

    其中 L_x, L_y 是一维拉普拉斯矩阵, D_y 是一阶导数矩阵,
    X² 是 x² 的对角矩阵.

    Args:
        Nx, Ny: x, y 方向内部格点数
        Lx, Ly: 系统尺寸
        B: 磁场强度 (原子单位)
        fd_order: 有限差分精度 (2, 4, 6)
        V_ext: 外势 (一维数组, 长度 (Nx+1)*(Ny+1)), 可选
        mass: 有效质量 (原子单位: 1.0)
    Returns:
        H: 稀疏哈密顿量矩阵 ((Nx+1)*(Ny+1) × (Nx+1)*(Ny+1))
    """
    Nx1 = Nx + 1
    Ny1 = Ny + 1
    N_total = Nx1 * Ny1
    hx = Lx / Nx
    hy = Ly / Ny
    inv2m = 1.0 / (2.0 * mass)

    # 一维算子 (Dirichlet 边界: 内部 N 个点)
    Lx_op = build_1d_laplacian(Nx, hx, fd_order)   # -d²/dx²
    Ly_op = build_1d_laplacian(Ny, hy, fd_order)   # -d²/dy²
    Dy_op = build_1d_first_derivative(Ny, hy, fd_order)  # d/dy

    Ix = sparse.eye(Nx, format='csr')
    Iy = sparse.eye(Ny, format='csr')

    # 动能项: -(1/2m)∇² = (1/2m)(-d²/dx² - d²/dy²)
    # 注意: Lx_op 已经是 -d²/dx², 所以 H_kin = inv2m * (Lx ⊗ Iy + Ix ⊗ Ly)
    H_kin = inv2m * (sparse.kron(Lx_op, Iy) + sparse.kron(Ix, Ly_op))

    # 交叉项: -(iBx/m)∂_y
    # x 坐标数组 (内部点)
    x_vals = np.array([(i + 1) * hx for i in range(Nx)])  # 不包括边界 0, Lx
    # Bx/m 的对角矩阵
    Bx_over_m = sparse.diags(B * x_vals / mass)
    # -i·(Bx/m) ⊗ ∂_y
    H_cross = -1j * sparse.kron(Bx_over_m, Dy_op)

    # 反磁性项: B²x²/(2m)
    x2_vals = (B ** 2) * (x_vals ** 2) / (2.0 * mass)
    X2_diag = sparse.diags(x2_vals)
    H_dia = sparse.kron(X2_diag, Iy)

    # 总哈密顿量
    H = H_kin + H_cross + H_dia

    # 外势
    if V_ext is not None:
        V_diag = sparse.diags(V_ext)
        H = H + V_diag

    # 确保厄米性 (消除数值误差)
    H = (H + H.conj().T) / 2.0

    return H.tocsr()


def assemble_hamiltonian_periodic(Nx: int, Ny: int, Lx: float, Ly: float,
                                   B: float, fd_order: int = 2,
                                   mass: float = 1.0) -> sparse.csr_matrix:
    """组装周期性边界条件下的哈密顿量

    周期性边界条件: ψ(x+L, y) = ψ(x, y)
    使用循环矩阵结构 (Toeplitz + 角元)

    适用于 Hofstadter 类型的计算, 其中磁通量子化要求
    Φ/Φ₀ = p/q (有理数).

    Args:
        Nx, Ny: x, y 方向格点数
        Lx, Ly: 系统尺寸
        B: 磁场强度
        fd_order: 精度阶数 (仅支持 2)
        mass: 有效质量
    Returns:
        H: 稀疏哈密顿量矩阵
    """
    N_total = Nx * Ny
    hx = Lx / Nx
    hy = Ly / Ny
    inv2m = 1.0 / (2.0 * mass)

    # 构建周期性格点坐标
    x_coords = np.array([i * hx for i in range(Nx)])
    y_coords = np.array([j * hy for j in range(Ny)])

    # 收集矩阵元 (行, 列, 值)
    rows = []
    cols = []
    vals = []

    for ix in range(Nx):
        for iy in range(Ny):
            idx = ix * Ny + iy
            x_val = x_coords[ix]

            # -d²/dx² 的 3点模板 (周期)
            ix_left = (ix - 1) % Nx
            ix_right = (ix + 1) % Nx
            idx_left = ix_left * Ny + iy
            idx_right = ix_right * Ny + iy

            # -d²/dx²
            rows.extend([idx, idx, idx])
            cols.extend([idx_left, idx, idx_right])
            vals.extend([-inv2m / hx**2, 2*inv2m / hx**2, -inv2m / hx**2])

            # -d²/dy²
            iy_left = (iy - 1) % Ny
            iy_right = (iy + 1) % Ny
            idx_left_y = ix * Ny + iy_left
            idx_right_y = ix * Ny + iy_right

            rows.extend([idx, idx, idx])
            cols.extend([idx_left_y, idx, idx_right_y])
            vals.extend([-inv2m / hy**2, 2*inv2m / hy**2, -inv2m / hy**2])

            # 交叉项 -iBx∂_y (反对称部分)
            Ay = B * x_val
            # 周期性边界: Peierls 相位包含磁通
            phase_right = np.exp(1j * Ay * hy)
            phase_left = np.exp(-1j * Ay * hy)

            cross_coeff = -1j * Ay / (2.0 * mass * hy)
            rows.extend([idx, idx])
            cols.extend([idx_right_y, idx_left_y])
            vals.extend([-cross_coeff * phase_right,
                         cross_coeff * phase_left])

            # 反磁性项 B²x²/(2m)
            dia_val = B**2 * x_val**2 / (2.0 * mass)
            rows.append(idx)
            cols.append(idx)
            vals.append(dia_val)

    H = sparse.coo_matrix((vals, (rows, cols)),
                          shape=(N_total, N_total)).tocsr()
    # 确保厄米性
    H = (H + H.conj().T) / 2.0
    return H.tocsr()


def ground_state_energy(H: sparse.csr_matrix,
                        num_states: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """求解哈密顿量的最低若干本征态

    使用 ARPACK 隐式重启 Lanczos 方法 (scipy.sparse.linalg.eigsh)
    仅计算最少数量的本征值/本征态.

    收敛准则:
        ||H·v - λ·v|| / |λ| < tol (默认 1e-10)

    Args:
        H: 稀疏厄米矩阵
        num_states: 要求的本征态数目
    Returns:
        eigenvalues: 本征值数组 (升序)
        eigenvectors: 本征态矩阵 (列向量)
    """
    N = H.shape[0]
    if num_states >= N - 1:
        # 小矩阵: 使用稠密对角化
        H_dense = H.toarray()
        evals, evecs = np.linalg.eigh(H_dense)
        return evals[:num_states], evecs[:, :num_states]

    evals, evecs = splinalg.eigsh(H, k=num_states, which='SA',
                                   tol=1e-10, maxiter=10000)
    # 排序
    idx = np.argsort(evals)
    return evals[idx], evecs[:, idx]


def spectral_density(H: sparse.csr_matrix, E_range: Tuple[float, float],
                     n_bins: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """计算谱密度 ρ(E) = (1/N) Σ δ(E - E_n)

    使用高斯展宽:
        δ(E - E_n) → (1/σ√(2π)) exp(-(E-E_n)²/(2σ²))
    其中 σ 是展宽参数 (自适应选择).

    Args:
        H: 哈密顿量
        E_range: 能量范围 (E_min, E_max)
        n_bins: 直方图 bin 数
    Returns:
        E_grid: 能量网格
        rho: 谱密度
    """
    evals, _ = ground_state_energy(H, num_states=min(H.shape[0], 50))
    E_min, E_max = E_range
    E_grid = np.linspace(E_min, E_max, n_bins)

    sigma = (E_max - E_min) / n_bins * 2.0
    rho = np.zeros(n_bins)
    for E_n in evals:
        rho += np.exp(-0.5 * ((E_grid - E_n) / sigma) ** 2) / (
            sigma * np.sqrt(2 * np.pi))
    rho /= len(evals)
    return E_grid, rho


def build_impurity_potential(Nx: int, Ny: int, hx: float, hy: float,
                              impurity_positions: List[Tuple[int, int]],
                              V0: float = 1.0,
                              sigma: float = 0.5) -> np.ndarray:
    """构建杂质势 V(r) = V₀ Σ exp(-|r-r_i|²/(2σ²))

    用于模拟量子霍尔系统中的杂质散射.
    杂质的存在导致朗道能级展宽, 并在能级之间产生局域态.

    Args:
        Nx, Ny: 格点数
        hx, hy: 网格间距
        impurity_positions: 杂质位置列表 [(ix, iy), ...]
        V0: 杂质势强度
        sigma: 杂质势宽度
    Returns:
        V: 势能数组 (一维展开)
    """
    Nx1 = Nx + 1
    Ny1 = Ny + 1
    V = np.zeros(Nx1 * Ny1)
    for ix in range(Nx1):
        for iy in range(Ny1):
            idx = ix * Ny1 + iy
            x = ix * hx
            y = iy * hy
            for (ix_imp, iy_imp) in impurity_positions:
                x_imp = ix_imp * hx
                y_imp = iy_imp * hy
                r2 = (x - x_imp)**2 + (y - y_imp)**2
                V[idx] += V0 * np.exp(-r2 / (2 * sigma**2))
    return V
