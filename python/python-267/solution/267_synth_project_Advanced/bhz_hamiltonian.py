"""
bhz_hamiltonian.py — BHZ 模型哈密顿量构造与稀疏矩阵 I/O
==========================================================

本模块实现 Bernevig-Hughes-Zhang (BHZ) 模型的实空间离散化,
构建拓扑绝缘体的紧束缚/有限差分哈密顿量矩阵。

BHZ 模型描述 HgTe/CdTe 量子阱中的量子自旋霍尔效应,
是二维拓扑绝缘体的标志性理论模型。

物理模型
--------
**BHZ 哈密顿量 (块对角形式):**

    H_BHZ(k) = [h(k)    0    ]
               [0    h*(-k)  ]

    h(k) = ε(k)·I₂ + d⃗(k)·σ⃗

其中:
    ε(k) = C - D(kx² + ky²)        — 带中心偏移
    d₁(k) = A·kx                     — Rashba 自旋轨道耦合
    d₂(k) = A·ky
    d₃(k) = M₀ - B(kx² + ky²)      — 质量项

    σ⃗ = (σx, σy, σz) 为 Pauli 矩阵

**典型参数 (HgTe/CdTe, 量子阱厚度 6.3 nm, 倒置区):**
    A = 3.42 eV·Å = 0.342 eV·nm
    B = -16.9 eV·Å² = -0.169 eV·nm²
    D = -5.74 eV·Å² = -0.0574 eV·nm²
    M₀ = -0.010 eV (倒置区)
    C = 0 eV (参考能量)

**实空间有限差分离散化:**
    kx → -i ∂/∂x ≈ -i D_x^{(2p)}
    ky → -i ∂/∂y ≈ -i D_y^{(2p)}
    kx² + ky² → -(D_xx^{(2p)} + D_yy^{(2p)})

其中 D_x^{(2p)}, D_xx^{(2p)} 为 2p 阶精度一阶/二阶导数 FD 矩阵。

来源映射
--------
- 131_c8lib: 复数矩阵高斯消元、Frobenius/L1/Inf 范数
- 781_msm_to_hb: 稀疏矩阵 Harwell-Boeing 格式输出
"""

import numpy as np
from scipy import sparse
from typing import Tuple, Dict, Optional
from fd_stencils import (
    first_derivative_coefficients,
    second_derivative_coefficients,
)


class BHZParameters:
    """
    BHZ 模型物理参数容器。

    默认值为 HgTe/CdTe 量子阱 (6.3 nm, 倒置区) 的实验拟合参数。

    物理量纲: 能量 (eV), 长度 (nm)
    """

    def __init__(self, A: float = 0.342, B: float = -0.169,
                 D: float = -0.0574, M0: float = 0.010,
                 C: float = 0.0):
        """
        Parameters
        ----------
        A : float
            自旋轨道耦合强度 (eV·nm)
        B : float
            质量项动能系数 (eV·nm²)
        D : float
            带中心偏移动能系数 (eV·nm²)
        M0 : float
            Dirac 质量 (eV), M0>0 且 B<0 对应拓扑非平庸相 (M0/B < 0)
        C : float
            带中心能量偏移 (eV)
        """
        self.A = A
        self.B = B
        self.D = D
        self.M0 = M0
        self.C = C

    @property
    def is_topological(self) -> bool:
        """
        判断系统是否处于拓扑非平庸相。

        拓扑判据: M0/B < 0 (即 M0 与 B 异号)
        对于 HgTe QW, B < 0, 所以 M0 < 0 时为拓扑相。

        更精确地, 拓扑相变发生在 M0 = 0 处 (忽略格点修正时)。
        """
        return (self.M0 / self.B) < 0

    @property
    def bulk_gap(self) -> float:
        """
        体带隙大小 (eV)。

        在 k=0 处, 带隙 = 2|M0|。
        考虑有限 k 修正后的最小带隙需要数值求解。
        """
        return 2.0 * abs(self.M0)

    @property
    def critical_thickness_indicator(self) -> float:
        """
        拓扑相变的临界指标 M0/B。

        当 M0/B 从正变负时, 系统经历拓扑相变。
        |M0/B| 越大, 体带隙越大, 边界态越稳定。
        """
        return self.M0 / self.B

    def to_dict(self) -> Dict:
        return {'A': self.A, 'B': self.B, 'D': self.D,
                'M0': self.M0, 'C': self.C}

    def __repr__(self) -> str:
        return (f"BHZParameters(A={self.A}, B={self.B}, D={self.D}, "
                f"M0={self.M0}, C={self.C})")


def bhz_h_k(kx: float, ky: float, params: BHZParameters) -> np.ndarray:
    """
    计算 BHZ 模型在动量空间中的 4×4 哈密顿量 H(k)。

    H(k) = [h(k)    0   ]
           [0    h*(-k) ]

    h(k) = [ε+M   A(kx-iky)]
           [A(kx+iky)  ε-M ]

    Parameters
    ----------
    kx, ky : float
        波矢分量 (nm⁻¹)
    params : BHZParameters
        物理参数

    Returns
    -------
    H : ndarray, shape (4, 4), complex
        BHZ 哈密顿量矩阵
    """
    k2 = kx * kx + ky * ky
    eps = params.C - params.D * k2
    M_k = params.M0 - params.B * k2

    # 上块 h(k)
    h11 = eps + M_k
    h22 = eps - M_k
    h12 = params.A * (kx - 1j * ky)
    h21 = params.A * (kx + 1j * ky)

    # 下块 h*(-k): kx→-kx, ky→-ky, 然后取复共轭
    # h*(-k) = [eps+M   A(-kx+iky)*]  = [eps+M   -A(kx+iky)]
    #          [A(-kx-iky)*  eps-M]     [-A(kx-iky)  eps-M]
    # 注意: 由于 ε 和 M 只依赖 k², 它们在 k→-k 下不变
    # h*(-k) 的非对角元: [A(-kx-i(-ky))]* = [-A(kx+iky)]* = -A(kx-iky)
    # 等等, 让我重新推导:
    # h(-k) = [eps+M   A(-kx-iky)] = [eps+M   -A(kx+iky)]
    #         [A(-kx+iky)  eps-M]   [-A(kx-iky)  eps-M]
    # h*(-k) = [eps+M   -A(kx-iky)]  (复共轭)
    #          [-A(kx+iky)  eps-M]

    h33 = eps + M_k
    h44 = eps - M_k
    h34 = -params.A * (kx - 1j * ky)
    h43 = -params.A * (kx + 1j * ky)

    H = np.array([
        [h11, h12, 0.0, 0.0],
        [h21, h22, 0.0, 0.0],
        [0.0, 0.0, h33, h34],
        [0.0, 0.0, h43, h44]
    ], dtype=np.complex128)

    return H


def bhz_h_k_analytic(kx: float, ky: float,
                     params: BHZParameters) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算 BHZ 模型的本征值 (解析)。

    由于块对角结构, 本征值为:
        E_{±}^{(1)}(k) = ε(k) ± sqrt(M(k)² + A²(kx²+ky²))
        E_{±}^{(2)}(k) = ε(k) ± sqrt(M(k)² + A²(kx²+ky²))

    两组简并 (时间反演对称性保证 Kramers 简并)。

    Returns
    -------
    eigenvalues : ndarray, shape (4,)
        升序排列的本征值
    """
    k2 = kx * kx + ky * ky
    eps = params.C - params.D * k2
    M_k = params.M0 - params.B * k2
    gap_half = np.sqrt(M_k ** 2 + params.A ** 2 * k2)

    eigenvalues = np.array([
        eps - gap_half,
        eps - gap_half,
        eps + gap_half,
        eps + gap_half
    ])
    return np.sort(eigenvalues)


def _build_fd_operator_2d(Nx: int, Ny: int, hx: float, hy: float,
                          p: int, bc_x: str = 'periodic',
                          bc_y: str = 'open') -> Dict[str, sparse.csr_matrix]:
    """
    构建二维有限差分算子的稀疏矩阵表示。

    在 Nx × Ny 网格上构造:
        Dx: 一阶 ∂/∂x 算子
        Dy: 一阶 ∂/∂y 算子
        Dxx: 二阶 ∂²/∂x² 算子
        Dyy: 二阶 ∂²/∂y² 算子

    使用 Kronecker 积:
        Dx_total = Dx_1d ⊗ I_Ny
        Dy_total = I_Nx ⊗ Dy_1d
        Dxx_total = Dxx_1d ⊗ I_Ny
        Dyy_total = I_Nx ⊗ Dyy_1d

    索引约定: (i, j) → i * Ny + j (行优先)

    Parameters
    ----------
    Nx, Ny : int
        x, y 方向网格点数
    hx, hy : float
        网格间距 (nm)
    p : int
        半带宽
    bc_x, bc_y : str
        x, y 方向边界条件 ('open' 或 'periodic')

    Returns
    -------
    ops : dict
        包含 'Dx', 'Dy', 'Dxx', 'Dyy' 的 CSR 稀疏矩阵
    """
    c1 = first_derivative_coefficients(p)
    d2, d2_diag = second_derivative_coefficients(p)

    # --- 1D 算子构造 ---
    # Dx_1d (Nx × Nx)
    rows_x, cols_x, vals_x = [], [], []
    for i in range(Nx):
        for j_idx in range(p):
            j = j_idx + 1
            ip = (i + j) % Nx if bc_x == 'periodic' else i + j
            im = (i - j) % Nx if bc_x == 'periodic' else i - j
            if 0 <= ip < Nx:
                rows_x.append(i); cols_x.append(ip)
                vals_x.append(c1[j_idx] / hx)
            if 0 <= im < Nx:
                rows_x.append(i); cols_x.append(im)
                vals_x.append(-c1[j_idx] / hx)

    Dx_1d = sparse.csr_matrix(
        (vals_x, (rows_x, cols_x)), shape=(Nx, Nx), dtype=np.complex128
    )

    # Dy_1d (Ny × Ny)
    rows_y, cols_y, vals_y = [], [], []
    for i in range(Ny):
        for j_idx in range(p):
            j = j_idx + 1
            ip = (i + j) % Ny if bc_y == 'periodic' else i + j
            im = (i - j) % Ny if bc_y == 'periodic' else i - j
            if 0 <= ip < Ny:
                rows_y.append(i); cols_y.append(ip)
                vals_y.append(c1[j_idx] / hy)
            if 0 <= im < Ny:
                rows_y.append(i); cols_y.append(im)
                vals_y.append(-c1[j_idx] / hy)

    Dy_1d = sparse.csr_matrix(
        (vals_y, (rows_y, cols_y)), shape=(Ny, Ny), dtype=np.complex128
    )

    # Dxx_1d (Nx × Nx)
    rows_xx, cols_xx, vals_xx = [], [], []
    for i in range(Nx):
        # 对角元
        rows_xx.append(i); cols_xx.append(i)
        vals_xx.append(d2_diag / (hx * hx))
        for j_idx in range(p):
            j = j_idx + 1
            ip = (i + j) % Nx if bc_x == 'periodic' else i + j
            im = (i - j) % Nx if bc_x == 'periodic' else i - j
            if 0 <= ip < Nx:
                rows_xx.append(i); cols_xx.append(ip)
                vals_xx.append(d2[j_idx] / (hx * hx))
            if 0 <= im < Nx:
                rows_xx.append(i); cols_xx.append(im)
                vals_xx.append(d2[j_idx] / (hx * hx))

    Dxx_1d = sparse.csr_matrix(
        (vals_xx, (rows_xx, cols_xx)), shape=(Nx, Nx), dtype=np.complex128
    )

    # Dyy_1d (Ny × Ny)
    rows_yy, cols_yy, vals_yy = [], [], []
    for i in range(Ny):
        rows_yy.append(i); cols_yy.append(i)
        vals_yy.append(d2_diag / (hy * hy))
        for j_idx in range(p):
            j = j_idx + 1
            ip = (i + j) % Ny if bc_y == 'periodic' else i + j
            im = (i - j) % Ny if bc_y == 'periodic' else i - j
            if 0 <= ip < Ny:
                rows_yy.append(i); cols_yy.append(ip)
                vals_yy.append(d2[j_idx] / (hy * hy))
            if 0 <= im < Ny:
                rows_yy.append(i); cols_yy.append(im)
                vals_yy.append(d2[j_idx] / (hy * hy))

    Dyy_1d = sparse.csr_matrix(
        (vals_yy, (rows_yy, cols_yy)), shape=(Ny, Ny), dtype=np.complex128
    )

    # --- 2D Kronecker 积 ---
    I_Nx = sparse.eye(Nx, format='csr', dtype=np.complex128)
    I_Ny = sparse.eye(Ny, format='csr', dtype=np.complex128)

    ops = {
        'Dx': sparse.kron(Dx_1d, I_Ny, format='csr'),
        'Dy': sparse.kron(I_Nx, Dy_1d, format='csr'),
        'Dxx': sparse.kron(Dxx_1d, I_Ny, format='csr'),
        'Dyy': sparse.kron(I_Nx, Dyy_1d, format='csr'),
    }

    return ops


def build_bhz_hamiltonian(Nx: int, Ny: int, hx: float, hy: float,
                          params: BHZParameters, p: int = 2,
                          bc_x: str = 'periodic',
                          bc_y: str = 'open',
                          disorder_potential: Optional[np.ndarray] = None
                          ) -> sparse.csr_matrix:
    """
    构建 BHZ 模型的 4-band 有限差分哈密顿量 (稀疏矩阵)。

    H_total = ε(k)·I₄ + M(k)·σz⊗τ0 + A·kx·σx⊗τ0 + A·ky·σy⊗τ0
              + [上块: σ, 下块: σ* (时间反演 partner)]

    在实空间有限差分下:
        kx → -i Dx,  ky → -i Dy
        k² → -(Dxx + Dyy)

    总希尔伯特空间维度: 4 × Nx × Ny

    Parameters
    ----------
    Nx, Ny : int
        网格点数
    hx, hy : float
        网格间距 (nm)
    params : BHZParameters
        BHZ 物理参数
    p : int
        FD 半带宽 (1=2阶, 2=4阶, 3=6阶, 4=8阶)
    bc_x, bc_y : str
        边界条件
    disorder_potential : ndarray, shape (Nx*Ny,), optional
        无序势 (每个格点的能量偏移)

    Returns
    -------
    H : csr_matrix, shape (4NxNy, 4NxNy)
        BHZ 哈密顿量
    """
    N_total = Nx * Ny
    dim = 4 * N_total

    # 构建 FD 算子
    ops = _build_fd_operator_2d(Nx, Ny, hx, hy, p, bc_x, bc_y)
    Dx = ops['Dx']
    Dy = ops['Dy']
    Dxx = ops['Dxx']
    Dyy = ops['Dyy']

    # 恒等矩阵
    I_N = sparse.eye(N_total, format='csr', dtype=np.complex128)

    # k² = -(Dxx + Dyy)
    k2_op = -(Dxx + Dyy)

    # ε(k) = C - D·k² = C·I + D·(Dxx+Dyy)
    eps_op = params.C * I_N - params.D * (-k2_op)
    # 注意: k2_op = -(Dxx+Dyy), 所以 -D*k2_op = D*(Dxx+Dyy)
    # ε = C - D*(-(Dxx+Dyy)) = C + D*(Dxx+Dyy)

    # 修正: ε(k) = C - D(kx²+ky²), 在 FD 下 k²→-(Dxx+Dyy)
    # 所以 ε = C·I - D·(-(Dxx+Dyy)) = C·I + D·(Dxx+Dyy)
    # 但注意 Dxx+Dyy 本身是负定的 (二阶导数)
    # 所以 ε = C·I + D·(Dxx+Dyy) 在 D<0 时使高动能态能量降低 ✓

    # M(k) = M0 - B·k² = M0·I + B·(Dxx+Dyy)
    M_op = params.M0 * I_N - params.B * (-k2_op)

    # A·kx → A·(-i·Dx) = -iA·Dx
    # A·ky → A·(-i·Dy) = -iA·Dy
    Akx = -1j * params.A * Dx
    Aky = -1j * params.A * Dy

    # 无序势
    if disorder_potential is not None:
        V_diag = sparse.diags(disorder_potential, 0, format='csr',
                              dtype=np.complex128)
        eps_op = eps_op + V_diag

    # Pauli 矩阵 (轨道空间 σ)
    # σ0 = I, σx, σy, σz
    # 自旋空间 τ (上块/下块)
    # 总结构: σ ⊗ τ ⊗ 格点

    # 块结构: 4×4 在轨道×自旋空间, 每个元素是 N_total×N_total 算子
    # 基底: (|e↑⟩, |h↑⟩, |e↓⟩, |h↓⟩) ⊗ |格点⟩
    # e=电子带, h=空穴带, ↑↓=自旋

    # 上块 h(k): 自旋 ↑ 子空间
    # h_11 = ε + M
    # h_12 = A(kx - iky) = -iA(Dx + iDy)·(-i) = A(-iDx + Dy)
    # 等等, 让我重新整理:
    # A·kx - iA·ky → -iA·Dx - i(-iA·Dy) = -iA·Dx - A·Dy
    # 不对, kx→-iDx, ky→-iDy
    # A(kx - iky) → A(-iDx - i(-iDy)) = A(-iDx - Dy) = -A(Dy + iDx)
    # A(kx + iky) → A(-iDx + i(-iDy)) = A(-iDx + Dy) = A(Dy - iDx)

    h11_op = eps_op + M_op  # (ε + M) 作用在 |e↑⟩
    h22_op = eps_op - M_op  # (ε - M) 作用在 |h↑⟩
    h12_op = -params.A * (Dy + 1j * Dx)  # A(kx-iky) → -A(Dy+iDx)
    h21_op = params.A * (Dy - 1j * Dx)   # A(kx+iky) → A(Dy-iDx)

    # 下块 h*(-k): 自旋 ↓ 子空间
    # h*(-k) 的非对角元:
    # 原来的 h12(-k) = A(-kx - i(-ky)) = A(-kx + iky) = -A(kx - iky)
    # 取共轭: -A(kx + iky) → -A(-iDx + (-i)(-iDy)) = -A(-iDx - Dy) = A(Dy + iDx)
    # 类似地 h21*(-k) = -A(kx-iky)* = -A(-iDx - i(-iDy))* = ...
    # 简化: h*(-k) 的非对角元 = -(h12 和 h21 的 kx→-kx,ky→-ky 版本取共轭)
    # = -(原始非对角元的负号版本)

    h33_op = eps_op + M_op
    h44_op = eps_op - M_op
    h34_op = params.A * (Dy + 1j * Dx)    # 负的 h12
    h43_op = -params.A * (Dy - 1j * Dx)   # 负的 h21

    # 组装 4N×4N 哈密顿量
    # 使用 block 结构
    blocks = [
        [h11_op, h12_op, sparse.csr_matrix((N_total, N_total), dtype=np.complex128),
         sparse.csr_matrix((N_total, N_total), dtype=np.complex128)],
        [h21_op, h22_op, sparse.csr_matrix((N_total, N_total), dtype=np.complex128),
         sparse.csr_matrix((N_total, N_total), dtype=np.complex128)],
        [sparse.csr_matrix((N_total, N_total), dtype=np.complex128),
         sparse.csr_matrix((N_total, N_total), dtype=np.complex128),
         h33_op, h34_op],
        [sparse.csr_matrix((N_total, N_total), dtype=np.complex128),
         sparse.csr_matrix((N_total, N_total), dtype=np.complex128),
         h43_op, h44_op],
    ]

    H = sparse.bmat(blocks, format='csr')

    # 验证 Hermiticity (数值修正微小非 Hermitian 部分)
    H_herm = 0.5 * (H + H.conj().T)

    return H_herm


def build_bulk_hamiltonian_k(kx_arr: np.ndarray, ky_arr: np.ndarray,
                             params: BHZParameters) -> np.ndarray:
    """
    构建体哈密顿量 H(k) 的网格 (用于能带计算)。

    Parameters
    ----------
    kx_arr, ky_arr : ndarray
        波矢网格 (通过 meshgrid 生成)
    params : BHZParameters

    Returns
    -------
    H_bulk : ndarray, shape (Nk, 4, 4), complex
    """
    kx_flat = kx_arr.ravel()
    ky_flat = ky_arr.ravel()
    Nk = len(kx_flat)

    H_bulk = np.zeros((Nk, 4, 4), dtype=np.complex128)
    for ik in range(Nk):
        H_bulk[ik] = bhz_h_k(kx_flat[ik], ky_flat[ik], params)

    return H_bulk


def hermiticity_error(H: sparse.csr_matrix) -> float:
    """
    计算哈密顿量的 Hermiticity 误差: ||H - H†||_F / ||H||_F。

    使用 Frobenius 范数 (借鉴 131_c8lib 的复矩阵范数)。

    Parameters
    ----------
    H : csr_matrix

    Returns
    -------
    error : float
        相对 Hermiticity 误差
    """
    diff = H - H.conj().T
    norm_diff = sparse.linalg.norm(diff, 'fro')
    norm_H = sparse.linalg.norm(H, 'fro')
    if norm_H < 1e-30:
        return 0.0
    return norm_diff / norm_H


def sparse_to_hb_format(H: sparse.csc_matrix, filename: str,
                        title: str = "BHZ_Hamiltonian") -> str:
    """
    将稀疏哈密顿量矩阵写入 Harwell-Boeing 格式文件。

    Harwell-Boeing 格式结构:
    Line 1: title (72 chars) + key (8 chars)
    Line 2: totcrd, ptrcrd, indcrd, valcrd, [rhscrd]
    Line 3: mxtype, nrow, ncol, nnzero, [neltvl]
    Line 4: [fmt for ptr], [fmt for ind], [fmt for val], [fmt for rhs]
    Data: column pointers, row indices, values

    借鉴 781_msm_to_hb 的 MSM→HB 转换算法。

    Parameters
    ----------
    H : csc_matrix
        稀疏矩阵 (CSC 格式)
    filename : str
        输出文件路径
    title : str
        标题

    Returns
    -------
    report : str
        写入报告的摘要
    """
    H_csc = sparse.csc_matrix(H)
    nrow, ncol = H_csc.shape
    nnz = H_csc.nnz

    # CSC 格式: col_ptr[j] 到 col_ptr[j+1]-1 是第 j 列的非零元
    col_ptr = H_csc.indptr  # length ncol+1
    row_ind = H_csc.indices  # length nnz
    values = H_csc.data  # length nnz, complex128

    # 判断矩阵类型
    is_complex = np.iscomplexobj(values) and np.any(np.imag(values) != 0)
    is_symmetric = False  # BHZ 哈密顿量一般不对称 (是 Hermitian)
    mxtype = "CUA" if is_complex else "RUA"

    # Fortran 格式字符串
    ptr_fmt = "(20I4)"
    ind_fmt = "(20I4)"
    val_fmt = "(5E16.8)" if not is_complex else "(5E16.8)"

    lines = []
    # Line 1
    lines.append(f"{title:<72s}{'BHZ':<8s}")
    # Line 2
    ptrcrd = (ncol + 1 + 19) // 20  # 每行 20 个整数
    indcrd = (nnz + 19) // 20
    if is_complex:
        valcrd = 2 * (nnz + 19) // 20  # 实部和虚部分开
    else:
        valcrd = (nnz + 19) // 20
    totcrd = ptrcrd + indcrd + valcrd
    lines.append(f"{totcrd:14d}{ptrcrd:14d}{indcrd:14d}{valcrd:14d}")
    # Line 3
    lines.append(f"{mxtype:<8s}{nrow:14d}{ncol:14d}{nnz:14d}")
    # Line 4
    lines.append(f"{ptr_fmt:<16s}{ind_fmt:<16s}{val_fmt:<20s}")

    # Column pointers (1-based for Fortran)
    ptr_data = col_ptr + 1  # 转为 1-based
    for i in range(0, len(ptr_data), 20):
        chunk = ptr_data[i:i+20]
        lines.append("".join(f"{v:4d}" for v in chunk))

    # Row indices (1-based)
    row_data = row_ind + 1
    for i in range(0, len(row_data), 20):
        chunk = row_data[i:i+20]
        lines.append("".join(f"{v:4d}" for v in chunk))

    # Values
    if is_complex:
        # 实部和虚部分开写入
        for i in range(0, len(values), 5):
            chunk = values[i:i+5]
            lines.append("".join(f"{v.real:16.8e}" for v in chunk))
        for i in range(0, len(values), 5):
            chunk = values[i:i+5]
            lines.append("".join(f"{v.imag:16.8e}" for v in chunk))
    else:
        for i in range(0, len(values), 5):
            chunk = values[i:i+5]
            lines.append("".join(f"{v:16.8e}" for v in chunk))

    with open(filename, 'w') as f:
        f.write("\n".join(lines) + "\n")

    report = (f"HB 写入完成: {filename}\n"
              f"  矩阵类型: {mxtype}\n"
              f"  维度: {nrow} × {ncol}\n"
              f"  非零元: {nnz}\n"
              f"  总行数: {totcrd}")
    return report


def complex_matrix_norms(H: sparse.csr_matrix) -> Dict[str, float]:
    """
    计算复数稀疏矩阵的多种范数 (借鉴 131_c8lib)。

    返回 Frobenius 范数, 1-范数 (最大列绝对值和), ∞-范数 (最大行绝对值和)。

    Parameters
    ----------
    H : csr_matrix

    Returns
    -------
    norms : dict
    """
    norm_fro = sparse.linalg.norm(H, 'fro')

    # 1-范数: max column sum of |elements|
    H_csc = sparse.csc_matrix(H)
    col_norms = np.zeros(H_csc.shape[1])
    for j in range(H_csc.shape[1]):
        col_data = H_csc.data[H_csc.indptr[j]:H_csc.indptr[j+1]]
        col_norms[j] = np.sum(np.abs(col_data))
    norm_1 = float(np.max(col_norms)) if len(col_norms) > 0 else 0.0

    # ∞-范数: max row sum of |elements|
    H_csr = sparse.csr_matrix(H)
    row_norms = np.zeros(H_csr.shape[0])
    for i in range(H_csr.shape[0]):
        row_data = H_csr.data[H_csr.indptr[i]:H_csr.indptr[i+1]]
        row_norms[i] = np.sum(np.abs(row_data))
    norm_inf = float(np.max(row_norms)) if len(row_norms) > 0 else 0.0

    return {
        'frobenius': norm_fro,
        'L1 (max_col_sum)': norm_1,
        'L_inf (max_row_sum)': norm_inf,
        'condition_estimate': norm_fro * norm_1 if norm_fro > 0 else 0.0
    }
