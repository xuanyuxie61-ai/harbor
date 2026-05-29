# -*- coding: utf-8 -*-
"""
helmholtz_fd.py
Helmholtz 方程有限差分求解器

核心公式与物理背景
------------------
1. 标量 Helmholtz 方程（柱坐标下的时谐电磁场）
   对于微环谐振腔的截面（忽略 z 向变化），光场 E 满足：
       ∇²E + k₀² n²(x,y) E = 0
   其中 k₀ = 2π/λ₀ 为真空波数，n(x,y) 为空间折射率分布。

2. 5 点拉普拉斯差分格式（均匀网格）
   在二维矩形网格上，步长 h：
       (∇²E)_{i,j} ≈ (E_{i+1,j} + E_{i-1,j} + E_{i,j+1} + E_{i,j-1} - 4E_{i,j}) / h²
   截断误差 O(h²)。

3. 9 点高精度格式（周期边界/环面）
   在 2D torus 上，9 点 stencil：
       (∇²u)_{i,j} ≈ [ -20·u_{i,j}
                       + 4·(u_{i+1,j}+u_{i-1,j}+u_{i,j+1}+u_{i,j-1})
                       + 1·(u_{i+1,j+1}+u_{i-1,j+1}+u_{i+1,j-1}+u_{i-1,j-1}) ] / (6h²)
   截断误差 O(h⁴)。

4. 非均匀网格 3 点格式（1D）
   对非均匀节点 x_{i-1}, x_i, x_{i+1}，间距 h_L = x_i - x_{i-1}, h_R = x_{i+1} - x_i：
       u''(x_i) ≈ 2·[ h_R·u_{i-1} - (h_L+h_R)·u_i + h_L·u_{i+1} ] / (h_L·h_R·(h_L+h_R))

5. 带状矩阵存储与求解
   5 点格式在 N×N 网格上产生带宽 ML=MU=N 的带状矩阵。
   采用紧凑存储 r8cb：A(ML+MU+1, n)。
   LU 分解（无 pivoting）用于直接求解。

融合来源
--------
- 647_laplacian   : 各类拉普拉斯差分算子（3点/5点/9点，均匀/非均匀，周期边界）
- 973_r8cb        : 压缩带状矩阵 LU 分解与前代回代
- 974_r8cbb       : 边界带状矩阵 Schur 补求解（用于处理边界条件引入的稠密耦合）
"""

import numpy as np
from typing import Optional, Tuple


class LaplacianStencils:
    """
    提供多种离散拉普拉斯算子 stencil。
    """

    @staticmethod
    def laplacian5_2d(u: np.ndarray, h: float, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        标准 5 点 stencil 在 2D 矩形域（Dirichlet 零边界）。
        边界外假设 u = 0。
        """
        if u.ndim != 2:
            raise ValueError("u 必须是二维数组")
        ny, nx = u.shape
        Lu = np.zeros_like(u)
        # 内部点
        Lu[1:-1, 1:-1] = (
            u[2:, 1:-1] + u[:-2, 1:-1] +
            u[1:-1, 2:] + u[1:-1, :-2] - 4.0 * u[1:-1, 1:-1]
        ) / (h * h)
        if mask is not None:
            Lu *= mask
        return Lu

    @staticmethod
    def laplacian5_torus(u: np.ndarray, h: float) -> np.ndarray:
        """
        5 点 stencil 在 2D torus（双周期边界）。
        通过 circshift / wrap-around 处理边界。
        """
        if u.ndim != 2:
            raise ValueError("u 必须是二维数组")
        ny, nx = u.shape
        Lu = np.zeros_like(u)
        # 利用 roll 实现周期 wrapping
        uxp = np.roll(u, shift=-1, axis=1)
        uxm = np.roll(u, shift=1, axis=1)
        uyp = np.roll(u, shift=-1, axis=0)
        uym = np.roll(u, shift=1, axis=0)
        Lu = (uxp + uxm + uyp + uym - 4.0 * u) / (h * h)
        return Lu

    @staticmethod
    def laplacian9_torus(u: np.ndarray, h: float) -> np.ndarray:
        """
        9 点高阶 stencil 在 2D torus，O(h⁴) 精度。
        """
        if u.ndim != 2:
            raise ValueError("u 必须是二维数组")
        # 4 邻域
        uxp = np.roll(u, shift=-1, axis=1)
        uxm = np.roll(u, shift=1, axis=1)
        uyp = np.roll(u, shift=-1, axis=0)
        uym = np.roll(u, shift=1, axis=0)
        # 4 对角
        uxpyp = np.roll(np.roll(u, shift=-1, axis=1), shift=-1, axis=0)
        uxmyp = np.roll(np.roll(u, shift=1, axis=1), shift=-1, axis=0)
        uxpym = np.roll(np.roll(u, shift=-1, axis=1), shift=1, axis=0)
        uxmym = np.roll(np.roll(u, shift=1, axis=1), shift=1, axis=0)
        Lu = (-20.0 * u
              + 4.0 * (uxp + uxm + uyp + uym)
              + 1.0 * (uxpyp + uxmyp + uxpym + uxmym)) / (6.0 * h * h)
        return Lu

    @staticmethod
    def laplacian3_uneven_1d(u: np.ndarray, x: np.ndarray) -> np.ndarray:
        """
        1D 非均匀网格 3 点拉普拉斯（二阶导数近似）。
        x 为节点坐标，长度与 u 相同。
        """
        if len(u) != len(x):
            raise ValueError("u 与 x 长度不匹配")
        n = len(u)
        Lu = np.zeros_like(u)
        for i in range(1, n - 1):
            hL = x[i] - x[i - 1]
            hR = x[i + 1] - x[i]
            if hL <= 0 or hR <= 0:
                raise ValueError("网格坐标必须严格递增")
            denom = hL * hR * (hL + hR)
            Lu[i] = 2.0 * (hR * u[i - 1] - (hL + hR) * u[i] + hL * u[i + 1]) / denom
        # 边界保持 0（Dirichlet）
        return Lu


class BandedMatrixSolver:
    """
    压缩带状矩阵（r8cb 格式）的 LU 分解与求解。
    矩阵 A 的半带宽为 ML（下）和 MU（上），紧凑存储为 A_band[ML+MU+1, N]。
    """

    def __init__(self, n: int, ml: int, mu: int):
        self.n = n
        self.ml = ml
        self.mu = mu
        self.lda = ml + mu + 1
        self._A_band = None
        self._factorized = False

    def from_dense(self, A_dense: np.ndarray):
        """从稠密矩阵提取带状部分并压缩存储"""
        if A_dense.shape != (self.n, self.n):
            raise ValueError("稠密矩阵维度不匹配")
        self._A_band = np.zeros((self.lda, self.n), dtype=float)
        for j in range(self.n):
            for i in range(max(0, j - self.mu), min(self.n, j + self.ml + 1)):
                self._A_band[self.mu + i - j, j] = A_dense[i, j]
        self._factorized = False

    def set_element(self, i: int, j: int, value: float):
        """设置 A[i,j]（仅允许在带宽内）"""
        if abs(i - j) > self.mu and abs(i - j) > self.ml:
            raise ValueError("(i,j) 超出带宽范围")
        if self._A_band is None:
            self._A_band = np.zeros((self.lda, self.n), dtype=float)
        self._A_band[self.mu + i - j, j] = value

    def factorize_np(self):
        """
        不带 pivoting 的带状 LU 分解（Doolittle 变体）。
        若遇到零主元，抛出 ValueError。
        """
        if self._A_band is None:
            raise ValueError("矩阵未初始化")
        A = self._A_band.copy()
        n = self.n
        ml = self.ml
        mu = self.mu
        lda = self.lda

        for k in range(n - 1):
            # 主元检查
            pivot = A[mu, k]
            if abs(pivot) < 1e-30:
                raise ValueError(f"零主元出现在第 {k} 行，LU 分解失败")
            # 计算 L 的列（严格下三角部分）
            n_rows = min(ml, n - k - 1)
            for i in range(n_rows):
                A[mu + i + 1, k] /= pivot
            # 更新 U 的尾随子矩阵
            n_cols = min(mu + ml, n - k - 1)
            for j in range(n_cols):
                factor = A[mu - j - 1, k + j + 1]
                for i in range(n_rows):
                    A[mu + i - j, k + j + 1] -= A[mu + i + 1, k] * factor
        self._A_band = A
        self._factorized = True

    def solve(self, b: np.ndarray) -> np.ndarray:
        """
        利用已分解的带状 LU 做前代回代求解 Ax = b。
        """
        if not self._factorized:
            raise ValueError("矩阵未分解，请先调用 factorize_np()")
        A = self._A_band
        n = self.n
        ml = self.ml
        mu = self.mu
        x = b.copy().astype(float)

        # 前代：L y = b
        for k in range(n - 1):
            n_rows = min(ml, n - k - 1)
            for i in range(n_rows):
                x[k + i + 1] -= A[mu + i + 1, k] * x[k]

        # 回代：U x = y
        for k in range(n - 1, -1, -1):
            pivot = A[mu, k]
            if abs(pivot) < 1e-30:
                raise ValueError("回代时遇到零主元")
            x[k] /= pivot
            n_rows = min(mu, k)
            for i in range(n_rows):
                x[k - i - 1] -= A[mu - i - 1, k] * x[k]
        return x

    def matvec(self, x: np.ndarray) -> np.ndarray:
        """
        带状矩阵-向量乘法 b = A x（使用紧凑存储）。
        """
        if self._A_band is None:
            raise ValueError("矩阵未初始化")
        A = self._A_band
        n = self.n
        ml = self.ml
        mu = self.mu
        b = np.zeros(n, dtype=float)
        x = np.asarray(x)
        for i in range(n):
            j_min = max(0, i - ml)
            j_max = min(n - 1, i + mu)
            for j in range(j_min, j_max + 1):
                b[i] += A[mu + i - j, j] * x[j]
        return b


class BorderedBandedSolver:
    """
    边界带状矩阵（Bordered Banded Matrix）求解器。
    矩阵分块为 [A1 A2; A3 A4]，其中 A1 为 n1×n1 带状，其余为稠密。
    使用 Schur 补分解：
        S = A4 - A3·A1^{-1}·A2
    """

    def __init__(self, n1: int, n2: int, ml: int, mu: int):
        self.n1 = n1
        self.n2 = n2
        self.ml = ml
        self.mu = mu
        self._A1_solver = BandedMatrixSolver(n1, ml, mu)
        self._A2 = None
        self._A3 = None
        self._A4 = None
        self._S_factor = None

    def set_blocks(self, A1_band: np.ndarray, A2: np.ndarray, A3: np.ndarray, A4: np.ndarray):
        """
        设置四个分块。A1 为 lda×n1 带状紧凑存储，其余为稠密。
        """
        self._A1_solver._A_band = A1_band.copy()
        self._A1_solver._factorized = False
        if A2.shape != (self.n1, self.n2):
            raise ValueError("A2 维度错误")
        if A3.shape != (self.n2, self.n1):
            raise ValueError("A3 维度错误")
        if A4.shape != (self.n2, self.n2):
            raise ValueError("A4 维度错误")
        self._A2 = A2.copy()
        self._A3 = A3.copy()
        self._A4 = A4.copy()

    def factorize(self):
        """
        Schur 补分解：
            1) A1 = LU
            2) A2' = -A1^{-1} A2   （逐列求解）
            3) S = A4 + A3 A2'
            4) S = LU_dense
        """
        self._A1_solver.factorize_np()
        # 求解 A1·X = -A2
        X = np.zeros_like(self._A2)
        for j in range(self.n2):
            rhs = -self._A2[:, j]
            X[:, j] = self._A1_solver.solve(rhs)
        self._A2 = X  # 现在存储的是 -A1^{-1} A2
        # Schur 补
        S = self._A4 + self._A3 @ self._A2
        # 稠密 LU（简单实现，无 pivoting）
        self._S_factor = self._dense_lu_factor(S)

    @staticmethod
    def _dense_lu_factor(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """稠密矩阵 Doolittle LU 分解，返回 (LU, pivot_indices)"""
        n = A.shape[0]
        LU = A.copy()
        piv = np.arange(n)
        for k in range(n):
            # 部分选主元（增强稳定性）
            max_row = np.argmax(np.abs(LU[k:, k])) + k
            if abs(LU[max_row, k]) < 1e-30:
                raise ValueError("稠密矩阵 LU 分解遇到奇异主元")
            if max_row != k:
                LU[[k, max_row], :] = LU[[max_row, k], :]
                piv[[k, max_row]] = piv[[max_row, k]]
            for i in range(k + 1, n):
                LU[i, k] /= LU[k, k]
                for j in range(k + 1, n):
                    LU[i, j] -= LU[i, k] * LU[k, j]
        return LU, piv

    @staticmethod
    def _dense_lu_solve(LU: np.ndarray, piv: np.ndarray, b: np.ndarray) -> np.ndarray:
        n = LU.shape[0]
        x = b[piv].copy().astype(float)
        # 前代
        for i in range(1, n):
            for j in range(i):
                x[i] -= LU[i, j] * x[j]
        # 回代
        for i in range(n - 1, -1, -1):
            for j in range(i + 1, n):
                x[i] -= LU[i, j] * x[j]
            x[i] /= LU[i, i]
        return x

    def solve(self, b: np.ndarray) -> np.ndarray:
        """
        求解 [A1 A2; A3 A4] [x1; x2] = [b1; b2]
        步骤：
            x1 = A1^{-1} b1
            b2' = b2 - A3 x1
            x2 = S^{-1} b2'
            x1 = x1 + (-A1^{-1} A2) x2
        """
        b1 = b[:self.n1]
        b2 = b[self.n1:]
        x1 = self._A1_solver.solve(b1)
        b2p = b2 - self._A3 @ x1
        x2 = self._dense_lu_solve(self._S_factor[0], self._S_factor[1], b2p)
        x1 = x1 + self._A2 @ x2
        return np.concatenate([x1, x2])


class HelmholtzSolver:
    """
    基于有限差分的 Helmholtz 方程求解器。
    求解  [∇² + k₀² n²] E = 0  在给定网格上，带 Dirichlet 边界条件。
    """

    def __init__(self, nx: int, ny: int, Lx: float, Ly: float,
                 k0: float, n_profile: np.ndarray,
                 boundary_value: float = 0.0,
                 damping: float = 0.05):
        """
        参数
        ----
        nx, ny : int
            x, y 方向网格点数（含边界）
        Lx, Ly : float
            物理域尺寸 [m]
        k0 : float
            真空波数 [rad/m]
        n_profile : np.ndarray, shape (ny, nx)
            折射率空间分布
        boundary_value : float
            Dirichlet 边界值
        """
        self.nx = nx
        self.ny = ny
        self.Lx = Lx
        self.Ly = Ly
        self.hx = Lx / (nx - 1)
        self.hy = Ly / (ny - 1)
        self.k0 = k0
        self.n_profile = n_profile.copy()
        self.boundary_value = boundary_value
        self.damping = damping
        self._N = nx * ny
        self._band_solver = None

    def _build_banded_system(self) -> BandedMatrixSolver:
        """
        用 5 点 stencil 构建 Helmholtz 离散系统：
            (L + k₀²n²) E = RHS
        其中 L 为离散拉普拉斯。带状半带宽 ML=MU=nx。
        返回 BandedMatrixSolver 实例。
        """
        n = self._N
        ml = self.nx
        mu = self.nx
        solver = BandedMatrixSolver(n, ml, mu)
        solver._A_band = np.zeros((solver.lda, n), dtype=float)
        hx2 = self.hx ** 2
        hy2 = self.hy ** 2

        for j in range(self.ny):
            for i in range(self.nx):
                idx = j * self.nx + i
                # 边界点固定
                if i == 0 or i == self.nx - 1 or j == 0 or j == self.ny - 1:
                    solver._A_band[mu, idx] = 1.0
                    continue
                # TODO(Hole 1): 实现 Helmholtz 方程内部点的 5 点 stencil 离散化
                # 提示：需要计算中心系数和四个邻接系数，并考虑阻尼项
                # 中心系数应包含：-2/hx² - 2/hy² + (k₀·n)² + damping·k₀²
                # 邻接系数为 1/hx²（左右）和 1/hy²（上下）
                # 使用 solver._A_band[mu + offset, idx] = value 设置带状矩阵元素
                raise NotImplementedError("Hole 1: 请补全 Helmholtz 5 点 stencil 离散化")
        solver._factorized = False
        return solver

    def solve_for_rhs(self, rhs: np.ndarray) -> np.ndarray:
        """
        求解 (L + k₀²n²) E = rhs，带 Dirichlet 边界。
        rhs 在边界上的值会被覆盖为 boundary_value。
        """
        if rhs.shape != (self.ny, self.nx):
            raise ValueError("rhs 形状必须与网格一致")
        rhs_flat = rhs.flatten().copy()
        # 边界固定
        for j in range(self.ny):
            for i in range(self.nx):
                idx = j * self.nx + i
                if i == 0 or i == self.nx - 1 or j == 0 or j == self.ny - 1:
                    rhs_flat[idx] = self.boundary_value

        if self._band_solver is None:
            self._band_solver = self._build_banded_system()
            self._band_solver.factorize_np()
        E_flat = self._band_solver.solve(rhs_flat)
        return E_flat.reshape(self.ny, self.nx)

    def compute_optical_intensity(self, E: np.ndarray) -> np.ndarray:
        """
        计算光强 I = 0.5·c·ε₀·n·|E|²。
        这里返回归一化强度 |E|²。
        """
        return np.abs(E) ** 2

    def apply_source_and_solve(self, source_mask: np.ndarray, source_amplitude: complex = 1.0) -> np.ndarray:
        """
        在 source_mask 标记的位置施加点源，求解稳态场分布。
        """
        rhs = np.zeros((self.ny, self.nx), dtype=complex)
        rhs[source_mask] = source_amplitude
        # 分别求解实部和虚部
        E_real = self.solve_for_rhs(rhs.real)
        E_imag = self.solve_for_rhs(rhs.imag)
        return E_real + 1j * E_imag
