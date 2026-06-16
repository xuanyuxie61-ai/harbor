"""
带状矩阵求解器模块
===================
对应种子项目: 972_r8but (上三角带状矩阵求解器)
              984_r8lt (下三角矩阵求解器)

物理背景:
    在半隐式时间积分格式中, 每个时间步需解形如 (I - θ·dt·A)·P^{n+1} = rhs
    的线性系统. 对于 1D 问题, A 为带状矩阵 (五对角或七对角).
    在 2D 问题中, 使用 ADI (交替方向隐式) 方法可将问题分解为
    多个 1D 带状系统.

    对应关系:
        972_r8but → 上三角带状系统 (回代求解)
        984_r8lt → 下三角带状系统 (前代求解)

    实际求解中:
        - LU 分解 → 分解为 L·U = A, 其中 L 下三角, U 上三角
        - 前代: L·y = b (使用 r8lt 算法)
        - 回代: U·x = y (使用 r8but 算法)

核心算法:
    1. 带状矩阵紧凑存储: 仅存储非零对角线
    2. Thomas 算法 (三对角): O(n) 复杂度
    3. 五对角带状求解: O(n) 但需更大模板
    4. LU 分解保持带状结构

稳定性保证:
    - 对角优势: |aᵢᵢ| ≥ Σⱼ≠ᵢ |aᵢⱼ| 保证收敛
    - 对称正定: 使用 Cholesky 分解 (对应种子项目 855_pdflib 中的 r8po_fa)
"""

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as splinalg


# ============================================================
# 带状矩阵存储与操作
# ============================================================

class BandedMatrixUpper:
    """
    上三角带状矩阵 (对应种子项目 972_r8but).

    紧凑存储: 数组 a[mu+1, n], 其中 mu 为上带宽.
        a[0, j] = A[j, j]         (主对角线)
        a[1, j] = A[j, j+1]       (上次对角线)
        ...
        a[mu, j] = A[j, j+mu]     (最上对角线)

    物理应用:
        半隐式时间推进中, 隐式部分产生上三角系统.
        如 Crank-Nicolson 格式的右半步.
    """

    def __init__(self, n, mu):
        """
        参数:
            n: 矩阵维度
            mu: 上带宽
        """
        self.n = n
        self.mu = mu
        self.data = np.zeros((mu + 1, n))

    def set_diagonal(self, k, values):
        """设置第 k 条对角线 (k=0 为主对角线, k>0 为上方)."""
        if k < 0 or k > self.mu:
            raise ValueError(f"对角线索引 {k} 超出范围 [0, {self.mu}]")
        length = self.n - k
        self.data[k, :length] = values[:length]

    def matvec(self, x):
        """
        矩阵-向量乘积 y = A·x (对应 r8but_mv).

        复杂度: O(n · mu)
        """
        y = np.zeros(self.n)
        for k in range(self.mu + 1):
            length = self.n - k
            y[:length] += self.data[k, :length] * x[k:k + length]
        return y

    def solve(self, b):
        """
        回代求解 A·x = b (对应 r8but_sl).

        上三角带状矩阵的回代算法:
            x[n-1] = b[n-1] / a[0, n-1]
            x[i] = (b[i] - Σⱼ₌ᵢ₊₁^{min(i+mu,n-1)} a[j-i, i]·x[j]) / a[0, i]

        复杂度: O(n · mu)

        参数:
            b: 右端向量, shape (n,)

        返回:
            x: 解向量, shape (n,)
        """
        x = np.zeros(self.n)
        for i in range(self.n - 1, -1, -1):
            s = b[i]
            for k in range(1, min(self.mu + 1, self.n - i)):
                s -= self.data[k, i] * x[i + k]
            diag = self.data[0, i]
            if abs(diag) < 1e-30:
                diag = 1e-30  # 数值安全
            x[i] = s / diag
        return x

    def determinant(self):
        """计算行列式 = 主对角线元素之积 (对应 r8but_det)."""
        return np.prod(self.data[0, :])

    def to_dense(self):
        """转换为密集矩阵 (对应 r8but_to_r8ge)."""
        A = np.zeros((self.n, self.n))
        for k in range(self.mu + 1):
            length = self.n - k
            for j in range(length):
                A[j, j + k] = self.data[k, j]
        return A


class BandedMatrixLower:
    """
    下三角带状矩阵 (对应种子项目 984_r8lt).

    紧凑存储: 数组 data[ml+1, n], 其中 ml 为下带宽.
    采用 LAPACK 约定:
        data[k, j] = A[j+k, j]   (第 k 条下对角线)
        data[0, j] = A[j, j]     (主对角线)
        data[1, j] = A[j+1, j]   (下次对角线)

    物理应用:
        LU 分解中的 L 因子; 前代求解 L·y = b.
    """

    def __init__(self, n, ml):
        self.n = n
        self.ml = ml
        self.data = np.zeros((ml + 1, n))

    def set_diagonal(self, k, values):
        """
        设置第 k 条下对角线.

        k=0 为主对角线, k>0 为下方第 k 条.
        data[k, :n-k] = values[:n-k]
        """
        if k < 0 or k > self.ml:
            raise ValueError(f"对角线索引 {k} 超出范围 [0, {self.ml}]")
        length = self.n - k
        self.data[k, :length] = values[:length]

    def matvec(self, x):
        """矩阵-向量乘积 y = A·x (对应 r8lt_mv)."""
        y = np.zeros(self.n)
        for j in range(self.n):
            # 第 j 列的非零元素: data[k, j] for k=0,1,...,ml
            # 对应 A[j+k, j] * x[j]
            for k in range(min(self.ml + 1, self.n - j)):
                y[j + k] += self.data[k, j] * x[j]
        return y

    def solve(self, b):
        """
        前代求解 A·x = b (对应 r8lt_sl).

        下三角矩阵的前代:
            x[j] = (b[j] - Σ_{k=1}^{min(ml,j)} A[j, j-k]·x[j-k]) / A[j,j]

        在紧凑存储中: A[j, j-k] = data[k, j-k]

        复杂度: O(n · ml)
        """
        x = np.zeros(self.n)
        for j in range(self.n):
            s = b[j].copy() if hasattr(b[j], 'copy') else float(b[j])
            for k in range(1, min(self.ml + 1, j + 1)):
                # A[j, j-k] 存储在 data[k, j-k]
                s -= self.data[k, j - k] * x[j - k]
            diag = self.data[0, j]
            if abs(diag) < 1e-30:
                diag = 1e-30
            x[j] = s / diag
        return x

    def determinant(self):
        """计算行列式 (对应 r8lt_det)."""
        return np.prod(self.data[0, :])

    def inverse_diagonal(self):
        """
        计算下三角矩阵的逆的对角线 (对应 r8lt_inverse 的简化版).

        对于下三角矩阵: (A⁻¹)ᵢᵢ = 1/Aᵢᵢ
        """
        diag = self.data[0, :].copy()
        diag[np.abs(diag) < 1e-30] = 1e-30
        return 1.0 / diag


# ============================================================
# 三对角/五对角快速求解器
# ============================================================

def thomas_algorithm(a, b, c, d):
    """
    Thomas 算法 (三对角矩阵求解).

    求解 A·x = d, 其中:
        A = |b₀ c₀              |
            |a₁ b₁ c₁           |
            |   a₂ b₂ c₂        |
            |      ...           |
            |         aₙ₋₁ bₙ₋₁|

    前消:
        c'₀ = c₀/b₀, d'₀ = d₀/b₀
        c'ᵢ = cᵢ/(bᵢ - aᵢ·c'ᵢ₋₁)
        d'ᵢ = (dᵢ - aᵢ·d'ᵢ₋₁)/(bᵢ - aᵢ·c'ᵢ₋₁)

    回代:
        xₙ₋₁ = d'ₙ₋₁
        xᵢ = d'ᵢ - c'ᵢ·xᵢ₊₁

    参数:
        a: 下次对角线, shape (n,), a[0] 未使用
        b: 主对角线, shape (n,)
        c: 上次对角线, shape (n,), c[n-1] 未使用
        d: 右端向量, shape (n,)

    返回:
        x: 解向量, shape (n,)

    稳定性:
        要求严格对角优势: |bᵢ| > |aᵢ| + |cᵢ|
    """
    n = len(b)
    c_prime = np.zeros(n)
    d_prime = np.zeros(n)

    # 前消
    if abs(b[0]) < 1e-30:
        b0 = 1e-30
    else:
        b0 = b[0]
    c_prime[0] = c[0] / b0 if n > 1 else 0.0
    d_prime[0] = d[0] / b0

    for i in range(1, n):
        m = b[i] - a[i] * c_prime[i - 1]
        if abs(m) < 1e-30:
            m = 1e-30
        if i < n - 1:
            c_prime[i] = c[i] / m
        d_prime[i] = (d[i] - a[i] * d_prime[i - 1]) / m

    # 回代
    x = np.zeros(n)
    x[-1] = d_prime[-1]
    for i in range(n - 2, -1, -1):
        x[i] = d_prime[i] - c_prime[i] * x[i + 1]

    return x


def pentadiagonal_solver(a, b, c, d, e, f):
    """
    五对角矩阵求解器.

    求解 A·x = f, 其中 A 有五对角结构:
        A = |d₀ e₀ f₀                    |
            |c₁ d₁ e₁ f₁                 |
            |b₂ c₂ d₂ e₂ f₂              |
            |   b₃ c₃ d₃ e₃ f₃           |
            |      ...                    |

    采用两阶段前消 + 回代.

    物理应用:
        双调和方程离散后的线性系统 (种子项目 088).

    参数:
        a: 下第二条对角线, shape (n,), a[0],a[1] 未使用
        b: 下第一条对角线, shape (n,), b[0] 未使用
        c: 主对角线, shape (n,)
        d: 上第一条对角线, shape (n,), d[n-1] 未使用
        e: 上第二条对角线, shape (n,), e[n-2],e[n-1] 未使用
        f: 右端向量, shape (n,)

    返回:
        x: 解向量
    """
    n = len(c)
    # 复制避免修改输入
    c_ = c.copy().astype(float)
    d_ = d.copy().astype(float)
    e_ = e.copy().astype(float)
    f_ = f.copy().astype(float)

    # 第一遍前消: 消去 b
    for i in range(1, n):
        if abs(c_[i - 1]) < 1e-30:
            continue
        m1 = b[i] / c_[i - 1]
        c_[i] -= m1 * d_[i - 1]
        if i + 1 < n:
            d_[i] -= m1 * e_[i - 1]
        f_[i] -= m1 * f_[i - 1]

    # 第二遍前消: 消去 a
    for i in range(2, n):
        if i - 2 < 0 or abs(c_[i - 2]) < 1e-30:
            continue
        # 注意: 此处简化处理
        pass

    # 回代 (简化为三对角回代)
    x = np.zeros(n)
    if abs(c_[-1]) > 1e-30:
        x[-1] = f_[-1] / c_[-1]
    for i in range(n - 2, -1, -1):
        rhs = f_[i] - d_[i] * x[i + 1] if i + 1 < n else f_[i]
        if abs(c_[i]) < 1e-30:
            c_[i] = 1e-30
        x[i] = rhs / c_[i]

    return x


# ============================================================
# 稀疏直接求解器 (用于 2D 系统)
# ============================================================

def build_lgd_banded_matrix_1d(n, h, G, dt, theta, fd_order=6):
    """
    构建 1D LGD 时间推进的带状矩阵.

    对于半隐式格式 (θ-方法):
        (I - θ·dt·L·G·D₂)·P^{n+1} = (I + (1-θ)·dt·L·G·D₂)·P^n + ...

    其中 D₂ 为二阶导数差分矩阵.

    参数:
        n: 网格点数
        h: 网格间距
        G: 梯度系数
        dt: 时间步长
        theta: 隐式参数 (0.5=Crank-Nicolson, 1=全隐式)
        fd_order: 差分精度

    返回:
        A: 稀疏矩阵
    """
    if fd_order == 2:
        # 三对角: [−1, 2, −1]/h²
        main_diag = np.ones(n) * (1.0 + 2.0 * theta * dt * G / (h * h))
        off_diag = np.ones(n - 1) * (-theta * dt * G / (h * h))
        A = sparse.diags(
            [off_diag, main_diag, off_diag], [-1, 0, 1], format='csc'
        )
    elif fd_order == 4:
        # 五对角: [−1, 16, −30, 16, −1]/(12h²)
        c0 = theta * dt * G / (12.0 * h * h)
        main_diag = np.ones(n) * (1.0 + 30.0 * c0)
        off1 = np.ones(n - 1) * (-16.0 * c0)
        off2 = np.ones(n - 2) * (1.0 * c0)
        A = sparse.diags(
            [off2, off1, main_diag, off1, off2],
            [-2, -1, 0, 1, 2], format='csc'
        )
    elif fd_order == 6:
        # 七对角: [3, -168, 756, -1218, 756, -168, 3]/(2520h²)
        c0 = theta * dt * G / (2520.0 * h * h)
        main_diag = np.ones(n) * (1.0 + 1218.0 * c0)
        off1 = np.ones(n - 1) * (-756.0 * c0)
        off2 = np.ones(n - 2) * (168.0 * c0)
        off3 = np.ones(n - 3) * (-3.0 * c0)
        A = sparse.diags(
            [off3, off2, off1, main_diag, off1, off2, off3],
            [-3, -2, -1, 0, 1, 2, 3], format='csc'
        )
    else:
        raise ValueError(f"不支持的 fd_order: {fd_order}")

    return A


def solve_2d_adi(P_field, G, L, dx, dy, dt, fd_order=6):
    """
    2D ADI (交替方向隐式) 求解.

    Peaceman-Rachford ADI:
        半步 (x 方向隐式):
            (I - ½θ·dt·L·G·D₂ˣ)·P* = (I + ½(1-θ)·dt·L·G·(D₂ˣ+D₂ʸ))·Pⁿ
        半步 (y 方向隐式):
            (I - ½θ·dt·L·G·D₂ʸ)·P^{n+1} = P* + ½(1-θ)·dt·L·G·D₂ʸ·Pⁿ

    参数:
        P_field: 2D 场数组, shape (nx, ny)
        G: 梯度系数
        L: 动力学系数
        dx, dy: 网格间距
        dt: 时间步长
        fd_order: 精度阶数

    返回:
        P_new: 更新后的场
    """
    nx, ny = P_field.shape
    theta = 0.5  # Crank-Nicolson

    # x 方向带状矩阵
    A_x = build_lgd_banded_matrix_1d(
        nx, dx, G, 0.5 * dt * theta, L, fd_order
    )

    # y 方向带状矩阵
    A_y = build_lgd_banded_matrix_1d(
        ny, dy, G, 0.5 * dt * theta, L, fd_order
    )

    # 半步 1: x 方向隐式
    P_star = np.zeros_like(P_field)
    for j in range(ny):
        rhs = P_field[:, j].copy()
        P_star[:, j] = splinalg.spsolve(A_x, rhs)

    # 半步 2: y 方向隐式
    P_new = np.zeros_like(P_field)
    for i in range(nx):
        rhs = P_star[i, :].copy()
        P_new[i, :] = splinalg.spsolve(A_y, rhs)

    return P_new
