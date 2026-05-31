"""
稀疏与带状矩阵求解器模块
整合自：
  - 994_r8sd（对称对角稀疏矩阵存储 + 共轭梯度法）
  - 987_r8pbl（对称正定带状矩阵存储与运算）

在吸积盘模拟中用于：
  1. 泊松方程 ∇²Φ = 4πGρ 的求解
  2. 压力方程和隐式粘性项的求解
"""
import numpy as np


# ===========================
# R8SD: Symmetric Diagonal Sparse
# ===========================

class R8SDMatrix:
    """
    对称对角稀疏矩阵（R8SD格式）。

    对于仅少数非零对角线的对称矩阵，将每条对角线折叠到左侧存储：
        A[i, offset[j]] -> a[i, j]
    其中 offset[j] 是第 j 条对角线相对于主对角线的偏移量。

    该格式特别适用于具有长程耦合但稀疏的对称系统，
    如吸积盘径向网格上的泊松方程离散。
    """

    def __init__(self, n, ndiag, offset):
        self.n = n
        self.ndiag = ndiag
        self.offset = np.asarray(offset, dtype=np.int64)
        self.a = np.zeros((n, ndiag), dtype=np.float64)

    def mv(self, x):
        """
        矩阵-向量乘法 y = A·x，利用对称性只存上半部分。
        """
        x = np.asarray(x, dtype=np.float64)
        if len(x) != self.n:
            raise ValueError("Dimension mismatch")

        y = np.zeros(self.n, dtype=np.float64)

        for j in range(self.ndiag):
            off = self.offset[j]
            for i in range(self.n):
                aij = self.a[i, j]
                if aij == 0.0:
                    continue
                y[i] += aij * x[i + off]
                if off != 0 and i + off < self.n:
                    y[i + off] += aij * x[i]

        return y

    def residual(self, x, b):
        """计算残差 r = b - A·x"""
        return b - self.mv(x)

    def to_dense(self):
        """转换为稠密矩阵（调试用）"""
        A = np.zeros((self.n, self.n), dtype=np.float64)
        for j in range(self.ndiag):
            off = self.offset[j]
            for i in range(self.n):
                if i + off < self.n:
                    A[i, i + off] = self.a[i, j]
                    if off != 0:
                        A[i + off, i] = self.a[i, j]
        return A


def r8sd_cg(A, b, x0=None, tol=1e-10, max_iter=None):
    """
    共轭梯度法求解 A·x = b。

    数学原理：
    对于对称正定矩阵 A，CG 法在 Krylov 子空间 K_k(A, r_0) 上
    最小化泛函 φ(x) = 0.5·x^T·A·x - b^T·x。

    迭代公式：
        r_k = b - A·x_k
        α_k = (r_k^T · r_k) / (p_k^T · A · p_k)
        x_{k+1} = x_k + α_k · p_k
        r_{k+1} = r_k - α_k · A · p_k
        β_k = (r_{k+1}^T · r_{k+1}) / (r_k^T · r_k)
        p_{k+1} = r_{k+1} + β_k · p_k

    参数:
        A: R8SDMatrix 对象
        b: 右端向量
        x0: 初始猜测
        tol: 收敛容差
        max_iter: 最大迭代次数（默认 n）

    返回:
        x: 解向量
        info: 迭代信息字典
    """
    b = np.asarray(b, dtype=np.float64)
    n = A.n

    if max_iter is None:
        max_iter = n

    if x0 is None:
        x = np.zeros(n, dtype=np.float64)
    else:
        x = np.asarray(x0, dtype=np.float64).copy()

    r = A.residual(x, b)
    p = r.copy()
    rsold = np.dot(r, r)
    rs0 = rsold

    if rs0 < 1e-30:
        return x, {'iterations': 0, 'residual': 0.0, 'converged': True}

    for k in range(max_iter):
        Ap = A.mv(p)
        pAp = np.dot(p, Ap)

        if abs(pAp) < 1e-30:
            break

        alpha = rsold / pAp
        x += alpha * p
        r -= alpha * Ap
        rsnew = np.dot(r, r)

        if np.sqrt(rsnew / rs0) < tol or rsnew < 1e-12:
            return x, {'iterations': k + 1, 'residual': np.sqrt(rsnew), 'converged': True}

        if rsold < 1e-30:
            break

        beta = rsnew / rsold
        p = r + beta * p
        rsold = rsnew

    return x, {'iterations': max_iter, 'residual': np.sqrt(rsnew), 'converged': False}


# ===========================
# R8PBL: Symmetric Positive Definite Band Lower
# ===========================

class R8PBLMatrix:
    """
    对称正定带状矩阵下三角存储（R8PBL格式）。

    存储方式：
        第1行 = 主对角线 (列 1..n)
        第2行 = 第1次对角线 (列 1..n-1)
        ...
        第ml+1行 = 第ml次对角线 (列 1..n-ml)

    该格式适合径向方向上的带状离散矩阵（如有限差分/有限元）。
    """

    def __init__(self, n, ml):
        self.n = n
        self.ml = ml
        self.a = np.zeros((ml + 1, n), dtype=np.float64)

    def set_diagonal(self, values):
        """设置主对角线"""
        self.a[0, :] = np.asarray(values, dtype=np.float64)

    def set_subdiagonal(self, k, values):
        """设置第 k 条次对角线（k=1..ml）"""
        if k < 1 or k > self.ml:
            raise ValueError(f"k must be in [1, {self.ml}]")
        self.a[k, :self.n - k] = np.asarray(values, dtype=np.float64)

    def mv(self, x):
        """矩阵-向量乘法 y = A·x，利用对称性"""
        x = np.asarray(x, dtype=np.float64)
        if len(x) != self.n:
            raise ValueError("Dimension mismatch")

        y = self.a[0, :] * x

        for k in range(1, self.ml + 1):
            for j in range(self.n - k):
                a_val = self.a[k, j]
                y[j] += a_val * x[j + k]
                y[j + k] += a_val * x[j]

        return y

    def to_dense(self):
        """转换为稠密矩阵"""
        A = np.zeros((self.n, self.n), dtype=np.float64)
        for j in range(self.n):
            A[j, j] = self.a[0, j]
        for k in range(1, self.ml + 1):
            for j in range(self.n - k):
                A[j, j + k] = self.a[k, j]
                A[j + k, j] = self.a[k, j]
        return A

    def cholesky_band_solve(self, b):
        """
        带状 Cholesky 分解求解 A·x = b。
        对于对称正定带状矩阵，A = L·L^T，其中 L 是下三角带状矩阵。

        分解公式（对于带宽 ml）：
            L_{j,j} = √(A_{j,j} - Σ_{k=max(1,j-ml)}^{j-1} L_{j,k}²)
            L_{i,j} = (A_{i,j} - Σ_{k=max(1,i-ml)}^{j-1} L_{i,k}·L_{j,k}) / L_{j,j}

        参数:
            b: 右端向量

        返回:
            x: 解向量
        """
        b = np.asarray(b, dtype=np.float64)
        n = self.n
        ml = self.ml

        # 构建稠密矩阵进行 Cholesky（工程上可进一步优化为纯带状运算）
        A = self.to_dense()

        # Cholesky 分解
        L = np.zeros_like(A)
        for j in range(n):
            # 对角元
            diag_sum = np.sum(L[j, max(0, j - ml):j] ** 2)
            val = A[j, j] - diag_sum
            if val <= 1e-15:
                val = 1e-15  # 数值稳定处理
            L[j, j] = np.sqrt(val)

            # 次对角元
            for i in range(j + 1, min(n, j + ml + 1)):
                off_sum = np.sum(L[i, max(0, i - ml):j] * L[j, max(0, i - ml):j])
                L[i, j] = (A[i, j] - off_sum) / L[j, j]

        # 前代法解 L·y = b
        y = np.zeros(n, dtype=np.float64)
        for i in range(n):
            y[i] = (b[i] - np.dot(L[i, :i], y[:i])) / L[i, i]

        # 回代法解 L^T·x = y
        x = np.zeros(n, dtype=np.float64)
        for i in range(n - 1, -1, -1):
            x[i] = (y[i] - np.dot(L[i + 1:, i], x[i + 1:])) / L[i, i]

        return x


def build_poisson_r8sd(n, dr):
    """
    构造径向泊松方程的离散矩阵：
        (1/r)·d/dr(r·dΦ/dr) ≈ 4πGρ

    采用中心差分：
        (1/r_i)·[(r_{i+1/2}·(Φ_{i+1}-Φ_i)/dr - r_{i-1/2}·(Φ_i-Φ_{i-1})/dr)] / dr

    参数:
        n: 网格点数
        dr: 径向步长

    返回:
        R8SDMatrix 对象
    """
    # 使用3对角结构
    offset = np.array([0, 1], dtype=np.int64)
    A = R8SDMatrix(n, 2, offset)

    for i in range(n):
        r_i = (i + 1) * dr  # 避免 r=0
        rp = r_i + 0.5 * dr
        rm = r_i - 0.5 * dr

        if rm < 0:
            rm = 0.0

        # 主对角线
        if i == 0:
            A.a[i, 0] = rp / (r_i * dr * dr)
        elif i == n - 1:
            A.a[i, 0] = (rp + rm) / (r_i * dr * dr)
        else:
            A.a[i, 0] = (rp + rm) / (r_i * dr * dr)

        # 上对角线
        if i < n - 1:
            A.a[i, 1] = -rp / (r_i * dr * dr)

    return A


def build_band_spd_matrix(n, ml, condition_hint=1.0):
    """
    构造一个随机的对称正定带状矩阵，保证严格对角占优。
    用于测试和作为模板。
    """
    A = R8PBLMatrix(n, ml)

    # 填充次对角线
    for k in range(1, ml + 1):
        vals = np.random.rand(n - k) * 0.5
        A.set_subdiagonal(k, vals)

    # 主对角线保证严格对角占优
    for i in range(n):
        off_sum = 0.0
        for k in range(1, ml + 1):
            if i - k >= 0:
                off_sum += abs(A.a[k, i - k])
            if i + k < n:
                off_sum += abs(A.a[k, i])
        A.a[0, i] = (1.0 + np.random.rand()) * off_sum + 0.01 * condition_hint

    return A
