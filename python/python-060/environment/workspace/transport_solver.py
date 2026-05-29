"""
平流-扩散传输方程求解器

本模块实现平流层中化学物种的垂直传输过程数值求解，包括：
- 垂直涡旋扩散方程离散化
- 平流项的迎风差分格式
- 大型稀疏线性系统的 GMRES 和 CG 求解
- 隐式时间积分 (后退 Euler)

科学公式:
1. 垂直扩散方程:
   ∂n/∂t = ∂/∂z (Kzz(z) × ∂n/∂z)
   其中 Kzz 为涡旋扩散系数 (m²/s)

2. 平流-扩散方程 (含化学源汇):
   ∂n/∂t + w × ∂n/∂z = ∂/∂z (Kzz × ∂n/∂z) + P - L
   其中 w 为垂直风速 (m/s)

3. 隐式离散化 (后退 Euler):
   (n^{k+1} - n^k) / Δt = L(n^{k+1}) + S^k
   (I - Δt L) n^{k+1} = n^k + Δt S^k
   A x = b

4. 扩散项中心差分:
   ∂/∂z(Kzz ∂n/∂z)|_i ≈ [K_{i+1/2}(n_{i+1}-n_i) - K_{i-1/2}(n_i-n_{i-1})] / dz²
   K_{i+1/2} = (K_i + K_{i+1}) / 2

5. 平流项迎风差分 (w > 0 向上):
   w ∂n/∂z|_i ≈ w (n_i - n_{i-1}) / dz

融入原项目: 760_mgmres (重启GMRES迭代), 149_cg (共轭梯度法)
"""

import numpy as np
from typing import Tuple, Optional, Callable


class SparseMatrixCRS:
    """
    压缩稀疏行 (CRS) 格式矩阵
    用于高效存储和操作大型稀疏矩阵
    """

    def __init__(self, n: int, nz_num: int):
        self.n = n
        self.nz_num = nz_num
        self.a = np.zeros(nz_num)      # 非零元素值
        self.ia = np.zeros(nz_num, dtype=int)  # 行索引
        self.ja = np.zeros(nz_num, dtype=int)  # 列索引

    def from_dense(self, A: np.ndarray, threshold: float = 1e-14) -> None:
        """
        从稠密矩阵构建 CRS 格式
        """
        if A.shape[0] != self.n or A.shape[1] != self.n:
            raise ValueError("矩阵维度不匹配")

        idx = 0
        for i in range(self.n):
            for j in range(self.n):
                if abs(A[i, j]) > threshold:
                    if idx >= self.nz_num:
                        raise ValueError("非零元素数量超过预分配")
                    self.a[idx] = A[i, j]
                    self.ia[idx] = i
                    self.ja[idx] = j
                    idx += 1

        # 如果实际非零元素更少，截断
        self.nz_num = idx
        self.a = self.a[:idx]
        self.ia = self.ia[:idx]
        self.ja = self.ja[:idx]


def sparse_matvec(a: np.ndarray, ia: np.ndarray, ja: np.ndarray,
                  x: np.ndarray, n: int, nz_num: int) -> np.ndarray:
    """
    CRS 格式稀疏矩阵-向量乘法 y = A x
    """
    y = np.zeros(n)
    for k in range(nz_num):
        i = ia[k]
        j = ja[k]
        y[i] += a[k] * x[j]
    return y


def mult_givens(c: float, s: float, k: int, g: np.ndarray) -> np.ndarray:
    """
    Givens 旋转应用于向量 g
    """
    g1 = c * g[k] - s * g[k + 1]
    g2 = s * g[k] + c * g[k + 1]
    g = g.copy()
    g[k] = g1
    g[k + 1] = g2
    return g


def mgmres(a: np.ndarray, ia: np.ndarray, ja: np.ndarray,
           x: np.ndarray, rhs: np.ndarray, n: int, nz_num: int,
           itr_max: int = 100, mr: int = 20,
           tol_abs: float = 1e-10, tol_rel: float = 1e-8) -> np.ndarray:
    """
    重启 GMRES 迭代求解稀疏线性系统 A x = rhs

    Parameters
    ----------
    a, ia, ja : ndarray
        CRS 格式矩阵
    x : ndarray
        初始猜测
    rhs : ndarray
        右端项
    n : int
        矩阵阶数
    nz_num : int
        非零元数
    itr_max : int
        最大外迭代次数
    mr : int
        Krylov 子空间维度
    tol_abs, tol_rel : float
        绝对/相对残差容差

    Returns
    -------
    x : ndarray
        解向量
    """
    if mr <= 0 or mr > n:
        mr = min(20, n)

    delta = 0.001
    verbose = 0

    rho_tol = tol_rel * np.linalg.norm(rhs) if np.linalg.norm(rhs) > 0 else tol_abs

    for itr in range(itr_max):
        r = rhs - sparse_matvec(a, ia, ja, x, n, nz_num)
        rho = np.linalg.norm(r)

        if rho <= tol_abs or rho <= rho_tol:
            break

        v = np.zeros((n, mr + 1))
        v[:, 0] = r / (rho + 1e-30)

        g = np.zeros(mr + 1)
        g[0] = rho

        h = np.zeros((mr + 1, mr))
        c = np.zeros(mr)
        s = np.zeros(mr)

        for k in range(mr):
            v[:, k + 1] = sparse_matvec(a, ia, ja, v[:, k], n, nz_num)
            av = np.linalg.norm(v[:, k + 1])

            for j in range(k + 1):
                h[j, k] = np.dot(v[:, j], v[:, k + 1])
                v[:, k + 1] -= h[j, k] * v[:, j]

            h[k + 1, k] = np.linalg.norm(v[:, k + 1])

            # 重正交化
            if av + delta * h[k + 1, k] == av:
                for j in range(k + 1):
                    htmp = np.dot(v[:, j], v[:, k + 1])
                    h[j, k] += htmp
                    v[:, k + 1] -= htmp * v[:, j]
                h[k + 1, k] = np.linalg.norm(v[:, k + 1])

            if h[k + 1, k] != 0.0:
                v[:, k + 1] /= h[k + 1, k]

            # Givens 旋转
            if k > 0:
                y = h[:k + 2, k].copy()
                for j in range(k):
                    y = mult_givens(c[j], s[j], j, y)
                h[:k + 2, k] = y

            mu = np.sqrt(h[k, k] ** 2 + h[k + 1, k] ** 2)
            if mu > 0:
                c[k] = h[k, k] / mu
                s[k] = -h[k + 1, k] / mu
                h[k, k] = c[k] * h[k, k] - s[k] * h[k + 1, k]
                h[k + 1, k] = 0.0
                g = mult_givens(c[k], s[k], k, g)

            rho = abs(g[k + 1])
            if rho <= tol_abs and rho <= rho_tol:
                # 回代求解上三角系统
                k_copy = k
                y = np.zeros(k_copy + 1)
                y[k_copy] = g[k_copy] / (h[k_copy, k_copy] + 1e-30)
                for i in range(k_copy - 1, -1, -1):
                    y[i] = (g[i] - np.dot(h[i, i + 1:k_copy + 1], y[i + 1:k_copy + 1])) / \
                           (h[i, i] + 1e-30)
                x += v[:, :k_copy + 1] @ y
                break
        else:
            # 达到 mr 次迭代
            k = mr - 1
            y = np.zeros(mr)
            y[mr - 1] = g[mr - 1] / (h[mr - 1, mr - 1] + 1e-30)
            for i in range(mr - 2, -1, -1):
                y[i] = (g[i] - np.dot(h[i, i + 1:mr], y[i + 1:mr])) / \
                       (h[i, i] + 1e-30)
            x += v[:, :mr] @ y

    return x


def conjugate_gradient(A: np.ndarray, b: np.ndarray,
                       x0: Optional[np.ndarray] = None,
                       max_iter: int = None,
                       tol: float = 1e-10) -> np.ndarray:
    """
    共轭梯度法求解对称正定系统 A x = b

    Parameters
    ----------
    A : ndarray
        SPD 矩阵 (n, n)
    b : ndarray
        右端项
    x0 : ndarray, optional
        初始猜测
    max_iter : int
        最大迭代次数
    tol : float
        残差容差

    Returns
    -------
    x : ndarray
        解向量
    """
    n = len(b)
    if max_iter is None:
        max_iter = n

    if x0 is None:
        x = np.zeros(n)
    else:
        x = x0.copy()

    r = b - A @ x
    p = r.copy()
    rs_old = np.dot(r, r)

    for _ in range(max_iter):
        Ap = A @ p
        pAp = np.dot(p, Ap)

        if abs(pAp) < 1e-30:
            break

        alpha = rs_old / pAp
        x += alpha * p
        r -= alpha * Ap
        rs_new = np.dot(r, r)

        if np.sqrt(rs_new) < tol:
            break

        beta = rs_new / (rs_old + 1e-30)
        p = r + beta * p
        rs_old = rs_new

    return x


class VerticalTransportSolver:
    """
    垂直传输方程求解器
    """

    def __init__(self, z: np.ndarray, Kzz: np.ndarray,
                 w: Optional[np.ndarray] = None):
        """
        Parameters
        ----------
        z : ndarray
            高度网格 (m), 单调递增
        Kzz : ndarray
            涡旋扩散系数 (m²/s)
        w : ndarray, optional
            垂直风速 (m/s), 正值向上
        """
        if len(z) < 3:
            raise ValueError("高度网格至少需要3个点")
        if len(Kzz) != len(z):
            raise ValueError("Kzz 和 z 长度必须相同")
        if not np.all(np.diff(z) > 0):
            raise ValueError("高度网格必须严格单调递增")

        self.nz = len(z)
        self.z = z.copy()
        self.dz = np.diff(z)
        self.Kzz = np.clip(Kzz, 1e-6, 1e4)

        if w is None:
            self.w = np.zeros(self.nz)
        else:
            self.w = w.copy()

        # 网格界面扩散系数
        self.K_face = np.zeros(self.nz - 1)
        for i in range(self.nz - 1):
            self.K_face[i] = 0.5 * (self.Kzz[i] + self.Kzz[i + 1])

    def _build_diffusion_matrix(self) -> np.ndarray:
        """
        构建扩散算子的离散矩阵 (中心差分)
        D_i = [K_{i+1/2}(n_{i+1}-n_i)/dz_{i+1} - K_{i-1/2}(n_i-n_{i-1})/dz_i] / dz_mid
        """
        nz = self.nz
        L = np.zeros((nz, nz))

        for i in range(1, nz - 1):
            dz_i = self.z[i] - self.z[i - 1]
            dz_ip1 = self.z[i + 1] - self.z[i]
            dz_mid = 0.5 * (dz_i + dz_ip1)

            K_m = self.K_face[i - 1]
            K_p = self.K_face[i]

            a_m = K_m / (dz_i * dz_mid)
            a_p = K_p / (dz_ip1 * dz_mid)

            L[i, i - 1] = a_m
            L[i, i] = -(a_m + a_p)
            L[i, i + 1] = a_p

        # 边界条件: 零通量 (Neumann)
        # 底部
        dz_0 = self.z[1] - self.z[0]
        L[0, 0] = -self.K_face[0] / (dz_0 ** 2)
        L[0, 1] = self.K_face[0] / (dz_0 ** 2)

        # 顶部
        dz_n = self.z[-1] - self.z[-2]
        L[-1, -2] = self.K_face[-1] / (dz_n ** 2)
        L[-1, -1] = -self.K_face[-1] / (dz_n ** 2)

        return L

    def _build_advection_matrix(self) -> np.ndarray:
        """
        构建平流算子离散矩阵 (迎风差分)
        """
        nz = self.nz
        A = np.zeros((nz, nz))

        for i in range(1, nz):
            dz_i = self.z[i] - self.z[i - 1]
            w_i = self.w[i]

            if w_i > 0:  # 向上风，用下方值
                A[i, i - 1] = -w_i / dz_i
                A[i, i] = w_i / dz_i
            elif w_i < 0:  # 向下风，用上方值
                if i < nz - 1:
                    dz_ip1 = self.z[i + 1] - self.z[i]
                    A[i, i] = -w_i / dz_ip1
                    A[i, i + 1] = w_i / dz_ip1

        # 底部边界: 固定通量
        A[0, 0] = 0.0

        return A

    def build_implicit_matrix(self, dt: float,
                              jac_diag: Optional[np.ndarray] = None) -> np.ndarray:
        """
        构建隐式时间步进矩阵
        A = I - dt * (L_diff + L_adv + J_chem)

        Parameters
        ----------
        dt : float
            时间步长
        jac_diag : ndarray, optional
            化学雅可比对角线

        Returns
        -------
        A : ndarray
            隐式矩阵 (nz, nz)
        """
        if dt <= 0:
            raise ValueError("dt 必须为正")

        nz = self.nz
        L_diff = self._build_diffusion_matrix()
        L_adv = self._build_advection_matrix()

        A = np.eye(nz) - dt * (L_diff + L_adv)

        if jac_diag is not None:
            if len(jac_diag) != nz:
                raise ValueError("jac_diag 长度必须与 nz 相同")
            for i in range(nz):
                A[i, i] -= dt * jac_diag[i]

        return A

    def solve_implicit(self, n_old: np.ndarray, dt: float,
                       source: Optional[np.ndarray] = None,
                       jac_diag: Optional[np.ndarray] = None,
                       use_cg: bool = True) -> np.ndarray:
        """
        隐式求解一步传输+化学
        (I - dt L) n_new = n_old + dt * S

        Parameters
        ----------
        n_old : ndarray
            上一时刻浓度
        dt : float
            时间步长
        source : ndarray, optional
            化学源汇项 (P - L)
        jac_diag : ndarray, optional
            化学雅可比对角线
        use_cg : bool
            若矩阵对称正定则使用 CG，否则使用 GMRES

        Returns
        -------
        n_new : ndarray
            新时刻浓度
        """
        nz = self.nz
        A = self.build_implicit_matrix(dt, jac_diag)
        b = n_old.copy()

        if source is not None:
            b += dt * source

        # 检查边界
        b = np.clip(b, -1e25, 1e25)

        if use_cg and self._is_symmetric_positive_definite(A):
            n_new = conjugate_gradient(A, b, max_iter=min(nz * 2, 1000),
                                       tol=1e-12)
        else:
            # 使用 GMRES
            nz_est = nz * 3
            sparse_A = SparseMatrixCRS(nz, nz_est)
            sparse_A.from_dense(A)
            n_new = mgmres(sparse_A.a, sparse_A.ia, sparse_A.ja,
                           n_old.copy(), b, nz, sparse_A.nz_num,
                           itr_max=200, mr=min(30, nz),
                           tol_abs=1e-12, tol_rel=1e-10)

        return np.clip(n_new, 0.0, 1e25)

    def _is_symmetric_positive_definite(self, A: np.ndarray,
                                         tol: float = 1e-10) -> bool:
        """
        检查矩阵是否近似对称正定
        """
        if A.shape[0] != A.shape[1]:
            return False
        # 检查对称性
        asym = np.max(np.abs(A - A.T))
        if asym > tol * np.max(np.abs(A)):
            return False
        # 检查正定性 (通过特征值或尝试 Cholesky)
        try:
            np.linalg.cholesky(A + 1e-12 * np.eye(A.shape[0]))
            return True
        except np.linalg.LinAlgError:
            return False

    def compute_flux_divergence(self, n: np.ndarray) -> np.ndarray:
        """
        计算通量散度 ∂/∂z(Kzz ∂n/∂z) - w ∂n/∂z
        """
        nz = self.nz
        div = np.zeros(nz)

        # 扩散通量散度
        for i in range(1, nz - 1):
            dz_i = self.z[i] - self.z[i - 1]
            dz_ip1 = self.z[i + 1] - self.z[i]
            dz_mid = 0.5 * (dz_i + dz_ip1)

            flux_p = self.K_face[i] * (n[i + 1] - n[i]) / dz_ip1
            flux_m = self.K_face[i - 1] * (n[i] - n[i - 1]) / dz_i
            div[i] = (flux_p - flux_m) / dz_mid

        # 平流通量散度
        for i in range(1, nz):
            dz_i = self.z[i] - self.z[i - 1]
            if self.w[i] > 0:
                div[i] -= self.w[i] * (n[i] - n[i - 1]) / dz_i
            elif i < nz - 1 and self.w[i] < 0:
                dz_ip1 = self.z[i + 1] - self.z[i]
                div[i] -= self.w[i] * (n[i + 1] - n[i]) / dz_ip1

        return div
