"""
spectral_solver.py
==================
基于 Vandermonde 矩阵的谱方法求解器。

融合种子项目：
  - 1004_r8vm : Vandermonde 矩阵的紧凑存储、求解、行列式计算等

科学应用：
  在非线性声学冲击波模拟中，使用谱方法进行高阶空间离散。
  Vandermonde 矩阵用于多项式插值和谱微分矩阵构造。
  采用 Chebyshev/Legendre 节点避免 Runge 现象。
"""

import numpy as np
from numpy.polynomial.legendre import leggauss
from numpy.polynomial.chebyshev import chebgauss


class VandermondeSolver:
    """
    封装 Vandermonde 矩阵相关运算。

    原始算法来自 1004_r8vm/r8vm_sl.m 等。
    R8VM 格式存储 Vandermonde 矩阵的第二行向量 x(1:n)，
    矩阵形式为 A[i,j] = x[j]^(i-1)。
    """

    def __init__(self, nodes):
        """
        Parameters
        ----------
        nodes : np.ndarray, shape (n,)
            Vandermonde 矩阵定义向量 x。
        """
        self.nodes = np.asarray(nodes, dtype=float).flatten()
        self.n = self.nodes.size
        if self.n < 2:
            raise ValueError("Vandermonde matrix dimension must be at least 2.")
        # 检查显式奇异性（重复节点）
        if len(np.unique(np.round(self.nodes, 12))) < self.n:
            raise ValueError("Vandermonde nodes contain duplicates (near-singular).")

    def solve(self, b):
        """
        求解 Vandermonde 线性系统 A * x = b。

        算法基于 Björck-Pereyra 算法，复杂度 O(n^2)。
        原始代码：r8vm_sl.m

        Parameters
        ----------
        b : np.ndarray, shape (n,) or (n, 1)
            右端向量。

        Returns
        -------
        np.ndarray, shape (n,)
            解向量 x。
        """
        b = np.asarray(b, dtype=float).flatten()
        if b.size != self.n:
            raise ValueError("b size must match nodes size.")

        a = self.nodes.copy()
        x = b.copy()

        # 检查显式奇异性
        for j in range(self.n - 1):
            for i in range(j + 1, self.n):
                if np.isclose(a[i], a[j]):
                    raise ValueError("Vandermonde matrix is singular: duplicate nodes.")

        # Björck-Pereyra 算法
        for j in range(self.n - 1):
            for i in range(self.n - 1, j, -1):
                x[i] -= a[j] * x[i - 1]

        for j in range(self.n - 2, -1, -1):
            for i in range(j + 1, self.n):
                x[i] /= (a[i] - a[i - j - 1])
            for i in range(j, self.n - 1):
                x[i] -= x[i + 1]

        return x

    def determinant(self):
        """
        计算 Vandermonde 行列式。

        .. math::
            \det(V) = \prod_{1 \le i < j \le n} (x_j - x_i)

        原始代码：r8vm_det.m

        Returns
        -------
        float
            行列式值。
        """
        det_val = 1.0
        for i in range(self.n):
            for j in range(i + 1, self.n):
                det_val *= (self.nodes[j] - self.nodes[i])
        return det_val

    def to_dense(self):
        """
        展开为稠密矩阵。

        Returns
        -------
        np.ndarray, shape (n, n)
            稠密 Vandermonde 矩阵。
        """
        V = np.vander(self.nodes, N=self.n, increasing=True)
        return V

    def apply_mv(self, v):
        """
        矩阵-向量乘法 V * v。

        原始代码：r8vm_mv.m，使用 Horner 法则，O(n^2)。

        Parameters
        ----------
        v : np.ndarray, shape (n,)
            向量。

        Returns
        -------
        np.ndarray, shape (n,)
            结果向量。
        """
        v = np.asarray(v, dtype=float).flatten()
        if v.size != self.n:
            raise ValueError("v size must match nodes size.")

        y = np.zeros(self.n, dtype=float)
        for i in range(self.n):
            # Horner 法则求多项式值
            p = v[self.n - 1]
            for j in range(self.n - 2, -1, -1):
                p = p * self.nodes[i] + v[j]
            y[i] = p
        return y


class SpectralDifferentiator:
    """
    基于多项式插值的谱微分矩阵构造器。

    科学背景：
      在谱方法中，微分算子通过微分矩阵 D 离散化：

      .. math::
          u'(x_i) \approx \sum_{j=0}^{N} D_{ij} u(x_j)

      对于 Legendre-Gauss-Lobatto (LGL) 节点，D 的显式公式为：

      .. math::
          D_{ij} = \frac{L_N(x_i)}{L_N(x_j)} \frac{1}{x_i - x_j}, \quad i \ne j

      .. math::
          D_{ii} = -\sum_{j \ne i} D_{ij}

      其中 :math:`L_N` 为 N 阶 Legendre 多项式。
    """

    def __init__(self, n, node_type='legendre_gauss_lobatto'):
        """
        Parameters
        ----------
        n : int
            节点数（多项式次数为 n-1）。
        node_type : str
            'legendre_gauss_lobatto', 'chebyshev_gauss_lobatto', 'legendre_gauss'。
        """
        self.n = int(n)
        if self.n < 2:
            raise ValueError("n must be at least 2.")
        self.node_type = node_type
        self.nodes = self._compute_nodes()
        self.differentiation_matrix = self._compute_dm()

    def _compute_nodes(self):
        """
        计算谱节点。

        Returns
        -------
        np.ndarray, shape (n,)
            节点坐标在 [-1, 1]。
        """
        if self.node_type == 'legendre_gauss':
            x, _ = leggauss(self.n)
            return x
        elif self.node_type == 'legendre_gauss_lobatto':
            # LGL 节点：Gauss-Legendre 节点加上 ±1
            if self.n == 2:
                return np.array([-1.0, 1.0])
            # 使用 Newton 法求 LGL 节点
            x, _ = leggauss(self.n - 2)
            nodes = np.concatenate([[-1.0], x, [1.0]])
            # 通过 Legendre 多项式导数零点精化
            nodes = self._refine_lgl_nodes(nodes)
            return np.sort(nodes)
        elif self.node_type == 'chebyshev_gauss_lobatto':
            # CGL 节点: x_j = cos(pi * j / (n-1))
            j = np.arange(self.n)
            return np.cos(np.pi * j / (self.n - 1))
        else:
            raise ValueError(f"Unknown node_type: {self.node_type}")

    def _refine_lgl_nodes(self, nodes, max_iter=10, tol=1e-14):
        """
        使用 Newton 迭代精化 LGL 节点。
        """
        from numpy.polynomial.legendre import legval, legder
        for _ in range(max_iter):
            # Legendre 多项式 P_{n-1}
            c = np.zeros(self.n)
            c[-1] = 1.0
            P = legval(nodes, c)
            dP = legval(nodes, legder(c))
            # LGL 条件: (1-x^2) P'_{n-1}(x) = 0
            # 内部节点满足 P'_{n-1}(x) = 0
            update = np.zeros_like(nodes)
            mask = np.abs(nodes) < 1.0 - 1e-12
            update[mask] = P[mask] / dP[mask]
            nodes = nodes - update
            if np.max(np.abs(update[mask])) < tol:
                break
        return np.clip(nodes, -1.0, 1.0)

    def _compute_dm(self):
        """
        构造谱微分矩阵 D。

        Returns
        -------
        np.ndarray, shape (n, n)
            微分矩阵。
        """
        n = self.n
        x = self.nodes
        D = np.zeros((n, n), dtype=float)

        if self.node_type.startswith('chebyshev'):
            # Chebyshev 微分矩阵
            c = np.ones(n)
            c[0] = 2.0
            c[-1] = 2.0
            for i in range(n):
                for j in range(n):
                    if i != j:
                        D[i, j] = (c[i] / c[j]) * ((-1) ** (i + j)) / (x[i] - x[j])
            for i in range(1, n - 1):
                D[i, i] = -x[i] / (2.0 * (1.0 - x[i] ** 2))
            D[0, 0] = (2.0 * (n - 1) ** 2 + 1.0) / 6.0
            D[-1, -1] = -D[0, 0]
        else:
            # Legendre 微分矩阵（通用多项式插值公式）
            # 使用重心坐标公式
            w = np.ones(n)
            if self.node_type == 'legendre_gauss_lobatto':
                # LGL 重心权重
                from numpy.polynomial.legendre import legval
                c = np.zeros(n)
                c[-1] = 1.0
                Pn_1 = legval(x, c)
                w = 1.0 / (np.ones(n) - x ** 2)
                w[0] = 0.5 * n * (n - 1)
                w[-1] = 0.5 * n * (n - 1)
                mask = (np.abs(x) < 1.0 - 1e-12)
                w[mask] = 1.0 / (Pn_1[mask] ** 2)
            else:
                # LG 节点：近似权重
                for j in range(n):
                    prod = 1.0
                    for k in range(n):
                        if k != j:
                            prod *= (x[j] - x[k])
                    w[j] = 1.0 / prod

            for i in range(n):
                for j in range(n):
                    if i != j:
                        D[i, j] = (w[j] / w[i]) / (x[i] - x[j])
            # 对角元由行和为零确定（对于常数函数的导数为零）
            for i in range(n):
                D[i, i] = -np.sum(D[i, :]) + D[i, i]

        return D

    def differentiate(self, u):
        """
        对函数值向量 u 进行谱微分。

        Parameters
        ----------
        u : np.ndarray, shape (n,)
            节点上的函数值。

        Returns
        -------
        np.ndarray, shape (n,)
            节点上的导数值。
        """
        u = np.asarray(u, dtype=float)
        if u.size != self.n:
            raise ValueError("u size must match n.")
        return self.differentiation_matrix @ u

    def second_derivative_matrix(self):
        """
        二阶微分矩阵 D^2。

        Returns
        -------
        np.ndarray, shape (n, n)
        """
        return self.differentiation_matrix @ self.differentiation_matrix


def map_nodes_to_interval(nodes, a, b):
    """
    将 [-1, 1] 上的节点映射到 [a, b]。

    Parameters
    ----------
    nodes : np.ndarray
        原节点。
    a, b : float
        目标区间。

    Returns
    -------
    np.ndarray
        映射后的节点。
    np.ndarray
        对应的 Jacobian 因子 dx/dxi（用于链式法则）。
    """
    if b <= a:
        raise ValueError("b must be greater than a.")
    x_mapped = 0.5 * (b - a) * nodes + 0.5 * (a + b)
    # TODO: 修复 Jacobian 因子（区间映射的链式法则）
    jacobian = 1.0
    return x_mapped, jacobian


def solve_burgers_spectral_1d(u0_func, x_a, x_b, N, t_span, nu,
                               n_time_steps=1000, node_type='legendre_gauss_lobatto'):
    """
    使用 Legendre/Chebyshev 谱方法求解 1D Burgers 方程。

    .. math::
        u_t + u u_x = \nu u_{xx}

    Parameters
    ----------
    u0_func : callable
        初始条件函数 u0(x)。
    x_a, x_b : float
        空间区间。
    N : int
        谱节点数。
    t_span : tuple(float, float)
        时间区间 (t0, tf)。
    nu : float
        粘性系数。
    n_time_steps : int
        时间步数。
    node_type : str
        节点类型。

    Returns
    -------
    np.ndarray, shape (n_time_steps+1, N)
        每个时间步的解。
    np.ndarray, shape (N,)
        空间节点。
    np.ndarray, shape (n_time_steps+1,)
        时间向量。
    """
    spec = SpectralDifferentiator(N, node_type=node_type)
    xi = spec.nodes
    x, jac = map_nodes_to_interval(xi, x_a, x_b)

    D = spec.differentiation_matrix / jac
    D2 = spec.second_derivative_matrix() / (jac ** 2)

    u = u0_func(x)
    if np.any(~np.isfinite(u)):
        raise ValueError("Initial condition produced non-finite values.")

    t0, tf = t_span
    dt = (tf - t0) / n_time_steps
    t_vec = np.linspace(t0, tf, n_time_steps + 1)

    # 存储解
    U = np.zeros((n_time_steps + 1, N), dtype=float)
    U[0, :] = u

    # 时间推进：RK4
    for n_step in range(n_time_steps):
        # 右端项: -u * u_x + nu * u_xx
        def rhs(v):
            v = np.asarray(v, dtype=float)
            v_x = D @ v
            v_xx = D2 @ v
            return -v * v_x + nu * v_xx

        k1 = rhs(u)
        k2 = rhs(u + 0.5 * dt * k1)
        k3 = rhs(u + 0.5 * dt * k2)
        k4 = rhs(u + dt * k3)

        u = u + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        # 边界处理：Dirichlet u=0 在两端
        u[0] = 0.0
        u[-1] = 0.0

        # 数值稳定性截断
        u_max = 10.0 * np.max(np.abs(U[0, :]))
        u = np.clip(u, -u_max, u_max)

        U[n_step + 1, :] = u

    return U, x, t_vec
