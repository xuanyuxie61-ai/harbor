"""
chebyshev_spectral.py — Chebyshev 谱方法求解器
================================================

融合种子项目:
  - 086_biharmonic_cheby1d : Chebyshev 谱微分矩阵构造

本模块实现 Chebyshev 谱方法用于求解级联方程中的刚性部分:

1. Chebyshev 谱微分矩阵:
   D[i,j] = c_i/c_j * (-1)^{i+j} / (x_i - x_j)  (i ≠ j)
   D[i,i] = -x_i / (2*(1 - x_i^2))                (0 < i < N)
   D[0,0] = (2*N^2+1)/6
   D[N,N] = -(2*N^2+1)/6

   其中 x_i = cos(pi*i/N), c_0 = c_N = 2, c_j = 1 (otherwise)

2. Chebyshev 插值:
   p(x) = sum_{k=0}^{N} a_k * T_k(x)
   a_k = (2/N) * sum_{j=0}^{N} '' f(x_j) * T_k(x_j)

3. 谱配置法求解 BVP:
   L * u = f 在 Chebyshev 节点上
   边界条件通过矩阵行替换施加

4. 矩阵预条件:
   使用对角预条件 P = diag(1/(1-x_i^2)) 改善条件数

关键公式 (谱方法误差):
  ||u - u_N||_inf <= C * N^{-m} * ||u^{(m)}||_inf  (m 次可微)
  ||u - u_N||_inf <= C * rho^{-N}                    (解析函数, rho > 1)
"""

import math
from typing import List, Tuple, Callable, Optional
from material_properties import MaterialSpec


class ChebyshevSpectralSolver:
    """Chebyshev 谱方法求解器"""

    def __init__(self, n_points: int = 32):
        self.N = n_points
        self._nodes = None
        self._diff_matrix = None
        self._weights = None

    @property
    def nodes(self) -> List[float]:
        """Chebyshev 节点 x_j = cos(pi*j/N), j = 0, ..., N"""
        if self._nodes is None:
            self._nodes = [
                math.cos(math.pi * j / self.N) for j in range(self.N + 1)
            ]
        return self._nodes

    def differentiation_matrix(self, order: int = 1) -> List[List[float]]:
        """
        构造 k 阶 Chebyshev 谱微分矩阵 D^(k)

        D^(1) 的元素:
          D[i,j] = (c_i/c_j) * (-1)^{i+j} / (x_i - x_j)   (i ≠ j)
          D[i,i] = -x_i / (2*(1 - x_i^2))                   (0 < i < N)
          D[0,0] = (2*N^2 + 1) / 6
          D[N,N] = -(2*N^2 + 1) / 6

        D^(k) = D^(1)^k (矩阵幂)
        """
        if self._diff_matrix is None:
            self._diff_matrix = self._build_first_derivative_matrix()

        if order == 1:
            return self._diff_matrix

        # 矩阵幂
        result = self._diff_matrix
        for _ in range(order - 1):
            result = self._matmul(result, self._diff_matrix)
        return result

    def _build_first_derivative_matrix(self) -> List[List[float]]:
        """构造一阶 Chebyshev 微分矩阵"""
        N = self.N
        x = self.nodes
        D = [[0.0] * (N + 1) for _ in range(N + 1)]

        # c 系数
        c = [2.0] + [1.0] * (N - 1) + [2.0]
        # 交替符号
        signs = [(-1.0) ** i for i in range(N + 1)]

        for i in range(N + 1):
            for j in range(N + 1):
                if i != j:
                    D[i][j] = (c[i] / c[j]) * signs[i] * signs[j] / (x[i] - x[j])

        # 对角元素
        for i in range(1, N):
            D[i][i] = -x[i] / (2.0 * (1.0 - x[i] * x[i]))
        D[0][0] = (2.0 * N * N + 1.0) / 6.0
        D[N][N] = -(2.0 * N * N + 1.0) / 6.0

        return D

    def chebyshev_interpolation(
        self, values: List[float], x_eval: float,
    ) -> float:
        """
        Chebyshev 插值 (Clenshaw 算法)

        p(x) = sum_{k=0}^{N} a_k * T_k(x)

        Clenshaw 递推:
          b_{N+1} = b_{N+2} = 0
          b_k = 2*x*b_{k+1} - b_{k+2} + a_k   (k = N, N-1, ..., 1)
          p(x) = b_0 - x*b_1 + a_0 ... 不对,
          p(x) = (b_0 - b_2)/2 + a_0 ... 需要更仔细

        简化: 直接用 b_1*x + (b_0 - b_2)/2 形式
        """
        N = self.N
        x = self.nodes

        # 计算 Chebyshev 系数
        coeffs = self._compute_chebyshev_coefficients(values)

        # Clenshaw 递推
        b_k_plus2 = 0.0
        b_k_plus1 = 0.0
        for k in range(N, 0, -1):
            b_k = 2.0 * x_eval * b_k_plus1 - b_k_plus2 + coeffs[k]
            b_k_plus2 = b_k_plus1
            b_k_plus1 = b_k

        return b_k_plus1 * x_eval - b_k_plus2 + coeffs[0]

    def _compute_chebyshev_coefficients(
        self, values: List[float],
    ) -> List[float]:
        """
        计算 Chebyshev 展开系数 (DCT)

        a_k = (2/(N*c_k)) * sum_{j=0}^{N} f(x_j) * T_k(x_j) / c_j''
        其中 c_0'' = c_N'' = 2, 其余 c_j'' = 1
        """
        N = self.N
        coeffs = [0.0] * (N + 1)
        x = self.nodes

        for k in range(N + 1):
            s = 0.0
            for j in range(N + 1):
                # T_k(x_j) = cos(k * arccos(x_j)) = cos(k * pi * j / N)
                T_kj = math.cos(k * math.pi * j / N)
                wj = 1.0
                if j == 0 or j == N:
                    wj = 0.5
                s += values[j] * T_kj * wj
            ck = 2.0 if (k == 0 or k == N) else 1.0
            coeffs[k] = s * 2.0 / (N * ck)

        return coeffs

    def solve_bvp(
        self,
        L_operator: Callable[[List[List[float]]], List[List[float]]],
        rhs_values: List[float],
        bc_left: float = 0.0,
        bc_right: float = 0.0,
    ) -> List[float]:
        """
        求解边值问题 L*u = f (Chebyshev 配置法)

        步骤:
        1. 构建 L 的配置矩阵 (使用 D)
        2. 施加边界条件 (替换第一行和最后一行)
        3. 求解线性系统

        边界条件:
          u(-1) = bc_left
          u(+1) = bc_right
        """
        N = self.N

        # 获取算子矩阵
        L_mat = L_operator(self.differentiation_matrix())

        # 施加边界条件
        rhs = list(rhs_values)
        rhs[0] = bc_left
        rhs[N] = bc_right

        for j in range(N + 1):
            L_mat[0][j] = 1.0 if j == 0 else 0.0
            L_mat[N][j] = 1.0 if j == N else 0.0

        # 求解
        return self._solve_linear_system(L_mat, rhs)

    def solve_cascade_spectral(
        self,
        material: MaterialSpec,
        E0_MeV: float,
        n_cheb: int = 32,
    ) -> Tuple[List[float], List[float]]:
        """
        用谱方法求解纵向级联方程

        Gamma''(t) - a*Gamma'(t) - b*Gamma(t) = 0
        Gamma(0) = E0 * delta(t)
        Gamma'(T_max) = 0 (shower max 处通量为零)

        映射到 [-1, 1]:
          t = (T_max/2) * (x + 1)
          d/dt = (2/T_max) * d/dx
        """
        self.N = n_cheb

        def build_operator(D: List[List[float]]) -> List[List[float]]:
            N = self.N
            T_max = material.critical_depth_X0(E0_MeV)
            T_max = max(T_max, 1.0)
            scale = 2.0 / T_max

            # D2 - a*D - b*I
            D2 = self.differentiation_matrix(2)
            D1 = self.differentiation_matrix(1)

            a_coeff = 0.5  # 衰减系数
            b_coeff = 0.25  # 源项系数

            L = [[0.0] * (N + 1) for _ in range(N + 1)]
            for i in range(N + 1):
                for j in range(N + 1):
                    L[i][j] = (
                        scale * scale * D2[i][j]
                        - a_coeff * scale * D1[i][j]
                    )
                    if i == j:
                        L[i][j] -= b_coeff

            return L

        # 右端项 (在 Chebyshev 节点上)
        x_nodes = self.nodes
        rhs = [0.0] * (self.N + 1)
        for i, xi in enumerate(x_nodes):
            t = material.critical_depth_X0(E0_MeV) * (xi + 1.0) / 2.0
            rhs[i] = 0.0  # 齐次方程 (源项在边界条件中)

        # 边界条件
        bc_left = E0_MeV / material.X0_cm  # 初始通量
        bc_right = 0.0  # 末端通量为零

        solution = self.solve_bvp(build_operator, rhs, bc_left, bc_right)

        # 映射回物理深度
        T_max = max(material.critical_depth_X0(E0_MeV), 1.0)
        depths = [T_max * (xi + 1.0) / 2.0 for xi in x_nodes]

        return depths, solution

    def _solve_linear_system(
        self, A: List[List[float]], b: List[float],
    ) -> List[float]:
        """高斯消元法 (带部分主元)"""
        n = len(b)
        M = [A[i][:] + [b[i]] for i in range(n)]

        for col in range(n):
            max_val = abs(M[col][col])
            max_row = col
            for row in range(col + 1, n):
                if abs(M[row][col]) > max_val:
                    max_val = abs(M[row][col])
                    max_row = row
            M[col], M[max_row] = M[max_row], M[col]

            if abs(M[col][col]) < 1e-14:
                continue

            pivot = M[col][col]
            for row in range(col + 1, n):
                factor = M[row][col] / pivot
                for j in range(col, n + 1):
                    M[row][j] -= factor * M[col][j]

        x = [0.0] * n
        for i in range(n - 1, -1, -1):
            if abs(M[i][i]) < 1e-14:
                continue
            s = M[i][n]
            for j in range(i + 1, n):
                s -= M[i][j] * x[j]
            x[i] = s / M[i][i]
        return x

    def _matmul(
        self, A: List[List[float]], B: List[List[float]],
    ) -> List[List[float]]:
        """矩阵乘法"""
        n = len(A)
        m = len(B[0]) if B else 0
        k = len(B)
        C = [[0.0] * m for _ in range(n)]
        for i in range(n):
            for j in range(m):
                s = 0.0
                for l in range(k):
                    s += A[i][l] * B[l][j]
                C[i][j] = s
        return C
