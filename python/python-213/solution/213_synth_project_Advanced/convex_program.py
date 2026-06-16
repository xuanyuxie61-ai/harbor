"""
convex_program.py - 凸优化问题建模模块
=======================================

融合种子项目:
  - 339_eternity: LP 稀疏矩阵建模
  - 882_polygon: 凸多边形几何与包含检测

核心数学: 凸优化标准形式

本模块将科学计算问题转化为标准凸优化形式:

  标准形式 (P):
    min  c^T x + 0.5 * x^T Q x
    s.t. A x = b       (线性等式约束 - 来自 PDE 离散化)
         l <= x <= u    (变量界约束)
         x in K         (锥约束 - 可选)

  对偶形式 (D):
    max  b^T y - 0.5 * (c - A^T y)^T Q^{-1} (c - A^T y)
         - l^T z^- + u^T z^+
    s.t. A^T y + z^- - z^+ = c
         z^-, z^+ >= 0

  KKT 最优性条件:
    1. 原始可行性:  A x = b, l <= x <= u
    2. 对偶可行性:  A^T y + z^- - z^+ = c
    3. 互补松弛:    z^-_i (x_i - l_i) = 0, z^+_i (u_i - x_i) = 0
    4. 强对偶性:    c^T x = b^T y + (对偶目标)

  内点法将不等式约束通过障碍函数松弛:
    min c^T x + 0.5 x^T Q x - mu sum [ln(x_i - l_i) + ln(u_i - x_i)]
    s.t. Ax = b

  对应的 KKT 系统 (Newton 步):
    [ Q + Theta    A^T ] [ dx ]   [ r_c ]
    [ A            0   ] [ dy ] = [ r_p ]

  其中 Theta = Z^- (X-L)^{-1} + Z^+ (U-X)^{-1} 是障碍 Hessian.

多边形约束:
  当可行域是多边形 Omega 时, FEM 离散化在 Omega 上进行.
  多边形凸性检测确保障碍函数的良好性质.
"""

import numpy as np
from scipy import sparse
from typing import Tuple, Optional, Dict, Any


class ConvexProgram:
    """
    凸优化问题的标准形式建模.

    支持:
      - 二次目标 (Q 半正定)
      - 线性等式约束 Ax = b
      - 变量界约束 l <= x <= u
      - 稀疏矩阵存储 (来自 eternity LP 建模思想)

    Attributes
    ----------
    n : int
        变量维度
    m : int
        等式约束个数
    c : ndarray
        线性目标系数
    Q : sparse matrix or None
        二次目标 Hessian (稀疏)
    A : sparse matrix
        等式约束矩阵
    b : ndarray
        等式约束右端
    lb : ndarray
        下界 (-inf 表示无界)
    ub : ndarray
        上界 (+inf 表示无界)
    """

    def __init__(self, n: int, m: int):
        """
        初始化凸优化问题.

        Parameters
        ----------
        n : int
            变量维度
        m : int
            等式约束个数
        """
        self.n = n
        self.m = m
        self.c = np.zeros(n)
        self.Q = None
        self.A = sparse.csc_matrix((m, n))
        self.b = np.zeros(m)
        self.lb = np.full(n, -np.inf)
        self.ub = np.full(n, np.inf)
        self._info: Dict[str, Any] = {}

    def set_objective(self, c: np.ndarray,
                       Q: Optional[sparse.spmatrix] = None):
        """设置目标函数 f(x) = c^T x + 0.5 x^T Q x."""
        self.c = c.copy()
        if Q is not None:
            self.Q = sparse.csc_matrix(Q)
            # 验证半正定性 (通过对称化)
            self.Q = 0.5 * (self.Q + self.Q.T)

    def set_equality_constraints(self, A: sparse.spmatrix, b: np.ndarray):
        """设置等式约束 Ax = b."""
        assert A.shape == (self.m, self.n), \
            f"A 的形状 {A.shape} 不匹配 (m={self.m}, n={self.n})"
        self.A = sparse.csc_matrix(A)
        self.b = b.copy()

    def set_bounds(self, lb: np.ndarray, ub: np.ndarray):
        """设置变量界约束 l <= x <= u."""
        self.lb = lb.copy()
        self.ub = ub.copy()
        # 验证有界性
        self._bounded_mask = np.isfinite(self.lb) & np.isfinite(self.ub)
        self._lower_only = np.isfinite(self.lb) & ~np.isfinite(self.ub)
        self._upper_only = ~np.isfinite(self.lb) & np.isfinite(self.ub)

    def check_strict_feasibility(self, x: np.ndarray,
                                  slack_tol: float = 1e-8) -> bool:
        """
        检查点 x 是否严格可行 (Slater 条件).

        Slater 条件是强对偶性的充分条件:
          存在 x0 使得 Ax0 = b, l < x0 < u

        若 Slater 条件满足, 则:
          1. 强对偶性成立 (原始最优 = 对偶最优)
          2. KKT 点是充要条件 (凸问题)
          3. 内点法保证收敛

        Parameters
        ----------
        x : ndarray
            待检查的点
        slack_tol : float
            严格可行性的最小松弛量

        Returns
        -------
        bool
            是否严格可行
        """
        # 等式约束
        eq_residual = np.max(np.abs(self.A @ x - self.b))
        if eq_residual > slack_tol:
            return False

        # 不等式约束
        if np.any(x[self._bounded_mask] <= self.lb[self._bounded_mask] + slack_tol):
            return False
        if np.any(x[self._bounded_mask] >= self.ub[self._bounded_mask] - slack_tol):
            return False

        return True

    def compute_residuals(self, x: np.ndarray, y: np.ndarray,
                           z_lower: np.ndarray, z_upper: np.ndarray
                           ) -> Dict[str, float]:
        """
        计算 KKT 残差.

        原始残差 (等式可行性):
          r_p = b - Ax

        对偶残差 (梯度条件):
          r_d = c + Qx - A^T y - z^- + z^+

        互补残差:
          r_cl = X Z^- e  (下界)
          r_cu = (U-X) Z^+ e  (上界)

        互补间隙:
          mu = (sum x_i z^-_i + sum (u_i-x_i) z^+_i) / n_bounded

        Parameters
        ----------
        x : ndarray
            原始变量
        y : ndarray
            等式约束对偶变量
        z_lower : ndarray
            下界对偶变量
        z_upper : ndarray
            上界对偶变量

        Returns
        -------
        dict
            各残差的范数
        """
        # 原始残差
        r_p = self.b - self.A @ x

        # 对偶残差
        grad = self.c.copy()
        if self.Q is not None:
            grad = grad + self.Q @ x
        r_d = grad - self.A.T @ y - z_lower + z_upper

        # 互补残差
        bounded = self._bounded_mask
        r_cl = x[bounded] * z_lower[bounded]
        r_cu = (self.ub[bounded] - x[bounded]) * z_upper[bounded]

        # 互补间隙
        gap = (np.dot(x[bounded], z_lower[bounded]) +
               np.dot(self.ub[bounded] - x[bounded], z_upper[bounded]))
        n_bounded = max(np.sum(bounded), 1)
        mu = gap / n_bounded

        return {
            'primal_residual': float(np.linalg.norm(r_p, np.inf)),
            'dual_residual': float(np.linalg.norm(r_d, np.inf)),
            'complementarity_lower': float(np.linalg.norm(r_cl, np.inf)),
            'complementarity_upper': float(np.linalg.norm(r_cu, np.inf)),
            'duality_gap': float(mu),
            'objective': float(self.c @ x +
                               0.5 * (x @ (self.Q @ x) if self.Q is not None else 0.0))
        }

    def initialize_ipm(self) -> Tuple[np.ndarray, np.ndarray,
                                       np.ndarray, np.ndarray]:
        """
        初始化内点法的起始点.

        使用启发式初始化 (Mehrotra 策略):
          1. 求最小二乘可行点: x0 = A^T (AA^T)^{-1} b
          2. 将 x0 投影到严格内部
          3. 初始化对偶变量为零
          4. 初始化对偶松弛使互补间隙 > 0

        具体策略:
          若 x0_i - l_i > u_i - x0_i:
              z^-_i = 1, z^+_i = 1 + (x0_i - l_i) - (u_i - x0_i)
          否则:
              z^+_i = 1, z^-_i = 1 + (u_i - x0_i) - (x0_i - l_i)

        Returns
        -------
        x0, y0, z_lower0, z_upper0 : ndarray
            初始点
        """
        n, m = self.n, self.m

        # 最小二乘可行点 (通过法方程)
        # x0 = A^T (A A^T)^{-1} b
        A_dense = self.A.toarray()
        try:
            AAT = A_dense @ A_dense.T
            reg = 1.0e-10 * np.eye(m)
            y0 = np.linalg.solve(AAT + reg, self.b)
            x0 = A_dense.T @ y0
        except np.linalg.LinAlgError:
            x0 = np.zeros(n)

        y0 = np.zeros(m)

        # 投影到严格内部
        x0 = np.clip(x0, self.lb + 1.0, self.ub - 1.0)
        # 处理无界方向
        x0 = np.where(np.isfinite(self.lb), np.maximum(x0, self.lb + 1.0), x0)
        x0 = np.where(np.isfinite(self.ub), np.minimum(x0, self.ub - 1.0), x0)

        # 初始化对偶变量
        z_lower = np.zeros(n)
        z_upper = np.zeros(n)

        bounded = self._bounded_mask
        if np.any(bounded):
            slack_l = x0[bounded] - self.lb[bounded]
            slack_u = self.ub[bounded] - x0[bounded]

            # Mehrotra 启发式
            for i in range(np.sum(bounded)):
                idx = np.where(bounded)[0][i]
                if slack_l[i] > slack_u[i]:
                    z_lower[idx] = 1.0
                    z_upper[idx] = max(1.0, slack_l[i] - slack_u[i])
                else:
                    z_upper[idx] = 1.0
                    z_lower[idx] = max(1.0, slack_u[i] - slack_l[i])
        else:
            z_lower = np.ones(n)
            z_upper = np.ones(n)

        return x0, y0, z_lower, z_upper


class PolyhedralDomain:
    """
    凸多边形可行域 (融合 882_polygon).

    在 PDE 约束优化中, 空间域 Omega 通常是多边形.
    本模块提供:
      - 凸性检测 (确保障碍函数良好定义)
      - 面积计算 (用于归一化目标)
      - 点包含检测 (用于初始化和验证)
      - 三角剖分 (用于 FEM 离散化)

    数学基础:
      多边形 Omega 由顶点 v_1, ..., v_N 按逆时针排列.
      凸性等价于: 所有转角同号 (叉积正定).

      面积: A = 0.5 * |sum_{i} (x_i y_{i+1} - x_{i+1} y_i)|
      质心: C = (1/(6A)) * sum_{i} (v_i + v_{i+1}) * (x_i y_{i+1} - x_{i+1} y_i)
    """

    def __init__(self, vertices: np.ndarray):
        """
        Parameters
        ----------
        vertices : ndarray, shape (N, 2)
            多边形顶点, 逆时针排列
        """
        self.vertices = np.asarray(vertices, dtype=float)
        self.n_vertices = len(self.vertices)
        assert self.n_vertices >= 3, "多边形至少需要 3 个顶点"

        self._is_convex: Optional[bool] = None
        self._area: Optional[float] = None
        self._centroid: Optional[np.ndarray] = None

    def is_convex(self) -> bool:
        """
        检测多边形是否为凸集.

        算法: 检查所有相邻边的叉积符号是否一致.
        对顶点 v_i, 边 e_i = v_{i+1} - v_i:
          cross_i = e_i x e_{i+1} = e_i.x * e_{i+1}.y - e_i.y * e_{i+1}.x

        若所有 cross_i > 0 (或都 < 0), 则为凸.

        凸性对内点法至关重要:
          - 凸域保证 FEM 刚度矩阵正定
          - 凸域上的线性约束形成多面体锥, 障碍函数自 concord

        Returns
        -------
        bool
            是否为凸多边形
        """
        if self._is_convex is not None:
            return self._is_convex

        n = self.n_vertices
        v = self.vertices
        sign = 0

        for i in range(n):
            # 边向量
            e1 = v[(i + 1) % n] - v[i]
            e2 = v[(i + 2) % n] - v[(i + 1) % n]

            cross = e1[0] * e2[1] - e1[1] * e2[0]

            if abs(cross) > 1.0e-12:
                if sign == 0:
                    sign = 1 if cross > 0 else -1
                elif (cross > 0 and sign < 0) or (cross < 0 and sign > 0):
                    self._is_convex = False
                    return False

        self._is_convex = True
        return True

    def area(self) -> float:
        """
        计算多边形面积 (Shoelace 公式).

        A = 0.5 * |sum_{i=0}^{n-1} (x_i * y_{i+1} - x_{i+1} * y_i)|

        Returns
        -------
        float
            面积
        """
        if self._area is not None:
            return self._area

        n = self.n_vertices
        v = self.vertices
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += v[i, 0] * v[j, 1] - v[j, 0] * v[i, 1]
        self._area = 0.5 * abs(area)
        return self._area

    def centroid(self) -> np.ndarray:
        """
        计算多边形质心.

        C_x = (1/(6A)) * sum_{i} (x_i + x_{i+1}) * (x_i y_{i+1} - x_{i+1} y_i)
        C_y = (1/(6A)) * sum_{i} (y_i + y_{i+1}) * (x_i y_{i+1} - x_{i+1} y_i)

        Returns
        -------
        ndarray, shape (2,)
            质心坐标
        """
        if self._centroid is not None:
            return self._centroid

        n = self.n_vertices
        v = self.vertices
        A = self.area()

        if A < 1.0e-15:
            self._centroid = np.mean(v, axis=0)
            return self._centroid

        cx = 0.0
        cy = 0.0
        for i in range(n):
            j = (i + 1) % n
            cross = v[i, 0] * v[j, 1] - v[j, 0] * v[i, 1]
            cx += (v[i, 0] + v[j, 0]) * cross
            cy += (v[i, 1] + v[j, 1]) * cross

        factor = 1.0 / (6.0 * A)
        self._centroid = np.array([cx * factor, cy * factor])
        return self._centroid

    def contains_point(self, p: np.ndarray) -> bool:
        """
        检测点 p 是否在凸多边形内部 (融合 882_polygon 的三角形分解法).

        算法: 将凸多边形分解为三角形扇 (v_0, v_i, v_{i+1}),
              若 p 在任一三角形内, 则在多边形内.

        点在三角形内的判断使用重心坐标:
          p = alpha * v_0 + beta * v_i + gamma * v_{i+1}
          alpha + beta + gamma = 1
          若 alpha, beta, gamma >= 0, 则 p 在三角形内.

        公式:
          det = (v_i - v_0) x (v_{i+1} - v_0)
          beta = ((p - v_0) x (v_{i+1} - v_0)) / det
          gamma = ((v_i - v_0) x (p - v_0)) / det
          alpha = 1 - beta - gamma

        Parameters
        ----------
        p : ndarray, shape (2,)
            测试点

        Returns
        -------
        bool
            是否在多边形内
        """
        n = self.n_vertices
        v = self.vertices

        for i in range(1, n - 1):
            # 三角形 (v[0], v[i], v[i+1])
            if self._triangle_contains(v[0], v[i], v[i + 1], p):
                return True
        return False

    @staticmethod
    def _triangle_contains(a: np.ndarray, b: np.ndarray,
                            c: np.ndarray, p: np.ndarray) -> bool:
        """重心坐标法判断点是否在三角形内."""
        v0 = c - a
        v1 = b - a
        v2 = p - a

        dot00 = np.dot(v0, v0)
        dot01 = np.dot(v0, v1)
        dot02 = np.dot(v0, v2)
        dot11 = np.dot(v1, v1)
        dot12 = np.dot(v1, v2)

        denom = dot00 * dot11 - dot01 * dot01
        if abs(denom) < 1.0e-15:
            return False

        inv_denom = 1.0 / denom
        u = (dot11 * dot02 - dot01 * dot12) * inv_denom
        v = (dot00 * dot12 - dot01 * dot02) * inv_denom

        return (u >= -1.0e-12) and (v >= -1.0e-12) and (u + v <= 1.0 + 1.0e-12)

    def triangulate(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        简单扇形三角剖分 (适用于凸多边形).

        Returns
        -------
        vertices : ndarray
            顶点坐标, shape (N, 2)
        triangles : ndarray
            三角形索引, shape (T, 3)
        """
        n = self.n_vertices
        triangles = np.array([[0, i, i + 1] for i in range(1, n - 1)])
        return self.vertices.copy(), triangles
