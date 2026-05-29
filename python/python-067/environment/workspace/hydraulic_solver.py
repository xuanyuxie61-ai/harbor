# -*- coding: utf-8 -*-
"""
hydraulic_solver.py
裂隙介质水力压力场求解模块

融合种子项目：
    - 451_gauss_seidel: 高斯-赛德尔迭代
    - 965_r83s: 三对角矩阵的 Gauss-Seidel 与 共轭梯度求解

在裂隙介质渗流模拟中，稳态水头场满足以下控制方程：
    
    裂隙网络中的质量守恒（节点方程）：
        Σ_j Q_{ij} = Q_source,i
    
    其中裂隙段流量（Cubic Law + Darcy）：
        Q_{ij} = T_{ij} * (h_i - h_j) / L_{ij}
    
    对于规则网格，可离散化为 Poisson 方程：
        ∇·(T ∇h) = -q_s
    
    即：
        ∂/∂x(T_x ∂h/∂x) + ∂/∂y(T_y ∂h/∂y) = -q_s

数值离散（五点差分格式）：
    T_{i+1/2,j} (h_{i+1,j} - h_{i,j}) / Δx² 
  - T_{i-1/2,j} (h_{i,j} - h_{i-1,j}) / Δx²
  + T_{i,j+1/2} (h_{i,j+1} - h_{i,j}) / Δy²
  - T_{i,j-1/2} (h_{i,j} - h_{i,j-1}) / Δy² = -q_{s,i,j}

边界条件：
    - Dirichlet: h = h_0 (定水头)
    - Neumann: ∂h/∂n = q_n / T (定流量)
"""

import numpy as np
from typing import Tuple, Optional, Callable


class HydraulicSolver:
    """
    裂隙介质水力压力场求解器

    采用有限差分离散和迭代方法求解稳态渗流方程。
    支持 Gauss-Seidel、Jacobi 和共轭梯度等多种求解算法。
    """

    def __init__(self, nx: int, ny: int, dx: float, dy: float):
        """
        Parameters
        ----------
        nx, ny : int
            网格数
        dx, dy : float
            网格间距 [m]
        """
        if nx <= 0 or ny <= 0:
            raise ValueError("nx 和 ny 必须为正")
        if dx <= 0 or dy <= 0:
            raise ValueError("dx 和 dy 必须为正")
        self.nx = nx
        self.ny = ny
        self.dx = dx
        self.dy = dy
        self.head = np.zeros((ny, nx))
        self.source = np.zeros((ny, nx))

    def _build_tridiagonal_system(self, T: np.ndarray,
                                  axis: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        构建一维三对角线性系统（用于行/列扫描）

        基于 r83s 存储格式：a_sub, a_diag, a_sup
        
        对于内部节点 i，方程为：
            a_sub * h_{i-1} + a_diag * h_i + a_sup * h_{i+1} = b_i

        Parameters
        ----------
        T : np.ndarray
            传导系数场 (ny, nx)
        axis : int
            0 为沿 y 方向（行扫描），1 为沿 x 方向（列扫描）

        Returns
        -------
        tuple
            (a_sub, a_diag, a_sup, rhs)
        """
        if axis == 0:
            n = self.nx
            d = self.dx
            m = self.ny
        else:
            n = self.ny
            d = self.dy
            m = self.nx

        a_sub = np.zeros(n)
        a_diag = np.zeros(n)
        a_sup = np.zeros(n)
        rhs = np.zeros(n)

        # 这里简化处理：构建均匀系数的近似三对角系统
        # 实际应用中应根据 T 场精确计算界面传导系数
        for i in range(1, n - 1):
            a_sub[i] = -1.0 / (d ** 2)
            a_diag[i] = 2.0 / (d ** 2)
            a_sup[i] = -1.0 / (d ** 2)

        # 边界处理（Dirichlet）
        a_diag[0] = 1.0
        a_sup[0] = 0.0
        a_sub[-1] = 0.0
        a_diag[-1] = 1.0

        return a_sub, a_diag, a_sup, rhs

    def solve_gauss_seidel(self, T: np.ndarray,
                           h_boundary: dict = None,
                           max_iter: int = 10000,
                           tol: float = 1.0e-8,
                           omega: float = 1.0) -> np.ndarray:
        """
        使用逐次超松弛 Gauss-Seidel 迭代求解水头场

        基于 gauss_seidel1 和 r83s_gs_sl 的算法融合：
            h_i^{(k+1)} = (1-ω) h_i^{(k)} + ω * (b_i - Σ_{j<i} a_{ij} h_j^{(k+1)} - Σ_{j>i} a_{ij} h_j^{(k)}) / a_{ii}

        Parameters
        ----------
        T : np.ndarray
            传导系数场 (ny, nx)
        h_boundary : dict
            边界条件 {'left': val, 'right': val, 'top': val, 'bottom': val}
        max_iter : int
            最大迭代次数
        tol : float
            收敛容差
        omega : float
            松弛因子 (0 < ω < 2)

        Returns
        -------
        np.ndarray
            水头场 (ny, nx)
        """
        if omega <= 0 or omega >= 2:
            raise ValueError("omega 必须在 (0, 2) 区间内")
        if T.shape != (self.ny, self.nx):
            raise ValueError("T 的形状必须与网格匹配")

        if h_boundary is None:
            h_boundary = {'left': 10.0, 'right': 0.0, 'top': 5.0, 'bottom': 5.0}

        h = np.zeros((self.ny, self.nx))
        # 初始化边界
        h[:, 0] = h_boundary.get('left', 0.0)
        h[:, -1] = h_boundary.get('right', 0.0)
        h[0, :] = h_boundary.get('top', 0.0)
        h[-1, :] = h_boundary.get('bottom', 0.0)

        # 界面传导系数（调和平均）
        Tx = np.zeros((self.ny, self.nx + 1))
        Ty = np.zeros((self.ny + 1, self.nx))

        Tx[:, 1:-1] = 2.0 * T[:, :-1] * T[:, 1:] / (T[:, :-1] + T[:, 1:] + 1e-20)
        Ty[1:-1, :] = 2.0 * T[:-1, :] * T[1:, :] / (T[:-1, :] + T[1:, :] + 1e-20)

        for it in range(max_iter):
            h_old = h.copy()

            for i in range(1, self.ny - 1):
                for j in range(1, self.nx - 1):
                    # 五点差分格式
                    coeff = (Tx[i, j] + Tx[i, j+1]) / self.dx**2 + (Ty[i, j] + Ty[i+1, j]) / self.dy**2
                    if coeff < 1e-20:
                        continue

                    rhs = (Tx[i, j] * h[i, j-1] + Tx[i, j+1] * h[i, j+1]) / self.dx**2
                    rhs += (Ty[i, j] * h[i-1, j] + Ty[i+1, j] * h[i+1, j]) / self.dy**2
                    rhs -= self.source[i, j]

                    h_new = rhs / coeff
                    h[i, j] = (1.0 - omega) * h[i, j] + omega * h_new

            # 检查收敛
            diff = np.max(np.abs(h - h_old))
            if diff < tol:
                break

        self.head = h
        return h

    def solve_conjugate_gradient(self, T: np.ndarray,
                                  h_boundary: dict = None,
                                  max_iter: int = None,
                                  tol: float = 1.0e-10) -> np.ndarray:
        """
        使用共轭梯度法 (CG) 求解水头场

        基于 r83s_cg 的算法：
            求解 Ax = b，其中 A 为对称正定矩阵。
            
            初始化：r_0 = b - Ax_0, p_0 = r_0
            迭代：
                α_k = (r_k^T r_k) / (p_k^T A p_k)
                x_{k+1} = x_k + α_k p_k
                r_{k+1} = r_k - α_k A p_k
                β_k = (r_{k+1}^T r_{k+1}) / (r_k^T r_k)
                p_{k+1} = r_{k+1} + β_k p_k

        Parameters
        ----------
        T : np.ndarray
            传导系数场
        h_boundary : dict
            边界条件
        max_iter : int
            最大迭代次数（默认 nx*ny）
        tol : float
            残差容差

        Returns
        -------
        np.ndarray
            水头场
        """
        if T.shape != (self.ny, self.nx):
            raise ValueError("T 的形状必须与网格匹配")

        if h_boundary is None:
            h_boundary = {'left': 10.0, 'right': 0.0, 'top': 5.0, 'bottom': 5.0}

        if max_iter is None:
            max_iter = self.nx * self.ny

        n = self.nx * self.ny

        # 构建矩阵-向量乘法函数（避免显式存储大矩阵）
        def matvec(h_flat: np.ndarray) -> np.ndarray:
            h = h_flat.reshape((self.ny, self.nx))
            Ah = np.zeros_like(h)

            Tx = np.zeros((self.ny, self.nx + 1))
            Ty = np.zeros((self.ny + 1, self.nx))
            Tx[:, 1:-1] = 2.0 * T[:, :-1] * T[:, 1:] / (T[:, :-1] + T[:, 1:] + 1e-20)
            Ty[1:-1, :] = 2.0 * T[:-1, :] * T[1:, :] / (T[:-1, :] + T[1:, :] + 1e-20)

            for i in range(1, self.ny - 1):
                for j in range(1, self.nx - 1):
                    Ah[i, j] = (
                        -(Tx[i, j] * h[i, j-1] + Tx[i, j+1] * h[i, j+1]) / self.dx**2
                        -(Ty[i, j] * h[i-1, j] + Ty[i+1, j] * h[i+1, j]) / self.dy**2
                        + ((Tx[i, j] + Tx[i, j+1]) / self.dx**2
                           + (Ty[i, j] + Ty[i+1, j]) / self.dy**2) * h[i, j]
                    )
            # 边界
            Ah[:, 0] = h[:, 0]
            Ah[:, -1] = h[:, -1]
            Ah[0, :] = h[0, :]
            Ah[-1, :] = h[-1, :]
            return Ah.ravel()

        # 初始猜测
        x = np.zeros(n)
        x = x.reshape((self.ny, self.nx))
        x[:, 0] = h_boundary.get('left', 0.0)
        x[:, -1] = h_boundary.get('right', 0.0)
        x[0, :] = h_boundary.get('top', 0.0)
        x[-1, :] = h_boundary.get('bottom', 0.0)
        x = x.ravel()

        # 右端项
        b = np.zeros(n)
        b = b.reshape((self.ny, self.nx))
        b[:, 0] = h_boundary.get('left', 0.0)
        b[:, -1] = h_boundary.get('right', 0.0)
        b[0, :] = h_boundary.get('top', 0.0)
        b[-1, :] = h_boundary.get('bottom', 0.0)
        for i in range(1, self.ny - 1):
            for j in range(1, self.nx - 1):
                b[i, j] = -self.source[i, j]
        b = b.ravel()

        # CG 迭代
        Ax = matvec(x)
        r = b - Ax
        p = r.copy()
        rs_old = np.dot(r, r)

        for it in range(max_iter):
            Ap = matvec(p)
            pAp = np.dot(p, Ap)
            if abs(pAp) < 1e-30:
                break

            alpha = rs_old / pAp
            x = x + alpha * p
            r = r - alpha * Ap
            rs_new = np.dot(r, r)

            if np.sqrt(rs_new) < tol:
                break

            beta = rs_new / rs_old
            p = r + beta * p
            rs_old = rs_new

        self.head = x.reshape((self.ny, self.nx))
        return self.head

    def compute_velocity(self, T: np.ndarray,
                         porosity: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算达西流速

        公式：
            v_x = -K_x/φ * ∂h/∂x
            v_y = -K_y/φ * ∂h/∂y

        其中 K = T / b （对于裂隙，b 为开度）

        Parameters
        ----------
        T : np.ndarray
            传导系数场
        porosity : np.ndarray
            孔隙度场（默认 1.0）

        Returns
        -------
        tuple
            (vx, vy) 流速分量
        """
        # TODO: 实现达西流速计算
        pass

    def compute_flow_rate(self, T: np.ndarray) -> dict:
        """
        计算边界总流量

        Returns
        -------
        dict
            各边界流量统计
        """
        if self.head is None or np.all(self.head == 0):
            raise RuntimeError("先求解水头场")

        h = self.head

        # 左边界流入
        Q_left = np.sum(T[:, 0] * (h[:, 1] - h[:, 0]) / self.dx * self.dy)
        # 右边界流出
        Q_right = np.sum(T[:, -1] * (h[:, -2] - h[:, -1]) / self.dx * self.dy)
        # 上边界
        Q_top = np.sum(T[0, :] * (h[1, :] - h[0, :]) / self.dy * self.dx)
        # 下边界
        Q_bottom = np.sum(T[-1, :] * (h[-2, :] - h[-1, :]) / self.dy * self.dx)

        return {
            'Q_left': float(Q_left),
            'Q_right': float(Q_right),
            'Q_top': float(Q_top),
            'Q_bottom': float(Q_bottom),
            'Q_net': float(Q_left + Q_right + Q_top + Q_bottom)
        }
