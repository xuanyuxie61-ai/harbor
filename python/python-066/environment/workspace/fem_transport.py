"""
fem_transport.py
================================================================================
一维/二维有限元溶质运移求解器

基于种子项目：
  - 391_fem1d_heat_implicit：一维热方程的隐式有限元求解

科学背景：
  地下水溶质运移遵循对流-弥散-反应方程（Advection-Dispersion-Reaction Equation, ADRE）：

      R ∂C/∂t = ∇·(D ∇C) - v·∇C - λ R C + S

  其中：
      C(x,t)  — 溶质浓度 [M/L³]
      R       — 滞留因子（retardation factor），R = 1 + ρ_b K_d / n
      D       — 水动力弥散系数张量 [L²/T]，D = α_L |v| I + (α_L - α_T) v v^T / |v|
      v       — 达西流速矢量 [L/T]，v = q / n = -K ∇h / n
      λ       — 一级衰变/降解速率 [1/T]
      S       — 源汇项 [M/(L³T)]
      ρ_b     —  bulk density [M/L³]
      K_d     — 分配系数 [L³/M]
      n       — 孔隙度
      α_L, α_T — 纵向、横向弥散度 [L]

  采用 Galerkin 有限元方法进行空间离散，向后 Euler 进行时间离散：

      (M/Δt + θ K) C^{n+1} = (M/Δt - (1-θ) K) C^n + b^{n+1/2}

  其中 θ = 1 对应完全隐式（无条件稳定），M 为质量矩阵，K 为刚度+对流矩阵。
================================================================================
"""

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
from typing import Optional, Callable


class FEMTransportSolver1D:
    """
    一维溶质运移方程的线性 Lagrange 有限元求解器。

    求解方程：
        R ∂C/∂t = D ∂²C/∂x² - v ∂C/∂x - λ R C + S(x,t)
    在区间 [x_a, x_b] 上，配合 Dirichlet 或 Neumann 边界条件。
    """

    def __init__(self, mesh_nodes: np.ndarray, theta: float = 1.0):
        """
        参数
        ----------
        mesh_nodes : np.ndarray
            升序排列的节点坐标数组，shape (n_nodes,)
        theta : float
            时间加权因子，0.5 ≤ θ ≤ 1.0（θ=1 为向后 Euler）
        """
        self.nodes = np.asarray(mesh_nodes, dtype=float)
        if len(self.nodes) < 2:
            raise ValueError("节点数至少为 2")
        if not np.all(np.diff(self.nodes) > 0):
            raise ValueError("节点坐标必须严格递增")
        if not (0.0 <= theta <= 1.0):
            raise ValueError("θ 必须在 [0,1] 区间内")
        self.n_nodes = len(self.nodes)
        self.n_elements = self.n_nodes - 1
        self.theta = float(theta)
        self._A = None  # 缓存系统矩阵
        self._M = None  # 质量矩阵

    def _element_matrices(self, e: int, D: float, v: float, R: float, lam: float):
        """
        计算单元 e 上的局部质量矩阵 m_e、刚度矩阵 k_e 和对流矩阵 a_e。

        在线性单元 [x_i, x_{i+1}] 上，基函数为：
            φ_i(x)   = (x_{i+1} - x) / h_e
            φ_{i+1}(x) = (x - x_i) / h_e

        局部质量矩阵（一致质量）：
            m_e = (h_e / 6) * [[2, 1], [1, 2]]
        局部刚度矩阵（弥散项）：
            k_e = (D / h_e) * [[1, -1], [-1, 1]]
        局部对流矩阵（采用迎风格式修正）：
            a_e = (v / 2) * [[-1, 1], [-1, 1]] + (|v| h_e / 2) * [[1, -1], [-1, 1]]
            第二项为人工扩散（SUPG 的一维简化）
        反应项矩阵：
            r_e = λ R * m_e
        """
        h = self.nodes[e + 1] - self.nodes[e]
        if h <= 0:
            raise ValueError(f"单元 {e} 的长度必须为正")

        # TODO: Hole 1 — 计算局部质量矩阵 m_e、刚度矩阵 k_e、对流矩阵 a_e 和反应项矩阵 r_e
        # 要求：
        #   1. 一致质量矩阵: m_e = (h/6) * [[2,1],[1,2]]
        #   2. 弥散刚度矩阵: k_e = (D/h) * [[1,-1],[-1,1]]，需考虑 Peclet 数过大时的人工扩散修正
        #   3. 对流矩阵: a_e = (v/2) * [[-1,1],[-1,1]]
        #   4. 反应项矩阵: r_e = λ R * m_e
        # 返回元组 (m_e, k_e + a_e + r_e)
        raise NotImplementedError("Hole 1: 请实现 FEM 局部单元矩阵计算")

        return m_e, k_e + a_e + r_e

    def assemble_system(self, D: float, v: float, R: float, lam: float,
                        dt: float) -> tuple[csr_matrix, csr_matrix]:
        """
        组装全局质量矩阵 M 和系统矩阵 A = M/dt + θ K。
        """
        if dt <= 0:
            raise ValueError("时间步长 dt 必须为正")
        if D < 0 or R <= 0 or lam < 0:
            raise ValueError("物理参数非法：D≥0, R>0, λ≥0")

        M_data = []
        K_data = []
        row_idx = []
        col_idx = []

        for e in range(self.n_elements):
            m_e, k_e = self._element_matrices(e, D, v, R, lam)
            nodes = [e, e + 1]
            for i_local in range(2):
                for j_local in range(2):
                    gi = nodes[i_local]
                    gj = nodes[j_local]
                    row_idx.append(gi)
                    col_idx.append(gj)
                    M_data.append(m_e[i_local, j_local])
                    K_data.append(k_e[i_local, j_local])

        M = csr_matrix((M_data, (row_idx, col_idx)), shape=(self.n_nodes, self.n_nodes))
        K = csr_matrix((K_data, (row_idx, col_idx)), shape=(self.n_nodes, self.n_nodes))

        self._M = M
        A = M / dt + self.theta * K
        self._A = A
        return M, A

    def solve_step(self, C_old: np.ndarray, dt: float,
                   bc_nodes: list[int], bc_values: list[float],
                   source: Optional[np.ndarray] = None) -> np.ndarray:
        """
        执行一个时间步的求解：从 C^n 计算 C^{n+1}。

        右端项：
            rhs = (M/dt - (1-θ) K) C^n + source
        然后施加 Dirichlet 边界条件。
        """
        if self._A is None or self._M is None:
            raise RuntimeError("请先调用 assemble_system")
        if len(C_old) != self.n_nodes:
            raise ValueError("C_old 长度必须与节点数一致")
        if len(bc_nodes) != len(bc_values):
            raise ValueError("边界节点数与边界值数必须一致")

        # 获取 K（从 A 和 M 反推）
        K = (self._A - self._M / dt) / self.theta
        rhs_matrix = self._M / dt - (1.0 - self.theta) * K
        rhs = rhs_matrix @ C_old

        if source is not None:
            if len(source) != self.n_nodes:
                raise ValueError("源项长度必须与节点数一致")
            rhs += source

        # 施加 Dirichlet BC
        A_mod = self._A.copy().tolil()
        rhs = np.array(rhs, dtype=float)
        for node, val in zip(bc_nodes, bc_values):
            A_mod[node, :] = 0.0
            A_mod[node, node] = 1.0
            rhs[node] = val

        C_new = spsolve(A_mod.tocsr(), rhs)
        return C_new

    def solve_transient(self, C0: np.ndarray, t_end: float, dt: float,
                        D: float, v: float, R: float, lam: float,
                        bc_nodes: list[int], bc_values: list[float],
                        source_func: Optional[Callable[[float], np.ndarray]] = None,
                        verbose: bool = False) -> tuple[np.ndarray, np.ndarray]:
        """
        求解完整的瞬态过程。

        返回
        -------
        t_history : np.ndarray
            时间序列
        C_history : np.ndarray
            每一时间步的浓度场，shape (n_steps, n_nodes)
        """
        if t_end <= 0 or dt <= 0:
            raise ValueError("t_end 和 dt 必须为正")
        n_steps = int(np.ceil(t_end / dt))
        dt = t_end / n_steps  # 调整步长使其精确整除

        self.assemble_system(D, v, R, lam, dt)

        t_history = np.zeros(n_steps + 1)
        C_history = np.zeros((n_steps + 1, self.n_nodes))
        C_history[0, :] = C0
        C = C0.copy()

        for step in range(1, n_steps + 1):
            t = step * dt
            source = source_func(t) if source_func is not None else None
            C = self.solve_step(C, dt, bc_nodes, bc_values, source)
            C_history[step, :] = C
            t_history[step] = t
            if verbose and step % max(1, n_steps // 10) == 0:
                print(f"  Step {step}/{n_steps}, t={t:.3f}, C_max={C.max():.6e}")

        return t_history, C_history

    def compute_mass_balance(self, C: np.ndarray) -> float:
        """
        计算总溶质质量（基于一致质量矩阵）：
            M_total = ∫ C(x) dx ≈ Σ_i Σ_j M_{ij} C_j
        """
        if self._M is None:
            raise RuntimeError("质量矩阵未组装")
        return float(np.sum(self._M @ C))


class FlowSolver1D:
    """
    一维稳态地下水流动求解器：

        d/dx (K(x) dh/dx) = 0   （无源汇）
    或
        d/dx (K(x) dh/dx) + q_s = 0

    配合 Dirichlet 边界条件 h(0)=h_L, h(L)=h_R。
    """

    def __init__(self, mesh_nodes: np.ndarray):
        self.nodes = np.asarray(mesh_nodes, dtype=float)
        self.n_nodes = len(self.nodes)

    def solve_steady(self, K: np.ndarray,
                     h_left: float, h_right: float,
                     source: Optional[np.ndarray] = None) -> np.ndarray:
        """
        求解稳态水头分布并返回流速场。

        采用中心差分离散：
            K_{i+1/2} (h_{i+1} - h_i)/Δx - K_{i-1/2} (h_i - h_{i-1})/Δx = -q_i Δx
        """
        n = self.n_nodes
        if len(K) != n:
            raise ValueError("K 数组长度必须等于节点数")

        A = np.zeros((n, n))
        b = np.zeros(n)
        if source is not None:
            b = -source.copy()

        for i in range(1, n - 1):
            dx_left = self.nodes[i] - self.nodes[i - 1]
            dx_right = self.nodes[i + 1] - self.nodes[i]
            K_left = 0.5 * (K[i] + K[i - 1])
            K_right = 0.5 * (K[i] + K[i + 1])

            A[i, i - 1] = -K_left / dx_left
            A[i, i] = K_left / dx_left + K_right / dx_right
            A[i, i + 1] = -K_right / dx_right

        # Dirichlet BC
        A[0, 0] = 1.0
        b[0] = h_left
        A[-1, -1] = 1.0
        b[-1] = h_right

        h = np.linalg.solve(A, b)
        return h

    def compute_velocity(self, K: np.ndarray, h: np.ndarray, porosity: float) -> np.ndarray:
        """
        计算达西流速：
            v = -K / n * dh/dx
        在节点上采用中心差分。
        """
        if porosity <= 0:
            raise ValueError("孔隙度必须为正")
        v = np.zeros(self.n_nodes)
        for i in range(1, self.n_nodes - 1):
            dx = self.nodes[i + 1] - self.nodes[i - 1]
            dh = (h[i + 1] - h[i - 1]) / dx
            v[i] = -K[i] * dh / porosity
        # 边界外延
        v[0] = v[1]
        v[-1] = v[-2]
        return v


if __name__ == "__main__":
    nodes = np.linspace(0.0, 100.0, 51)
    solver = FEMTransportSolver1D(nodes, theta=1.0)
    C0 = np.zeros_like(nodes)
    C0[nodes <= 10.0] = 1.0
    t_hist, C_hist = solver.solve_transient(
        C0, t_end=50.0, dt=2.0,
        D=1.0, v=0.5, R=1.0, lam=0.01,
        bc_nodes=[0, -1], bc_values=[0.0, 0.0]
    )
    assert C_hist.shape[0] == len(t_hist)
    print("fem_transport: 自测试通过")
