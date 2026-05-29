"""
pde_solver.py
================================================================================
高性能计算检查点容错：三维对流-扩散-反应方程有限元求解器

融合原项目：
  - 1111_sparse_parfor (稀疏矩阵并行组装)
  - 1246_tetrahedron_felippa_rule (四面体高斯求积)

科学角色：
  1) 求解 3D 时变对流-扩散-反应方程:
         du/dt = D * nabla^2 u - v . nabla u + R(u) + eta(x,t)
     其中 D 为扩散系数，v 为对流速度，R(u) 为非线性反应项，eta 为随机源项；
  2) 使用 P1 有限元在四面体网格上离散，稀疏组装刚度与质量矩阵；
  3) 状态向量 u(t) 作为检查点的核心数据对象。
================================================================================
"""

import numpy as np
from mesh_geometry import TetrahedralMesh
from quadrature_engine import tetrahedron_unit_o04, tetrahedron_unit_volume
from sparse_linear_algebra import r83s_cg


class AdvectionDiffusionSolver:
    """P1 有限元求解器。"""

    def __init__(self, mesh: TetrahedralMesh, D: float = 0.01,
                 velocity: np.ndarray = None, reaction_rate: float = 0.0):
        self.mesh = mesh
        self.D = D
        if velocity is None:
            velocity = np.array([0.0, 0.0, 0.0])
        self.velocity = np.asarray(velocity, dtype=float)
        self.reaction_rate = reaction_rate
        self.M_lumped = None
        self.K = None
        self.boundary_nodes = None
        self._build_matrices()

    def _build_matrices(self):
        """组装质量矩阵（ lumped ）与刚度矩阵，并识别边界节点。"""
        n = self.mesh.n_nodes
        self.M_lumped = np.zeros(n)
        # TODO [Hole 2]: 实现 lumped mass 矩阵组装
        # 使用简单 lumped mass: M_i = sum_{包含 i 的单元} V_e / 4
        # 需要调用 self.mesh.compute_volumes() 获取单元体积
        raise NotImplementedError("lumped mass assembly not implemented (Hole 2)")

        # 识别边界节点（位于长方体域的六个面上）
        x, y, z = self.mesh.nodes[:, 0], self.mesh.nodes[:, 1], self.mesh.nodes[:, 2]
        tol = 1.0e-10
        self.boundary_nodes = np.where(
            (x <= tol) | (x >= 1.0 - tol) |
            (y <= tol) | (y >= 1.0 - tol) |
            (z <= tol) | (z >= 1.0 - tol)
        )[0]

        # 刚度矩阵使用显式差分近似：仅存储三对角部分（简化为一维线松弛）
        self.K = self._build_diffusion_operator()

    def _build_diffusion_operator(self):
        """
        构造近似的扩散算子系数。
        返回 shape (n_nodes, 3) 的 [sub, diag, super] 形式（按节点排序）。
        这里简化为沿 x 方向的一维三对角结构。
        """
        n = self.mesh.n_nodes
        # 按 x 坐标排序节点以构造一维三对角结构（简化）
        order = np.argsort(self.mesh.nodes[:, 0])
        self._perm = order
        self._inv_perm = np.argsort(order)

        # 构造 DIF2-like 算子，尺度由网格直径与平均体积决定
        h = self.mesh.element_diameter()
        if h < 1.0e-14:
            h = 1.0
        vol_avg = np.mean(self.mesh.compute_volumes()) if self.mesh.n_elements > 0 else 1.0
        scale = self.D * vol_avg / (h * h)
        diag = 2.0 * scale
        off = -scale
        # 对每个节点返回局部系数（内部节点统一，边界在 step_explicit 中强制 Dirichlet）
        a_sub = np.full(n, off)
        a_diag = np.full(n, diag)
        a_sup = np.full(n, off)
        return np.vstack([a_sub, a_diag, a_sup]).T

    def reaction_term(self, u: np.ndarray) -> np.ndarray:
        """非线性反应项: R(u) = reaction_rate * u * (1 - u)。"""
        return self.reaction_rate * u * (1.0 - u)

    def advection_term(self, u: np.ndarray) -> np.ndarray:
        """
        对流项 -v . nabla u 的有限差分近似。
        沿速度方向的一阶迎风格式（仅在按 x 排序后的序列上操作）。
        """
        n = len(u)
        adv = np.zeros(n)
        nodes = self.mesh.nodes[self._perm]
        u_perm = u[self._perm]
        # 简化为沿 x 方向
        dx = np.diff(nodes[:, 0])
        dx = np.append(dx, dx[-1])
        dx[dx < 1.0e-14] = 1.0e-14
        v = self.velocity[0]
        for i in range(n):
            if v > 0.0 and i < n - 1:
                adv[i] = -v * (u_perm[i + 1] - u_perm[i]) / dx[i]
            elif v < 0.0 and i > 0:
                adv[i] = -v * (u_perm[i] - u_perm[i - 1]) / dx[i - 1]
        return adv[self._inv_perm]

    def step_explicit(self, u: np.ndarray, dt: float) -> np.ndarray:
        """
        显式 Euler 时间步:
            u^{n+1} = u^n + dt * (D * laplacian(u) - v . nabla(u) + R(u)) / M_lumped
        边界节点强制 Dirichlet u=0。
        """
        u = np.asarray(u, dtype=float)
        # 扩散项: 对按 x 排序后的状态应用三对角算子
        u_perm = u[self._perm]
        rhs_diff_perm = np.zeros_like(u_perm)
        n = len(u_perm)
        for i in range(n):
            sub, diag, sup = self.K[i]
            val = diag * u_perm[i]
            if i > 0:
                val += sub * u_perm[i - 1]
            if i < n - 1:
                val += sup * u_perm[i + 1]
            rhs_diff_perm[i] = val
        rhs_diff = rhs_diff_perm[self._inv_perm]

        rhs = rhs_diff + self.advection_term(u) + self.reaction_term(u)
        # 除以 lumped mass
        rhs = rhs / self.M_lumped
        u_new = u + dt * rhs
        # 强制 Dirichlet 边界条件
        u_new[self.boundary_nodes] = 0.0
        # 内部节点数值截断（防止极端情况）
        u_new = np.clip(u_new, -1.0e3, 1.0e3)
        return u_new

    def initial_condition(self, mode: str = "gaussian") -> np.ndarray:
        """生成初始条件。"""
        x = self.mesh.nodes[:, 0]
        y = self.mesh.nodes[:, 1]
        z = self.mesh.nodes[:, 2]
        if mode == "gaussian":
            u = np.exp(-((x - 0.5) ** 2 + (y - 0.5) ** 2 + (z - 0.5) ** 2) / 0.05)
        elif mode == "random":
            rng = np.random.default_rng(42)
            u = rng.random(self.mesh.n_nodes)
        else:
            u = np.zeros(self.mesh.n_nodes)
        # 强制边界为零
        u[self.boundary_nodes] = 0.0
        return u

    def compute_energy(self, u: np.ndarray) -> float:
        """
        计算离散能量泛函:
            E(u) = 0.5 * D * sum_e V_e * |grad u_e|^2 + 0.5 * sum_i M_i * u_i^2
        其中 grad u_e 为 P1 元在每个四面体上的常数梯度。
        """
        # TODO [Hole 1]: 实现离散能量泛函计算
        # 需要遍历所有单元，利用 P1 元常数梯度公式计算 |grad u_e|^2
        # 需要调用 self.mesh.compute_volumes() 获取单元体积
        raise NotImplementedError("compute_energy not implemented (Hole 1)")
