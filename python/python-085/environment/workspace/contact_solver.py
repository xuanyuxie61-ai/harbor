"""
contact_solver.py
接触与摩擦非线性求解器模块
核心科学内容：Signorini 接触条件 + Coulomb 摩擦定律的增广 Lagrange 求解
"""
import numpy as np
from typing import Tuple, List, Optional
from mesh_generator import TriMesh2D
from fem_assembler import ElasticFEM2D, assemble_contact_gaps, assemble_contact_normals
from banded_solver import BandedSolver
from utils import macaulay_bracket, solve_2x2_symmetric, safe_divide


class SignoriniCoulombContact:
    r"""
    Signorini-Coulomb 接触问题的增广 Lagrange 求解器。

    控制方程：
    1. 平衡方程：K u = f_ext + f_contact
    2. Signorini 条件（法向）：
       g_n \ge 0, \quad p_n \ge 0, \quad g_n \cdot p_n = 0
    3. Coulomb 摩擦定律（切向）：
       \|p_t\| \le \mu p_n, \quad
       \text{if } \|p_t\| < \mu p_n \Rightarrow \dot{u}_t = 0, \quad
       \text{if } \|p_t\| = \mu p_n \Rightarrow p_t = -\mu p_n \frac{\dot{u}_t}{\|\dot{u}_t\|}

    增广 Lagrange 泛函：
    \mathcal{L}(u, \lambda_n, \lambda_t) =
    \frac{1}{2} u^T K u - u^T f_{ext}
    + \sum_{i \in \mathcal{C}} \left[
        \lambda_n^{(i)} g_n^{(i)} + \frac{c_n}{2} (g_n^{(i)})^2
        + \lambda_t^{(i)} g_t^{(i)} + \frac{c_t}{2} (g_t^{(i)})^2
    \right]_+

    其中 [...]_+ 表示正部投影，g_t^{(i)} 为切向相对位移。
    """

    def __init__(self, fem: ElasticFEM2D, contact_nodes: np.ndarray,
                 friction_coeff: float = 0.3, aug_lag_penalty: float = 1e9,
                 max_iter: int = 100, tol: float = 1e-8):
        self.fem = fem
        self.mesh = fem.mesh
        self.contact_nodes = np.array(contact_nodes, dtype=int)
        self.n_contact = len(contact_nodes)
        self.mu_friction = friction_coeff
        self.c_n = aug_lag_penalty
        self.c_t = aug_lag_penalty * 0.5
        self.max_iter = max_iter
        self.tol = tol
        self.normals = assemble_contact_normals(self.mesh, self.contact_nodes)
        # Lagrange 乘子
        self.lambda_n = np.zeros(self.n_contact)
        self.lambda_t = np.zeros(self.n_contact)

    def _compute_local_gap_and_slip(self, u: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        r"""
        计算接触节点的法向间隙 g_n 和切向位移 g_t。
        对水平刚性基础：
        g_n = (y + u_y) - y_rigid
        g_t = u_x
        """
        g_n = assemble_contact_gaps(self.mesh, u, self.contact_nodes, rigid_surface_y=0.0)
        g_t = np.zeros(self.n_contact)
        for idx, node in enumerate(self.contact_nodes):
            g_t[idx] = u[2 * node]  # x方向位移即切向位移
        return g_n, g_t

    def _augmented_lagrange_update(self, g_n: np.ndarray, g_t: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        r"""
        更新 Lagrange 乘子：
        \lambda_n^{new} = \langle \lambda_n + c_n g_n \rangle_+
        \lambda_t^{new} = \text{proj}_{[-\mu \lambda_n, \mu \lambda_n]}(\lambda_t + c_t g_t)
        """
        # [HOLE 2]: 请实现增广 Lagrange 乘子更新公式。
        # 法向乘子更新：lambda_n_new = max(lambda_n - c_n * g_n, 0)
        # 切向乘子更新：trial_t = lambda_t + c_t * g_t, 然后投影到 [-mu*lambda_n_new, mu*lambda_n_new]
        pass

    def _assemble_contact_force(self, u: np.ndarray) -> np.ndarray:
        r"""
        组装接触力向量（作为外力施加到结构）。
        法向接触力：f_n = \lambda_n + c_n g_n（若激活）
        切向摩擦力：f_t = \lambda_t + c_t g_t（若激活）
        对水平基础：f_contact[node, 0] += f_t, f_contact[node, 1] -= f_n
        """
        f_contact = np.zeros(2 * self.mesh.n_nodes)
        g_n, g_t = self._compute_local_gap_and_slip(u)
        active = g_n < 1e-6  # 近似激活集
        for idx, node in enumerate(self.contact_nodes):
            if active[idx]:
                p_n = max(self.lambda_n[idx] - self.c_n * g_n[idx], 0.0)
                trial_t = self.lambda_t[idx] + self.c_t * g_t[idx]
                max_t = self.mu_friction * p_n
                p_t = np.clip(trial_t, -max_t, max_t)
                # 水平基础：法向反力向上(+y)，切向沿x
                f_contact[2 * node] += p_t
                f_contact[2 * node + 1] += p_n
        return f_contact

    def _contact_stiffness_penalty(self) -> np.ndarray:
        r"""
        构造接触罚刚度矩阵（仅对激活接触节点）。
        K_c[2*node+1, 2*node+1] += c_n
        K_c[2*node, 2*node] += c_t（近似）
        """
        n_dof = 2 * self.mesh.n_nodes
        K_c = np.zeros((n_dof, n_dof))
        # 为简化，对所有接触节点施加罚刚度
        for idx, node in enumerate(self.contact_nodes):
            K_c[2 * node + 1, 2 * node + 1] += self.c_n
            K_c[2 * node, 2 * node] += self.c_t
        return K_c

    def solve_static(self, f_ext: np.ndarray,
                     fixed_nodes: Optional[np.ndarray] = None,
                     fixed_values: Optional[np.ndarray] = None,
                     dof_mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, dict]:
        r"""
        求解静态接触问题。
        采用主动集 + 增广 Lagrange 迭代。

        算法：
        1. 初始化 u = K^{-1} f_ext（无接触）
        2. 对每个 AL 迭代：
           a. 组装接触力 f_contact(u, \lambda)
           b. 求解 K u^{new} = f_ext + f_contact
           c. 计算 g_n, g_t
           d. 更新 \lambda
           e. 检查收敛
        """
        n_dof = 2 * self.mesh.n_nodes
        K = self.fem.assemble_global_stiffness()
        # 初始位移（无接触）
        if fixed_nodes is not None and fixed_values is not None:
            K_mod, F_mod = self.fem.apply_dirichlet_bc(K, f_ext, fixed_nodes, fixed_values, dof_mask=dof_mask)
        else:
            K_mod, F_mod = K.copy(), f_ext.copy()
        try:
            u = np.linalg.solve(K_mod, F_mod)
        except np.linalg.LinAlgError:
            u = np.zeros(n_dof)

        self.lambda_n = np.zeros(self.n_contact)
        self.lambda_t = np.zeros(self.n_contact)

        history = {"residuals": [], "active_set_sizes": []}

        for it in range(self.max_iter):
            g_n, g_t = self._compute_local_gap_and_slip(u)
            active = g_n < 1e-4
            n_active = int(np.sum(active))
            history["active_set_sizes"].append(n_active)

            # 更新乘子
            lambda_n_new, lambda_t_new = self._augmented_lagrange_update(g_n, g_t)

            # 组装接触力
            f_contact = self._assemble_contact_force(u)

            # 求解
            F_total = f_ext + f_contact
            if fixed_nodes is not None and fixed_values is not None:
                K_mod, F_mod = self.fem.apply_dirichlet_bc(K, F_total, fixed_nodes, fixed_values, dof_mask=dof_mask)
            else:
                K_mod, F_mod = K.copy(), F_total.copy()
            try:
                u_new = np.linalg.solve(K_mod, F_mod)
            except np.linalg.LinAlgError:
                break

            # 收敛检查
            du = np.linalg.norm(u_new - u) / max(np.linalg.norm(u_new), 1e-12)
            dlambda = np.linalg.norm(lambda_n_new - self.lambda_n) / max(np.linalg.norm(lambda_n_new), 1e-12)
            res = max(du, dlambda)
            history["residuals"].append(res)

            u = u_new
            self.lambda_n = lambda_n_new
            self.lambda_t = lambda_t_new

            if res < self.tol:
                break

        history["iterations"] = it + 1
        history["final_residual"] = res if it > 0 else 0.0
        return u, history

    def compute_contact_pressure(self, u: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        r"""
        计算接触节点上的法向压力和切向牵引力。
        p_n = max(0, \lambda_n + c_n g_n)
        p_t = clip(\lambda_t + c_t g_t, -\mu p_n, \mu p_n)
        返回 (p_n, p_t)。
        """
        g_n, g_t = self._compute_local_gap_and_slip(u)
        p_n = np.maximum(self.lambda_n + self.c_n * g_n, 0.0)
        trial_t = self.lambda_t + self.c_t * g_t
        max_t = self.mu_friction * p_n
        p_t = np.clip(trial_t, -max_t, max_t)
        return p_n, p_t

    def compute_friction_dissipation(self, u: np.ndarray, v: np.ndarray) -> float:
        r"""
        计算摩擦耗散功率：
        D = \sum_{i \in \mathcal{C}} p_t^{(i)} \cdot v_t^{(i)}
        其中 v_t 为接触节点切向速度。
        """
        _, p_t = self.compute_contact_pressure(u)
        diss = 0.0
        for idx, node in enumerate(self.contact_nodes):
            v_t = v[2 * node]
            diss += p_t[idx] * v_t
        return diss


def active_set_newton_contact(fem: ElasticFEM2D, contact_nodes: np.ndarray,
                               f_ext: np.ndarray, friction_coeff: float = 0.3,
                               max_iter: int = 50, tol: float = 1e-10,
                               fixed_nodes: Optional[np.ndarray] = None,
                               fixed_values: Optional[np.ndarray] = None,
                               dof_mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, dict]:
    r"""
    基于主动集策略的 Newton 迭代接触求解器（博士级算法）。

    在每个 Newton 步：
    1. 确定当前主动集 \mathcal{A} = {i | g_n^{(i)} \le 0}
    2. 对主动集节点施加约束：
       - 法向：u_y^{(i)} = -g_n^{(i), ref}
       - 切向：根据 Coulomb 条件判断粘着/滑动
    3. 求解约化线性系统
    4. 更新主动集
    """
    mesh = fem.mesh
    n_dof = 2 * mesh.n_nodes
    K = fem.assemble_global_stiffness()
    u = np.zeros(n_dof)

    history = {"residuals": [], "active_sets": []}

    for it in range(max_iter):
        g_n = assemble_contact_gaps(mesh, u, contact_nodes, rigid_surface_y=0.0)
        active = g_n <= 1e-6
        history["active_sets"].append(int(np.sum(active)))

        # 构造约束条件
        n_con = np.sum(active)
        if n_con > 0:
            active_nodes = contact_nodes[active]
            # 法向位移约束：固定 y 位移使间隙为零
            bc_nodes = active_nodes.copy()
            bc_values = np.zeros((n_con, 2))
            for idx, node in enumerate(active_nodes):
                bc_values[idx, 1] = -mesh.nodes[node, 1]  # 将节点压到 y=0
            # 合并外部固定节点与接触约束节点
            if fixed_nodes is not None and fixed_values is not None:
                all_nodes = np.concatenate([fixed_nodes, bc_nodes])
                all_values = np.vstack([fixed_values, bc_values])
                if dof_mask is not None:
                    bc_mask = np.ones((len(bc_nodes), 2), dtype=bool)
                    all_mask = np.vstack([dof_mask, bc_mask])
                else:
                    all_mask = None
                K_mod, F_mod = fem.apply_dirichlet_bc(K, f_ext, all_nodes, all_values, dof_mask=all_mask)
            else:
                K_mod, F_mod = fem.apply_dirichlet_bc(K, f_ext, bc_nodes, bc_values)
        else:
            if fixed_nodes is not None and fixed_values is not None:
                K_mod, F_mod = fem.apply_dirichlet_bc(K, f_ext, fixed_nodes, fixed_values, dof_mask=dof_mask)
            else:
                K_mod, F_mod = K.copy(), f_ext.copy()

        try:
            u_new = np.linalg.solve(K_mod, F_mod)
        except np.linalg.LinAlgError:
            break

        res = np.linalg.norm(u_new - u) / max(np.linalg.norm(u_new), 1e-12)
        history["residuals"].append(res)
        u = u_new
        if res < tol:
            break

    history["iterations"] = it + 1
    return u, history
