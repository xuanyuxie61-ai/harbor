
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
from typing import Optional, Callable


class FEMTransportSolver1D:

    def __init__(self, mesh_nodes: np.ndarray, theta: float = 1.0):
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
        self._A = None
        self._M = None

    def _element_matrices(self, e: int, D: float, v: float, R: float, lam: float):
        h = self.nodes[e + 1] - self.nodes[e]
        if h <= 0:
            raise ValueError(f"单元 {e} 的长度必须为正")








        raise NotImplementedError("Hole 1: 请实现 FEM 局部单元矩阵计算")

        return m_e, k_e + a_e + r_e

    def assemble_system(self, D: float, v: float, R: float, lam: float,
                        dt: float) -> tuple[csr_matrix, csr_matrix]:
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
        if self._A is None or self._M is None:
            raise RuntimeError("请先调用 assemble_system")
        if len(C_old) != self.n_nodes:
            raise ValueError("C_old 长度必须与节点数一致")
        if len(bc_nodes) != len(bc_values):
            raise ValueError("边界节点数与边界值数必须一致")


        K = (self._A - self._M / dt) / self.theta
        rhs_matrix = self._M / dt - (1.0 - self.theta) * K
        rhs = rhs_matrix @ C_old

        if source is not None:
            if len(source) != self.n_nodes:
                raise ValueError("源项长度必须与节点数一致")
            rhs += source


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
        if t_end <= 0 or dt <= 0:
            raise ValueError("t_end 和 dt 必须为正")
        n_steps = int(np.ceil(t_end / dt))
        dt = t_end / n_steps

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
        if self._M is None:
            raise RuntimeError("质量矩阵未组装")
        return float(np.sum(self._M @ C))


class FlowSolver1D:

    def __init__(self, mesh_nodes: np.ndarray):
        self.nodes = np.asarray(mesh_nodes, dtype=float)
        self.n_nodes = len(self.nodes)

    def solve_steady(self, K: np.ndarray,
                     h_left: float, h_right: float,
                     source: Optional[np.ndarray] = None) -> np.ndarray:
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


        A[0, 0] = 1.0
        b[0] = h_left
        A[-1, -1] = 1.0
        b[-1] = h_right

        h = np.linalg.solve(A, b)
        return h

    def compute_velocity(self, K: np.ndarray, h: np.ndarray, porosity: float) -> np.ndarray:
        if porosity <= 0:
            raise ValueError("孔隙度必须为正")
        v = np.zeros(self.n_nodes)
        for i in range(1, self.n_nodes - 1):
            dx = self.nodes[i + 1] - self.nodes[i - 1]
            dh = (h[i + 1] - h[i - 1]) / dx
            v[i] = -K[i] * dh / porosity

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
