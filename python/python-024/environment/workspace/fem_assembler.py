
import numpy as np
from typing import Tuple, Optional, Callable
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import spsolve


class FEM1DAssembler:

    def __init__(self, n_elements: int, domain: Tuple[float, float] = (0.0, 1.0)):
        if n_elements < 2:
            raise ValueError("n_elements 至少为 2")
        self.n = n_elements
        self.domain = domain
        self.h = (domain[1] - domain[0]) / n_elements
        self.nodes = np.linspace(domain[0], domain[1], n_elements + 1)

    def mass_matrix(self, sparse: bool = True):
        main = np.full(self.n + 1, 4.0 * self.h / 6.0)
        main[0] = 2.0 * self.h / 6.0
        main[-1] = 2.0 * self.h / 6.0
        off = np.full(self.n, self.h / 6.0)
        if sparse:
            M = diags([off, main, off], offsets=[-1, 0, 1], format='csr')
        else:
            M = np.diag(main) + np.diag(off, k=1) + np.diag(off, k=-1)
        return M

    def stiffness_matrix(self, sparse: bool = True):
        main = np.full(self.n + 1, 2.0 / self.h)
        off = np.full(self.n, -1.0 / self.h)
        if sparse:
            K = diags([off, main, off], offsets=[-1, 0, 1], format='csr')
        else:
            K = np.diag(main) + np.diag(off, k=1) + np.diag(off, k=-1)
        return K

    def hat_function(self, x: np.ndarray, node_idx: int) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        xi = self.nodes[node_idx]
        if node_idx == 0:
            val = np.where((x >= xi) & (x <= xi + self.h),
                           1.0 - np.abs(x - xi) / self.h, 0.0)
        elif node_idx == self.n:
            val = np.where((x >= xi - self.h) & (x <= xi),
                           1.0 - np.abs(x - xi) / self.h, 0.0)
        else:
            val = np.where((x >= xi - self.h) & (x <= xi + self.h),
                           1.0 - np.abs(x - xi) / self.h, 0.0)
        return val

    def project_initial(self, w0_func: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
        M = self.mass_matrix(sparse=False)
        wr = np.zeros(self.n + 1)
        for i in range(self.n + 1):
            if i == 0:
                xq = np.array([self.nodes[0], self.nodes[0] + self.h / 2.0, self.nodes[1]])
            elif i == self.n:
                xq = np.array([self.nodes[self.n - 1],
                               self.nodes[self.n - 1] + self.h / 2.0,
                               self.nodes[self.n]])
            else:
                xq = np.array([self.nodes[i - 1],
                               self.nodes[i],
                               self.nodes[i + 1]])

            fvals = w0_func(xq) * self.hat_function(xq, i)
            if len(xq) == 3:
                wr[i] = self.h / 6.0 * (fvals[0] + 4.0 * fvals[1] + fvals[2])
            else:
                wr[i] = np.trapz(fvals, xq)

        w0 = np.linalg.solve(M, wr)
        return w0

    def solve_steady(self,
                     K: np.ndarray,
                     F: np.ndarray,
                     sparse: bool = True) -> np.ndarray:
        if sparse:
            Kd = K.toarray().copy()
        else:
            Kd = K.copy()
        Fd = F.copy()

        Kd[0, :] = 0.0
        Kd[:, 0] = 0.0
        Kd[0, 0] = 1.0
        Fd[0] = 0.0
        u = np.linalg.solve(Kd, Fd)
        return u


class STtoGEAssembler:

    @staticmethod
    def assemble(nst: int,
                 ist: np.ndarray,
                 jst: np.ndarray,
                 ast: np.ndarray,
                 m: Optional[int] = None,
                 n: Optional[int] = None) -> np.ndarray:
        ist = np.asarray(ist, dtype=int)
        jst = np.asarray(jst, dtype=int)
        ast = np.asarray(ast, dtype=float)

        if m is None:
            m = int(np.max(ist)) if len(ist) > 0 else 0
        if n is None:
            n = int(np.max(jst)) if len(jst) > 0 else 0

        if m <= 0 or n <= 0:
            raise ValueError("矩阵维度必须为正")

        Age = np.zeros((m, n))
        for k in range(nst):
            i = ist[k] - 1
            j = jst[k] - 1
            if 0 <= i < m and 0 <= j < n:
                Age[i, j] += ast[k]
            else:
                raise IndexError(f"ST 索引越界: ({i+1}, {j+1}) 超出 ({m}, {n})")
        return Age

    @staticmethod
    def assemble_2d_stiffness_from_quads(nodes: np.ndarray,
                                         elements: np.ndarray) -> np.ndarray:
        nnodes = len(nodes)
        nelems = len(elements)
        K = np.zeros((nnodes, nnodes))

        for e in range(nelems):
            idx = elements[e]
            if len(idx) != 4:
                raise ValueError("每个单元必须是四边形（4个节点）")

            x = nodes[idx, 0]
            y = nodes[idx, 1]

            area = 0.5 * abs((x[1]-x[0])*(y[2]-y[0]) - (x[2]-x[0])*(y[1]-y[0])) + \
                   0.5 * abs((x[2]-x[0])*(y[3]-y[0]) - (x[3]-x[0])*(y[2]-y[0]))

            for a in range(4):
                for b in range(4):

                    K[idx[a], idx[b]] += area / 12.0 if a != b else area / 6.0
        return K


def demo_fem():
    print("\n[FEMAssembler] 演示: 1D Poisson Neumann")
    n = 64
    fem = FEM1DAssembler(n, domain=(0.0, 1.0))
    M = fem.mass_matrix(sparse=False)
    K = fem.stiffness_matrix(sparse=False)


    w0_func = lambda x: np.sin(np.pi * x)
    w0 = fem.project_initial(w0_func)


    f = lambda x: np.sin(np.pi * x)
    x_nodes = fem.nodes
    F_rhs = M @ f(x_nodes)
    u_num = fem.solve_steady(K, F_rhs, sparse=False)
    u_exact = np.sin(np.pi * x_nodes) / (np.pi ** 2)
    err = np.max(np.abs(u_num - u_exact))
    print(f"  网格数 n={n}, L_inf 误差 = {err:.3e}")


    print("\n[FEMAssembler] 演示: ST->GE 组装")
    ist = np.array([1, 2, 2, 3, 3, 3])
    jst = np.array([1, 1, 2, 2, 3, 1])
    ast = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    A = STtoGEAssembler.assemble(len(ist), ist, jst, ast)
    print("  组装后矩阵:")
    print(A)


if __name__ == "__main__":
    demo_fem()
