r"""
fem_assembler.py
===============
有限元稀疏矩阵组装模块，用于磁重联问题中质量矩阵、刚度矩阵
和散度-自由约束矩阵的构造。

核心数学模型
------------
1D 反应-扩散方程的弱形式（Neumann 边界）:
    integral_Omega (partial u/partial t) v dx + D integral_Omega nabla u . nabla v dx = integral_Omega f(u) v dx

离散后得到:
    M * du/dt + K * u = F(u)

其中质量矩阵 M 和刚度矩阵 K 通过帽子函数（hat function）组装:
    M_{ij} = integral_Omega phi_i phi_j dx
    K_{ij} = integral_Omega nabla phi_i . nabla phi_j dx

对于均匀网格 h = 1/n，线性帽子函数的矩阵为:
    M = (h/6) * tridiag(1, 4, 1)   （内部节点）
    K = (1/h) * tridiag(-1, 2, -1)

二维情况下的 ST（Coordinate List）到 GE（General Dense）组装:
    对每个单元 e，计算局部贡献 K^e_{ab}
    通过自由度映射组装到全局矩阵:
        K_{I(a), I(b)} += K^e_{ab}

融入原项目:
- 377_fem_neumann: 帽子函数、质量/刚度矩阵、Neumann 边界处理
- 1156_st_to_ge: 稀疏矩阵从 ST 格式到稠密格式的累加组装
"""

import numpy as np
from typing import Tuple, Optional, Callable
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import spsolve


class FEM1DAssembler:
    """
    一维有限元组装器，带齐次 Neumann 边界条件。
    """

    def __init__(self, n_elements: int, domain: Tuple[float, float] = (0.0, 1.0)):
        if n_elements < 2:
            raise ValueError("n_elements 至少为 2")
        self.n = n_elements
        self.domain = domain
        self.h = (domain[1] - domain[0]) / n_elements
        self.nodes = np.linspace(domain[0], domain[1], n_elements + 1)

    def mass_matrix(self, sparse: bool = True):
        """
        组装质量矩阵 M（使用线性帽子函数）。
        内部节点: M_ii = 4h/6, M_{i,i+/-1} = h/6
        边界节点: M_00 = M_nn = 2h/6
        """
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
        """
        组装刚度矩阵 K。
        K_ii = 2/h, K_{i,i+/-1} = -1/h
        """
        main = np.full(self.n + 1, 2.0 / self.h)
        off = np.full(self.n, -1.0 / self.h)
        if sparse:
            K = diags([off, main, off], offsets=[-1, 0, 1], format='csr')
        else:
            K = np.diag(main) + np.diag(off, k=1) + np.diag(off, k=-1)
        return K

    def hat_function(self, x: np.ndarray, node_idx: int) -> np.ndarray:
        """
        计算第 node_idx 个帽子函数在点 x 处的值。
        定义域: [x_{i-1}, x_{i+1}]，在 x_i 处值为 1。
        """
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
        """
        将初始条件投影到有限元空间: w_0 = M^{-1} <w0, phi>。
        使用 Simpson 规则近似内积积分。
        """
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
            # Simpson 积分
            fvals = w0_func(xq) * self.hat_function(xq, i)
            if len(xq) == 3:
                wr[i] = self.h / 6.0 * (fvals[0] + 4.0 * fvals[1] + fvals[2])
            else:
                wr[i] = np.trapz(fvals, xq)
        # 解线性系统
        w0 = np.linalg.solve(M, wr)
        return w0

    def solve_steady(self,
                     K: np.ndarray,
                     F: np.ndarray,
                     sparse: bool = True) -> np.ndarray:
        """
        求解稳态方程 K u = F（Neumann 边界需要处理零空间）。
        通过固定第一个节点的值消除奇异性。
        """
        if sparse:
            Kd = K.toarray().copy()
        else:
            Kd = K.copy()
        Fd = F.copy()
        # 固定第一个节点
        Kd[0, :] = 0.0
        Kd[:, 0] = 0.0
        Kd[0, 0] = 1.0
        Fd[0] = 0.0
        u = np.linalg.solve(Kd, Fd)
        return u


class STtoGEAssembler:
    """
    将 ST（Coordinate List）格式的单元贡献组装为全局稠密矩阵。
    对应原项目 st_to_ge 的核心思想。
    """

    @staticmethod
    def assemble(nst: int,
                 ist: np.ndarray,
                 jst: np.ndarray,
                 ast: np.ndarray,
                 m: Optional[int] = None,
                 n: Optional[int] = None) -> np.ndarray:
        """
        将 ST 格式的三元组 (i, j, value) 组装为 m x n 稠密矩阵。
        相同 (i, j) 位置的值累加。
        """
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
            i = ist[k] - 1  # 转换为 0-based
            j = jst[k] - 1
            if 0 <= i < m and 0 <= j < n:
                Age[i, j] += ast[k]
            else:
                raise IndexError(f"ST 索引越界: ({i+1}, {j+1}) 超出 ({m}, {n})")
        return Age

    @staticmethod
    def assemble_2d_stiffness_from_quads(nodes: np.ndarray,
                                         elements: np.ndarray) -> np.ndarray:
        """
        从四边形网格组装二维刚度矩阵（Poisson 算子）。
        使用双线性四边形单元的简化一阶积分。
        """
        nnodes = len(nodes)
        nelems = len(elements)
        K = np.zeros((nnodes, nnodes))

        for e in range(nelems):
            idx = elements[e]
            if len(idx) != 4:
                raise ValueError("每个单元必须是四边形（4个节点）")
            # 单元顶点坐标
            x = nodes[idx, 0]
            y = nodes[idx, 1]
            # 计算单元面积（简化：用两个三角形面积之和）
            area = 0.5 * abs((x[1]-x[0])*(y[2]-y[0]) - (x[2]-x[0])*(y[1]-y[0])) + \
                   0.5 * abs((x[2]-x[0])*(y[3]-y[0]) - (x[3]-x[0])*(y[2]-y[0]))
            # 简化的局部刚度（假设单元近似矩形）
            for a in range(4):
                for b in range(4):
                    # 简化的双线性形函数梯度内积
                    K[idx[a], idx[b]] += area / 12.0 if a != b else area / 6.0
        return K


def demo_fem():
    """
    演示：求解一维泊松方程 -u'' = f，Neumann 边界，并与解析解比较。
    """
    print("\n[FEMAssembler] 演示: 1D Poisson Neumann")
    n = 64
    fem = FEM1DAssembler(n, domain=(0.0, 1.0))
    M = fem.mass_matrix(sparse=False)
    K = fem.stiffness_matrix(sparse=False)

    # 初始条件投影
    w0_func = lambda x: np.sin(np.pi * x)
    w0 = fem.project_initial(w0_func)

    # 求解稳态: -D*u'' = sin(pi*x)，解析解 u = sin(pi*x)/(pi^2)
    f = lambda x: np.sin(np.pi * x)
    x_nodes = fem.nodes
    F_rhs = M @ f(x_nodes)
    u_num = fem.solve_steady(K, F_rhs, sparse=False)
    u_exact = np.sin(np.pi * x_nodes) / (np.pi ** 2)
    err = np.max(np.abs(u_num - u_exact))
    print(f"  网格数 n={n}, L_inf 误差 = {err:.3e}")

    # ST 组装演示
    print("\n[FEMAssembler] 演示: ST->GE 组装")
    ist = np.array([1, 2, 2, 3, 3, 3])
    jst = np.array([1, 1, 2, 2, 3, 1])
    ast = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    A = STtoGEAssembler.assemble(len(ist), ist, jst, ast)
    print("  组装后矩阵:")
    print(A)


if __name__ == "__main__":
    demo_fem()
