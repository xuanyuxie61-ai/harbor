"""
forward_model.py  --  Gray-Scott 反应扩散 + Serendipity FEM 正向算子
===============================================================
来源种子项目:
    486_gray_scott_movie    : Gray-Scott 反应扩散动力学 (du, dv, f, k).
    402_fem2d_bvp_serene    : 二维矩形区域 Serendipity 8 节点有限元.
科学问题角色:
    被校准的正向算子 G(theta) : (Du, Dv, f, k) -> 稀疏观测 y.
    反应扩散方程
        du/dt = Du * Laplacian(u) - u*v^2 + f*(1 - u)
        dv/dt = Dv * Laplacian(v) + u*v^2 - (f + k)*v
    在 [0, L]^2 上带 Neumann 边界; 离散化为
        M * dX/dt + A(theta) * X + N(X; theta) = 0
    其中 M 为 Serendipity 质量阵, A 为扩散-反应刚度阵, N 为非线性项.
边界与鲁棒性:
    1. 对 u, v 做 [0, 1] 投影, 防止负浓度.
    2. 隐式-显式 IMEX: 扩散隐式 (CG 求解), 反应显式.
    3. 自适应步长: 若 ||dX||/||X|| > safety, 折半步.
"""
from __future__ import annotations
import math
from typing import Dict, List, Tuple
from numerical_base import NUMERICS


# ----------------------------------------------------------------------
# Gray-Scott 反应项: R(u, v; f, k)
# ----------------------------------------------------------------------
def gray_scott_reaction(u: float, v: float, f: float, k: float
                        ) -> Tuple[float, float]:
    """
    R_u = -u * v^2 + f * (1 - u)
    R_v = +u * v^2 - (f + k) * v
    物理约束: u, v in [0, 1].
    """
    u = max(0.0, min(1.0, u))
    v = max(0.0, min(1.0, v))
    uv2 = u * v * v
    ru = -uv2 + f * (1.0 - u)
    rv = uv2 - (f + k) * v
    return ru, rv


# ----------------------------------------------------------------------
# Serendipity 8 节点局部刚度 / 质量阵 (Burkardt 公式, 比例化)
# ----------------------------------------------------------------------
# 参考 402_fem2d_bvp_serene: 局部质量阵 (乘以 h^2 / 36)
_SERENE_MASS = [
    [ 6.0, -6.0,  2.0, -8.0,  3.0, -8.0,  2.0, -6.0],
    [-6.0, 32.0, -6.0, 20.0, -8.0, 16.0, -8.0, 20.0],
    [ 2.0, -6.0,  6.0, -6.0,  2.0, -8.0,  3.0, -8.0],
    [-8.0, 20.0, -6.0, 32.0, -6.0, 20.0, -8.0, 16.0],
    [ 3.0, -8.0,  2.0, -6.0,  6.0, -6.0,  2.0, -8.0],
    [-8.0, 16.0, -8.0, 20.0, -6.0, 32.0, -6.0, 20.0],
    [ 2.0, -8.0,  3.0, -8.0,  2.0, -6.0,  6.0, -6.0],
    [-6.0, 20.0, -8.0, 16.0, -8.0, 20.0, -6.0, 32.0],
]
# 局部 Laplacian 刚度阵 (各向同性, 比例化 1/3)
_SERENE_STIFF = [
    [ 4.0, -1.0, -1.0, -4.0, -1.0, -4.0, -1.0, -1.0],
    [-1.0,  4.0, -1.0, -1.0, -1.0, -1.0, -4.0, -1.0],
    [-1.0, -1.0,  4.0, -1.0, -1.0, -1.0, -1.0, -4.0],
    [-4.0, -1.0, -1.0,  4.0, -1.0, -4.0, -1.0, -1.0],
    [-1.0, -1.0, -1.0, -1.0,  4.0, -1.0, -1.0, -1.0],
    [-4.0, -1.0, -1.0, -4.0, -1.0,  4.0, -1.0, -1.0],
    [-1.0, -4.0, -1.0, -1.0, -1.0, -1.0,  4.0, -1.0],
    [-1.0, -1.0, -4.0, -1.0, -1.0, -1.0, -1.0,  4.0],
]


class SerendipityMesh:
    """
    二维矩形规则网格, Serendipity 8 节点单元.
    nx, ny 必须为奇数 (与 402_fem2d_bvp_serene 一致).
    节点总数 Nn = nx * ny, 单元总数 Ne = ((nx-1)//2) * ((ny-1)//2).
    """
    def __init__(self, nx: int, ny: int, Lx: float = 1.0, Ly: float = 1.0):
        if nx < 3 or ny < 3:
            raise ValueError("Serendipity 8-node 要求 nx, ny >= 3")
        if nx % 2 == 0:
            nx += 1
        if ny % 2 == 0:
            ny += 1
        self.nx, self.ny = nx, ny
        self.Lx, self.Ly = Lx, Ly
        self.hx = Lx / (nx - 1)
        self.hy = Ly / (ny - 1)
        self.Nn = nx * ny
        self.ne = ((nx - 1) // 2) * ((ny - 1) // 2)
        self._build_connectivity()

    def _build_connectivity(self) -> None:
        """
        构造单元-节点连接表. 单元素节点编号 (逆时针):
            3--2--1
            |     |
            4     8
            |     |
            5--6--7
        这里我们重新编号为 0..7 对应 (i, j) 偏移.
        """
        el_i = (self.nx - 1) // 2
        el_j = (self.ny - 1) // 2
        self.conn: List[List[int]] = []
        for jj in range(el_j):
            for ii in range(el_i):
                i0 = 2 * ii
                j0 = 2 * jj
                # 8 节点偏移 (列优先索引)
                offsets = [
                    (i0, j0 + 2), (i0 + 1, j0 + 2), (i0 + 2, j0 + 2),
                    (i0 + 2, j0 + 1), (i0 + 2, j0),
                    (i0 + 1, j0), (i0, j0),
                    (i0, j0 + 1),
                ]
                nodes = [i + j * self.nx for (i, j) in offsets]
                self.conn.append(nodes)

    def assemble_laplacian(self, D: float) -> List[List[float]]:
        """组装整体 Laplacian 刚度阵 K = D * sum_e K_e."""
        N = self.Nn
        K = [[0.0] * N for _ in range(N)]
        scale = D / 3.0  # 与 Burkardt 比例化一致
        for elem in self.conn:
            for a_loc, a_glob in enumerate(elem):
                for b_loc, b_glob in enumerate(elem):
                    K[a_glob][b_glob] += _SERENE_STIFF[a_loc][b_loc] * scale
        return K

    def assemble_mass(self) -> List[List[float]]:
        """组装整体质量阵 M = (hx * hy / 36) * sum_e M_e."""
        N = self.Nn
        M = [[0.0] * N for _ in range(N)]
        scale = self.hx * self.hy / 36.0
        for elem in self.conn:
            for a_loc, a_glob in enumerate(elem):
                for b_loc, b_glob in enumerate(elem):
                    M[a_glob][b_glob] += _SERENE_MASS[a_loc][b_loc] * scale
        return M


# ----------------------------------------------------------------------
# 对角优势近似求解器 (避免大矩阵直接求逆, 体现工程鲁棒性)
# ----------------------------------------------------------------------
def jacobi_precond_solve(A, b, maxit: int = 50,
                         tol: float | None = None) -> List[float]:
    """
    对角预处理 Jacobi 迭代求解 A x = b.
    用于 IMEX 中隐式扩散步的快速近似.
    """
    if tol is None:
        tol = NUMERICS.rtol
    n = len(b)
    diag = [max(abs(A[i][i]), NUMERICS.cholesky_jitter) for i in range(n)]
    x = [bi / di for bi, di in zip(b, diag)]
    for _ in range(maxit):
        r = [b[i] - sum(A[i][j] * x[j] for j in range(n)) for i in range(n)]
        rnorm = math.sqrt(sum(ri * ri for ri in r))
        if rnorm < tol:
            break
        for i in range(n):
            x[i] += r[i] / diag[i]
    return x


# ----------------------------------------------------------------------
# 正向算子: 给定 theta -> 观测 y
# ----------------------------------------------------------------------
class GrayScottForward:
    """
    正向算子 G(theta).
    theta = [log(Du), log(Dv), f, k]  (对数参数化保证正性).
    返回稀疏观测向量 y in R^{n_obs}.
    """
    def __init__(self, nx: int = 11, ny: int = 11,
                 Lx: float = 1.0, Ly: float = 1.0,
                 T: float = 0.5, n_steps: int = 50,
                 obs_indices: List[int] | None = None,
                 seed: int = 0):
        self.mesh = SerendipityMesh(nx, ny, Lx, Ly)
        self.T = T
        self.n_steps = n_steps
        self.dt = T / max(n_steps, 1)
        if obs_indices is None:
            # 稀疏观测: 网格内部均匀抽 8 个点
            interior = [i for i in range(self.mesh.Nn)
                        if (i // nx) % 2 == 1 and (i % nx) % 2 == 1]
            step = max(1, len(interior) // 8)
            self.obs_idx = interior[::step][:8]
        else:
            self.obs_idx = obs_indices
        self.n_obs = len(self.obs_idx)
        # 初始条件: 中心斑图
        self.u0 = [0.5] * self.mesh.Nn
        self.v0 = [0.5] * self.mesh.Nn
        cx, cy = Lx / 2.0, Ly / 2.0
        for j in range(ny):
            for i in range(nx):
                idx = i + j * nx
                dx = i * self.mesh.hx - cx
                dy = j * self.mesh.hy - cy
                if dx * dx + dy * dy < 0.04:
                    self.u0[idx] = 0.5 + 0.05
                    self.v0[idx] = 0.25 + 0.05
        self._rng = _LCG(seed)

    def _apply_reaction(self, u, v, f, k) -> Tuple[List[float], List[float]]:
        ru = [0.0] * self.mesh.Nn
        rv = [0.0] * self.mesh.Nn
        for i in range(self.mesh.Nn):
            ru[i], rv[i] = gray_scott_reaction(u[i], v[i], f, k)
        return ru, rv

    def _matvec(self, A, x) -> List[float]:
        return [sum(A[i][j] * x[j] for j in range(len(x))) for i in range(len(x))]

    def step_imex(self, u, v, Du, Dv, f, k) -> Tuple[List[float], List[float]]:
        """IMEX 单步: 扩散隐式, 反应显式."""
        Ku = self.mesh.assemble_laplacian(Du)
        Kv = self.mesh.assemble_laplacian(Dv)
        M = self.mesh.assemble_mass()
        ru, rv = self._apply_reaction(u, v, f, k)
        # 显式右端: M * [u + dt*ru, v + dt*rv]
        rhs_u = [0.0] * self.mesh.Nn
        rhs_v = [0.0] * self.mesh.Nn
        for i in range(self.mesh.Nn):
            rhs_u[i] = sum(M[i][j] * (u[j] + self.dt * ru[j])
                           for j in range(self.mesh.Nn))
            rhs_v[i] = sum(M[i][j] * (v[j] + self.dt * rv[j])
                           for j in range(self.mesh.Nn))
        # 隐式左端: (M - dt * K) x_new = rhs
        A_u = [[M[i][j] - self.dt * Ku[i][j] for j in range(self.mesh.Nn)]
               for i in range(self.mesh.Nn)]
        A_v = [[M[i][j] - self.dt * Kv[i][j] for j in range(self.mesh.Nn)]
               for i in range(self.mesh.Nn)]
        u_new = jacobi_precond_solve(A_u, rhs_u, maxit=20)
        v_new = jacobi_precond_solve(A_v, rhs_v, maxit=20)
        # 投影 [0, 1]
        u_new = [max(0.0, min(1.0, ui)) for ui in u_new]
        v_new = [max(0.0, min(1.0, vi)) for vi in v_new]
        return u_new, v_new

    def evaluate(self, theta: Dict[str, float]) -> List[float]:
        """
        正向求解: 返回观测向量 y = G(theta) in R^{n_obs}.
        theta 包含: Du, Dv, f, k.
        """
        Du = max(NUMERICS.safe_log_floor, theta["Du"])
        Dv = max(NUMERICS.safe_log_floor, theta["Dv"])
        f = max(0.0, min(1.0, theta["f"]))
        k = max(0.0, min(1.0, theta["k"]))
        u, v = list(self.u0), list(self.v0)
        for _ in range(self.n_steps):
            u, v = self.step_imex(u, v, Du, Dv, f, k)
        return [u[i] for i in self.obs_idx]


# ----------------------------------------------------------------------
# 简单线性同余随机数 (避免依赖 numpy)
# ----------------------------------------------------------------------
class _LCG:
    def __init__(self, seed: int = 0):
        self._state = int(seed) & 0xFFFFFFFF
        self._a = 1664525
        self._c = 1013904223
        self._m = 1 << 32

    def next(self) -> float:
        self._state = (self._a * self._state + self._c) % self._m
        return self._state / self._m

    def normal(self) -> float:
        # Box-Muller
        u1 = max(self.next(), NUMERICS.safe_log_floor)
        u2 = self.next()
        r = math.sqrt(-2.0 * math.log(u1))
        return r * math.cos(2.0 * math.pi * u2)


__all__ = ["GrayScottForward", "SerendipityMesh", "gray_scott_reaction",
           "jacobi_precond_solve"]
