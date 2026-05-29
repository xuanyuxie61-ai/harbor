r"""
detector_field.py
探测器内部电场有限元求解模块

本模块实现一维泊松方程的有限元求解，用于计算暗物质探测器
（如低温锗探测器、液氙 TPC）内部的电场分布。

控制方程：
    -\frac{d}{dz}\left( \epsilon(z) \frac{d\phi}{dz} \right) = \rho(z), \quad z \in [0, L]

边界条件：
    \phi(0) = V_0  （阴极电压）
    \phi(L) = V_L  （阳极电压）

离散化：
    采用分段线性有限元（P1-FEM），单元 e = [z_i, z_{i+1}]，
    局部基函数：
        N_1^e(\xi) = 1 - \xi, \quad N_2^e(\xi) = \xi, \quad \xi = \frac{z - z_i}{h_e}

    单元刚度矩阵：
        K^e = \frac{\epsilon_e}{h_e} \begin{bmatrix} 1 & -1 \\ -1 & 1 \end{bmatrix}

    单元载荷向量：
        F^e = \frac{\rho_e h_e}{2} \begin{bmatrix} 1 \\ 1 \end{bmatrix}

全局组装后得到三对角（五对角若 ε 变化剧烈）线性系统：
    A \Phi = F

本模块同时包含 R85 格式的五对角矩阵求解器（来自 968_r85），
用于处理高阶差分或耦合方程组的情形。

参考文献：
- Cheney, W., & Kincaid, D. (1985). Numerical Mathematics and Computing.
- Hughes, T. J. R. (2000). The Finite Element Method.
"""

import numpy as np
from typing import Tuple
from utils import r8vec_bracket4


# ============================================================================
# 一维有限元求解器
# ============================================================================

class FEM1DSolver:
    """
    一维有限元泊松方程求解器。
    """

    def __init__(self, nodes: np.ndarray, permittivity: np.ndarray, charge_density: np.ndarray):
        """
        参数：
            nodes: (N,) 网格节点坐标 [m]
            permittivity: (N-1,) 或 (N,) 介电常数分布
            charge_density: (N-1,) 或 (N,) 空间电荷密度 [C/m^3]
        """
        self.nodes = np.asarray(nodes, dtype=float)
        self.n = len(self.nodes)
        if self.n < 2:
            raise ValueError("FEM1DSolver: 节点数至少为 2")

        self.permittivity = np.asarray(permittivity, dtype=float)
        if len(self.permittivity) == self.n - 1:
            self.epsilon = self.permittivity
        elif len(self.permittivity) == self.n:
            self.epsilon = 0.5 * (self.permittivity[:-1] + self.permittivity[1:])
        else:
            raise ValueError("FEM1DSolver: permittivity 长度不匹配")

        self.charge_density = np.asarray(charge_density, dtype=float)
        if len(self.charge_density) == self.n - 1:
            self.rho = self.charge_density
        elif len(self.charge_density) == self.n:
            self.rho = 0.5 * (self.charge_density[:-1] + self.charge_density[1:])
        else:
            raise ValueError("FEM1DSolver: charge_density 长度不匹配")

    def _assemble(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        组装全局刚度矩阵 A（三对角稀疏）和载荷向量 F。

        返回：
            K: (N, N) 稠密矩阵（简化实现，实际应为稀疏）
            F: (N,) 载荷向量
        """
        K = np.zeros((self.n, self.n))
        F = np.zeros(self.n)

        for e in range(self.n - 1):
            h_e = self.nodes[e + 1] - self.nodes[e]
            if h_e <= 0.0:
                raise ValueError(f"FEM1DSolver: 单元 {e} 长度非正")

            eps_e = self.epsilon[e]
            rho_e = self.rho[e]

            # 单元刚度矩阵
            ke = (eps_e / h_e) * np.array([[1.0, -1.0], [-1.0, 1.0]])
            # 单元载荷向量
            fe = (rho_e * h_e / 2.0) * np.array([1.0, 1.0])

            # 组装到全局
            K[e, e] += ke[0, 0]
            K[e, e + 1] += ke[0, 1]
            K[e + 1, e] += ke[1, 0]
            K[e + 1, e + 1] += ke[1, 1]

            F[e] += fe[0]
            F[e + 1] += fe[1]

        return K, F

    def solve_dirichlet(self, phi_left: float, phi_right: float) -> np.ndarray:
        """
        在 Dirichlet 边界条件下求解电势。

        参数：
            phi_left: z = nodes[0] 处电势 [V]
            phi_right: z = nodes[-1] 处电势 [V]

        返回：
            phi: (N,) 节点电势 [V]
        """
        K, F = self._assemble()

        # 施加 Dirichlet 边界条件（消去法）
        F[0] = phi_left
        F[1] -= K[1, 0] * phi_left
        K[0, :] = 0.0
        K[:, 0] = 0.0
        K[0, 0] = 1.0

        F[-1] = phi_right
        F[-2] -= K[-2, -1] * phi_right
        K[-1, :] = 0.0
        K[:, -1] = 0.0
        K[-1, -1] = 1.0

        # 求解线性系统
        phi = np.linalg.solve(K, F)
        return phi

    def compute_electric_field(self, phi: np.ndarray) -> np.ndarray:
        """
        由电势计算电场 E = -dφ/dz（单元常数近似）。

        参数：
            phi: (N,) 节点电势

        返回：
            E: (N-1,) 单元电场 [V/m]
        """
        E = np.zeros(self.n - 1)
        for e in range(self.n - 1):
            h_e = self.nodes[e + 1] - self.nodes[e]
            E[e] = -(phi[e + 1] - phi[e]) / h_e
        return E

    def evaluate_at_points(self, phi: np.ndarray, query_points: np.ndarray) -> np.ndarray:
        """
        将有限元解插值到任意查询点（参考 fem1d_sample）。

        算法：
            对每个查询点 z_q，找到所在单元 [z_i, z_{i+1}]，
            计算局部坐标 ξ = (z_q - z_i) / h_e，
            线性插值：φ(z_q) = (1-ξ) φ_i + ξ φ_{i+1}。
        """
        query_points = np.asarray(query_points, dtype=float)
        nq = len(query_points)
        result = np.zeros(nq)
        for iq in range(nq):
            zq = query_points[iq]
            if zq <= self.nodes[0]:
                result[iq] = phi[0]
                continue
            if zq >= self.nodes[-1]:
                result[iq] = phi[-1]
                continue
            idx = r8vec_bracket4(self.n, self.nodes, zq)
            h_e = self.nodes[idx + 1] - self.nodes[idx]
            if h_e <= 0.0:
                result[iq] = phi[idx]
                continue
            xi = (zq - self.nodes[idx]) / h_e
            result[iq] = (1.0 - xi) * phi[idx] + xi * phi[idx + 1]
        return result


# ============================================================================
# 五对角矩阵求解器（R85 格式）
# ============================================================================

def r85_np_fs(n: int, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    求解五对角线性系统 A x = b，不选主元的高斯消去。

    矩阵存储格式（R85）：
        a 为 (5, n) 数组，各行分别存储：
        a[0, :] : 主对角线 A[i, i]
        a[1, :] : 第一上对角线 A[i, i+1]
        a[2, :] : 第二上对角线 A[i, i+2]
        a[3, :] : 第一下对角线 A[i, i-1]
        a[4, :] : 第二下对角线 A[i, i-2]

    参数：
        n: 矩阵维度
        a: (5, n) 五对角带矩阵
        b: (n,) 右端项

    返回：
        x: (n,) 解向量

    注意：
        此实现会覆盖输入矩阵 a！
        边界条件自动处理：不足的对角线元素视为 0。

    参考文献：
        Cheney & Kincaid, Numerical Mathematics and Computing, 1985.
    """
    if n < 1:
        raise ValueError("r85_np_fs: n 必须 >= 1")
    a = np.copy(a)
    b = np.copy(b)
    x = np.zeros(n)

    # 前向消去
    for i in range(n):
        # 主元
        pivot = a[0, i]
        if abs(pivot) < 1.0e-30:
            # 数值稳定性保护：若主元过小，设为极小值避免除零
            pivot = 1.0e-30 if pivot >= 0 else -1.0e-30
            a[0, i] = pivot

        # 消去第一下对角线
        if i + 1 < n:
            factor1 = a[3, i + 1] / pivot
            a[3, i + 1] = factor1
            a[0, i + 1] -= factor1 * a[1, i]
            if i + 2 < n:
                a[1, i + 1] -= factor1 * a[2, i]
            b[i + 1] -= factor1 * b[i]

        # 消去第二下对角线
        if i + 2 < n:
            factor2 = a[4, i + 2] / pivot
            a[4, i + 2] = factor2
            a[3, i + 2] -= factor2 * a[1, i]
            a[0, i + 2] -= factor2 * a[2, i]
            b[i + 2] -= factor2 * b[i]

    # 回代
    x[-1] = b[-1] / a[0, -1]
    if n >= 2:
        x[-2] = (b[-2] - a[1, -2] * x[-1]) / a[0, -2]
    for i in range(n - 3, -1, -1):
        x[i] = (b[i] - a[1, i] * x[i + 1] - a[2, i] * x[i + 2]) / a[0, i]

    return x


def r85_dif2(n: int) -> np.ndarray:
    """
    构造五对角二阶差分矩阵（用于扩散方程离散）。

    矩阵形式：
        A[i, i]   = 2
        A[i, i+1] = A[i, i-1] = -1
        A[i, i+2] = A[i, i-2] = 0  （边界附近自动截断）

    参数：
        n: 维度

    返回：
        a: (5, n) R85 格式矩阵
    """
    a = np.zeros((5, n))
    a[0, :] = 2.0
    if n > 1:
        a[1, :-1] = -1.0
        a[3, 1:] = -1.0
    return a


def solve_diffusion_1d(
    n: int,
    D: float,
    sigma_a: float,
    source: np.ndarray,
    dx: float,
    bc_left: float,
    bc_right: float,
) -> np.ndarray:
    """
    求解一维稳态扩散方程：

        -D \frac{d^2 \phi}{dx^2} + \Sigma_a \phi = S(x), \quad x \in [0, L]

    采用中心差分，生成五对角系统后用 R85 求解器求解。

    差分格式：
        -D \frac{\phi_{i-1} - 2\phi_i + \phi_{i+1}}{\Delta x^2}
        + \Sigma_a \phi_i = S_i

    整理得：
        (-\frac{D}{\Delta x^2}) \phi_{i-1}
        + (\frac{2D}{\Delta x^2} + \Sigma_a) \phi_i
        + (-\frac{D}{\Delta x^2}) \phi_{i+1} = S_i

    参数：
        n: 网格点数
        D: 扩散系数
        sigma_a: 吸收截面
        source: (n,) 源项
        dx: 网格间距
        bc_left, bc_right: Dirichlet 边界值

    返回：
        phi: (n,) 解
    """
    if n < 3:
        raise ValueError("solve_diffusion_1d: n 必须 >= 3")
    if len(source) != n:
        raise ValueError("solve_diffusion_1d: source 长度必须等于 n")

    a = np.zeros((5, n))
    diag = 2.0 * D / (dx * dx) + sigma_a
    offdiag = -D / (dx * dx)

    a[0, :] = diag
    a[1, :-1] = offdiag
    a[3, 1:] = offdiag

    b = np.copy(source)
    # 施加边界
    b[0] = bc_left
    b[-1] = bc_right
    a[0, 0] = 1.0
    a[1, 0] = 0.0
    a[0, -1] = 1.0
    a[3, -1] = 0.0
    if n > 1:
        a[3, 1] = 0.0
        a[1, -2] = 0.0

    return r85_np_fs(n, a, b)


# ============================================================================
# 自测
# ============================================================================

if __name__ == "__main__":
    # 测试 FEM：线性电势，无电荷，phi(0)=0, phi(1)=1
    nodes = np.linspace(0.0, 1.0, 11)
    eps = np.ones(10)
    rho = np.zeros(10)
    solver = FEM1DSolver(nodes, eps, rho)
    phi = solver.solve_dirichlet(0.0, 1.0)
    assert abs(phi[0]) < 1e-10, "左边界条件未满足"
    assert abs(phi[-1] - 1.0) < 1e-10, "右边界条件未满足"
    # 线性插值误差
    for i in range(len(nodes)):
        expected = nodes[i]
        assert abs(phi[i] - expected) < 1e-10, f"FEM 线性解偏差: {phi[i]} vs {expected}"

    # 测试 R85 求解器
    a = r85_dif2(5)
    b = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    x = r85_np_fs(5, a, b)
    # 验证 A x = b
    full = np.zeros((5, 5))
    for i in range(5):
        full[i, i] = a[0, i]
        if i + 1 < 5:
            full[i, i + 1] = a[1, i]
        if i - 1 >= 0:
            full[i, i - 1] = a[3, i]
    residual = np.linalg.norm(full @ x - b)
    assert residual < 1e-10, f"R85 残差过大: {residual}"

    # 测试扩散方程
    n = 21
    dx = 0.05
    phi_diff = solve_diffusion_1d(n, D=1.0, sigma_a=0.1, source=np.zeros(n), dx=dx, bc_left=0.0, bc_right=1.0)
    assert abs(phi_diff[0]) < 1e-10
    assert abs(phi_diff[-1] - 1.0) < 1e-10
    assert np.all(np.isfinite(phi_diff)), "扩散方程解含非有限值"

    print("detector_field.py: 所有自测通过")
