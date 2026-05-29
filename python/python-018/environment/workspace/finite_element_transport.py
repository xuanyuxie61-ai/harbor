"""
finite_element_transport.py

基于种子项目 391_fem1d_heat_implicit（一维热方程隐式有限元）
和 923_pwc_plot_1d（一维分段常数函数），实现拓扑超导体中的
准粒子输运与热传导的有限元求解。

物理模型：
    将一维热方程类比为非平衡马约拉纳准粒子的扩散过程：
        ∂_t n(x,t) - D ∂_x^2 n(x,t) = S(x,t)

    其中：
        n(x,t): 准粒子数密度
        D: 扩散系数（与跃迁强度t和散射率相关）
        S(x,t): 源项（如局域光泵浦或电注入）

    在拓扑超导体中，由于马约拉纳零能模的存在，
    边界处的有效扩散系数被修正为：
        D_eff = D_0 * (1 - P_A)
    其中P_A为Andreev反射概率。

    分段常数势垒模型用于描述超导能隙的阶跃变化：
        Δ(x) = Δ_0 * Θ(x - x_0)
    其中Θ为Heaviside阶跃函数。

有限元离散化：
    采用线性帽函数（hat function）作为基函数V_i(x)，
    在单元[x_i, x_{i+1}]上求解弱形式：

        ∫ (n_t V_i + D n_x V_{i,x}) dx = ∫ S V_i dx

    隐式Euler时间离散：
        (M + dt*K) n^{k+1} = M n^k + dt * F
    其中M为质量矩阵，K为刚度矩阵。
"""

import numpy as np
from typing import Callable, Tuple, Optional


class PiecewiseConstantPotential:
    """
    一维分段常数势垒/能隙模型（基于923_pwc_plot_1d）。

    用于描述超导纳米线中化学势或能隙的空间变化：
        V(x) = V_i,  x_i ≤ x < x_{i+1}
    """

    def __init__(self, breakpoints: np.ndarray, values: np.ndarray):
        """
        初始化分段常数函数。

        Args:
            breakpoints: 断点x坐标，长度为N+1
            values: 每个区间的函数值，长度为N
        """
        if len(breakpoints) != len(values) + 1:
            raise ValueError("断点数必须等于区间数加1")
        if np.any(np.diff(breakpoints) <= 0):
            raise ValueError("断点必须严格递增")

        self.x = np.asarray(breakpoints, dtype=np.float64)
        self.y = np.asarray(values, dtype=np.float64)
        self.n = len(values)

    def evaluate(self, x_query: np.ndarray) -> np.ndarray:
        """
        在查询点处求值。
        """
        x_query = np.asarray(x_query, dtype=np.float64)
        result = np.zeros_like(x_query)

        for i in range(self.n):
            mask = (x_query >= self.x[i]) & (x_query < self.x[i + 1])
            result[mask] = self.y[i]

        # 右端点处理
        result[x_query >= self.x[-1]] = self.y[-1]
        result[x_query < self.x[0]] = self.y[0]

        return result

    def get_interval_values(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        返回分段线段的坐标表示。
        """
        xp = np.zeros(2 * self.n + 2)
        yp = np.zeros(2 * self.n + 2)
        k = 0
        for i in range(self.n + 1):
            if i == 0:
                xp[k] = self.x[i]
                yp[k] = 0.0
                k += 1
                xp[k] = self.x[i]
                yp[k] = self.y[i]
                k += 1
            elif i < self.n:
                xp[k] = self.x[i]
                yp[k] = self.y[i - 1]
                k += 1
                xp[k] = self.x[i]
                yp[k] = self.y[i]
                k += 1
            else:
                xp[k] = self.x[i]
                yp[k] = self.y[i - 1]
                k += 1
                xp[k] = self.x[i]
                yp[k] = 0.0
                k += 1
                break

        return xp[:k], yp[:k]


class FEM1DTransport:
    """
    一维有限元输运求解器（基于391_fem1d_heat_implicit）。
    """

    def __init__(self, x_nodes: np.ndarray,
                 diffusion_coeff: float,
                 boundary_type: str = 'dirichlet'):
        """
        初始化FEM求解器。

        Args:
            x_nodes: 节点坐标数组
            diffusion_coeff: 扩散系数D
            boundary_type: 'dirichlet'或'neumann'
        """
        self.x = np.asarray(x_nodes, dtype=np.float64)
        self.n = len(x_nodes)
        self.D = diffusion_coeff
        self.boundary_type = boundary_type

        if self.n < 3:
            raise ValueError("至少需要3个节点")
        if np.any(np.diff(self.x) <= 0):
            raise ValueError("节点坐标必须严格递增")

        self._build_element_info()

    def _build_element_info(self) -> None:
        """
        构建单元信息。
        """
        self.num_elements = self.n - 1
        self.element_nodes = np.zeros((2, self.num_elements), dtype=int)
        for e in range(self.num_elements):
            self.element_nodes[0, e] = e
            self.element_nodes[1, e] = e + 1

    def _basis_function(self, xi: float, node: int) -> float:
        """
        参考单元[-1,1]上的线性基函数。
        """
        if node == 0:
            return 0.5 * (1.0 - xi)
        elif node == 1:
            return 0.5 * (1.0 + xi)
        else:
            return 0.0

    def _basis_gradient(self, xi: float, node: int) -> float:
        """
        参考单元上的基函数梯度。
        """
        if node == 0:
            return -0.5
        elif node == 1:
            return 0.5
        else:
            return 0.0

    def _assemble_matrices(self, t: float,
                           source_fn: Optional[Callable] = None
                           ) -> Tuple[np.ndarray, np.ndarray]:
        """
        组装质量矩阵M和刚度矩阵K。

        对于线性单元，质量矩阵（一致质量）：
            M_{ii} = h_e/3, M_{i,i+1} = h_e/6
        刚度矩阵：
            K_{ii} = D/h_e, K_{i,i+1} = -D/h_e
        """
        M = np.zeros((self.n, self.n))
        K = np.zeros((self.n, self.n))
        F = np.zeros(self.n)

        for e in range(self.num_elements):
            n1 = self.element_nodes[0, e]
            n2 = self.element_nodes[1, e]
            h_e = self.x[n2] - self.x[n1]

            if h_e < 1e-15:
                continue

            # 质量矩阵（梯形规则近似）
            M[n1, n1] += h_e / 3.0
            M[n1, n2] += h_e / 6.0
            M[n2, n1] += h_e / 6.0
            M[n2, n2] += h_e / 3.0

            # 刚度矩阵
            K[n1, n1] += self.D / h_e
            K[n1, n2] -= self.D / h_e
            K[n2, n1] -= self.D / h_e
            K[n2, n2] += self.D / h_e

            # 源项（使用节点值）
            if source_fn is not None:
                x_mid = 0.5 * (self.x[n1] + self.x[n2])
                s_mid = source_fn(x_mid, t)
                F[n1] += s_mid * h_e / 2.0
                F[n2] += s_mid * h_e / 2.0

        return M, K, F

    def solve_implicit(self, u_old: np.ndarray,
                       dt: float, t: float,
                       source_fn: Optional[Callable] = None,
                       bc_left: float = 0.0,
                       bc_right: float = 0.0) -> np.ndarray:
        """
        隐式Euler求解一步。

        系统方程：
            (M + dt*K) u^{new} = M*u^{old} + dt*F
        """
        if len(u_old) != self.n:
            raise ValueError("旧解向量长度必须与节点数匹配")
        if dt <= 0:
            raise ValueError("时间步长必须为正")

        M, K, F = self._assemble_matrices(t, source_fn)
        A = M + dt * K
        b = M @ u_old + dt * F

        # 施加Dirichlet边界条件
        if self.boundary_type == 'dirichlet':
            A[0, :] = 0.0
            A[0, 0] = 1.0
            b[0] = bc_left
            A[-1, :] = 0.0
            A[-1, -1] = 1.0
            b[-1] = bc_right

        # 求解
        u_new = np.linalg.solve(A, b)
        return u_new

    def solve_time_dependent(self, u0: np.ndarray,
                              t_final: float,
                              num_steps: int,
                              source_fn: Optional[Callable] = None,
                              bc_left: float = 0.0,
                              bc_right: float = 0.0) -> Tuple[np.ndarray,
                                                                 np.ndarray]:
        """
        时间依赖求解。

        Returns:
            t_array: 时间数组
            u_history: 形状为(num_steps+1, n)的解历史
        """
        if len(u0) != self.n:
            raise ValueError("初始条件长度必须匹配")

        dt = t_final / num_steps
        t_array = np.linspace(0.0, t_final, num_steps + 1)
        u_history = np.zeros((num_steps + 1, self.n))
        u_history[0] = u0

        for k in range(num_steps):
            u_history[k + 1] = self.solve_implicit(
                u_history[k], dt, t_array[k + 1],
                source_fn, bc_left, bc_right
            )

        return t_array, u_history

    def compute_heat_flux(self, u: np.ndarray) -> np.ndarray:
        """
        计算热流/粒子流（Fourier/Fick定律）。

        J = -D ∂_x u
        """
        flux = np.zeros(self.n)
        for i in range(self.n - 1):
            h_e = self.x[i + 1] - self.x[i]
            if h_e > 1e-15:
                grad = (u[i + 1] - u[i]) / h_e
                flux[i] -= self.D * grad
                flux[i + 1] -= self.D * grad
        return flux

    def effective_diffusion_with_majorana(self,
                                           delta_profile: PiecewiseConstantPotential,
                                           energy: float = 0.0) -> float:
        """
        计算考虑Andreev反射的有效扩散系数。

        D_eff = D_0 * <1 - P_A(x)>
        其中P_A(x) = |Δ(x)|^2 / (E^2 + |Δ(x)|^2)
        """
        x_mid = 0.5 * (self.x[:-1] + self.x[1:])
        delta_vals = delta_profile.evaluate(x_mid)
        p_andreev = delta_vals ** 2 / (energy ** 2 + delta_vals ** 2 + 1e-15)
        d_eff = self.D * np.mean(1.0 - p_andreev)
        return float(d_eff)


def demo():
    """演示有限元输运求解。"""
    x_nodes = np.linspace(0.0, 10.0, 51)
    fem = FEM1DTransport(x_nodes, diffusion_coeff=0.1)

    # 分段常数势
    pwc = PiecewiseConstantPotential(
        breakpoints=np.array([0.0, 3.0, 7.0, 10.0]),
        values=np.array([0.8, 0.0, 0.8])
    )

    # 初始条件：高斯波包
    u0 = np.exp(-(x_nodes - 5.0) ** 2 / 0.5)

    # 源项
    def source(x, t):
        return 0.1 * np.exp(-(x - 5.0) ** 2 / 1.0)

    t_array, u_history = fem.solve_time_dependent(
        u0=u0, t_final=2.0, num_steps=100,
        source_fn=source, bc_left=0.0, bc_right=0.0
    )

    print("FEM transport final total density:", np.trapezoid(u_history[-1], x_nodes))

    d_eff = fem.effective_diffusion_with_majorana(pwc, energy=0.0)
    print("Effective diffusion coefficient:", d_eff)


if __name__ == "__main__":
    demo()
