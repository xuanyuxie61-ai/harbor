"""
一维稳态热扩散模块：语义嵌入空间中的信息扩散模拟

原项目映射: 362_fd1d_heat_steady
科学背景: 使用有限差分法求解一维稳态热传导方程，
          模拟语义信息在嵌入空间中的稳态扩散分布。

数学模型:
    稳态热传导方程:
        -d/dx ( K(x) dU/dx ) = F(x),  x \in [A, B]
    
    边界条件:
        U(A) = U_A,  U(B) = U_B
    
    其中:
        K(x) 是热导率（对应语义扩散系数）
        F(x) 是热源项（对应语义信息注入率）
    
    有限差分离散（二阶精度）:
        对内部节点 i = 2,...,N-1:
            -K_{i-1/2} * U_{i-1} / dx^2
            + (K_{i-1/2} + K_{i+1/2}) * U_i / dx^2
            - K_{i+1/2} * U_{i+1} / dx^2
            = F(x_i)
        
        其中 K_{i\pm 1/2} = K(x_i \pm dx/2)

在NLP语义嵌入中的应用:
    将语义嵌入空间的一维切片视为导热介质，
    研究语义信息从高密度区域向低密度区域的稳态扩散分布，
    为理解语义传播提供物理模型。
"""

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve


class SemanticHeatDiffusion:
    """
    语义信息稳态热扩散系统。
    
    模拟语义信息在嵌入空间一维路径上的稳态扩散过程。
    """

    def __init__(self, n: int = 41, a: float = 0.0, b: float = 1.0):
        """
        初始化扩散系统。
        
        Parameters
        ----------
        n : int
            网格节点数，必须 >= 3。
        a, b : float
            空间区间端点，a < b。
        """
        if n < 3:
            raise ValueError(f"n must be at least 3, got {n}")
        if a >= b:
            raise ValueError(f"must have a < b, got a={a}, b={b}")

        self.n = int(n)
        self.a = float(a)
        self.b = float(b)
        self.dx = (b - a) / (n - 1)
        self.x = np.linspace(a, b, n)

    def solve(self, ua: float, ub: float,
              conductivity_func, source_func) -> np.ndarray:
        """
        求解稳态热扩散方程。
        
        Parameters
        ----------
        ua, ub : float
            左右边界条件 U(a) = ua, U(b) = ub。
        conductivity_func : callable
            热导率函数 K(x)。
        source_func : callable
            热源项函数 F(x)。
            
        Returns
        -------
        np.ndarray
            稳态解 U(x)。
        """
        # TODO [Hole 1]: 实现稳态热扩散方程的有限差分离散求解
        # 需要构建并求解三对角线性系统 A U = rhs
        # 内部节点使用变系数离散格式:
        #   -K_{i-1/2} * U_{i-1}/dx^2 + (K_{i-1/2}+K_{i+1/2}) * U_i/dx^2 - K_{i+1/2} * U_{i+1}/dx^2 = F(x_i)
        # 边界节点使用 Dirichlet 条件: U(a)=ua, U(b)=ub
        # 返回稳态解 U
        raise NotImplementedError("Hole 1: 热扩散求解器尚未实现")

    def compute_flux(self, U: np.ndarray, conductivity_func) -> np.ndarray:
        """
        计算热流密度（语义信息流）。
        
        物理公式:
            q(x) = -K(x) * dU/dx
            
        使用中心差分:
            dU/dx|_i \approx (U_{i+1} - U_{i-1}) / (2*dx)
        """
        flux = np.zeros(self.n)
        for i in range(1, self.n - 1):
            dUdx = (U[i + 1] - U[i - 1]) / (2.0 * self.dx)
            flux[i] = -conductivity_func(self.x[i]) * dUdx

        # 边界热流使用单侧差分
        flux[0] = -conductivity_func(self.x[0]) * (U[1] - U[0]) / self.dx
        flux[self.n - 1] = -conductivity_func(self.x[self.n - 1]) * (U[self.n - 1] - U[self.n - 2]) / self.dx

        return flux

    def solve_nonuniform_conductivity(self, ua: float, ub: float,
                                       K_values: np.ndarray, F_values: np.ndarray) -> np.ndarray:
        """
        求解非均匀热导率的稳态扩散方程。
        
        Parameters
        ----------
        K_values : np.ndarray
            各节点上的热导率值，长度必须为 n。
        F_values : np.ndarray
            各节点上的热源项值，长度必须为 n。
        """
        if len(K_values) != self.n or len(F_values) != self.n:
            raise ValueError("K_values and F_values must have length n")

        if np.any(K_values <= 0.0):
            raise ValueError("all conductivity values must be positive")

        # 使用节点平均值近似界面值
        data_main = np.zeros(self.n)
        data_lower = np.zeros(self.n - 1)
        data_upper = np.zeros(self.n - 1)
        rhs = np.zeros(self.n)

        data_main[0] = 1.0
        rhs[0] = ua

        for i in range(1, self.n - 1):
            k_left = 0.5 * (K_values[i - 1] + K_values[i])
            k_right = 0.5 * (K_values[i] + K_values[i + 1])

            data_lower[i - 1] = -k_left / (self.dx * self.dx)
            data_main[i] = (k_left + k_right) / (self.dx * self.dx)
            data_upper[i] = -k_right / (self.dx * self.dx)
            rhs[i] = F_values[i]

        data_main[self.n - 1] = 1.0
        rhs[self.n - 1] = ub

        A = csr_matrix(
            (np.concatenate([data_lower, data_main, data_upper]),
             (np.concatenate([np.arange(1, self.n), np.arange(self.n), np.arange(self.n - 1)]),
              np.concatenate([np.arange(self.n - 1), np.arange(self.n), np.arange(1, self.n)]))),
            shape=(self.n, self.n)
        )

        U = spsolve(A, rhs)
        return U


def demo():
    """模块功能演示"""
    print("=" * 60)
    print("语义信息稳态热扩散演示")
    print("=" * 60)

    diffusion = SemanticHeatDiffusion(n=41, a=0.0, b=1.0)
    print(f"\n网格数: {diffusion.n}")
    print(f"空间步长: {diffusion.dx:.6f}")

    # 边界条件: 左端语义密度高，右端低
    ua = 1.0
    ub = 0.0

    # 热导率: 中间区域扩散快
    def conductivity(x):
        return 1.0 + 0.5 * np.sin(np.pi * x) ** 2

    # 热源: 在 x=0.3 附近有语义信息注入
    def source(x):
        return 2.0 * np.exp(-((x - 0.3) ** 2) / 0.01)

    U = diffusion.solve(ua, ub, conductivity, source)
    flux = diffusion.compute_flux(U, conductivity)

    print(f"\n稳态解范围: [{U.min():.6f}, {U.max():.6f}]")
    print(f"热流范围: [{flux.min():.6f}, {flux.max():.6f}]")
    print(f"边界热流 (左): {flux[0]:.6f}")
    print(f"边界热流 (右): {flux[-1]:.6f}")

    # 质量守恒检查: 积分(F dx) = q(B) - q(A) = flux[-1] - flux[0]
    total_source = np.trapezoid([source(xi) for xi in diffusion.x], diffusion.x)
    net_flux = flux[-1] - flux[0]
    print(f"\n质量守恒检查:")
    print(f"  总热源: {total_source:.6f}")
    print(f"  净热流: {net_flux:.6f}")
    print(f"  相对偏差: {abs(total_source - net_flux) / (abs(total_source) + 1e-15):.6e}")

    print("\n模块运行完成")
    return diffusion, U, flux


if __name__ == "__main__":
    demo()
