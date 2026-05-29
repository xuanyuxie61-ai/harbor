"""
双向反应动力学模块：语义嵌入的化学交互演化

原项目映射: 1018_reaction_twoway_ode
科学背景: 描述两种化学物质之间的双向反应动力学:
            W1 --k1--> W2
            W2 --k2--> W1
          
          精确解:
            w1(t) = [k2*(w10+w20) + exp(-(k1+k2)*t)*(k1*w10 - k2*w20)] / (k1+k2)
            w2(t) = [k1*(w10+w20) - exp(-(k1+k2)*t)*(k1*w10 - k2*w20)] / (k1+k2)

在NLP语义嵌入中的应用:
    将语义概念视为化学反应中的物质，描述语义信息在
    不同概念域之间的双向转化过程。
    
    例如:
        - 语义漂移: 词义随时间从 W1 转化为 W2
        - 语义回流: 通过上下文影响，部分 W2 又转回 W1
        - 稳态: 当 t -> infinity 时达到动态平衡
"""

import numpy as np
from scipy.integrate import solve_ivp


class SemanticReactionDynamics:
    """
    语义双向反应动力学系统。
    
    模拟语义概念之间的双向转化过程。
    """

    def __init__(self, k1: float = 0.1, k2: float = 0.05,
                 w10: float = 1.0, w20: float = 0.0,
                 t0: float = 0.0, tstop: float = 100.0):
        """
        初始化反应动力学参数。
        
        Parameters
        ----------
        k1, k2 : float
            正向和逆向反应速率常数，必须 >= 0。
        w10, w20 : float
            初始语义浓度。
        t0, tstop : float
            时间区间。
        """
        if k1 < 0.0 or k2 < 0.0:
            raise ValueError(f"reaction rates must be non-negative, got k1={k1}, k2={k2}")

        self.k1 = float(k1)
        self.k2 = float(k2)
        self.w10 = float(w10)
        self.w20 = float(w20)
        self.t0 = float(t0)
        self.tstop = float(tstop)

    def exact_solution(self, t: np.ndarray) -> tuple:
        """
        计算精确解。
        
        Parameters
        ----------
        t : np.ndarray
            时间点数组。
            
        Returns
        -------
        tuple
            (w1, w2) 精确解数组。
            
        数学公式:
            w1(t) = [k2*(w10+w20) + exp(-(k1+k2)*t)*(k1*w10 - k2*w20)] / (k1+k2)
            w2(t) = [k1*(w10+w20) - exp(-(k1+k2)*t)*(k1*w10 - k2*w20)] / (k1+k2)
            
        守恒量:
            w1(t) + w2(t) = w10 + w20  (常数)
        """
        # TODO [Hole 2]: 实现双向反应动力学的精确解析解
        # 对于线性 ODE 系统:
        #   dw1/dt = -k1*w1 + k2*w2
        #   dw2/dt =  k1*w1 - k2*w2
        # 初始条件: w1(0)=w10, w2(0)=w20
        # 守恒量: w1(t) + w2(t) = w10 + w20
        # 需要给出 w1(t) 和 w2(t) 的闭式表达式
        # 考虑退化情况 k1=k2=0
        raise NotImplementedError("Hole 2: 反应动力学精确解尚未实现")

    def derivative(self, t: float, y: np.ndarray) -> np.ndarray:
        """
        计算ODE右端项。
        
        方程组:
            dw1/dt = -k1*w1 + k2*w2
            dw2/dt =  k1*w1 - k2*w2
            
        守恒:
            d(w1+w2)/dt = 0
        """
        w1, w2 = y[0], y[1]
        return np.array([
            -self.k1 * w1 + self.k2 * w2,
            self.k1 * w1 - self.k2 * w2
        ])

    def solve_numerical(self, num_points: int = 200) -> tuple:
        """
        数值求解ODE系统。
        
        Returns
        -------
        tuple
            (t_array, w1_array, w2_array)
        """
        t_eval = np.linspace(self.t0, self.tstop, num_points)
        y0 = np.array([self.w10, self.w20])

        sol = solve_ivp(
            fun=self.derivative,
            t_span=[self.t0, self.tstop],
            y0=y0,
            t_eval=t_eval,
            method='RK45',
            rtol=1e-9,
            atol=1e-12
        )

        return sol.t, sol.y[0, :], sol.y[1, :]

    def equilibrium(self) -> tuple:
        """
        计算稳态平衡值。
        
        当 t -> infinity:
            w1_eq = k2 * (w10 + w20) / (k1 + k2)
            w2_eq = k1 * (w10 + w20) / (k1 + k2)
        """
        k_sum = self.k1 + self.k2
        if abs(k_sum) < 1e-15:
            return self.w10, self.w20
        w1_eq = self.k2 * (self.w10 + self.w20) / k_sum
        w2_eq = self.k1 * (self.w10 + self.w20) / k_sum
        return w1_eq, w2_eq

    def relaxation_time(self) -> float:
        """
        计算弛豫时间。
        
        特征时间尺度:
            tau = 1 / (k1 + k2)
        """
        k_sum = self.k1 + self.k2
        if abs(k_sum) < 1e-15:
            return float('inf')
        return 1.0 / k_sum

    def conserved_quantity(self, w1: np.ndarray, w2: np.ndarray) -> np.ndarray:
        """
        验证守恒量。
        
        守恒量:
            C = w1 + w2 = w10 + w20
        """
        return w1 + w2


class MultiConceptReactionNetwork:
    """
    多概念反应网络。
    
    将双向反应动力学扩展到 N 个语义概念之间的反应网络。
    """

    def __init__(self, n_concepts: int, rate_matrix: np.ndarray, initial_state: np.ndarray):
        """
        初始化多概念反应网络。
        
        Parameters
        ----------
        n_concepts : int
            概念数量。
        rate_matrix : np.ndarray
            n x n 反应速率矩阵，rate_matrix[i,j] 表示从概念 j 到概念 i 的转化速率。
            对角线元素应为 -(其他列之和)。
        initial_state : np.ndarray
            初始语义浓度。
        """
        self.n = int(n_concepts)
        self.K = np.asarray(rate_matrix, dtype=float)
        self.y0 = np.asarray(initial_state, dtype=float)

        if self.K.shape != (self.n, self.n):
            raise ValueError(f"rate_matrix must be {self.n}x{self.n}")
        if len(self.y0) != self.n:
            raise ValueError(f"initial_state must have length {self.n}")

        # 验证守恒性: 每列之和应为 0
        col_sums = np.sum(self.K, axis=0)
        if np.max(np.abs(col_sums)) > 1e-10:
            raise ValueError("rate_matrix columns must sum to 0 for mass conservation")

    def derivative(self, t: float, y: np.ndarray) -> np.ndarray:
        """
        dy/dt = K @ y
        """
        return self.K @ y

    def solve(self, t_span: tuple, num_points: int = 500) -> tuple:
        """
        求解多概念反应网络。
        """
        t_eval = np.linspace(t_span[0], t_span[1], num_points)
        sol = solve_ivp(
            fun=self.derivative,
            t_span=t_span,
            y0=self.y0,
            t_eval=t_eval,
            method='RK45',
            rtol=1e-9,
            atol=1e-12
        )
        return sol.t, sol.y

    def equilibrium_state(self) -> np.ndarray:
        """
        计算稳态: 解 K @ y_eq = 0, sum(y_eq) = sum(y0)
        """
        # 构造增广系统
        A = np.zeros((self.n + 1, self.n))
        A[:self.n, :] = self.K
        A[self.n, :] = 1.0  # 守恒条件
        b = np.zeros(self.n + 1)
        b[self.n] = np.sum(self.y0)

        # 最小二乘求解
        y_eq, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
        return y_eq


def demo():
    """模块功能演示"""
    print("=" * 60)
    print("语义双向反应动力学演示")
    print("=" * 60)

    reaction = SemanticReactionDynamics(k1=0.3, k2=0.1, w10=1.0, w20=0.0,
                                        t0=0.0, tstop=50.0)
    print(f"\n反应参数:")
    print(f"  k1 = {reaction.k1}, k2 = {reaction.k2}")
    print(f"  w10 = {reaction.w10}, w20 = {reaction.w20}")

    t_num, w1_num, w2_num = reaction.solve_numerical(num_points=200)

    # 与精确解比较
    w1_exact, w2_exact = reaction.exact_solution(t_num)
    error1 = np.max(np.abs(w1_num - w1_exact))
    error2 = np.max(np.abs(w2_num - w2_exact))

    print(f"\n数值解与精确解最大偏差:")
    print(f"  w1: {error1:.6e}")
    print(f"  w2: {error2:.6e}")

    # 守恒量检查
    conserved = reaction.conserved_quantity(w1_num, w2_num)
    conserved_exact = reaction.w10 + reaction.w20
    print(f"\n守恒量检查 (w1+w2):")
    print(f"  理论值: {conserved_exact:.6f}")
    print(f"  数值范围: [{conserved.min():.10f}, {conserved.max():.10f}]")
    print(f"  最大偏差: {np.max(np.abs(conserved - conserved_exact)):.6e}")

    # 稳态
    w1_eq, w2_eq = reaction.equilibrium()
    print(f"\n稳态平衡:")
    print(f"  w1_eq = {w1_eq:.6f}")
    print(f"  w2_eq = {w2_eq:.6f}")
    print(f"  弛豫时间 tau = {reaction.relaxation_time():.6f}")

    # 多概念网络演示
    print("\n" + "-" * 40)
    print("多概念反应网络演示")
    print("-" * 40)

    n = 4
    # 构造一个环状反应网络
    K = np.zeros((n, n))
    for i in range(n):
        j_next = (i + 1) % n
        K[j_next, i] = 0.2  # i -> j_next
        K[i, i] -= 0.2
        j_prev = (i - 1) % n
        K[j_prev, i] = 0.1  # i -> j_prev
        K[i, i] -= 0.1

    y0 = np.array([1.0, 0.0, 0.0, 0.0])
    network = MultiConceptReactionNetwork(n, K, y0)
    t_net, y_net = network.solve((0.0, 50.0), num_points=200)

    print(f"概念数: {n}")
    print(f"稳态分布: {network.equilibrium_state()}")
    print(f"最终状态: {y_net[:, -1]}")

    print("\n模块运行完成")
    return reaction, network


if __name__ == "__main__":
    demo()
