
import numpy as np
from scipy.integrate import solve_ivp


class SemanticReactionDynamics:

    def __init__(self, k1: float = 0.1, k2: float = 0.05,
                 w10: float = 1.0, w20: float = 0.0,
                 t0: float = 0.0, tstop: float = 100.0):
        if k1 < 0.0 or k2 < 0.0:
            raise ValueError(f"reaction rates must be non-negative, got k1={k1}, k2={k2}")

        self.k1 = float(k1)
        self.k2 = float(k2)
        self.w10 = float(w10)
        self.w20 = float(w20)
        self.t0 = float(t0)
        self.tstop = float(tstop)

    def exact_solution(self, t: np.ndarray) -> tuple:








        raise NotImplementedError("Hole 2: 反应动力学精确解尚未实现")

    def derivative(self, t: float, y: np.ndarray) -> np.ndarray:
        w1, w2 = y[0], y[1]
        return np.array([
            -self.k1 * w1 + self.k2 * w2,
            self.k1 * w1 - self.k2 * w2
        ])

    def solve_numerical(self, num_points: int = 200) -> tuple:
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
        k_sum = self.k1 + self.k2
        if abs(k_sum) < 1e-15:
            return self.w10, self.w20
        w1_eq = self.k2 * (self.w10 + self.w20) / k_sum
        w2_eq = self.k1 * (self.w10 + self.w20) / k_sum
        return w1_eq, w2_eq

    def relaxation_time(self) -> float:
        k_sum = self.k1 + self.k2
        if abs(k_sum) < 1e-15:
            return float('inf')
        return 1.0 / k_sum

    def conserved_quantity(self, w1: np.ndarray, w2: np.ndarray) -> np.ndarray:
        return w1 + w2


class MultiConceptReactionNetwork:

    def __init__(self, n_concepts: int, rate_matrix: np.ndarray, initial_state: np.ndarray):
        self.n = int(n_concepts)
        self.K = np.asarray(rate_matrix, dtype=float)
        self.y0 = np.asarray(initial_state, dtype=float)

        if self.K.shape != (self.n, self.n):
            raise ValueError(f"rate_matrix must be {self.n}x{self.n}")
        if len(self.y0) != self.n:
            raise ValueError(f"initial_state must have length {self.n}")


        col_sums = np.sum(self.K, axis=0)
        if np.max(np.abs(col_sums)) > 1e-10:
            raise ValueError("rate_matrix columns must sum to 0 for mass conservation")

    def derivative(self, t: float, y: np.ndarray) -> np.ndarray:
        return self.K @ y

    def solve(self, t_span: tuple, num_points: int = 500) -> tuple:
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

        A = np.zeros((self.n + 1, self.n))
        A[:self.n, :] = self.K
        A[self.n, :] = 1.0
        b = np.zeros(self.n + 1)
        b[self.n] = np.sum(self.y0)


        y_eq, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
        return y_eq


def demo():
    print("=" * 60)
    print("语义双向反应动力学演示")
    print("=" * 60)

    reaction = SemanticReactionDynamics(k1=0.3, k2=0.1, w10=1.0, w20=0.0,
                                        t0=0.0, tstop=50.0)
    print(f"\n反应参数:")
    print(f"  k1 = {reaction.k1}, k2 = {reaction.k2}")
    print(f"  w10 = {reaction.w10}, w20 = {reaction.w20}")

    t_num, w1_num, w2_num = reaction.solve_numerical(num_points=200)


    w1_exact, w2_exact = reaction.exact_solution(t_num)
    error1 = np.max(np.abs(w1_num - w1_exact))
    error2 = np.max(np.abs(w2_num - w2_exact))

    print(f"\n数值解与精确解最大偏差:")
    print(f"  w1: {error1:.6e}")
    print(f"  w2: {error2:.6e}")


    conserved = reaction.conserved_quantity(w1_num, w2_num)
    conserved_exact = reaction.w10 + reaction.w20
    print(f"\n守恒量检查 (w1+w2):")
    print(f"  理论值: {conserved_exact:.6f}")
    print(f"  数值范围: [{conserved.min():.10f}, {conserved.max():.10f}]")
    print(f"  最大偏差: {np.max(np.abs(conserved - conserved_exact)):.6e}")


    w1_eq, w2_eq = reaction.equilibrium()
    print(f"\n稳态平衡:")
    print(f"  w1_eq = {w1_eq:.6f}")
    print(f"  w2_eq = {w2_eq:.6f}")
    print(f"  弛豫时间 tau = {reaction.relaxation_time():.6f}")


    print("\n" + "-" * 40)
    print("多概念反应网络演示")
    print("-" * 40)

    n = 4

    K = np.zeros((n, n))
    for i in range(n):
        j_next = (i + 1) % n
        K[j_next, i] = 0.2
        K[i, i] -= 0.2
        j_prev = (i - 1) % n
        K[j_prev, i] = 0.1
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
