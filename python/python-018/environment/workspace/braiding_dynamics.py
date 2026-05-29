"""
braiding_dynamics.py

基于种子项目 1023_rigid_body_ode（刚体欧拉方程）和
908_predator_prey_ode（Lotka-Volterra方程），实现马约拉纳
费米子的非阿贝尔编织动力学与耦合系统演化。

物理模型：
    1) 刚体欧拉方程映射到马约拉纳算符的编织动力学：
        将马约拉纳算符 {γ_1, γ_2, γ_3} 视为SO(3)群上的“角动量”变量，
        其交换操作对应刚体在力矩作用下的转动。

        两个马约拉纳算符的交换产生Bogoliubov变换：
            U_{ij} = exp(π/4 * γ_i γ_j)
        在编织操作下：
            γ_i → γ_j
            γ_j → -γ_i

    2) Lotka-Volterra方程映射到马约拉纳-杂质耦合系统：
        将马约拉纳束缚态（猎物）与杂质态（捕食者）的
        占据数耦合动力学类比为：
            dn_M/dt = α n_M - β n_M n_I  （马约拉纳被杂质散射）
            dn_I/dt = -γ n_I + δ n_M n_I  （杂质被马约拉纳激发）

核心公式：
    编织矩阵（对于4个马约拉纳的TQC）：
        σ_1 = exp(π/4 * γ_1 γ_2)
        σ_2 = exp(π/4 * γ_2 γ_3)
        σ_3 = exp(π/4 * γ_3 γ_4)

    这些编织操作生成Clifford代数，实现拓扑量子门。
"""

import numpy as np
from typing import Tuple, Optional


class MajoranaBraidingDynamics:
    """
    马约拉纳费米子的非阿贝尔编织动力学。

    将马约拉纳算符表示为Majorana矩阵（实反对称矩阵），
    编织操作对应SO(2N)群的旋转。
    """

    def __init__(self, num_majorana: int = 4):
        """
        初始化编织系统。

        Args:
            num_majorana: 马约拉纳算符数目（必须为偶数，≥4）
        """
        if num_majorana < 4 or num_majorana % 2 != 0:
            raise ValueError("马约拉纳数目必须为≥4的偶数")
        self.N = num_majorana

    def _majorana_matrix(self, i: int, j: int) -> np.ndarray:
        """
        构造马约拉纳矩阵 γ_i γ_j 的矩阵表示。

        利用Clifford代数的矩阵表示（Dirac表示），
        对于2N个马约拉纳算符，需要 2^N × 2^N 的矩阵。
        这里采用简化的SO(2N)生成元表示。
        """
        # 使用实反对称矩阵表示
        mat = np.zeros((self.N, self.N))
        if 0 <= i < self.N and 0 <= j < self.N and i != j:
            mat[i, j] = 1.0
            mat[j, i] = -1.0
        return mat

    def braid_operator(self, i: int, j: int) -> np.ndarray:
        """
        计算交换两个马约拉纳算符的编织算符。

        对于顺时针交换（braid）：
            B_{ij} = exp(π/4 * γ_i γ_j)

        在矩阵表示下，对于实反对称生成元A = γ_i γ_j：
            exp(θ A) = I + sin(θ) A + (1-cos(θ)) A^2
        由于A^2 = -I（当i≠j时），因此：
            exp(π/4 * A) = (1/√2)(I + A)
        """
        if i == j or not (0 <= i < self.N and 0 <= j < self.N):
            raise ValueError("编织指标必须不同且在有效范围内")

        A = self._majorana_matrix(i, j)
        # A^2 = -I_{N×N} 在(i,j)子空间内
        I = np.eye(self.N)
        theta = np.pi / 4.0
        # 对于反对称A，A^3 = -A，因此可以用Rodrigues公式
        B = I + np.sin(theta) * A + (1.0 - np.cos(theta)) * (A @ A)
        return B

    def apply_braid_sequence(self, sequence: list) -> np.ndarray:
        """
        应用编织序列。

        Args:
            sequence: [(i1,j1), (i2,j2), ...] 编织操作序列

        Returns:
            累积的变换矩阵
        """
        U = np.eye(self.N)
        for i, j in sequence:
            B = self.braid_operator(i, j)
            U = B @ U
        return U

    def rigid_body_mapping_derivative(self, xyz: np.ndarray,
                                       I1: float, I2: float, I3: float
                                       ) -> np.ndarray:
        """
        将刚体欧拉方程映射到马约拉纳算符的“转动”动力学。

        刚体欧拉方程（自由转动）：
            dω_1/dt = (1/I3 - 1/I2) ω_3 ω_2
            dω_2/dt = (1/I1 - 1/I3) ω_1 ω_3
            dω_3/dt = (1/I2 - 1/I1) ω_2 ω_1

        映射关系：
            ω_i ↔ <γ_i γ_{i+1}> （马约拉纳对关联）
            I_i ↔ 局域化长度 ξ_i
        """
        x, y, z = xyz[0], xyz[1], xyz[2]

        # 边界处理：防止数值爆炸
        max_val = 1e6
        x = np.clip(x, -max_val, max_val)
        y = np.clip(y, -max_val, max_val)
        z = np.clip(z, -max_val, max_val)

        dxdt = (1.0 / I3 - 1.0 / I2) * z * y
        dydt = (1.0 / I1 - 1.0 / I3) * x * z
        dzdt = (1.0 / I2 - 1.0 / I1) * y * x

        return np.array([dxdt, dydt, dzdt])

    def integrate_rigid_body(self, xyz0: np.ndarray,
                              t_span: np.ndarray,
                              I1: float, I2: float, I3: float) -> np.ndarray:
        """
        使用RK4积分刚体方程。
        """
        n_steps = len(t_span)
        xyz = np.zeros((n_steps, 3))
        xyz[0] = xyz0

        for i in range(n_steps - 1):
            dt = t_span[i + 1] - t_span[i]
            k1 = self.rigid_body_mapping_derivative(xyz[i], I1, I2, I3)
            k2 = self.rigid_body_mapping_derivative(
                xyz[i] + 0.5 * dt * k1, I1, I2, I3)
            k3 = self.rigid_body_mapping_derivative(
                xyz[i] + 0.5 * dt * k2, I1, I2, I3)
            k4 = self.rigid_body_mapping_derivative(
                xyz[i] + dt * k3, I1, I2, I3)

            xyz[i + 1] = xyz[i] + (dt / 6.0) * (k1 + 2.0 * k2
                                                  + 2.0 * k3 + k4)
            # 数值稳定性截断
            xyz[i + 1] = np.clip(xyz[i + 1], -1e6, 1e6)

        return xyz

    def compute_braid_group_relation(self, i: int, j: int, k: int
                                      ) -> float:
        """
        验证编织群的Yang-Baxter关系。

        对于相邻编织生成元，应满足：
            σ_i σ_{i+1} σ_i = σ_{i+1} σ_i σ_{i+1}

        计算左右两边的Frobenius范数差异作为验证指标。
        """
        if not all(0 <= idx < self.N for idx in [i, j, k]):
            return -1.0

        B_i = self.braid_operator(i, j)
        B_j = self.braid_operator(j, k)

        # 注意：这里需要调整指标使得它们相邻
        # 简化为验证 B_ab B_bc B_ab = B_bc B_ab B_bc
        left = B_i @ B_j @ B_i
        right = B_j @ B_i @ B_j

        diff = np.linalg.norm(left - right, ord='fro')
        return float(diff)


class MajoranaImpurityDynamics:
    """
    马约拉纳-杂质耦合系统的Lotka-Volterra动力学。
    """

    def __init__(self, alpha: float = 1.0, beta: float = 0.5,
                 gamma: float = 0.8, delta: float = 0.3):
        """
        初始化耦合参数。

        Args:
            alpha: 马约拉纳产生率
            beta: 马约拉纳-杂质散射率
            gamma: 杂质衰减率
            delta: 杂质被马约拉纳激发率
        """
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta

    def derivatives(self, t: float, y: np.ndarray) -> np.ndarray:
        """
        Lotka-Volterra型导数。

        y[0] = n_M: 马约拉ana束缚态占据数
        y[1] = n_I: 杂质态占据数

        dn_M/dt = α n_M - β n_M n_I
        dn_I/dt = -γ n_I + δ n_M n_I
        """
        n_m, n_i = y[0], y[1]

        # 边界处理：占据数非负
        n_m = max(n_m, 0.0)
        n_i = max(n_i, 0.0)

        dn_m = self.alpha * n_m - self.beta * n_m * n_i
        dn_i = -self.gamma * n_i + self.delta * n_m * n_i

        return np.array([dn_m, dn_i])

    def integrate(self, y0: np.ndarray, t_span: np.ndarray) -> np.ndarray:
        """
        使用RK4积分耦合动力学。
        """
        n_steps = len(t_span)
        y = np.zeros((n_steps, 2))
        y[0] = np.maximum(y0, 0.0)

        for i in range(n_steps - 1):
            dt = t_span[i + 1] - t_span[i]
            k1 = self.derivatives(t_span[i], y[i])
            k2 = self.derivatives(t_span[i] + 0.5 * dt, y[i] + 0.5 * dt * k1)
            k3 = self.derivatives(t_span[i] + 0.5 * dt, y[i] + 0.5 * dt * k2)
            k4 = self.derivatives(t_span[i] + dt, y[i] + dt * k3)

            y[i + 1] = y[i] + (dt / 6.0) * (k1 + 2.0 * k2
                                             + 2.0 * k3 + k4)
            # 非负约束
            y[i + 1] = np.maximum(y[i + 1], 0.0)

        return y

    def conserved_quantity(self, y: np.ndarray) -> np.ndarray:
        """
        计算Lotka-Volterra系统的守恒量（类比能量）。

        H = δ n_M + β n_I - γ ln(n_M) - α ln(n_I)
        """
        n_m, n_i = y[:, 0], y[:, 1]
        # 避免log(0)
        n_m_safe = np.maximum(n_m, 1e-15)
        n_i_safe = np.maximum(n_i, 1e-15)

        H = (self.delta * n_m_safe + self.beta * n_i_safe
             - self.gamma * np.log(n_m_safe)
             - self.alpha * np.log(n_i_safe))
        return H


def demo():
    """演示编织动力学。"""
    braid = MajoranaBraidingDynamics(num_majorana=4)

    # 验证Yang-Baxter关系
    diff = braid.compute_braid_group_relation(0, 1, 2)
    print("Yang-Baxter relation difference:", diff)

    # 编织序列
    seq = [(0, 1), (1, 2), (2, 3)]
    U = braid.apply_braid_sequence(seq)
    print("Braid sequence matrix trace:", np.trace(U))

    # 刚体映射动力学
    t_span = np.linspace(0, 10, 101)
    xyz = braid.integrate_rigid_body(
        xyz0=np.array([1.0, 0.5, 0.2]),
        t_span=t_span, I1=1.0, I2=0.8, I3=0.6
    )
    print("Rigid-body mapped dynamics final state:", xyz[-1])

    # 耦合系统
    impurity = MajoranaImpurityDynamics(
        alpha=1.0, beta=0.5, gamma=0.8, delta=0.3
    )
    y = impurity.integrate(y0=np.array([2.0, 1.0]),
                           t_span=np.linspace(0, 20, 201))
    H = impurity.conserved_quantity(y)
    print("Conserved quantity variation:", np.max(H) - np.min(H))


if __name__ == "__main__":
    demo()
