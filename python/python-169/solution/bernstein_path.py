"""
Bernstein 多项式与 Bézier 轨迹生成模块
==========================================
基于种子项目:
  - 077_bernstein_approximation : Bernstein基函数与de Casteljau递推

核心数学模型:
  1. Bernstein 基多项式（定义在 [A, B] 上）:
     B_i^n(t) = C(n,i) * (B-t)^{n-i} * (t-A)^i / (B-A)^n

  2. Bézier 曲线（向量值）:
     P(t) = Σ_{i=0}^{n} P_i * B_i^n(t),   t ∈ [A, B]

  3. de Casteljau 递推（数值稳定）:
     P_i^{(0)}(t) = P_i
     P_i^{(r)}(t) = (B-t)/(B-A) * P_i^{(r-1)}(t) + (t-A)/(B-A) * P_{i+1}^{(r-1)}(t)
     P(t) = P_0^{(n)}(t)

  4. 导数公式:
     P'(t) = n/(B-A) * Σ_{i=0}^{n-1} (P_{i+1} - P_i) * B_i^{n-1}(t)

  5. 凸包性质（边界安全）:
     曲线始终位于控制点凸包内 ⇒ 通过对控制点加框即可保证轨迹不超出安全区域。
"""

import numpy as np
from typing import List, Tuple


def bernstein_basis(n: int, a: float, b: float, t: float) -> np.ndarray:
    r"""
    计算 n 次 Bernstein 基函数在参数 t 处的值。
    使用稳定的递推关系，避免直接计算大阶乘。

    返回数组 B[0..n]，其中 B[i] = B_i^n(t)。
    """
    if b <= a:
        raise ValueError("区间右端点b必须大于左端点a")
    if t < a - 1e-12 or t > b + 1e-12:
        # 外推时给出警告并截断
        t = np.clip(t, a, b)
    # 归一化参数 s ∈ [0,1]
    s = (t - a) / (b - a)
    # 递推初始化（一次基）
    B = np.array([1.0 - s, s], dtype=float)
    if n == 0:
        return np.array([1.0])
    if n == 1:
        return B
    # 递推升至 n 次
    for degree in range(2, n + 1):
        B_new = np.zeros(degree + 1, dtype=float)
        B_new[0] = (1.0 - s) * B[0]
        for i in range(1, degree):
            B_new[i] = (1.0 - s) * B[i] + s * B[i - 1]
        B_new[degree] = s * B[degree - 1]
        B = B_new
    return B


def de_casteljau(control_points: np.ndarray, a: float, b: float, t: float) -> np.ndarray:
    r"""
    de Casteljau 算法数值稳定地计算 Bézier 曲线在 t 处的值。

    control_points: (n+1, d) 数组，d 为空间维度（通常为3或7，对应关节空间）。
    """
    if b <= a:
        raise ValueError("区间右端点b必须大于左端点a")
    t = np.clip(t, a, b)
    s = (t - a) / (b - a)
    # 复制控制点
    P = np.array(control_points, dtype=float)
    n = P.shape[0] - 1
    if n < 0:
        raise ValueError("控制点数量不能为零")
    # 递推消去
    for r in range(1, n + 1):
        for i in range(n - r + 1):
            P[i] = (1.0 - s) * P[i] + s * P[i + 1]
    return P[0]


def bezier_derivative(control_points: np.ndarray, a: float, b: float,
                      order: int = 1) -> np.ndarray:
    r"""
    计算 Bézier 曲线的控制点表示的导数。
    对于 n 次 Bézier 曲线 P(t)，其 k 阶导数仍是 Bézier 曲线，次数为 n-k。

    一阶导数控制点:
      D_i = n/(B-A) * (P_{i+1} - P_i),   i = 0,...,n-1
    """
    P = np.array(control_points, dtype=float)
    n = P.shape[0] - 1
    if n < order:
        raise ValueError(f"阶数{n}不足以求{order}阶导数")
    for _ in range(order):
        n = P.shape[0] - 1
        scale = n / (b - a)
        D = np.zeros((n, P.shape[1]), dtype=float)
        for i in range(n):
            D[i] = scale * (P[i + 1] - P[i])
        P = D
    return P


class JointSpaceBezierTrajectory:
    r"""
    基于Bernstein多项式的关节空间轨迹类。
    每个关节独立使用一个标量Bézier曲线，或统一使用一个向量值Bézier曲线。
    提供位置、速度、加速度的解析计算。
    """

    def __init__(self, control_points: np.ndarray, t0: float = 0.0, tf: float = 1.0):
        """
        control_points: (n+1, n_dof) 数组
        t0, tf: 时间区间 [t0, tf]
        """
        self.P = np.array(control_points, dtype=float)
        self.n = self.P.shape[0] - 1          # Bézier 次数
        self.n_dof = self.P.shape[1]
        self.t0 = float(t0)
        self.tf = float(tf)
        if self.tf <= self.t0:
            raise ValueError("终止时间tf必须大于起始时间t0")
        # 预计算速度、加速度的控制点（用于快速查询）
        self.dP = bezier_derivative(self.P, self.t0, self.tf, order=1)
        self.ddP = bezier_derivative(self.P, self.t0, self.tf, order=2)

    def position(self, t: float) -> np.ndarray:
        """关节位置 q(t)。"""
        return de_casteljau(self.P, self.t0, self.tf, t)

    def velocity(self, t: float) -> np.ndarray:
        """关节速度 \dot{q}(t) = dP/dt。"""
        return de_casteljau(self.dP, self.t0, self.tf, t)

    def acceleration(self, t: float) -> np.ndarray:
        """关节加速度 \ddot{q}(t) = d²P/dt²。"""
        return de_casteljau(self.ddP, self.t0, self.tf, t)

    def evaluate(self, t: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """同时返回 (q, dq, ddq)。"""
        t = np.clip(t, self.t0, self.tf)
        return self.position(t), self.velocity(t), self.acceleration(t)

    def max_velocity_bound(self) -> np.ndarray:
        r"""
        利用凸包性质给出速度上界:
        \|\dot{q}(t)\|_∞ ≤ max_i \|dP_i\|_∞
        """
        return np.max(np.abs(self.dP), axis=0)

    def max_acceleration_bound(self) -> np.ndarray:
        r"""
        利用凸包性质给出加速度上界:
        \|\ddot{q}(t)\|_∞ ≤ max_i \|ddP_i\|_∞
        """
        return np.max(np.abs(self.ddP), axis=0)

    def split_at(self, t_split: float) -> Tuple['JointSpaceBezierTrajectory', 'JointSpaceBezierTrajectory']:
        r"""
        利用de Casteljau算法在 t_split 处分割曲线为两段。
        这是轨迹重规划的关键操作。
        """
        t_split = np.clip(t_split, self.t0, self.tf)
        s = (t_split - self.t0) / (self.tf - self.t0)
        P_left = []
        P_right = []
        P = np.array(self.P, dtype=float)
        n = self.n
        P_left.append(P[0].copy())
        P_right.append(P[n].copy())
        for r in range(1, n + 1):
            for i in range(n - r + 1):
                P[i] = (1.0 - s) * P[i] + s * P[i + 1]
            P_left.append(P[0].copy())
            P_right.append(P[n - r].copy())
        # 反转右侧控制点顺序使其保持正向
        P_right.reverse()
        traj_left = JointSpaceBezierTrajectory(np.array(P_left), self.t0, t_split)
        traj_right = JointSpaceBezierTrajectory(np.array(P_right), t_split, self.tf)
        return traj_left, traj_right


def generate_minimum_jerk_bezier(q_start: np.ndarray, q_end: np.ndarray,
                                  t0: float = 0.0, tf: float = 1.0,
                                  degree: int = 5) -> JointSpaceBezierTrajectory:
    r"""
    基于最小加加速度（minimum jerk）边界条件生成Bézier轨迹。
    边界条件:
      q(t0)=q_start, q(tf)=q_end
      \dot{q}(t0)=\dot{q}(tf)=0
      \ddot{q}(t0)=\ddot{q}(tf)=0
    对于5次Bézier，恰好有6个控制点，可由6个边界条件唯一确定。

    控制点闭式解（标量情形，推广到各关节独立）:
      P0 = q0
      P1 = q0
      P2 = q0
      P3 = (10 q1 - 4 q0) / 6   ... 实际上直接用最小范数解
    更简洁的方法：使用多项式插值后转换为Bernstein基。
    """
    q0 = np.asarray(q_start, dtype=float).reshape(-1)
    q1 = np.asarray(q_end, dtype=float).reshape(-1)
    n_dof = q0.size
    if degree < 3:
        raise ValueError("minimum-jerk轨迹至少需要3次")
    # 使用5次多项式（6个控制点）
    if degree != 5:
        degree = 5
    # 构造多项式系数 a0 + a1 t + ... + a5 t^5
    # 边界条件矩阵（在 t0=0, tf=1 下）
    T = np.array([
        [1, 0, 0, 0, 0, 0],    # q(0)
        [0, 1, 0, 0, 0, 0],    # dq(0)
        [0, 0, 2, 0, 0, 0],    # ddq(0)
        [1, 1, 1, 1, 1, 1],    # q(1)
        [0, 1, 2, 3, 4, 5],    # dq(1)
        [0, 0, 2, 6, 12, 20],  # ddq(1)
    ], dtype=float)
    # 对每个关节求解
    P_list = []
    for j in range(n_dof):
        rhs = np.array([q0[j], 0.0, 0.0, q1[j], 0.0, 0.0], dtype=float)
        a = np.linalg.solve(T, rhs)
        # 转换为Bernstein控制点（5次）
        # P_i = Σ_{k=0}^{i} C(i,k) / C(5,k) * a_k  (当t∈[0,1]时)
        # 更直接：在均匀参数处采样后拟合，或使用基变换矩阵
        P = np.zeros(6, dtype=float)
        for i in range(6):
            # P_i 是 Bézier 控制点，通过将幂基转换为Bernstein基得到
            # B_i^5(t) = Σ_{k=0}^5 (-1)^{k-i} C(5,i) C(i,k) t^k
            # 反解: t^k = Σ_{i=0}^k C(k,i) / C(5,i) B_i^5(t)
            # 因此 a_k 的贡献分布在 i=0..k
            for k in range(i + 1):
                if k <= 5:
                    P[i] += (np.math.comb(k, i) / np.math.comb(5, i)) * a[k] if i <= k else 0.0
        # 修正：正确的基变换矩阵 M 满足 Power = M * Bernstein
        # 对于5次，Bernstein→Power 的逆变换为：
        # P_i = Σ_{k=0}^i (-1)^{i-k} * C(5,k) * C(5-k, i-k) / C(5,i) * a_k
        # 这里我们用数值积分/采样方法更稳健
        pass
    # 使用更稳健的数值方法：在Legendre节点采样幂基多项式，再最小二乘拟合Bernstein控制点
    # 实际上对5次多项式，6个控制点可由6个样本精确确定
    ts = np.linspace(0.0, 1.0, 6)
    V = np.zeros((6, 6), dtype=float)
    for i in range(6):
        for j in range(6):
            V[i, j] = ts[i] ** j
    # Bernstein 基在 ts 处的值
    B_mat = np.zeros((6, 6), dtype=float)
    for idx, t in enumerate(ts):
        B_mat[idx, :] = bernstein_basis(5, 0.0, 1.0, t)
    # 控制点 = B_mat^{-1} * V * a
    # 由于 B_mat 是6×6 Vandermonde-like，可直接求逆
    try:
        B_inv = np.linalg.inv(B_mat)
    except np.linalg.LinAlgError:
        B_inv = np.linalg.pinv(B_mat)
    P_all = np.zeros((6, n_dof), dtype=float)
    for j in range(n_dof):
        rhs = np.array([q0[j], 0.0, 0.0, q1[j], 0.0, 0.0], dtype=float)
        a = np.linalg.solve(T, rhs)
        y_samples = V @ a
        P_all[:, j] = B_inv @ y_samples
    return JointSpaceBezierTrajectory(P_all, t0, tf)


def clamp_control_points_to_joint_limits(P: np.ndarray,
                                         q_min: np.ndarray,
                                         q_max: np.ndarray) -> np.ndarray:
    r"""
    将控制点裁剪到关节限位内，利用Bernstein曲线的凸包性质:
    若所有控制点 P_i ∈ [q_min, q_max]，则整条曲线 q(t) ∈ [q_min, q_max]。
    """
    q_min = np.asarray(q_min).reshape(1, -1)
    q_max = np.asarray(q_max).reshape(1, -1)
    return np.clip(P, q_min, q_max)
