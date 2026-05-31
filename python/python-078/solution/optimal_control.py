"""
optimal_control.py
壁面剪切应力（WSS）最优控制：血管药物释放策略优化

融合来源:
- 215_control_bio: Pontryagin极大值原理、前向-后向扫描法、收敛判据

科学背景:
内皮细胞对WSS极为敏感：
    - WSS < 1 Pa: 促动脉粥样硬化表型（低剪切诱导炎症）
    - WSS > 7 Pa: 促动脉瘤表型（高剪切诱导基质降解）
    - 1 < WSS < 7 Pa: 生理稳态范围

通过局部释放血管活性物质（如NO供体、ACE抑制剂），
可以调控血管半径，进而调节WSS。这是一个典型的最优控制问题。

状态方程（简化的血管半径动力学）:
    dr/dt = k_g · (r_eq - r) + k_u · u(t) · r

其中:
    r(t): 血管半径 [m]
    r_eq: 平衡半径 [m]
    u(t): 药物释放率（控制变量）
    k_g: 弹性恢复系数
    k_u: 药物效能系数

目标泛函（最小化WSS偏离 + 控制代价）:
    J = ∫_0^T [ 0.5 (WSS(t) - WSS_target)² + 0.5 B u(t)² ] dt

伴随方程:
    dλ/dt = -∂H/∂r

最优性条件:
    ∂H/∂u = 0  →  u*(t) = λ(t) k_u r(t) / B
"""

import numpy as np
from scipy.integrate import odeint
from typing import Tuple, Callable


# ======================================================================
# 来自 215_control_bio 的前向-后向扫描法框架
# ======================================================================

def forward_backward_sweep(state_rhs: Callable, costate_rhs: Callable,
                           control_update: Callable,
                           r0: float, lambda_T: float,
                           time: np.ndarray,
                           u_guess: np.ndarray,
                           max_iter: int = 100,
                           tol: float = 1e-6,
                           alpha_step: float = 0.1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    前向-后向扫描法求解最优控制问题。

    算法步骤:
    1. 初始化控制猜测 u⁽⁰⁾(t)
    2. 前向扫描: 用odeint求解状态方程 dr/dt = g(t, r, u)
    3. 后向扫描: 反转时间，求解伴随方程 dλ/dt = -∂H/∂r
    4. 控制更新: 由最优性条件 u* = argmin H
    5. 收敛检验: 若 ||u^{k+1} - u^k||_1 / ||u^k||_1 < tol，则停止

    参数:
        state_rhs: 状态方程右端项 g(t, r, u)
        costate_rhs: 伴随方程右端项 h(t, r, lambda, u)
        control_update: 最优控制更新函数 u_new = f(r, lambda)
        r0: 初始状态
        lambda_T: 伴随终值条件
        time: 时间网格
        u_guess: 初始控制猜测
        max_iter: 最大迭代次数
        tol: 收敛容差
        alpha_step: 控制更新松弛因子

    返回:
        r_opt: 最优状态轨迹
        lambda_opt: 最优伴随变量
        u_opt: 最优控制
    """
    n = len(time)
    u = u_guess.copy()
    r = np.zeros(n)
    lam = np.zeros(n)

    for it in range(max_iter):
        # 前向扫描
        def fwd_ode(y, t_idx):
            # odeint期望 y' = f(y, t)，这里时间索引映射
            idx = min(int(t_idx), n - 1)
            return state_rhs(time[idx], y, u[idx])

        # 使用简单Euler（因为odeint对离散控制不友好）
        r[0] = r0
        for i in range(1, n):
            dt = time[i] - time[i - 1]
            r[i] = r[i - 1] + dt * state_rhs(time[i - 1], r[i - 1], u[i - 1])
            # 边界：半径不能为负
            if r[i] < 0.1e-3:
                r[i] = 0.1e-3

        # 后向扫描（时间反转）
        lam[-1] = lambda_T
        for i in range(n - 2, -1, -1):
            dt = time[i + 1] - time[i]
            lam[i] = lam[i + 1] + dt * costate_rhs(time[i + 1], r[i + 1], lam[i + 1], u[i + 1])

        # 控制更新（带松弛）
        u_new = np.zeros(n)
        for i in range(n):
            u_new[i] = control_update(r[i], lam[i])
            # 控制变量边界：0 ≤ u ≤ u_max
            u_new[i] = np.clip(u_new[i], 0.0, 1.0)

        # 收敛检验（L1相对误差）
        u_old_norm = np.linalg.norm(u, ord=1) + 1e-15
        diff = np.linalg.norm(u_new - u, ord=1) / u_old_norm
        u = (1.0 - alpha_step) * u + alpha_step * u_new

        if diff < tol:
            break

    return r, lam, u


# ======================================================================
# WSS最优控制具体实现
# ======================================================================

class WSSOptimalControl:
    """
    壁面剪切应力最优控制问题。

    状态方程:
        dr/dt = k_g (r_eq - r) + k_u u r

    WSS与半径的关系（Poiseuille流近似）:
        WSS = 4 μ Q / (π r³)

    目标: 使WSS跟踪目标值 WSS_target
    """
    def __init__(self, equilibrium_radius: float = 0.005,
                 target_wss_pa: float = 2.5,
                 blood_viscosity_pa_s: float = 0.0035,
                 flow_rate_m3_s: float = 5.0e-5,
                 k_growth: float = 0.5,
                 k_drug: float = 0.3,
                 control_penalty: float = 0.1):
        """
        参数:
            equilibrium_radius: 平衡半径 [m]
            target_wss_pa: 目标WSS [Pa]（生理范围1-7 Pa）
            blood_viscosity_pa_s: 血液粘度 [Pa·s]
            flow_rate_m3_s: 恒定流量 [m³/s]
            k_growth: 弹性恢复系数
            k_drug: 药物效能系数
            control_penalty: 控制代价权重B
        """
        self.r_eq = equilibrium_radius
        self.wss_target = target_wss_pa
        self.mu = blood_viscosity_pa_s
        self.Q = flow_rate_m3_s
        self.k_g = k_growth
        self.k_u = k_drug
        self.B = control_penalty

    def wss_from_radius(self, r: float) -> float:
        """由半径计算WSS（Poiseuille近似）。"""
        if r < 1e-6:
            return 0.0
        return 4.0 * self.mu * self.Q / (np.pi * r ** 3)

    def state_rhs(self, t: float, r: float, u: float) -> float:
        """
        状态方程右端项: dr/dt = k_g(r_eq - r) + k_u u r
        """
        return self.k_g * (self.r_eq - r) + self.k_u * u * r

    def costate_rhs(self, t: float, r: float, lam: float, u: float) -> float:
        """
        伴随方程右端项: dλ/dt = -∂H/∂r

        Hamiltonian:
            H = 0.5 (WSS(r) - WSS_t)² + 0.5 B u² + λ [k_g(r_eq - r) + k_u u r]

        ∂H/∂r = (WSS - WSS_t) · dWSS/dr + λ (-k_g + k_u u)
        dWSS/dr = -12 μ Q / (π r⁴)
        """
        wss = self.wss_from_radius(r)
        dwss_dr = -12.0 * self.mu * self.Q / (np.pi * r ** 4 + 1e-20)
        dH_dr = (wss - self.wss_target) * dwss_dr + lam * (-self.k_g + self.k_u * u)
        return -dH_dr

    def control_update(self, r: float, lam: float) -> float:
        """
        最优性条件: ∂H/∂u = B u + λ k_u r = 0
        → u* = - λ k_u r / B
        """
        u_star = -lam * self.k_u * r / (self.B + 1e-15)
        return u_star

    def solve(self, r0: float, time: np.ndarray,
              u_guess: np.ndarray = None,
              max_iter: int = 100) -> dict:
        """
        求解WSS最优控制问题。

        参数:
            r0: 初始半径 [m]
            time: 时间数组 [s]
            u_guess: 初始控制猜测
            max_iter: 最大迭代次数

        返回:
            包含r, lambda, u, wss轨迹的字典
        """
        n = len(time)
        if u_guess is None:
            u_guess = np.zeros(n)

        r, lam, u = forward_backward_sweep(
            self.state_rhs,
            self.costate_rhs,
            self.control_update,
            r0, 0.0, time, u_guess,
            max_iter=max_iter, tol=1e-5, alpha_step=0.3
        )

        wss = np.array([self.wss_from_radius(ri) for ri in r])

        return {
            "radius": r,
            "costate": lam,
            "control": u,
            "wss": wss,
            "target_wss": self.wss_target,
            "time": time
        }


def compute_control_cost(wss_trajectory: np.ndarray,
                         target_wss: float,
                         control_trajectory: np.ndarray,
                         B: float = 0.1) -> float:
    """
    计算目标泛函值:
        J = ∫ [0.5 (WSS - WSS_t)² + 0.5 B u²] dt

    使用梯形法则数值积分。
    """
    n = len(wss_trajectory)
    if n < 2:
        return 0.0

    integrand = 0.5 * (wss_trajectory - target_wss) ** 2 + 0.5 * B * control_trajectory ** 2
    # 梯形法则
    J = np.trapezoid(integrand)
    return float(J)


def wss_physiological_score(wss_pa: float) -> float:
    """
    评估WSS的生理健康度。

    评分规则:
        1 < WSS < 7: 健康（满分1.0）
        WSS < 0.5 或 WSS > 10: 危险（0.0）
        其他: 线性插值
    """
    if 1.0 <= wss_pa <= 7.0:
        return 1.0
    elif wss_pa < 0.5 or wss_pa > 10.0:
        return 0.0
    elif wss_pa < 1.0:
        return (wss_pa - 0.5) / 0.5
    else:
        return (10.0 - wss_pa) / 3.0
