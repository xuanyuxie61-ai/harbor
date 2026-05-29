"""
power_flow.py
牛顿-拉夫逊潮流计算
融合种子项目：diff2_center（数值微分验证）, r8st_cg（稀疏求解）
"""

import numpy as np
from typing import Optional, Tuple
from sparse_matrix import SparseMatrix, conjugate_gradient
from utils import diff2_center


class PowerFlowSolver:
    """
    基于牛顿-拉夫逊法（Newton-Raphson）的潮流计算求解器。

    核心方程（节点功率方程，极坐标形式）：
        P_i = |V_i| · Σ_{j∈i} |V_j| · (G_{ij} cos θ_{ij} + B_{ij} sin θ_{ij})
        Q_i = |V_i| · Σ_{j∈i} |V_j| · (G_{ij} sin θ_{ij} - B_{ij} cos θ_{ij})
    其中 θ_{ij} = θ_i - θ_j，G_{ij} + jB_{ij} = Y_{ij} 为导纳矩阵元素。

    牛顿迭代格式：
        [ ΔP ]   [ J11  J12 ] [ Δθ   ]
        [ ΔQ ] = [ J21  J22 ] [ Δ|V|/|V| ]

    雅可比子块：
        J11_{ij} = ∂P_i/∂θ_j = |V_i||V_j|(G_{ij} sinθ_{ij} - B_{ij} cosθ_{ij})   (i≠j)
        J11_{ii} = -Q_i - B_{ii}|V_i|^2
        J12_{ij} = ∂P_i/∂|V_j| · |V_j| = |V_i||V_j|(G_{ij} cosθ_{ij} + B_{ij} sinθ_{ij})   (i≠j)
        J12_{ii} = P_i + G_{ii}|V_i|^2
        J21_{ij} = ∂Q_i/∂θ_j = -|V_i||V_j|(G_{ij} cosθ_{ij} + B_{ij} sinθ_{ij})   (i≠j)
        J21_{ii} = P_i - G_{ii}|V_i|^2
        J22_{ij} = ∂Q_i/∂|V_j| · |V_j| = |V_i||V_j|(G_{ij} sinθ_{ij} - B_{ij} cosθ_{ij})   (i≠j)
        J22_{ii} = Q_i - B_{ii}|V_i|^2
    """

    def __init__(self, y_bus: np.ndarray, bus_types: np.ndarray,
                 v_magnitude: np.ndarray, v_angle: np.ndarray,
                 p_spec: np.ndarray, q_spec: np.ndarray):
        """
        参数：
            y_bus:     n×n 复数导纳矩阵
            bus_types: n 维数组，0=PQ, 1=PV, 2=Slack
            v_magnitude, v_angle: 初始电压幅值与相角（弧度）
            p_spec, q_spec: 节点注入有功/无功（发电为正，负荷为负）
        """
        self.n = y_bus.shape[0]
        self.y_bus = np.array(y_bus, dtype=np.complex128)
        self.bus_types = np.array(bus_types, dtype=np.int32)
        self.vm = np.array(v_magnitude, dtype=np.float64)
        self.va = np.array(v_angle, dtype=np.float64)
        self.p_spec = np.array(p_spec, dtype=np.float64)
        self.q_spec = np.array(q_spec, dtype=np.float64)
        self.g = self.y_bus.real
        self.b = self.y_bus.imag

    def compute_power_mismatch(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算功率失配 ΔP, ΔQ。
        """
        n = self.n
        p_calc = np.zeros(n, dtype=np.float64)
        q_calc = np.zeros(n, dtype=np.float64)
        # TODO: Hole 1 — 实现节点功率失配计算（极坐标形式）
        # 对每一节点 i，计算注入有功 P_i 和注入无功 Q_i：
        #   P_i = |V_i| * Σ_j |V_j| * (G_ij * cos(θ_i - θ_j) + B_ij * sin(θ_i - θ_j))
        #   Q_i = |V_i| * Σ_j |V_j| * (G_ij * sin(θ_i - θ_j) - B_ij * cos(θ_i - θ_j))
        # 然后返回 ΔP = P_spec - P_calc, ΔQ = Q_spec - Q_calc
        raise NotImplementedError("Hole 1: compute_power_mismatch 待实现")
        dp = self.p_spec - p_calc
        dq = self.q_spec - q_calc
        return dp, dq

    def build_jacobian(self) -> np.ndarray:
        """
        构建完整雅可比矩阵（稠密形式，用于中小规模系统）。
        大规模系统可改用稀疏矩阵+CG求解。
        """
        n = self.n
        j11 = np.zeros((n, n), dtype=np.float64)
        j12 = np.zeros((n, n), dtype=np.float64)
        j21 = np.zeros((n, n), dtype=np.float64)
        j22 = np.zeros((n, n), dtype=np.float64)

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                angle_diff = self.va[i] - self.va[j]
                j11[i, j] = self.vm[i] * self.vm[j] * (
                    self.g[i, j] * np.sin(angle_diff)
                    - self.b[i, j] * np.cos(angle_diff)
                )
                j12[i, j] = self.vm[i] * self.vm[j] * (
                    self.g[i, j] * np.cos(angle_diff)
                    + self.b[i, j] * np.sin(angle_diff)
                )
                j21[i, j] = -j12[i, j]
                j22[i, j] = j11[i, j]

            j11[i, i] = -self.q_calc_at_bus(i) - self.b[i, i] * self.vm[i]**2
            j12[i, i] = self.p_calc_at_bus(i) + self.g[i, i] * self.vm[i]**2
            j21[i, i] = self.p_calc_at_bus(i) - self.g[i, i] * self.vm[i]**2
            j22[i, i] = self.q_calc_at_bus(i) - self.b[i, i] * self.vm[i]**2

        # 根据节点类型裁剪
        pq_idx = np.where(self.bus_types == 0)[0]
        pv_idx = np.where(self.bus_types == 1)[0]
        slack_idx = np.where(self.bus_types == 2)[0]

        # 构造修正方程的未知量索引
        # 未知量：所有非 slack 节点的 θ，所有 PQ 节点的 |V|
        theta_idx = np.concatenate([pq_idx, pv_idx])
        v_idx = pq_idx

        rows = []
        cols = []
        # ΔP 对所有非 slack 节点
        for i in theta_idx:
            row = []
            for j in theta_idx:
                row.append(j11[i, j])
            for j in v_idx:
                row.append(j12[i, j])
            rows.append(row)
        # ΔQ 对所有 PQ 节点
        for i in v_idx:
            row = []
            for j in theta_idx:
                row.append(j21[i, j])
            for j in v_idx:
                row.append(j22[i, j])
            rows.append(row)

        if len(rows) == 0:
            return np.zeros((0, 0), dtype=np.float64)
        return np.array(rows, dtype=np.float64)

    def p_calc_at_bus(self, i: int) -> float:
        """计算节点 i 的注入有功。"""
        p = 0.0
        for j in range(self.n):
            angle_diff = self.va[i] - self.va[j]
            p += self.vm[i] * self.vm[j] * (
                self.g[i, j] * np.cos(angle_diff)
                + self.b[i, j] * np.sin(angle_diff)
            )
        return p

    def q_calc_at_bus(self, i: int) -> float:
        """计算节点 i 的注入无功。"""
        q = 0.0
        for j in range(self.n):
            angle_diff = self.va[i] - self.va[j]
            q += self.vm[i] * self.vm[j] * (
                self.g[i, j] * np.sin(angle_diff)
                - self.b[i, j] * np.cos(angle_diff)
            )
        return q

    def solve(self, tol: float = 1e-8, max_iter: int = 50) -> dict:
        """
        牛顿-拉夫逊潮流求解主循环（带阻尼步长控制）。

        收敛判据：
            max(|ΔP|, |ΔQ|) < tol

        局部二次收敛定理：若初始点 x0 充分靠近真解 x*，
        且雅可比矩阵 J(x*) 非奇异，则
            ||e_{k+1}|| ≤ C · ||e_k||^2
        其中 e_k = x_k - x*。

        引入阻尼因子 α∈(0,1] 保证全局收敛：
            x_{k+1} = x_k + α · Δx
        α 初始为 1.0，若目标函数（失配范数）不下降则减半。
        """
        n = self.n
        pq_idx = np.where(self.bus_types == 0)[0]
        pv_idx = np.where(self.bus_types == 1)[0]
        slack_idx = np.where(self.bus_types == 2)[0]

        theta_idx = np.concatenate([pq_idx, pv_idx])
        v_idx = pq_idx

        for it in range(max_iter):
            dp, dq = self.compute_power_mismatch()
            # 裁剪 slack/PV 对应的已知量
            dp[slack_idx] = 0.0
            dq[slack_idx] = 0.0
            dq[pv_idx] = 0.0

            mismatch = np.concatenate([dp[theta_idx], dq[v_idx]])
            norm_mis = np.linalg.norm(mismatch, ord=np.inf)
            if norm_mis < tol:
                return {
                    "converged": True,
                    "iterations": it,
                    "vm": self.vm.copy(),
                    "va": self.va.copy(),
                    "mismatch": norm_mis
                }

            # 构建并求解修正方程
            jacobian = self.build_jacobian()
            if jacobian.size == 0:
                break
            try:
                dx = np.linalg.solve(jacobian, mismatch)
            except np.linalg.LinAlgError:
                # 若雅可比奇异，采用最小二乘
                dx = np.linalg.lstsq(jacobian, mismatch, rcond=None)[0]

            n_theta = len(theta_idx)
            dtheta = dx[:n_theta]
            dvm = dx[n_theta:]

            # 阻尼线搜索
            alpha = 1.0
            vm_old = self.vm.copy()
            va_old = self.va.copy()
            best_norm = norm_mis
            for _ in range(5):
                self.va = va_old.copy()
                self.vm = vm_old.copy()
                self.va[theta_idx] += alpha * dtheta
                self.vm[v_idx] *= (1.0 + alpha * dvm)
                self.vm = np.clip(self.vm, 0.8, 1.2)

                dp_t, dq_t = self.compute_power_mismatch()
                dp_t[slack_idx] = 0.0
                dq_t[slack_idx] = 0.0
                dq_t[pv_idx] = 0.0
                mis_t = np.concatenate([dp_t[theta_idx], dq_t[v_idx]])
                norm_t = np.linalg.norm(mis_t, ord=np.inf)
                if norm_t < best_norm:
                    best_norm = norm_t
                    break
                alpha *= 0.5
            else:
                # 线搜索失败，接受最小步长
                self.va = va_old.copy()
                self.vm = vm_old.copy()
                self.va[theta_idx] += alpha * dtheta
                self.vm[v_idx] *= (1.0 + alpha * dvm)
                self.vm = np.clip(self.vm, 0.8, 1.2)

        dp, dq = self.compute_power_mismatch()
        dp[slack_idx] = 0.0
        dq[slack_idx] = 0.0
        dq[pv_idx] = 0.0
        mismatch = np.concatenate([dp[theta_idx], dq[v_idx]])
        norm_mis = np.linalg.norm(mismatch, ord=np.inf)
        return {
            "converged": False,
            "iterations": max_iter,
            "vm": self.vm.copy(),
            "va": self.va.copy(),
            "mismatch": norm_mis
        }


def build_y_bus(n: int, edges: np.ndarray,
                r_line: np.ndarray, x_line: np.ndarray,
                b_shunt: Optional[np.ndarray] = None) -> np.ndarray:
    """
    构建导纳矩阵 Y_bus。

    对于每条线路 (i,j)，串联阻抗 z = r + jx，导纳 y = 1/z = g + jb。
    节点导纳矩阵元素：
        Y_{ii} = Σ_{k∈N(i)} y_{ik} + j·b_{shunt,i}/2
        Y_{ij} = -y_{ij}   (i≠j)

    其中 N(i) 为与节点 i 相邻的节点集合。
    """
    y_bus = np.zeros((n, n), dtype=np.complex128)
    if b_shunt is None:
        b_shunt = np.zeros(len(edges), dtype=np.float64)

    for idx, (i, j) in enumerate(edges):
        i, j = int(i), int(j)
        z = complex(r_line[idx], x_line[idx])
        if abs(z) < 1e-12:
            z = 1e-12 + 1j * 1e-12
        y = 1.0 / z
        y_bus[i, i] += y + 1j * b_shunt[idx] * 0.5
        y_bus[j, j] += y + 1j * b_shunt[idx] * 0.5
        y_bus[i, j] -= y
        y_bus[j, i] -= y
    return y_bus
