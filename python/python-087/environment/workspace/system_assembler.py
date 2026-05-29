"""
system_assembler.py
===================
刚柔耦合多体系统全局矩阵组装与耦合约束

本模块将以下种子项目的核心算法融入结构力学：
  - 141_cavity_flow_display : 结构网格向量场处理思想 → 系统状态向量组装与节点力映射

核心物理模型：
  - 柔性梁的模态叠加降阶模型：
        w(x,t) = Σ_{i=1}^{N_m} η_i(t) W_i(x)
    其中 η_i(t) 为模态坐标，W_i(x) 为空间模态振型。
  
  - 刚柔耦合系统的广义坐标：
        q = [q_r^T,  η^T]^T
    q_r 为刚体自由度（位置 + 姿态），η 为柔性模态坐标。
  
  - 全局质量矩阵（块对角）：
        M = diag(M_r,  M_f)
    M_r = diag(m_r I,  J_r)   — 刚体平动与转动惯量
    M_f = ρA L · I            — 模态质量（已归一化）
  
  - 全局刚度矩阵：
        K = diag(0,  K_f)
    刚体无刚度，柔性部分 K_f = diag(EI β_i⁴)。
  
  - 约束方程（铰接约束）：
        Φ(q) = r_j + u_j(L_j, t) - r_k - u_k(0, t) = 0
    其中 r_j, r_k 为刚体/节点位置，u 为柔性梁端部位移。
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from flexible_beam_modal import EulerBernoulliBeam
from rotation_mechanics import so3_exp, so3_log, skew_symmetric


class RigidBody:
    """刚体节点（桁架 hub），含三维平动与转动自由度。"""
    def __init__(self, mass: float, inertia: np.ndarray,
                 position: np.ndarray, orientation: np.ndarray = None):
        self.mass = float(mass)
        self.inertia = np.asarray(inertia, dtype=np.float64)  # 3×3 转动惯量张量
        self.position = np.asarray(position, dtype=np.float64).flatten()[:3]
        if orientation is None:
            self.orientation = np.eye(3)
        else:
            self.orientation = np.asarray(orientation, dtype=np.float64)
        if self.inertia.shape != (3, 3):
            raise ValueError("惯性张量必须为 3×3")

    def dof_count(self) -> int:
        return 6  # 3 平动 + 3 转动（旋转向量参数化）

    def get_state(self) -> np.ndarray:
        """返回当前状态 [position; so3_log(orientation)]。"""
        theta = so3_log(self.orientation)
        return np.concatenate([self.position, theta])

    def set_state(self, state: np.ndarray):
        """从状态向量更新。"""
        state = np.asarray(state, dtype=np.float64).flatten()
        if len(state) != 6:
            raise ValueError("刚体状态长度必须为 6")
        self.position = state[:3].copy()
        self.orientation = so3_exp(state[3:6])

    def mass_matrix(self) -> np.ndarray:
        """当前位形下的质量矩阵 M_r = diag(mI, J)。"""
        M = np.zeros((6, 6), dtype=np.float64)
        M[:3, :3] = self.mass * np.eye(3)
        M[3:6, 3:6] = self.inertia
        return M


class FlexibleMember:
    """柔性梁构件，采用模态叠加降阶模型。"""
    def __init__(self, beam: EulerBernoulliBeam, n_modes: int,
                 node_i: int, node_j: int):
        self.beam = beam
        self.n_modes = n_modes
        self.node_i = node_i
        self.node_j = node_j
        self.eta = np.zeros(n_modes, dtype=np.float64)
        self.eta_dot = np.zeros(n_modes, dtype=np.float64)
        # 预计算模态质量与刚度
        self.M_f, self.K_f = beam.modal_stiffness_mass(n_modes)

    def dof_count(self) -> int:
        return self.n_modes

    def get_state(self) -> np.ndarray:
        return self.eta.copy()

    def set_state(self, state: np.ndarray):
        self.eta = np.asarray(state, dtype=np.float64).flatten()[:self.n_modes].copy()

    def endpoint_displacement(self, end: str = "tip") -> float:
        """
        计算端部位移（仅考虑横向，简化）。
        w(L) = Σ η_i W_i(L)
        """
        x_eval = np.array([self.beam.L]) if end == "tip" else np.array([0.0])
        disp = 0.0
        for i in range(self.n_modes):
            W = self.beam.modal_shape(x_eval, i)
            disp += self.eta[i] * W[0]
        return disp

    def endpoint_rotation(self, end: str = "tip") -> float:
        """端部转角近似：dw/dx。"""
        x_eval = np.array([0.0, self.beam.L * 1e-4, self.beam.L])
        if end == "root":
            x_eval = np.array([0.0, self.beam.L * 1e-4])
        w = np.zeros_like(x_eval)
        for i in range(self.n_modes):
            W = self.beam.modal_shape(x_eval, i)
            w += self.eta[i] * W
        dw = np.gradient(w, x_eval)
        return dw[0] if end == "root" else dw[-1]


class MultibodySystem:
    """
    刚柔耦合多体系统：由若干刚体节点和柔性梁构件组成。
    """
    def __init__(self):
        self.rigid_bodies: List[RigidBody] = []
        self.flexible_members: List[FlexibleMember] = []
        self.constraints: List[Tuple[int, int, str]] = []  # (body_idx, member_idx, end)
        self._damping_ratio: float = 0.02

    def add_rigid_body(self, body: RigidBody) -> int:
        self.rigid_bodies.append(body)
        return len(self.rigid_bodies) - 1

    def add_flexible_member(self, member: FlexibleMember) -> int:
        self.flexible_members.append(member)
        return len(self.flexible_members) - 1

    def add_constraint(self, body_idx: int, member_idx: int, end: str):
        """
        添加铰接约束：柔性梁的 end("root" 或 "tip") 与刚体 body_idx 固连。
        """
        if end not in ("root", "tip"):
            raise ValueError("end 必须为 'root' 或 'tip'")
        self.constraints.append((body_idx, member_idx, end))

    @property
    def n_dof(self) -> int:
        return sum(rb.dof_count() for rb in self.rigid_bodies) \
               + sum(fm.dof_count() for fm in self.flexible_members)

    @property
    def n_constr(self) -> int:
        """每个铰接约束提供 1 个横向位移约束（简化一维梁模型）。"""
        return len(self.constraints)

    def assemble_state(self) -> np.ndarray:
        """将所有状态组装为一个长向量。"""
        parts = []
        for rb in self.rigid_bodies:
            parts.append(rb.get_state())
        for fm in self.flexible_members:
            parts.append(fm.get_state())
        return np.concatenate(parts)

    def disassemble_state(self, q: np.ndarray):
        """将长向量写回各部件。"""
        offset = 0
        for rb in self.rigid_bodies:
            rb.set_state(q[offset:offset + 6])
            offset += 6
        for fm in self.flexible_members:
            fm.set_state(q[offset:offset + fm.dof_count()])
            offset += fm.dof_count()

    def assemble_mass_matrix(self) -> np.ndarray:
        """组装全局质量矩阵。"""
        M = np.zeros((self.n_dof, self.n_dof), dtype=np.float64)
        offset = 0
        for rb in self.rigid_bodies:
            m = rb.mass_matrix()
            M[offset:offset + 6, offset:offset + 6] = m
            offset += 6
        for fm in self.flexible_members:
            nm = fm.dof_count()
            M[offset:offset + nm, offset:offset + nm] = fm.M_f
            offset += nm
        return M

    def assemble_stiffness_matrix(self) -> np.ndarray:
        """组装全局刚度矩阵（仅柔性部分非零）。"""
        K = np.zeros((self.n_dof, self.n_dof), dtype=np.float64)
        offset = sum(rb.dof_count() for rb in self.rigid_bodies)
        for fm in self.flexible_members:
            nm = fm.dof_count()
            K[offset:offset + nm, offset:offset + nm] = fm.K_f
            offset += nm
        return K

    def constraint_function(self, q: np.ndarray) -> np.ndarray:
        """
        约束残差 Φ(q)。
        对每个约束：柔性梁端部位移引起的空间偏移应等于刚体连接点位置。
        简化模型：仅约束端部横向位移为 0（固支端）。
        """
        self.disassemble_state(q)
        phi_list = []
        for body_idx, member_idx, end in self.constraints:
            fm = self.flexible_members[member_idx]
            # 约束：端部位移 = 0（一维横向位移约束）
            disp = fm.endpoint_displacement(end)
            phi_list.append(np.array([disp]))
        if len(phi_list) == 0:
            return np.zeros(0)
        return np.concatenate(phi_list)

    def constraint_jacobian(self, q: np.ndarray) -> np.ndarray:
        """
        约束雅可比 Φ_q = ∂Φ/∂q。
        使用数值差分近似。
        """
        eps = np.sqrt(np.finfo(float).eps)
        phi0 = self.constraint_function(q)
        n_c = len(phi0)
        n_d = len(q)
        J = np.zeros((n_c, n_d), dtype=np.float64)
        for j in range(n_d):
            q1 = q.copy()
            h = eps * max(1.0, abs(q[j]))
            q1[j] += h
            phi1 = self.constraint_function(q1)
            J[:, j] = (phi1 - phi0) / h
        return J

    def force_function(self, q: np.ndarray, p: np.ndarray, t: float) -> np.ndarray:
        """
        广义力 Q = -K q - C q̇ + F_ext。
        此处 F_ext 包含重力与简谐激励。
        """
        self.disassemble_state(q)
        # 恢复速度（对模态坐标，p = M η̇）
        qdot = np.zeros_like(q)
        M = self.assemble_mass_matrix()
        # 安全求解 q̇ = M^{-1} p
        try:
            qdot = np.linalg.solve(M, p)
        except np.linalg.LinAlgError:
            qdot = np.linalg.lstsq(M, p, rcond=None)[0]
        K = self.assemble_stiffness_matrix()
        # TODO: Hole 3 — 请根据 Rayleigh 阻尼模型与多体系统外力模型实现广义力 Q 的组装
        raise NotImplementedError("Hole 3: force_function 中的广义力计算待实现")
