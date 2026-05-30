
import numpy as np
from typing import List, Tuple, Optional, Dict
from flexible_beam_modal import EulerBernoulliBeam
from rotation_mechanics import so3_exp, so3_log, skew_symmetric


class RigidBody:
    def __init__(self, mass: float, inertia: np.ndarray,
                 position: np.ndarray, orientation: np.ndarray = None):
        self.mass = float(mass)
        self.inertia = np.asarray(inertia, dtype=np.float64)
        self.position = np.asarray(position, dtype=np.float64).flatten()[:3]
        if orientation is None:
            self.orientation = np.eye(3)
        else:
            self.orientation = np.asarray(orientation, dtype=np.float64)
        if self.inertia.shape != (3, 3):
            raise ValueError("惯性张量必须为 3×3")

    def dof_count(self) -> int:
        return 6

    def get_state(self) -> np.ndarray:
        theta = so3_log(self.orientation)
        return np.concatenate([self.position, theta])

    def set_state(self, state: np.ndarray):
        state = np.asarray(state, dtype=np.float64).flatten()
        if len(state) != 6:
            raise ValueError("刚体状态长度必须为 6")
        self.position = state[:3].copy()
        self.orientation = so3_exp(state[3:6])

    def mass_matrix(self) -> np.ndarray:
        M = np.zeros((6, 6), dtype=np.float64)
        M[:3, :3] = self.mass * np.eye(3)
        M[3:6, 3:6] = self.inertia
        return M


class FlexibleMember:
    def __init__(self, beam: EulerBernoulliBeam, n_modes: int,
                 node_i: int, node_j: int):
        self.beam = beam
        self.n_modes = n_modes
        self.node_i = node_i
        self.node_j = node_j
        self.eta = np.zeros(n_modes, dtype=np.float64)
        self.eta_dot = np.zeros(n_modes, dtype=np.float64)

        self.M_f, self.K_f = beam.modal_stiffness_mass(n_modes)

    def dof_count(self) -> int:
        return self.n_modes

    def get_state(self) -> np.ndarray:
        return self.eta.copy()

    def set_state(self, state: np.ndarray):
        self.eta = np.asarray(state, dtype=np.float64).flatten()[:self.n_modes].copy()

    def endpoint_displacement(self, end: str = "tip") -> float:
        x_eval = np.array([self.beam.L]) if end == "tip" else np.array([0.0])
        disp = 0.0
        for i in range(self.n_modes):
            W = self.beam.modal_shape(x_eval, i)
            disp += self.eta[i] * W[0]
        return disp

    def endpoint_rotation(self, end: str = "tip") -> float:
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
    def __init__(self):
        self.rigid_bodies: List[RigidBody] = []
        self.flexible_members: List[FlexibleMember] = []
        self.constraints: List[Tuple[int, int, str]] = []
        self._damping_ratio: float = 0.02

    def add_rigid_body(self, body: RigidBody) -> int:
        self.rigid_bodies.append(body)
        return len(self.rigid_bodies) - 1

    def add_flexible_member(self, member: FlexibleMember) -> int:
        self.flexible_members.append(member)
        return len(self.flexible_members) - 1

    def add_constraint(self, body_idx: int, member_idx: int, end: str):
        if end not in ("root", "tip"):
            raise ValueError("end 必须为 'root' 或 'tip'")
        self.constraints.append((body_idx, member_idx, end))

    @property
    def n_dof(self) -> int:
        return sum(rb.dof_count() for rb in self.rigid_bodies) \
               + sum(fm.dof_count() for fm in self.flexible_members)

    @property
    def n_constr(self) -> int:
        return len(self.constraints)

    def assemble_state(self) -> np.ndarray:
        parts = []
        for rb in self.rigid_bodies:
            parts.append(rb.get_state())
        for fm in self.flexible_members:
            parts.append(fm.get_state())
        return np.concatenate(parts)

    def disassemble_state(self, q: np.ndarray):
        offset = 0
        for rb in self.rigid_bodies:
            rb.set_state(q[offset:offset + 6])
            offset += 6
        for fm in self.flexible_members:
            fm.set_state(q[offset:offset + fm.dof_count()])
            offset += fm.dof_count()

    def assemble_mass_matrix(self) -> np.ndarray:
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
        K = np.zeros((self.n_dof, self.n_dof), dtype=np.float64)
        offset = sum(rb.dof_count() for rb in self.rigid_bodies)
        for fm in self.flexible_members:
            nm = fm.dof_count()
            K[offset:offset + nm, offset:offset + nm] = fm.K_f
            offset += nm
        return K

    def constraint_function(self, q: np.ndarray) -> np.ndarray:
        self.disassemble_state(q)
        phi_list = []
        for body_idx, member_idx, end in self.constraints:
            fm = self.flexible_members[member_idx]

            disp = fm.endpoint_displacement(end)
            phi_list.append(np.array([disp]))
        if len(phi_list) == 0:
            return np.zeros(0)
        return np.concatenate(phi_list)

    def constraint_jacobian(self, q: np.ndarray) -> np.ndarray:
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
        self.disassemble_state(q)

        qdot = np.zeros_like(q)
        M = self.assemble_mass_matrix()

        try:
            qdot = np.linalg.solve(M, p)
        except np.linalg.LinAlgError:
            qdot = np.linalg.lstsq(M, p, rcond=None)[0]
        K = self.assemble_stiffness_matrix()

        raise NotImplementedError("Hole 3: force_function 中的广义力计算待实现")
