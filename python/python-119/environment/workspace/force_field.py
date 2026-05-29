"""
force_field.py
力场定义模块

融合原项目:
- 146_ccvt_reflect: 反射边界约束思想用于非键相互作用截断
- 252_cvt_box: 盒子约束投影思想用于 PBC 下的力计算

功能:
1. Lennard-Jones (LJ) 非键势
2. FENE (Finite Extensible Nonlinear Elastic) 键合势
3. 弯曲角势（Angle Bending Potential）
4. 截断处理与邻居列表
"""

import numpy as np
from typing import Tuple, Optional
from numeric_utils import safe_divide, soft_cutoff, distance_matrix_pbc


class ForceField:
    """
    粗粒化分子动力学力场。
    
    势能组成:
        U_total = U_LJ + U_FENE + U_angle
    
    1. Lennard-Jones 势（非键相互作用）:
        U_LJ(r) = 4ε[(σ/r)^12 - (σ/r)^6]
        截断于 r_c，采用最小像约定
    
    2. FENE 键合势:
        U_FENE(r) = -0.5 k R_0^2 ln[1 - (r/R_0)^2]
        其中 R_0 为最大伸长度，k 为弹簧常数
    
    3. 弯曲角势（保持链刚性）:
        U_angle(θ) = k_θ (θ - θ_0)^2
        θ 为三个连续单体形成的角度
    """
    
    def __init__(
        self,
        epsilon: float = 1.0,
        sigma: float = 1.0,
        rcutoff: float = 2.5,
        fene_k: float = 30.0,
        fene_R0: float = 1.5,
        angle_k: float = 5.0,
        angle_theta0: float = np.pi,
    ):
        """
        初始化力场参数。
        
        参数:
            epsilon: LJ 势阱深度 ε
            sigma: LJ 特征长度 σ
            rcutoff: LJ 截断半径 r_c（单位: σ）
            fene_k: FENE 弹簧常数 k
            fene_R0: FENE 最大伸长度 R_0
            angle_k: 弯曲角力常数 k_θ
            angle_theta0: 平衡角 θ_0（弧度）
        """
        if epsilon <= 0 or sigma <= 0:
            raise ValueError("epsilon, sigma 必须 > 0")
        if rcutoff <= sigma:
            raise ValueError("rcutoff 必须 > sigma")
        if fene_k <= 0 or fene_R0 <= 0:
            raise ValueError("fene_k, fene_R0 必须 > 0")
        
        self.epsilon = epsilon
        self.sigma = sigma
        self.rcutoff = rcutoff
        self.fene_k = fene_k
        self.fene_R0 = fene_R0
        self.angle_k = angle_k
        self.angle_theta0 = angle_theta0
        
        # 预计算 LJ 截断处的势能和力偏移，保证连续性
        self._compute_lj_shift()
    
    def _compute_lj_shift(self):
        """
        计算截断偏移量，使势能在 r_c 处连续。
        
        U_shift = U_LJ(r_c)
        """
        sr = self.sigma / self.rcutoff
        sr6 = sr ** 6
        sr12 = sr6 ** 2
        self.u_shift = 4.0 * self.epsilon * (sr12 - sr6)
    
    def lj_potential(self, r: np.ndarray) -> np.ndarray:
        """
        计算 Lennard-Jones 势能。
        
        公式:
            U_LJ(r) = 4ε[(σ/r)^12 - (σ/r)^6] - U_shift,  r <= r_c
            U_LJ(r) = 0,                                 r > r_c
        
        参数:
            r: 距离数组
        
        返回:
            势能数组
        """
        r = np.asarray(r)
        u = np.zeros_like(r, dtype=float)
        
        mask = (r > 1e-10) & (r <= self.rcutoff)
        if np.any(mask):
            sr = self.sigma / r[mask]
            sr6 = sr ** 6
            sr12 = sr6 ** 2
            u[mask] = 4.0 * self.epsilon * (sr12 - sr6) - self.u_shift
        
        return u
    
    def lj_force_scalar(self, r: float) -> float:
        """
        LJ 力的大小（标量），方向为排斥/吸引方向。
        
        公式:
            F(r) = -dU/dr = 24ε/σ [2(σ/r)^13 - (σ/r)^7]
        
        参数:
            r: 距离
        
        返回:
            力的大小（正为排斥，负为吸引）
        """
        if r <= 1e-10 or r > self.rcutoff:
            return 0.0
        sr = self.sigma / r
        sr7 = sr ** 7
        sr13 = sr ** 13
        return 24.0 * self.epsilon / self.sigma * (2.0 * sr13 - sr7)
    
    def lj_forces_vector(self, positions: np.ndarray, box: np.ndarray) -> np.ndarray:
        """
        计算所有非键 LJ 力（向量形式）。
        
        使用最小像约定处理 PBC。
        
        参数:
            positions: (N, 3) 位置数组
            box: (3,) 盒子尺寸
        
        返回:
            (N, 3) 力数组
        """
        N = positions.shape[0]
        forces = np.zeros_like(positions)
        
        for i in range(N):
            for j in range(i + 1, N):
                dr = positions[i] - positions[j]
                dr = dr - box * np.rint(dr / box)
                r = np.linalg.norm(dr)
                
                if r > 1e-10 and r <= self.rcutoff:
                    f_mag = self.lj_force_scalar(r)
                    f_vec = f_mag * dr / r
                    forces[i] += f_vec
                    forces[j] -= f_vec
        
        return forces
    
    def fene_potential(self, r: float) -> float:
        """
        FENE 键合势能。
        
        公式:
            U_FENE(r) = -0.5 k R_0^2 ln[1 - (r/R_0)^2],  r < R_0
            U_FENE(r) = +∞,                              r >= R_0
        
        参数:
            r: 键长
        
        返回:
            势能值
        """
        if r >= self.fene_R0:
            # 数值鲁棒性: 返回大数而非无穷
            return 1e10
        if r <= 0:
            return 0.0
        ratio = r / self.fene_R0
        return -0.5 * self.fene_k * self.fene_R0 ** 2 * np.log(1.0 - ratio ** 2)
    
    def fene_force_scalar(self, r: float) -> float:
        """
        FENE 力的大小。
        
        公式:
            F(r) = -dU/dr = -k r / [1 - (r/R_0)^2]
        
        参数:
            r: 键长
        
        返回:
            力的大小（负值表示吸引力）
        """
        if r >= self.fene_R0 or r <= 0:
            return 0.0
        ratio_sq = (r / self.fene_R0) ** 2
        return -self.fene_k * r / (1.0 - ratio_sq)
    
    def fene_forces(self, positions: np.ndarray, box: np.ndarray, chain_starts: np.ndarray) -> np.ndarray:
        """
        计算所有 FENE 键合力。
        
        参数:
            positions: (N, 3) 位置数组
            box: (3,) 盒子尺寸
            chain_starts: 每条链的起始索引数组
        
        返回:
            (N, 3) 力数组
        """
        N = positions.shape[0]
        forces = np.zeros_like(positions)
        beads_per_chain = chain_starts[1] - chain_starts[0] if len(chain_starts) > 1 else N
        
        for c in range(len(chain_starts)):
            start = chain_starts[c]
            end = min(start + beads_per_chain, N)
            for i in range(start, end - 1):
                j = i + 1
                dr = positions[i] - positions[j]
                dr = dr - box * np.rint(dr / box)
                r = np.linalg.norm(dr)
                
                if r > 1e-10:
                    f_mag = self.fene_force_scalar(r)
                    f_vec = f_mag * dr / r
                    forces[i] += f_vec
                    forces[j] -= f_vec
        
        return forces
    
    def angle_potential(self, theta: float) -> float:
        """
        弯曲角势能。
        
        公式:
            U_angle(θ) = k_θ (θ - θ_0)^2
        
        参数:
            theta: 角度（弧度）
        
        返回:
            势能值
        """
        dtheta = theta - self.angle_theta0
        return self.angle_k * dtheta ** 2
    
    def angle_forces(self, positions: np.ndarray, box: np.ndarray, chain_starts: np.ndarray) -> np.ndarray:
        """
        计算弯曲角力。
        
        对三个连续单体 i-j-k，角力作用于保持 θ_ijk ≈ θ_0。
        
        参数:
            positions: (N, 3) 位置数组
            box: (3,) 盒子尺寸
            chain_starts: 每条链的起始索引
        
        返回:
            (N, 3) 力数组
        """
        N = positions.shape[0]
        forces = np.zeros_like(positions)
        beads_per_chain = chain_starts[1] - chain_starts[0] if len(chain_starts) > 1 else N
        
        for c in range(len(chain_starts)):
            start = chain_starts[c]
            end = min(start + beads_per_chain, N)
            for j in range(start + 1, end - 1):
                i = j - 1
                k = j + 1
                
                # 向量 r_ji 和 r_jk
                rji = positions[i] - positions[j]
                rji = rji - box * np.rint(rji / box)
                rjk = positions[k] - positions[j]
                rjk = rjk - box * np.rint(rjk / box)
                
                norm_ji = np.linalg.norm(rji)
                norm_jk = np.linalg.norm(rjk)
                
                if norm_ji < 1e-10 or norm_jk < 1e-10:
                    continue
                
                # 角度
                cos_theta = np.dot(rji, rjk) / (norm_ji * norm_jk)
                cos_theta = np.clip(cos_theta, -1.0, 1.0)
                theta = np.arccos(cos_theta)
                
                # 力常数乘以角度偏差
                dtheta = theta - self.angle_theta0
                
                # 简化角力计算: 对 i 和 k 施加恢复力
                # 力大小与 dtheta 成正比，方向垂直于键
                force_mag = 2.0 * self.angle_k * dtheta
                
                # 对 i 的力: 推向使角度增大的方向
                if norm_ji > 1e-10:
                    perp_i = rji - np.dot(rji, rjk) / (norm_jk ** 2) * rjk
                    perp_norm = np.linalg.norm(perp_i)
                    if perp_norm > 1e-10:
                        forces[i] += force_mag * perp_i / perp_norm
                        forces[j] -= force_mag * perp_i / perp_norm
                
                # 对 k 的力
                if norm_jk > 1e-10:
                    perp_k = rjk - np.dot(rjk, rji) / (norm_ji ** 2) * rji
                    perp_norm = np.linalg.norm(perp_k)
                    if perp_norm > 1e-10:
                        forces[k] += force_mag * perp_k / perp_norm
                        forces[j] -= force_mag * perp_k / perp_norm
        
        return forces
    
    def compute_total_forces(
        self,
        positions: np.ndarray,
        box: np.ndarray,
        chain_starts: np.ndarray,
    ) -> np.ndarray:
        """
        计算总力 = LJ 非键力 + FENE 键合力 + 弯曲角力。
        
        参数:
            positions: (N, 3) 位置数组
            box: (3,) 盒子尺寸
            chain_starts: 每条链的起始索引
        
        返回:
            (N, 3) 总力数组
        """
        f_lj = self.lj_forces_vector(positions, box)
        f_fene = self.fene_forces(positions, box, chain_starts)
        f_angle = self.angle_forces(positions, box, chain_starts)
        
        return f_lj + f_fene + f_angle
    
    def total_potential_energy(
        self,
        positions: np.ndarray,
        box: np.ndarray,
        chain_starts: np.ndarray,
    ) -> float:
        """
        计算系统总势能。
        
        参数:
            positions: (N, 3) 位置数组
            box: (3,) 盒子尺寸
            chain_starts: 每条链的起始索引
        
        返回:
            总势能
        """
        N = positions.shape[0]
        beads_per_chain = chain_starts[1] - chain_starts[0] if len(chain_starts) > 1 else N
        
        # LJ 势能
        u_lj = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                dr = positions[i] - positions[j]
                dr = dr - box * np.rint(dr / box)
                r = np.linalg.norm(dr)
                u_lj += self.lj_potential(r)
        
        # FENE 势能
        u_fene = 0.0
        for c in range(len(chain_starts)):
            start = chain_starts[c]
            end = min(start + beads_per_chain, N)
            for i in range(start, end - 1):
                j = i + 1
                dr = positions[i] - positions[j]
                dr = dr - box * np.rint(dr / box)
                r = np.linalg.norm(dr)
                u_fene += self.fene_potential(r)
        
        # 角势能
        u_angle = 0.0
        for c in range(len(chain_starts)):
            start = chain_starts[c]
            end = min(start + beads_per_chain, N)
            for j in range(start + 1, end - 1):
                i = j - 1
                k = j + 1
                rji = positions[i] - positions[j]
                rji = rji - box * np.rint(rji / box)
                rjk = positions[k] - positions[j]
                rjk = rjk - box * np.rint(rjk / box)
                
                norm_ji = np.linalg.norm(rji)
                norm_jk = np.linalg.norm(rjk)
                if norm_ji > 1e-10 and norm_jk > 1e-10:
                    cos_t = np.dot(rji, rjk) / (norm_ji * norm_jk)
                    cos_t = np.clip(cos_t, -1.0, 1.0)
                    theta = np.arccos(cos_t)
                    u_angle += self.angle_potential(theta)
        
        return u_lj + u_fene + u_angle
