"""
symplectic_dynamics.py
======================
约束多体系统辛几何时间积分与状态稀疏化

本模块将以下种子项目的核心算法融入结构力学：
  - 171_chirikov_iteration : Chirikov 标准映射（保面积辛映射）→ 哈密顿系统辛积分器
  - 141_cavity_flow_display : 向量场稀疏化（thinning）→ 大规模状态向量输出降采样

核心物理模型：
  - 受约束多体系统的哈密顿方程：
        dq/dt =  M^{-1} p
        dp/dt =  Q(q, p, t) - Φ_q^T λ
        0     =  Φ(q, t)          (位置约束)
    其中 q 为广义坐标，p 为广义动量，Φ 为代数约束，λ 为拉格朗日乘子。
  
  - 辛欧拉/Verlet 积分器：
        p_{n+1/2} = p_n + (h/2) [Q_n - Φ_q^T λ_n]
        q_{n+1}   = q_n + h M^{-1} p_{n+1/2}
        0         = Φ(q_{n+1})    (修正 λ)
        p_{n+1}   = p_{n+1/2} + (h/2) [Q_{n+1} - Φ_q^T λ_{n+1}]
    
  - 能量守恒检验：
        H = 0.5 p^T M^{-1} p + V(q)  应在机器精度附近守恒
  
  - 约束漂移修正（Baumgarte 稳定化）：
        Φ̈ + 2α Φ̇ + β² Φ = 0
    将其嵌入加速度级约束以抑制数值漂移。
"""

import numpy as np
from scipy.linalg import solve_triangular
from typing import Callable, Tuple, Optional, List


class ConstrainedHamiltonianSystem:
    """
    受约束哈密顿系统的数值积分器，支持辛 Verlet 格式与 Baumgarte 约束稳定化。
    """
    def __init__(self, n_dof: int, n_constr: int,
                 mass_matrix: np.ndarray,
                 force_func: Callable[[np.ndarray, np.ndarray, float], np.ndarray],
                 constraint_func: Callable[[np.ndarray], np.ndarray],
                 constraint_jacobian: Callable[[np.ndarray], np.ndarray],
                 alpha_baumgarte: float = 5.0,
                 beta_baumgarte: float = 5.0,
                 potential_func: Optional[Callable[[np.ndarray], float]] = None):
        """
        参数
        ----
        n_dof : 自由度数目
        n_constr : 约束方程数目
        mass_matrix : (n_dof, n_dof) 质量矩阵（正定）
        force_func : (q, p, t) -> Q  广义力函数（可含耗散、外力）
        constraint_func : q -> Φ  约束函数，返回 (n_constr,)
        constraint_jacobian : q -> Φ_q  约束雅可比，返回 (n_constr, n_dof)
        alpha_baumgarte, beta_baumgarte : Baumgarte 稳定化参数
        """
        self.n_dof = n_dof
        self.n_constr = n_constr
        self.M = np.asarray(mass_matrix, dtype=np.float64)
        self.M_inv = np.linalg.inv(self.M)
        self.force_func = force_func
        self.potential_func = potential_func
        self.phi_func = constraint_func
        self.phi_q_func = constraint_jacobian
        self.alpha = alpha_baumgarte
        self.beta = beta_baumgarte
        # 预分解质量矩阵（后续可改用 Cholesky）
        self._M_chol = None
        try:
            self._M_chol = np.linalg.cholesky(self.M)
        except np.linalg.LinAlgError:
            pass

    def _solve_mass(self, b: np.ndarray) -> np.ndarray:
        """利用 M 的逆或 Cholesky 分解求解 M x = b。"""
        if self._M_chol is not None:
            # 先解 L y = b，再解 L^T x = y
            y = solve_triangular(self._M_chol, b, lower=True)
            x = solve_triangular(self._M_chol.T, y, lower=False)
            return x
        return self.M_inv @ b

    def potential_energy(self, q: np.ndarray) -> float:
        """势能为零或由外部传入 potential_func 计算。"""
        if self.potential_func is not None:
            return float(self.potential_func(q))
        return 0.0

    def kinetic_energy(self, p: np.ndarray) -> float:
        """动能 T = 0.5 p^T M^{-1} p。"""
        v = self._solve_mass(p)
        return 0.5 * float(p @ v)

    def total_energy(self, q: np.ndarray, p: np.ndarray) -> float:
        return self.kinetic_energy(p) + self.potential_energy(q)

    def compute_lagrange_multipliers(self, q: np.ndarray, p: np.ndarray,
                                     t: float, qdot: Optional[np.ndarray] = None) -> np.ndarray:
        """
        求解加速度级约束中的拉格朗日乘子 λ。
        
        运动方程：M q̈ + Φ_q^T λ = Q
        对约束求二阶导：
            Φ_q q̈ + Φ_qq (q̇, q̇) = 0
        代入 Baumgarte 稳定化：
            Φ_q q̈ = -2α Φ̇ - β² Φ - Φ_qq(q̇,q̇)
        联立得：
            [Φ_q M^{-1} Φ_q^T] λ = Φ_q M^{-1} Q + 2α Φ̇ + β² Φ + Φ_qq(q̇,q̇)
        
        返回 λ
        """
        phi_q = self.phi_q_func(q)
        Q = self.force_func(q, p, t)
        if qdot is None:
            qdot = self._solve_mass(p)
        phi = self.phi_func(q)
        phi_dot = phi_q @ qdot
        # 简化处理：忽略 Φ_qq（二阶导数项），对较小系统可接受
        # 计算 Schur 补矩阵 S = Φ_q M^{-1} Φ_q^T
        # 为数值稳定，逐行计算
        S = phi_q @ self._solve_mass(phi_q.T)
        rhs = phi_q @ self._solve_mass(Q) + 2.0 * self.alpha * phi_dot + self.beta ** 2 * phi
        # 正则化避免奇异
        reg = 1e-12 * np.eye(self.n_constr)
        lam = np.linalg.solve(S + reg, rhs)
        return lam

    def step_symplectic_euler(self, q: np.ndarray, p: np.ndarray,
                              t: float, h: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        辛欧拉半步格式（Stormer-Verlet 的显隐混合版）。
        
        步骤：
          1. p_{1/2} = p_n + (h/2) (Q_n - Φ_q^T λ_n)
          2. q_{n+1} = q_n + h M^{-1} p_{1/2}
          3. 投影修正 q_{n+1} 使约束满足（可选）
          4. p_{n+1} = p_{1/2} + (h/2) (Q_{n+1} - Φ_q^T λ_{n+1})
        """
        # TODO: Hole 2 — 请实现受约束哈密顿系统的辛 Verlet 积分步
        raise NotImplementedError("Hole 2: step_symplectic_euler 待实现")

    def _project_position(self, q: np.ndarray, max_iter: int = 3) -> np.ndarray:
        """一步 Newton 投影使位置约束满足 Φ(q)=0。"""
        q_proj = q.copy()
        for _ in range(max_iter):
            phi = self.phi_func(q_proj)
            if np.linalg.norm(phi) < 1e-12:
                break
            phi_q = self.phi_q_func(q_proj)
            S = phi_q @ self._solve_mass(phi_q.T)
            reg = 1e-12 * np.eye(self.n_constr)
            delta_lam = np.linalg.solve(S + reg, phi)
            q_proj -= self._solve_mass(phi_q.T @ delta_lam)
        return q_proj

    def integrate(self, q0: np.ndarray, p0: np.ndarray,
                  t_span: Tuple[float, float], h: float,
                  thinning_factor: int = 1) -> dict:
        """
        执行完整时间积分，返回轨迹与能量历史。
        
        参数
        ----
        q0, p0 : 初始状态
        t_span : (t0, tf)
        h : 时间步长
        thinning_factor : 输出稀疏化因子（每 thinning_factor 步保存一次）
        
        返回
        ----
        dict : {
            "t": 时间数组,
            "q": 广义坐标轨迹,
            "p": 广义动量轨迹,
            "energy": 总能量,
            "energy_drift": 能量相对漂移
        }
        """
        t0, tf = t_span
        if h <= 0:
            raise ValueError("步长 h 必须为正")
        n_steps = int(np.ceil((tf - t0) / h))
        q = np.asarray(q0, dtype=np.float64).copy()
        p = np.asarray(p0, dtype=np.float64).copy()
        t = t0
        e0 = self.total_energy(q, p)
        # 预分配（稀疏存储）
        save_every = max(1, thinning_factor)
        n_save = n_steps // save_every + 2
        ts = np.zeros(n_save)
        qs = np.zeros((n_save, self.n_dof))
        ps = np.zeros((n_save, self.n_dof))
        es = np.zeros(n_save)
        idx = 0
        ts[idx] = t
        qs[idx] = q
        ps[idx] = p
        es[idx] = e0
        idx += 1
        for step in range(n_steps):
            q, p = self.step_symplectic_euler(q, p, t, h)
            t = min(t + h, tf)
            if (step + 1) % save_every == 0 or step == n_steps - 1:
                if idx < n_save:
                    ts[idx] = t
                    qs[idx] = q
                    ps[idx] = p
                    es[idx] = self.total_energy(q, p)
                    idx += 1
        # 裁剪
        ts = ts[:idx]
        qs = qs[:idx]
        ps = ps[:idx]
        es = es[:idx]
        energy_scale = max(abs(e0), np.max(np.abs(es)), 1e-12)
        energy_drift = np.abs((es - e0) / energy_scale)
        return {
            "t": ts,
            "q": qs,
            "p": ps,
            "energy": es,
            "energy_drift": energy_drift,
            "max_drift": float(np.max(energy_drift)),
            "mean_drift": float(np.mean(energy_drift))
        }


def thin_state_vectors(states: np.ndarray, thin_factor: int,
                       method: str = "uniform") -> np.ndarray:
    """
    基于 141_cavity_flow_display 中向量场稀疏化思想，对高维状态向量序列进行降采样。
    
    method="uniform" : 等间隔保留
    method="energy"  : 保留能量变化率较大的时刻（自适应）
    """
    if thin_factor <= 1:
        return states
    n = states.shape[0]
    if method == "uniform":
        keep = np.arange(0, n, thin_factor)
        return states[keep]
    elif method == "energy":
        if states.ndim == 1:
            return states[::thin_factor]
        # 用相邻状态差分范数作为变化率指标
        diffs = np.linalg.norm(np.diff(states, axis=0), axis=1)
        # 分箱，每箱取最大变化率点
        n_bins = max(1, n // thin_factor)
        bin_size = n // n_bins
        keep = [0]
        for b in range(n_bins):
            start = b * bin_size
            end = min((b + 1) * bin_size, n - 1)
            if start >= end:
                continue
            local_max = start + np.argmax(diffs[start:end])
            keep.append(local_max)
        keep = sorted(set(keep))
        return states[keep]
    else:
        raise ValueError(f"未知稀疏化方法: {method}")
