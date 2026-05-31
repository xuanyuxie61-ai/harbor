# -*- coding: utf-8 -*-
"""
nonlinear_solver.py
非线性方程组 Newton-Raphson 求解器

融合种子项目:
  - 1286_trapezoidal: 隐式梯形积分与残差构造思想
  - 908_predator_prey_ode: 动力学方程守恒量检测

科学背景:
  壳体非线性平衡方程:
    R(u, λ) = F_int(u) - λ F_ext = 0

  Newton-Raphson 迭代:
    K_T(u_k) Δu = λ F_ext - F_int(u_k)
    u_{k+1} = u_k + Δu

  其中 K_T 为切线刚度矩阵:
    K_T = K_m + K_b + K_σ + K_g

  收敛判据:
    ||R|| / ||λ F_ext|| < ε  (力收敛)
    ||Δu|| / ||u|| < ε       (位移收敛)

  能量守恒检测 (基于 predator_prey_conserved 思想):
    Π = U_int - W_ext
    ΔΠ 在弹性问题中应单调递减并趋于稳定
"""

import numpy as np
from scipy.sparse.linalg import spsolve
from scipy.sparse import csr_matrix


class NewtonRaphsonSolver:
    """
    带线搜索的修正 Newton-Raphson 求解器
    """

    def __init__(self, max_iter: int = 50, tol_force: float = 1e-6,
                 tol_disp: float = 1e-8, line_search: bool = True):
        self.max_iter = max_iter
        self.tol_force = tol_force
        self.tol_disp = tol_disp
        self.line_search = line_search
        self.history = []

    def solve(self, fem_model, external_force: np.ndarray, lambda_load: float,
              u0: np.ndarray = None) -> dict:
        """
        求解非线性方程组 F_int(u) = lambda_load * F_ext

        Parameters
        ----------
        fem_model : ShellFEModel
        external_force : (n_dof,) ndarray
            参考外载荷向量 F_ext
        lambda_load : float
            载荷因子 λ
        u0 : (n_dof,) ndarray, optional
            初始猜测

        Returns
        -------
        result : dict
            包含 'disp', 'converged', 'iterations', 'residual_norm', 'energy'
        """
        n_dof = fem_model.n_dof
        f_ext = external_force
        if u0 is None:
            u = np.zeros(n_dof)
        else:
            u = np.array(u0, dtype=float)

        f_target = lambda_load * f_ext
        energy_history = []

        for it in range(self.max_iter):
            f_int = fem_model.internal_force(u)
            residual = f_target - f_int

            # 切线刚度 (近似: K_linear + K_geometric)
            K_lin = fem_model.assemble_linear_stiffness()
            K_geo = fem_model.assemble_geometric_stiffness(u)
            K_t = K_lin + K_geo

            # 施加 Dirichlet 边界条件
            # 底部固支 (u=v=w=0); 顶部限制面内位移 (x=y=0)，允许轴向 z
            bottom, top = fem_model.mesh.get_boundary_nodes()
            fixed_dofs = []
            for nid in bottom:
                fixed_dofs.extend([nid * 3 + 0, nid * 3 + 1, nid * 3 + 2])
            for nid in top:
                fixed_dofs.extend([nid * 3 + 0, nid * 3 + 1])  # 仅固定 x, y
            # 额外约束: 固定底部第一个节点的 z 转动等效约束 (简化)
            if len(bottom) > 0:
                fixed_dofs.append(bottom[0] * 3 + 2)
            fixed_dofs = np.unique(fixed_dofs)
            free_dofs = np.setdiff1d(np.arange(n_dof), fixed_dofs)

            if len(free_dofs) == 0:
                return {
                    'disp': u,
                    'converged': True,
                    'iterations': it,
                    'residual_norm': 0.0,
                    'energy': 0.0,
                    'history': self.history
                }

            # 降维求解
            K_ff = K_t[free_dofs][:, free_dofs]
            R_f = residual[free_dofs]

            # 检查矩阵正定性 (尝试 Cholesky，失败则转 LU)
            try:
                du_f = spsolve(K_ff, R_f)
            except Exception:
                # 添加小量正则化
                K_reg = K_ff + 1e-8 * csr_matrix(np.eye(len(free_dofs)))
                du_f = spsolve(K_reg, R_f)

            du = np.zeros(n_dof)
            du[free_dofs] = du_f

            # 线搜索 (Armijo 条件)
            alpha = 1.0
            if self.line_search:
                alpha = self._armijo_line_search(fem_model, u, du, f_target, f_int)

            u_new = u + alpha * du

            # 能量计算
            u_mid = 0.5 * (u + u_new)
            f_int_mid = fem_model.internal_force(u_mid)
            energy = 0.5 * np.dot(u_new, fem_model.internal_force(u_new)) - np.dot(u_new, f_target)
            energy_history.append(energy)

            # 收敛判据
            norm_r = np.linalg.norm(R_f)
            norm_f = np.linalg.norm(f_target[free_dofs]) + 1e-14
            norm_du = np.linalg.norm(du_f)
            norm_u = np.linalg.norm(u[free_dofs]) + 1e-14

            self.history.append({
                'iter': it,
                'residual': norm_r,
                'disp_inc': norm_du,
                'energy': energy
            })

            if norm_r / norm_f < self.tol_force and norm_du / norm_u < self.tol_disp:
                return {
                    'disp': u_new,
                    'converged': True,
                    'iterations': it + 1,
                    'residual_norm': norm_r,
                    'energy': energy,
                    'energy_history': energy_history,
                    'history': self.history
                }

            u = u_new

        # 未收敛
        return {
            'disp': u,
            'converged': False,
            'iterations': self.max_iter,
            'residual_norm': norm_r,
            'energy': energy_history[-1] if energy_history else 0.0,
            'history': self.history
        }

    def _armijo_line_search(self, fem_model, u, du, f_target, f_int_old,
                            c1: float = 1e-4, alpha_max: float = 1.0,
                            max_ls_iter: int = 10) -> float:
        """
        Armijo 线搜索

        要求:
          ||R(u + α du)||² ≤ ||R(u)||² + 2 α c1 <R(u), K_T du>
        由于 K_T du = R(u), 简化为:
          ||R(u + α du)||² ≤ (1 - 2 α c1) ||R(u)||²
        """
        alpha = alpha_max
        norm_r0 = np.linalg.norm(f_target - f_int_old)
        for _ in range(max_ls_iter):
            u_trial = u + alpha * du
            f_int_trial = fem_model.internal_force(u_trial)
            norm_r = np.linalg.norm(f_target - f_int_trial)
            if norm_r <= (1.0 - 2.0 * alpha * c1) * norm_r0:
                return alpha
            alpha *= 0.5
        return alpha


class PseudoTimeSolver:
    """
    伪时间步进动态松弛求解器 (基于 predator_prey_ode 的动态演化思想)

    将静力问题嵌入到动态方程:
      M ü + C u̇ + R(u, λ) = 0
    通过添加人工阻尼使系统收敛到稳态。
    """

    def __init__(self, damping_ratio: float = 0.9, dt: float = 0.01,
                 max_steps: int = 2000):
        self.damping_ratio = damping_ratio
        self.dt = dt
        self.max_steps = max_steps

    def solve(self, fem_model, external_force: np.ndarray, lambda_load: float,
              mass_lumping: np.ndarray = None) -> dict:
        """
        使用显式伪动力松弛求解

        Parameters
        ----------
        fem_model : ShellFEModel
        external_force : (n_dof,) ndarray
        lambda_load : float
        mass_lumping : (n_dof,) ndarray, optional
            集中质量向量

        Returns
        -------
        result : dict
        """
        n_dof = fem_model.n_dof
        f_target = lambda_load * external_force
        u = np.zeros(n_dof)
        v = np.zeros(n_dof)

        # 集中质量 (简化)
        if mass_lumping is None:
            rho = fem_model.mat.rho
            t = fem_model.mesh.geom.t
            area = fem_model.mesh.geom.surface_area()
            m_val = rho * t * area / n_dof
            mass = np.full(n_dof, m_val)
        else:
            mass = np.array(mass_lumping)

        # 边界条件
        bottom, top = fem_model.mesh.get_boundary_nodes()
        fixed_dofs = []
        for nid in bottom:
            fixed_dofs.extend([nid * 3 + 0, nid * 3 + 1, nid * 3 + 2])
        for nid in top:
            fixed_dofs.extend([nid * 3 + 0, nid * 3 + 1])
        if len(bottom) > 0:
            fixed_dofs.append(bottom[0] * 3 + 2)
        fixed_dofs = np.unique(fixed_dofs)
        free_dofs = np.setdiff1d(np.arange(n_dof), fixed_dofs)

        # 守恒量: 伪能量 E = 0.5 vᵀ M v + Π(u)
        energy_history = []

        for step in range(self.max_steps):
            f_int = fem_model.internal_force(u)
            R = f_target - f_int
            # 显式更新
            a = np.zeros(n_dof)
            m_f = mass[free_dofs]
            m_f = np.where(m_f < 1e-14, 1e-14, m_f)
            a[free_dofs] = R[free_dofs] / m_f
            # 阻尼 (质量比例阻尼)
            a[free_dofs] -= 2.0 * self.damping_ratio * np.sqrt(np.max(mass)) * v[free_dofs] / m_f
            # NaN/inf 保护
            if not np.all(np.isfinite(a[free_dofs])):
                break
            v += a * self.dt
            u += v * self.dt

            # 固定边界
            u[fixed_dofs] = 0.0
            v[fixed_dofs] = 0.0

            # 伪能量
            kinetic = 0.5 * np.sum(mass[free_dofs] * v[free_dofs] ** 2)
            potential = 0.5 * np.dot(u, f_int) - np.dot(u, f_target)
            if np.isfinite(kinetic) and np.isfinite(potential):
                energy_history.append(kinetic + potential)
            else:
                break

            # 稳态检测
            if step > 50 and len(energy_history) >= 30:
                recent = np.abs(energy_history[-30:])
                avg_energy = np.mean(recent)
                max_energy = np.max(recent)
                if avg_energy > 1e-14 and max_energy / avg_energy < 1.05:
                    rel_res = np.linalg.norm(R[free_dofs]) / (np.linalg.norm(f_target[free_dofs]) + 1e-14)
                    if rel_res < 1e-3:
                        return {
                            'disp': u,
                            'converged': True,
                            'steps': step + 1,
                            'energy_history': energy_history,
                            'residual_norm': rel_res
                        }

        rel_res = np.linalg.norm(R[free_dofs]) / (np.linalg.norm(f_target[free_dofs]) + 1e-14)
        if not np.isfinite(rel_res):
            rel_res = 999.0
        return {
            'disp': u,
            'converged': False,
            'steps': step + 1,
            'energy_history': energy_history,
            'residual_norm': rel_res
        }
