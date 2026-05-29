# -*- coding: utf-8 -*-
"""
arc_length_tracker.py
弧长法后屈曲路径跟踪与分岔检测

融合种子项目:
  - 171_chirikov_iteration: 迭代映射与混沌检测思想
  - 199_collatz_recursive: 递归步长细分策略

科学背景:
  壳体后屈曲平衡路径由参数方程 (u(s), λ(s)) 描述，s 为弧长参数。

  控制方程:
    R(u, λ) = F_int(u) - λ F_ext = 0          ... (1)
    g(Δu, Δλ) = ΔuᵀΔu + ψ² Δλ² (F_extᵀF_ext) = Δs²  ... (2)

  其中 ψ 为缩放因子，通常取 ψ² = (uᵀu) / (λ² F_extᵀF_ext)。

  弧长约束的球面形式 (Riks-Wempner):
    g = (u - u₀)ᵀ (u - u₀) + ψ² (λ - λ₀)² - Δs² = 0

  扩展的 Newton-Raphson 迭代 (在 (u, λ) 空间):
    [ K_T   -F_ext ] [ Δu ]   [ -R(u, λ) ]
    [ 2Δuᵀ  2ψ²Δλ ] [ Δλ ] = [ -g(Δu, Δλ) ]

  分岔检测指标:
    det(K_T) 的符号变化 → 极限点/分岔点
    最小特征值 λ_min(K_T) → 刚度退化程度
"""

import numpy as np
from scipy.sparse.linalg import spsolve, eigsh
from scipy.sparse import csr_matrix


class ArcLengthTracker:
    """
    圆柱壳屈曲后屈曲路径弧长法跟踪器
    """

    def __init__(self, initial_arc_length: float = 0.01, min_arc_length: float = 1e-5,
                 max_arc_length: float = 0.5, adaptivity: float = 0.5,
                 psi_scale: float = 1.0, max_recursion_depth: int = 5):
        self.ds = initial_arc_length
        self.ds_min = min_arc_length
        self.ds_max = max_arc_length
        self.adaptivity = adaptivity
        self.psi_scale = psi_scale
        self.max_recursion = max_recursion_depth
        self.path_history = []

    def _compute_psi(self, u: np.ndarray, lambda_val: float,
                     f_ext_norm: float) -> float:
        """
        计算载荷-位移缩放因子 ψ
        单位: [位移/力]

        为防止数值不稳定，psi 设置下限 1/f_norm，
        确保弧长约束中载荷项至少与位移项可比。
        """
        u_norm = np.linalg.norm(u) + 1e-14
        f_norm = f_ext_norm + 1e-14
        if abs(lambda_val) < 1e-10:
            psi = self.psi_scale * u_norm / f_norm
        else:
            psi = self.psi_scale * u_norm / (abs(lambda_val) * f_norm)
        # 数值保护: 确保 psi^2 * dlambda^2 与 ||du||^2 同量级
        psi_min = 1.0 / f_norm
        return max(psi, psi_min)

    def _solve_correction(self, fem_model, u_pred: np.ndarray, lambda_pred: float,
                          f_ext: np.ndarray, u0: np.ndarray, lambda0: float,
                          free_dofs: np.ndarray) -> tuple:
        """
        球面弧长约束下的 Newton 修正步

        必须与以下模块协同:
          - shell_fem_element.py 中的 _b_matrix_membrane (Hole 1):
            切线刚度 K_T = K_lin + K_geo 的物理正确性依赖于 B_m 矩阵。
          - nonlinear_solver.py 中的 solve (Hole 2):
            边界条件处理方式和切线刚度组装逻辑必须一致。

        Returns
        -------
        du_corr, dlambda_corr
        """
        # === HOLE 3 ===
        # 请实现球面弧长约束下的 Newton 修正步。
        #
        # 数学背景:
        #   控制方程: [ K_T   -F_ext ] [ Δu ]   [ -R(u, λ) ]
        #             [ 2Δuᵀ  2ψ²Δλ ] [ Δλ ] = [ -g(Δu, Δλ) ]
        #   其中 g = (u-u₀)ᵀ(u-u₀) + ψ²(λ-λ₀)² - Δs² = 0
        #
        # 步骤:
        #   1. 计算内力 f_int = fem_model.internal_force(u_pred)
        #   2. 计算残差 R = lambda_pred * f_ext - f_int
        #   3. 组装切线刚度 K_T = K_lin + K_geo
        #   4. 降维求解 delta_u_R = K_ff^{-1} R_f, delta_u_F = K_ff^{-1} f_f
        #   5. 计算球面约束残差 g
        #   6. 求解 dlambda_corr 和 du_corr
        #   7. 注意数值保护 (正则化、除零检查)
        raise NotImplementedError("Hole 3: 请实现 _solve_correction")
        # ==============

    def track_path(self, fem_model, external_force: np.ndarray,
                   n_steps: int = 20, lambda_max: float = 2.0) -> dict:
        """
        跟踪后屈曲平衡路径

        Parameters
        ----------
        fem_model : ShellFEModel
        external_force : (n_dof,) ndarray
        n_steps : int
            期望载荷步数
        lambda_max : float
            最大载荷因子 (相对于屈曲载荷)

        Returns
        -------
        result : dict
            包含 path (list of dict), bifurcation_points
        """
        n_dof = fem_model.n_dof
        f_ext = external_force
        f_norm = np.linalg.norm(f_ext) + 1e-14

        # 边界条件: 底部固支, 顶部限制面内位移
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

        if len(free_dofs) == 0:
            return {'path': [], 'bifurcation_points': [], 'n_steps': 0}

        # 初始线性解
        K0 = fem_model.assemble_linear_stiffness()
        K0_ff = K0[free_dofs][:, free_dofs]
        u0 = np.zeros(n_dof)
        try:
            u0_f = spsolve(K0_ff, f_ext[free_dofs])
        except Exception:
            u0_f = spsolve(K0_ff + 1e-8 * csr_matrix(np.eye(len(free_dofs))), f_ext[free_dofs])
        u0[free_dofs] = u0_f
        lambda0 = 0.0

        path = [{
            'lambda': lambda0,
            'disp': u0.copy(),
            'max_disp': 0.0,
            'det_sign': 1.0,
            'min_eig': 1.0
        }]

        u = u0.copy()
        lam = lambda0

        prev_det_sign = 1.0
        prev_min_eig = 1.0
        bifurcation_points = []

        for step in range(n_steps):
            if abs(lam) >= lambda_max:
                break

            # 切线方向: K_T * tangent_f = F_ext
            K_lin = fem_model.assemble_linear_stiffness()
            K_geo = fem_model.assemble_geometric_stiffness(u)
            K_T = K_lin + K_geo
            K_ff = K_T[free_dofs][:, free_dofs]
            try:
                delta_u_F = spsolve(K_ff, f_ext[free_dofs])
            except Exception:
                delta_u_F = spsolve(K_ff + 1e-8 * csr_matrix(np.eye(len(free_dofs))), f_ext[free_dofs])

            psi = self._compute_psi(u[free_dofs], lam, np.linalg.norm(f_ext[free_dofs]) + 1e-14)

            # 归一化切向量: (u_dot, lambda_dot)
            # 动态调整弧长使得目标 dlambda ≈ 0.05
            target_dlambda = 0.05
            tangent_norm_sq = np.dot(delta_u_F, delta_u_F) + psi ** 2
            if tangent_norm_sq < 1e-20:
                tangent_norm_sq = 1e-20
            tangent_norm = np.sqrt(tangent_norm_sq)
            self.ds = target_dlambda * tangent_norm
            self.ds = max(self.ds_min, min(self.ds, self.ds_max))

            # lambda_dot 符号选择 (保持与前一步同向)
            lambda_dot = 1.0 / tangent_norm
            if step > 0 and len(path) >= 2:
                if (path[-1]['lambda'] - path[-2]['lambda']) < 0:
                    lambda_dot = -abs(lambda_dot)
                else:
                    lambda_dot = abs(lambda_dot)

            u_dot = np.zeros(n_dof)
            u_dot[free_dofs] = lambda_dot * delta_u_F

            # 预测步
            du_pred = self.ds * u_dot
            dlambda_pred = self.ds * lambda_dot
            u_pred = u + du_pred
            lam_pred = lam + dlambda_pred

            # 修正步
            converged = False
            for corr in range(15):
                du_corr, dlambda_corr = self._solve_correction(
                    fem_model, u_pred, lam_pred, f_ext, u, lam, free_dofs)
                u_pred += du_corr
                lam_pred += dlambda_corr

                # 收敛检查
                f_int = fem_model.internal_force(u_pred)
                R = lam_pred * f_ext - f_int
                norm_r = np.linalg.norm(R[free_dofs])
                norm_f = np.linalg.norm((lam_pred * f_ext)[free_dofs]) + 1e-14

                # 约束残差
                du_c = u_pred[free_dofs] - u[free_dofs]
                dl_c = lam_pred - lam
                g_res = abs(np.dot(du_c, du_c) + psi ** 2 * dl_c ** 2 - self.ds ** 2)

                if norm_r / norm_f < 1e-5 and g_res < self.ds ** 2 * 1e-3:
                    converged = True
                    break

            # 若未收敛或 lambda 异常，递归减小步长重试
            if not converged or abs(lam_pred) > 10.0 * lambda_max or np.isnan(lam_pred):
                self.ds = max(self.ds * 0.25, self.ds_min)
                if self.ds <= self.ds_min:
                    break
                continue

            u = u_pred
            lam = lam_pred

            # 自适应步长
            n_iter = corr + 1
            if n_iter <= 3:
                self.ds = min(self.ds * 1.5, self.ds_max)
            elif n_iter >= 10:
                self.ds = max(self.ds * 0.5, self.ds_min)

            # 分岔检测
            K_T_corr = fem_model.assemble_linear_stiffness() + fem_model.assemble_geometric_stiffness(u)
            K_T_ff = K_T_corr[free_dofs][:, free_dofs]
            try:
                eigvals = eigsh(K_T_ff, k=1, which='SM', return_eigenvectors=False, tol=1e-2)
                min_eig = float(eigvals[0])
            except Exception:
                min_eig = float(np.linalg.cond(K_T_ff.toarray()))
                min_eig = 1.0 / min_eig if min_eig > 0 else 0.0

            det_sign = np.sign(min_eig) if abs(min_eig) > 1e-10 else 0.0
            if prev_det_sign * det_sign < 0 and step > 0:
                bifurcation_points.append({
                    'step': step,
                    'lambda': float(lam),
                    'min_eig': float(min_eig),
                    'prev_min_eig': float(prev_min_eig)
                })
            prev_det_sign = det_sign
            prev_min_eig = min_eig

            max_w = np.max(np.abs(u[2::3])) if n_dof >= 3 else 0.0
            path.append({
                'lambda': float(lam),
                'disp': u.copy(),
                'max_disp': float(max_w),
                'det_sign': float(det_sign),
                'min_eig': float(min_eig),
                'arc_length': float(self.ds)
            })

        return {
            'path': path,
            'bifurcation_points': bifurcation_points,
            'n_steps': len(path) - 1
        }

    def chirikov_stability_indicator(self, path: list) -> np.ndarray:
        """
        使用 Chirikov 共振重叠判据思想检测后屈曲路径中的混沌区域

        将载荷-位移响应离散映射:
          (u_k, λ_k) -> (u_{k+1}, λ_{k+1})
        计算相邻步的 "扭转数" (winding number):
          ν_k = arctan2(Δλ_{k+1}, Δw_{k+1}) - arctan2(Δλ_k, Δw_k)
        若 ν_k 的方差剧烈增大，预示进入混沌区域。

        Returns
        -------
        indicators : (n-2,) ndarray
            局部稳定性指标 (0=稳定, 1=混沌)
        """
        n = len(path)
        if n < 4:
            return np.array([])
        indicators = np.zeros(n - 2)
        winding = []
        for i in range(1, n):
            dw = path[i]['max_disp'] - path[i - 1]['max_disp']
            dl = path[i]['lambda'] - path[i - 1]['lambda']
            winding.append(np.arctan2(dl, dw + 1e-14))

        for i in range(1, len(winding) - 1):
            nu = winding[i] - winding[i - 1]
            nu_next = winding[i + 1] - winding[i]
            jump = abs(nu_next - nu)
            if jump > np.pi / 4.0:
                indicators[i - 1] = 1.0
        return indicators
