"""
convergence_utils.py
====================
数值收敛性分析、误差范数估计与统计矩计算工具。

本模块融合项目 113_box_distance（长方体内随机点距离统计）与
814_norm_loo（L∞ 范数估计）的核心算法，应用于超构表面仿真结果
的数值精度评估与不确定性统计。

科学背景：
在电磁仿真中，需要对以下量进行严格的收敛性检验：
1. 网格收敛性：当网格加密时，相位响应是否趋于稳定？
2. 远场精度：多极展开截断后的残差范数
3. Monte-Carlo 统计收敛：制造误差分析的采样充分性

核心公式：

L² 范数误差（场误差）：
    ||E - E_h||_{L²} = sqrt( ∫ |E - E_h|² dΩ )

L∞ 范数误差（最大误差）：
    ||E - E_h||_{L∞} = max_{x∈Ω} |E(x) - E_h(x)|

H¹ 半范数误差（梯度误差）：
    |E - E_h|_{H¹} = sqrt( ∫ |∇E - ∇E_h|² dΩ )

长方体内距离统计（用于随机采样质量评估）：
    设长方体边长 a ≤ b ≤ c，两个随机点间距离的
    均值 μ、方差 σ² 和二阶矩 M₂ 可通过 Monte-Carlo 估计：
        μ ≈ (1/N) Σ_i ||p_i - q_i||
        σ² ≈ (1/(N-1)) Σ_i (||p_i - q_i|| - μ)²
        M₂ ≈ (1/N) Σ_i ||p_i - q_i||²
"""

import numpy as np


class ConvergenceAnalysis:
    """
    数值收敛性与误差分析工具。
    """

    def __init__(self):
        pass

    # ------------------------------------------------------------------
    # 范数计算（源自 814_norm_loo）
    # ------------------------------------------------------------------
    @staticmethod
    def norm_l2(field_diff, weights=None):
        """
        计算 L² 范数：||f||₂ = sqrt(∫ |f|² dΩ)。

        Parameters
        ----------
        field_diff : ndarray
            场差异值（可复数）
        weights : ndarray, optional
            积分权重（如单元面积）
        """
        if weights is None:
            weights = np.ones_like(field_diff)
        val = np.sum(weights * np.abs(field_diff) ** 2)
        return np.sqrt(val)

    @staticmethod
    def norm_linfty(field_diff, sample_points=None):
        """
        计算 L∞ 范数：||f||_∞ = max |f(x)|。
        若提供 sample_points，同时返回最大值位置。

        源自 814_norm_loo 的核心思想：在离散采样点上估计最大范数。
        """
        abs_vals = np.abs(field_diff)
        max_val = np.max(abs_vals)
        max_idx = np.argmax(abs_vals)
        if sample_points is not None:
            return max_val, sample_points[max_idx]
        return max_val

    @staticmethod
    def norm_h1_semi(grad_diff_x, grad_diff_y, weights=None):
        """
        计算 H¹ 半范数：|f|_{H¹} = sqrt(∫ |∇f|² dΩ)。
        """
        if weights is None:
            weights = np.ones_like(grad_diff_x)
        val = np.sum(weights * (np.abs(grad_diff_x) ** 2 + np.abs(grad_diff_y) ** 2))
        return np.sqrt(val)

    # ------------------------------------------------------------------
    # 长方体距离统计（源自 113_box_distance）
    # ------------------------------------------------------------------
    @staticmethod
    def box_distance_stats(n_pairs, a, b, c):
        """
        估计长方体内两随机点间距离的统计矩。

        Parameters
        ----------
        n_pairs : int
            随机点对数
        a, b, c : float
            长方体边长（a ≤ b ≤ c）

        Returns
        -------
        mu : float
            均值
        variance : float
            方差
        moment2 : float
            二阶矩
        """
        p = np.random.rand(n_pairs, 3) * np.array([a, b, c])
        q = np.random.rand(n_pairs, 3) * np.array([a, b, c])
        t = np.linalg.norm(p - q, axis=1)
        mu = np.mean(t)
        if n_pairs > 1:
            variance = np.sum((t - mu) ** 2) / (n_pairs - 1)
        else:
            variance = 0.0
        moment2 = np.mean(t ** 2)
        return mu, variance, moment2

    @staticmethod
    def box_distance_analytical(a, b, c):
        """
        长方体内随机点距离的解析均值近似（Monte-Carlo 验证用）。
        近似公式：μ ≈ 0.6617 * (a + b + c) / 3
        """
        # 这是一个经验近似，用于验证 Monte-Carlo 结果
        return 0.6617 * (a + b + c) / 3.0

    # ------------------------------------------------------------------
    # 收敛阶估计
    # ------------------------------------------------------------------
    @staticmethod
    def estimate_convergence_rate(errors, resolutions):
        """
        通过最小二乘拟合估计收敛阶：
            error ≈ C * h^p
            log(error) = log(C) + p * log(h)

        Parameters
        ----------
        errors : ndarray
            不同分辨率下的误差值
        resolutions : ndarray
            对应的网格尺寸 h

        Returns
        -------
        p : float
            收敛阶
        C : float
            常数因子
        r2 : float
            拟合 R²
        """
        log_e = np.log(errors)
        log_h = np.log(resolutions)
        # 线性拟合
        A = np.vstack([np.ones_like(log_h), log_h]).T
        coeffs, residuals, rank, s = np.linalg.lstsq(A, log_e, rcond=None)
        log_C = coeffs[0]
        p = coeffs[1]
        C = np.exp(log_C)

        # R²
        ss_res = np.sum((log_e - (log_C + p * log_h)) ** 2)
        ss_tot = np.sum((log_e - np.mean(log_e)) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-15 else 1.0
        return p, C, r2

    @staticmethod
    def richardson_extrapolation(f_h, f_h2, f_h4, order=2):
        """
        Richardson 外推：利用不同网格上的解估计精确解。
            f_exact ≈ (2^p f_{h/2} - f_h) / (2^p - 1)
        """
        ratio = 2.0 ** order
        f_extrap = (ratio * f_h2 - f_h) / (ratio - 1.0)
        # 进一步外推
        if f_h4 is not None:
            f_extrap2 = (ratio * f_h4 - f_h2) / (ratio - 1.0)
            f_extrap = (ratio * f_extrap2 - f_extrap) / (ratio - 1.0)
        return f_extrap

    # ------------------------------------------------------------------
    # Monte-Carlo 统计收敛检验
    # ------------------------------------------------------------------
    @staticmethod
    def mc_convergence_test(samples, batch_size=100):
        """
        检验 Monte-Carlo 估计的收敛性。
        计算累积均值及其标准误差随样本量增加的变化。

        Returns
        -------
        cumulative_mean : ndarray
        std_error : ndarray
        """
        N = len(samples)
        n_batches = N // batch_size
        cumulative_mean = []
        std_error = []
        for i in range(1, n_batches + 1):
            batch = samples[:i * batch_size]
            cm = np.mean(batch)
            se = np.std(batch) / np.sqrt(len(batch))
            cumulative_mean.append(cm)
            std_error.append(se)
        return np.array(cumulative_mean), np.array(std_error)

    @staticmethod
    def gci_calculation(fine, medium, coarse, r=2.0, p=2.0):
        """
        计算 Grid Convergence Index (GCI) —— Roache 方法。
        用于网格无关性验证。

            GCI = F_s |ε| / (r^p - 1)
        其中 ε = (f_medium - f_fine) / f_fine
        """
        F_s = 1.25  # 安全因子
        epsilon = (medium - fine) / fine if abs(fine) > 1e-15 else 0.0
        gci = F_s * abs(epsilon) / (r ** p - 1.0)
        return gci

    # ------------------------------------------------------------------
    # 超构表面专用评估
    # ------------------------------------------------------------------
    def evaluate_phase_error(self, phi_exact, phi_numeric, x_grid, y_grid):
        """
        计算相位分布的多范数误差。
        """
        dx = x_grid[1] - x_grid[0]
        dy = y_grid[1] - y_grid[0]
        area = dx * dy

        # 相位差（考虑 2π 缠绕）
        diff = np.angle(np.exp(1.0j * (phi_numeric - phi_exact)))
        dphi_dx, dphi_dy = np.gradient(diff, dx, dy)

        l2_err = self.norm_l2(diff, weights=np.ones_like(diff) * area)
        linf_err = self.norm_linfty(diff)
        h1_err = self.norm_h1_semi(dphi_dx, dphi_dy, weights=np.ones_like(diff) * area)

        return {
            'L2_error': l2_err,
            'Linf_error': linf_err,
            'H1_error': h1_err,
            'relative_L2': l2_err / self.norm_l2(phi_exact, weights=np.ones_like(diff) * area)
        }

    def diffraction_efficiency_analysis(self, target_efficiency,
                                         simulated_efficiency):
        """
        评估衍射效率的数值误差。
        """
        abs_err = abs(simulated_efficiency - target_efficiency)
        rel_err = abs_err / target_efficiency if target_efficiency > 1e-15 else abs_err
        return {'absolute_error': abs_err, 'relative_error': rel_err}


def demo():
    """演示：收敛性分析与统计检验。"""
    ca = ConvergenceAnalysis()

    # 1. 长方体距离统计
    mu, var, m2 = ca.box_distance_stats(50000, 1.0e-6, 2.0e-6, 3.0e-6)
    mu_ana = ca.box_distance_analytical(1.0e-6, 2.0e-6, 3.0e-6)
    print(f"[convergence_utils] 长方体距离统计 (Monte-Carlo):")
    print(f"  μ={mu:.6e}, σ²={var:.6e}, M₂={m2:.6e}")
    print(f"  解析近似 μ≈{mu_ana:.6e}, 偏差={abs(mu-mu_ana)/mu_ana*100:.2f}%")

    # 2. 收敛阶估计
    h_vals = np.array([0.4, 0.2, 0.1, 0.05]) * 1e-6
    # 假设二阶收敛
    errors = 0.1 * h_vals ** 2
    errors *= (1 + 0.05 * np.random.randn(len(errors)))  # 加噪声
    p, C, r2 = ca.estimate_convergence_rate(errors, h_vals)
    print(f"[convergence_utils] 拟合收敛阶 p={p:.3f}, R²={r2:.4f}")

    # 3. GCI 计算
    fine, medium, coarse = 0.951, 0.943, 0.920
    gci = ca.gci_calculation(fine, medium, coarse, r=2.0, p=p)
    print(f"[convergence_utils] GCI(fine-medium)={gci*100:.3f}%")

    # 4. Monte-Carlo 收敛
    samples = np.random.randn(10000)
    cm, se = ca.mc_convergence_test(samples, batch_size=200)
    print(f"[convergence_utils] MC 最终均值={cm[-1]:.4f}±{se[-1]:.4f}")

    return mu, p, gci


if __name__ == "__main__":
    demo()
