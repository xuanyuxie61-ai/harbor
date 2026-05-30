
import numpy as np


class ConvergenceAnalysis:

    def __init__(self):
        pass




    @staticmethod
    def norm_l2(field_diff, weights=None):
        if weights is None:
            weights = np.ones_like(field_diff)
        val = np.sum(weights * np.abs(field_diff) ** 2)
        return np.sqrt(val)

    @staticmethod
    def norm_linfty(field_diff, sample_points=None):
        abs_vals = np.abs(field_diff)
        max_val = np.max(abs_vals)
        max_idx = np.argmax(abs_vals)
        if sample_points is not None:
            return max_val, sample_points[max_idx]
        return max_val

    @staticmethod
    def norm_h1_semi(grad_diff_x, grad_diff_y, weights=None):
        if weights is None:
            weights = np.ones_like(grad_diff_x)
        val = np.sum(weights * (np.abs(grad_diff_x) ** 2 + np.abs(grad_diff_y) ** 2))
        return np.sqrt(val)




    @staticmethod
    def box_distance_stats(n_pairs, a, b, c):
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

        return 0.6617 * (a + b + c) / 3.0




    @staticmethod
    def estimate_convergence_rate(errors, resolutions):
        log_e = np.log(errors)
        log_h = np.log(resolutions)

        A = np.vstack([np.ones_like(log_h), log_h]).T
        coeffs, residuals, rank, s = np.linalg.lstsq(A, log_e, rcond=None)
        log_C = coeffs[0]
        p = coeffs[1]
        C = np.exp(log_C)


        ss_res = np.sum((log_e - (log_C + p * log_h)) ** 2)
        ss_tot = np.sum((log_e - np.mean(log_e)) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-15 else 1.0
        return p, C, r2

    @staticmethod
    def richardson_extrapolation(f_h, f_h2, f_h4, order=2):
        ratio = 2.0 ** order
        f_extrap = (ratio * f_h2 - f_h) / (ratio - 1.0)

        if f_h4 is not None:
            f_extrap2 = (ratio * f_h4 - f_h2) / (ratio - 1.0)
            f_extrap = (ratio * f_extrap2 - f_extrap) / (ratio - 1.0)
        return f_extrap




    @staticmethod
    def mc_convergence_test(samples, batch_size=100):
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
        F_s = 1.25
        epsilon = (medium - fine) / fine if abs(fine) > 1e-15 else 0.0
        gci = F_s * abs(epsilon) / (r ** p - 1.0)
        return gci




    def evaluate_phase_error(self, phi_exact, phi_numeric, x_grid, y_grid):
        dx = x_grid[1] - x_grid[0]
        dy = y_grid[1] - y_grid[0]
        area = dx * dy


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
        abs_err = abs(simulated_efficiency - target_efficiency)
        rel_err = abs_err / target_efficiency if target_efficiency > 1e-15 else abs_err
        return {'absolute_error': abs_err, 'relative_error': rel_err}


def demo():
    ca = ConvergenceAnalysis()


    mu, var, m2 = ca.box_distance_stats(50000, 1.0e-6, 2.0e-6, 3.0e-6)
    mu_ana = ca.box_distance_analytical(1.0e-6, 2.0e-6, 3.0e-6)
    print(f"[convergence_utils] 长方体距离统计 (Monte-Carlo):")
    print(f"  μ={mu:.6e}, σ²={var:.6e}, M₂={m2:.6e}")
    print(f"  解析近似 μ≈{mu_ana:.6e}, 偏差={abs(mu-mu_ana)/mu_ana*100:.2f}%")


    h_vals = np.array([0.4, 0.2, 0.1, 0.05]) * 1e-6

    errors = 0.1 * h_vals ** 2
    errors *= (1 + 0.05 * np.random.randn(len(errors)))
    p, C, r2 = ca.estimate_convergence_rate(errors, h_vals)
    print(f"[convergence_utils] 拟合收敛阶 p={p:.3f}, R²={r2:.4f}")


    fine, medium, coarse = 0.951, 0.943, 0.920
    gci = ca.gci_calculation(fine, medium, coarse, r=2.0, p=p)
    print(f"[convergence_utils] GCI(fine-medium)={gci*100:.3f}%")


    samples = np.random.randn(10000)
    cm, se = ca.mc_convergence_test(samples, batch_size=200)
    print(f"[convergence_utils] MC 最终均值={cm[-1]:.4f}±{se[-1]:.4f}")

    return mu, p, gci


if __name__ == "__main__":
    demo()
