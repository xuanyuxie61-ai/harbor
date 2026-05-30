
import numpy as np
from utils import validate_positive


class ProcessOptimizer:

    def __init__(self, amine_type="MEA"):
        self.amine_type = amine_type

    def objective_cost_avoided(self, T, P, c_amine, L_G_ratio, capture_rate_target=0.9):
        validate_positive(T, "Temperature")
        validate_positive(P, "Pressure")
        validate_positive(c_amine, "Amine concentration")
        validate_positive(L_G_ratio, "L/G ratio")


        eta = self._absorption_efficiency(T, P, c_amine, L_G_ratio)
        eta = np.clip(eta, 0.1, 0.999)



        reboiler_duty = self._reboiler_duty(T, c_amine, eta)
        steam_cost = reboiler_duty * 15.0 / 1e6


        power_cost = L_G_ratio * 0.5


        degradation_rate = np.exp(-5000.0 / T) * 0.01
        makeup_cost = c_amine * degradation_rate * 2.0

        operating_cost = steam_cost + power_cost + makeup_cost


        capital_cost = 50.0 * (P / 1e5) ** 0.3 * (c_amine / 5000.0) ** 0.5

        total_cost = operating_cost + capital_cost


        co2_captured = eta * 100.0

        cost_avoided = total_cost / co2_captured * 1000.0
        return cost_avoided

    def _absorption_efficiency(self, T, P, c_amine, L_G_ratio):



        T_ref = 313.15
        P_ref = 1.0e5
        term_T = np.exp(-0.02 * (T - T_ref))
        term_P = (P / P_ref) ** 0.3
        term_c = np.tanh(c_amine / 3000.0)
        term_LG = np.tanh(L_G_ratio / 3.0)
        eta = term_T * term_P * term_c * term_LG
        return np.clip(eta, 0.0, 1.0)

    def _reboiler_duty(self, T, c_amine, eta):
        base_duty = 3500.0
        T_penalty = max(0, (T - 313.15) * 20.0)
        c_penalty = max(0, (c_amine - 5000.0) * 0.1)
        return base_duty + T_penalty + c_penalty

    def gradient_optimization(self, x0, learning_rate=0.01, max_iter=100, tol=1e-6):
        x = np.array(x0, dtype=float)
        bounds = np.array([
            [298.15, 353.15],
            [1.0e5, 3.0e5],
            [1000.0, 8000.0],
            [0.5, 5.0]
        ])

        history = []
        h = 1e-5

        for iteration in range(max_iter):
            f_val = self.objective_cost_avoided(*x)
            history.append((x.copy(), f_val))

            gradient = np.zeros(4)
            for i in range(4):
                x_plus = x.copy()
                x_plus[i] += h
                f_plus = self.objective_cost_avoided(*x_plus)
                gradient[i] = (f_plus - f_val) / h


            alpha = learning_rate
            for _ in range(10):
                x_new = x - alpha * gradient
                x_new = np.clip(x_new, bounds[:, 0], bounds[:, 1])
                f_new = self.objective_cost_avoided(*x_new)
                if f_new < f_val:
                    break
                alpha *= 0.5

            x = np.clip(x - alpha * gradient, bounds[:, 0], bounds[:, 1])

            if np.linalg.norm(gradient) < tol:
                print(f"  Gradient optimization converged at iteration {iteration + 1}")
                break

        return x, history

    def grid_search_2d(self, var1_idx, var2_idx, fixed_values, n_grid=20):
        bounds = [
            np.linspace(298.15, 353.15, n_grid),
            np.linspace(1.0e5, 3.0e5, n_grid),
            np.linspace(1000.0, 8000.0, n_grid),
            np.linspace(0.5, 5.0, n_grid)
        ]

        grid1 = bounds[var1_idx]
        grid2 = bounds[var2_idx]
        cost_surface = np.zeros((n_grid, n_grid))

        best_cost = float('inf')
        best_point = None

        for i, v1 in enumerate(grid1):
            for j, v2 in enumerate(grid2):
                x = np.array(fixed_values)
                x[var1_idx] = v1
                x[var2_idx] = v2
                cost = self.objective_cost_avoided(*x)
                cost_surface[i, j] = cost
                if cost < best_cost:
                    best_cost = cost
                    best_point = x.copy()

        return cost_surface, grid1, grid2, best_point, best_cost


def optimize_additive_package():
    additives = [
        "corrosion_inhibitor_A", "corrosion_inhibitor_B",
        "activator_piperazine", "activator_AMP",
        "antioxidant_sulphite", "antioxidant_hydroquinone",
        "foam_suppressor", "oxygen_scavenger"
    ]
    costs = [5.0, 3.5, 12.0, 8.0, 2.0, 4.0, 1.5, 3.0]
    benefits = [0.15, 0.10, 0.35, 0.25, 0.08, 0.12, 0.05, 0.10]
    budget = 20.0

    from degradation_pathways import knapsack_additive_selection
    return knapsack_additive_selection(additives, costs, benefits, budget)


class SensitivityAnalysis:

    def __init__(self, optimizer):
        self.opt = optimizer

    def local_sensitivity(self, x_base, delta_frac=0.05):
        f_base = self.opt.objective_cost_avoided(*x_base)
        sensitivities = {}
        names = ["Temperature", "Pressure", "Amine_conc", "L_G_ratio"]

        for i, name in enumerate(names):
            dx = x_base[i] * delta_frac
            x_pert = x_base.copy()
            x_pert[i] += dx
            f_pert = self.opt.objective_cost_avoided(*x_pert)
            df = f_pert - f_base
            S = (df / dx) * (x_base[i] / f_base) if f_base != 0 else 0.0
            sensitivities[name] = S

        return sensitivities

    def monte_carlo_uncertainty(self, x_mean, x_std, n_samples=1000):
        costs = []
        for _ in range(n_samples):
            x_sample = np.array([
                np.random.normal(x_mean[i], x_std[i]) for i in range(4)
            ])
            x_sample = np.clip(x_sample,
                [298.15, 1.0e5, 1000.0, 0.5],
                [353.15, 3.0e5, 8000.0, 5.0])
            costs.append(self.opt.objective_cost_avoided(*x_sample))

        return {
            "mean_cost": np.mean(costs),
            "std_cost": np.std(costs),
            "p5": np.percentile(costs, 5),
            "p95": np.percentile(costs, 95)
        }
