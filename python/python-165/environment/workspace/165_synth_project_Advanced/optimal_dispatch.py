
import numpy as np
from typing import List, Tuple, Optional
from utils import polynomial_multiply


class EconomicDispatch:

    def __init__(self, a: np.ndarray, b: np.ndarray, c: np.ndarray,
                 p_min: np.ndarray, p_max: np.ndarray):
        self.a = np.array(a, dtype=np.float64)
        self.b = np.array(b, dtype=np.float64)
        self.c = np.array(c, dtype=np.float64)
        self.p_min = np.array(p_min, dtype=np.float64)
        self.p_max = np.array(p_max, dtype=np.float64)
        self.n_gen = len(a)

    def incremental_cost(self, p: np.ndarray) -> np.ndarray:
        return 2.0 * self.a * p + self.b

    def solve_lambda(self, P_demand: float, lambda_bounds: Tuple[float, float] = (0.0, 200.0),
                     tol: float = 1e-6, max_iter: int = 100) -> dict:
        if P_demand < 0:
            raise ValueError("P_demand must be non-negative")
        lam_lo, lam_hi = lambda_bounds

        for _ in range(max_iter):
            lam = (lam_lo + lam_hi) * 0.5
            p = (lam - self.b) / (2.0 * self.a)
            p = np.clip(p, self.p_min, self.p_max)
            total = float(np.sum(p))
            if abs(total - P_demand) < tol:
                break
            if total > P_demand:
                lam_hi = lam
            else:
                lam_lo = lam

        cost = np.sum(self.a * p**2 + self.b * p + self.c)
        return {
            "pg": p,
            "lambda": lam,
            "total_cost": float(cost),
            "total_generation": float(total)
        }


class UnitCommitmentDP:

    def __init__(self, n_gen: int, T: int,
                 startup_cost: np.ndarray, shutdown_cost: np.ndarray,
                 min_up: np.ndarray, min_down: np.ndarray):
        self.n_gen = n_gen
        self.T = T
        self.startup_cost = np.array(startup_cost, dtype=np.float64)
        self.shutdown_cost = np.array(shutdown_cost, dtype=np.float64)
        self.min_up = np.array(min_up, dtype=np.int32)
        self.min_down = np.array(min_down, dtype=np.int32)

    def solve_single_unit_dp(self, gen_idx: int,
                             ed_cost_on: np.ndarray,
                             ed_cost_off: float = 0.0) -> dict:
        T = self.T
        mu = int(self.min_up[gen_idx])
        md = int(self.min_down[gen_idx])
        INF = 1e18



        n_states = mu + md


        def idx_map(u, tau):
            if u == 0:
                return tau - 1
            return md + tau - 1


        V_prev = np.full(n_states, INF, dtype=np.float64)
        V_prev[idx_map(1, 1)] = ed_cost_on[0] if len(ed_cost_on) > 0 else 0.0
        V_prev[idx_map(0, 1)] = ed_cost_off

        policy = []

        for t in range(1, T):
            V_curr = np.full(n_states, INF, dtype=np.float64)
            best_prev = np.full(n_states, -1, dtype=np.int32)
            for u in [0, 1]:
                max_tau = mu if u == 1 else md
                for tau in range(1, max_tau + 1):
                    idx = idx_map(u, tau)
                    best_cost = INF
                    best_state = -1
                    for u_prev in [0, 1]:
                        max_tau_prev = mu if u_prev == 1 else md
                        for tau_prev in range(1, max_tau_prev + 1):
                            idx_prev = idx_map(u_prev, tau_prev)
                            if V_prev[idx_prev] >= INF:
                                continue

                            if u_prev == 1 and u == 1 and tau != tau_prev + 1:
                                continue
                            if u_prev == 0 and u == 0 and tau != tau_prev + 1:
                                continue
                            if u_prev == 1 and u == 0 and tau_prev < mu:
                                continue
                            if u_prev == 0 and u == 1 and tau_prev < md:
                                continue
                            if u_prev == 1 and u == 0 and tau != 1:
                                continue
                            if u_prev == 0 and u == 1 and tau != 1:
                                continue

                            switch_cost = 0.0
                            if u_prev == 0 and u == 1:
                                switch_cost = self.startup_cost[gen_idx]
                            if u_prev == 1 and u == 0:
                                switch_cost = self.shutdown_cost[gen_idx]

                            stage_cost = ed_cost_on[t] if u == 1 else ed_cost_off
                            total = V_prev[idx_prev] + stage_cost + switch_cost
                            if total < best_cost:
                                best_cost = total
                                best_state = idx_prev
                    V_curr[idx] = best_cost
                    best_prev[idx] = best_state
            V_prev = V_curr.copy()
            policy.append(best_prev)


        final_idx = int(np.argmin(V_prev))
        path = [final_idx]
        for t in range(T - 2, -1, -1):
            final_idx = int(policy[t][final_idx])
            if final_idx < 0:
                break
            path.append(final_idx)
        path.reverse()

        schedule = []
        for idx in path:
            if idx < md:
                schedule.append(0)
            else:
                schedule.append(1)
        return {
            "schedule": np.array(schedule, dtype=np.int32),
            "total_cost": float(np.min(V_prev))
        }

    def solve_aggregated_dp(self, demand_series: np.ndarray,
                            ed_solver: EconomicDispatch) -> dict:

        n_levels = 5
        max_demand = int(np.ceil(np.max(demand_series)))

        agg = np.array([1.0])
        gen_polys = []
        for i in range(ed_solver.n_gen):
            p_levels = np.linspace(ed_solver.p_min[i], ed_solver.p_max[i], n_levels)
            poly = np.zeros(int(np.ceil(ed_solver.p_max[i])) + 1)
            for pl in p_levels:
                idx = int(round(pl))
                if idx < len(poly):
                    poly[idx] += 1.0
            gen_polys.append(poly)
            agg = polynomial_multiply(agg, poly)


        feasible = []
        for d in demand_series:
            idx = int(round(d))
            if idx < len(agg) and agg[idx] > 0:
                feasible.append(True)
            else:
                feasible.append(False)

        return {
            "capacity_distribution": agg,
            "feasible_per_period": np.array(feasible),
            "all_feasible": all(feasible)
        }
