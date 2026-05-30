
import numpy as np
from typing import List, Tuple, Dict





def knapsack_01(weights: np.ndarray, values: np.ndarray, capacity: float) -> Tuple[float, List[int]]:
    n = len(weights)
    if n != len(values):
        raise ValueError("weights and values must have same length")
    if capacity < 0:
        raise ValueError("capacity must be non-negative")
    if n == 0:
        return 0.0, []


    scale = 100.0
    W_int = int(np.ceil(capacity * scale))
    w_int = np.maximum((weights * scale).astype(int), 0)
    v = np.maximum(values, 0.0)


    dp = np.zeros(W_int + 1, dtype=float)

    choice = np.full((n, W_int + 1), -1, dtype=int)

    for i in range(n):
        wi = w_int[i]
        vi = v[i]

        for w in range(W_int, wi - 1, -1):
            if dp[w - wi] + vi > dp[w]:
                dp[w] = dp[w - wi] + vi
                choice[i, w] = 1
            else:
                choice[i, w] = 0


    selected = []
    w = W_int
    for i in range(n - 1, -1, -1):
        if choice[i, w] == 1:
            selected.append(i)
            w -= w_int[i]

    max_val = dp[W_int]
    return max_val, selected[::-1]






def optimize_dose_allocation(total_dose: float,
                              organ_volumes: np.ndarray,
                              organ_sensitivities: np.ndarray,
                              organ_toxicities: np.ndarray,
                              max_toxicity: float) -> Tuple[np.ndarray, float]:
    n_organs = len(organ_volumes)
    if n_organs != len(organ_sensitivities) or n_organs != len(organ_toxicities):
        raise ValueError("Array lengths must match")
    if total_dose <= 0 or max_toxicity < 0:
        raise ValueError("Invalid dose or toxicity limits")



    n_levels = 20
    delta = total_dose / n_levels


    weights = []
    values = []
    organ_idx = []
    for i in range(n_organs):
        for j in range(1, n_levels + 1):
            dose_j = j * delta

            if organ_toxicities[i] * dose_j > max_toxicity / n_organs:
                continue
            weights.append(dose_j)
            values.append(organ_sensitivities[i] * dose_j)
            organ_idx.append(i)

    weights = np.array(weights)
    values = np.array(values)
    max_val, selected = knapsack_01(weights, values, total_dose)


    allocation = np.zeros(n_organs)
    for idx in selected:
        allocation[organ_idx[idx]] += weights[idx]

    total_efficacy = np.sum(organ_sensitivities * allocation)
    return allocation, total_efficacy


def optimize_dosing_schedule(horizon_hours: float,
                              dose_units: np.ndarray,
                              efficacy_values: np.ndarray,
                              toxicity_values: np.ndarray,
                              min_interval: float = 4.0,
                              max_daily_dose: float = 1000.0) -> Tuple[np.ndarray, float]:
    n_slots = int(horizon_hours / min_interval)
    if n_slots < 1:
        raise ValueError("Horizon too short")
    if len(dose_units) != len(efficacy_values) or len(dose_units) != len(toxicity_values):
        raise ValueError("Dose arrays must have same length")


    n_options = len(dose_units)

    dp = np.full(n_slots + 1, -np.inf)
    dp[0] = 0.0
    choice = np.full(n_slots + 1, -1, dtype=int)

    lambda_penalty = 0.5
    for i in range(1, n_slots + 1):

        if dp[i - 1] > dp[i]:
            dp[i] = dp[i - 1]
            choice[i] = -1

        for opt in range(n_options):
            prev = max(0, i - 1)
            if dp[prev] > -np.inf:
                val = dp[prev] + efficacy_values[opt] - lambda_penalty * toxicity_values[opt]
                if val > dp[i]:
                    dp[i] = val
                    choice[i] = opt


    schedule = np.zeros(n_slots, dtype=int)
    i = n_slots
    while i > 0:
        if choice[i] >= 0:
            schedule[i - 1] = 1
            i -= 1
        i -= 1

    return schedule, dp[n_slots]






def optimize_drug_combination(n_drugs: int, budget: float,
                               drug_costs: np.ndarray,
                               drug_efficacies: np.ndarray,
                               synergy_matrix: np.ndarray = None) -> Tuple[List[int], float]:
    if len(drug_costs) != n_drugs or len(drug_efficacies) != n_drugs:
        raise ValueError("Array lengths must match n_drugs")

    values = drug_efficacies.copy()
    if synergy_matrix is not None:
        for i in range(n_drugs):
            for j in range(i + 1, n_drugs):
                if synergy_matrix[i, j] > 0:
                    values[i] += 0.5 * synergy_matrix[i, j]
                    values[j] += 0.5 * synergy_matrix[i, j]

    max_val, selected = knapsack_01(drug_costs, values, budget)
    return selected, max_val






if __name__ == "__main__":
    w = np.array([2.0, 3.0, 4.0, 5.0])
    v = np.array([3.0, 4.0, 5.0, 6.0])
    cap = 5.0
    max_val, sel = knapsack_01(w, v, cap)
    print(f"Knapsack max value: {max_val:.2f}, selected: {sel}")

    alloc, eff = optimize_dose_allocation(
        500.0,
        np.array([1.5, 0.3, 30.0, 10.0, 0.5]),
        np.array([0.8, 0.6, 0.3, 0.2, 1.0]),
        np.array([0.05, 0.08, 0.02, 0.01, 0.10]),
        50.0
    )
    print(f"Dose allocation: {alloc}, total efficacy: {eff:.2f}")

    sched, benefit = optimize_dosing_schedule(
        72.0,
        np.array([100.0, 200.0]),
        np.array([10.0, 18.0]),
        np.array([2.0, 5.0])
    )
    print(f"Schedule length: {len(sched)}, doses given: {sched.sum()}, net benefit: {benefit:.2f}")
