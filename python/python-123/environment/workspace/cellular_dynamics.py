
import numpy as np
from typing import Tuple



STATE_PROLIFERATION = 0
STATE_QUIESCENCE = 1
STATE_APOPTOSIS = 2
STATE_NECROSIS = 3


def cell_transition_matrix(
    p_prolif_to_quies: float = 0.15,
    p_prolif_to_apop: float = 0.05,
    p_quies_to_prolif: float = 0.20,
    p_quies_to_apop: float = 0.10,
    p_apop_to_quies: float = 0.02,
    p_necrosis_recovery: float = 0.0
) -> np.ndarray:
    for p in [p_prolif_to_quies, p_prolif_to_apop, p_quies_to_prolif,
              p_quies_to_apop, p_apop_to_quies, p_necrosis_recovery]:
        if not (0.0 <= p <= 1.0):
            raise ValueError(f"cell_transition_matrix: 概率 {p} 超出 [0,1]")

    M = np.zeros((4, 4))


    M[0, 0] = 1.0 - p_prolif_to_quies - p_prolif_to_apop
    M[0, 1] = p_prolif_to_quies
    M[0, 2] = p_prolif_to_apop
    M[0, 3] = 0.0


    M[1, 0] = p_quies_to_prolif
    M[1, 1] = 1.0 - p_quies_to_prolif - p_quies_to_apop
    M[1, 2] = p_quies_to_apop
    M[1, 3] = 0.0


    M[2, 0] = 0.0
    M[2, 1] = p_apop_to_quies
    M[2, 2] = 1.0 - p_apop_to_quies
    M[2, 3] = 0.0


    M[3, 0] = 0.0
    M[3, 1] = 0.0
    M[3, 2] = 0.0
    M[3, 3] = 1.0


    row_sums = M.sum(axis=1)
    for i in range(4):
        if abs(row_sums[i] - 1.0) > 1e-12:
            if row_sums[i] > 1e-15:
                M[i, :] /= row_sums[i]
            else:
                M[i, i] = 1.0

    return M


def evolve_cell_population_markov(
    initial_counts: np.ndarray, trans_matrix: np.ndarray, steps: int
) -> np.ndarray:
    initial_counts = np.asarray(initial_counts, dtype=float)
    if initial_counts.shape[0] != 4:
        raise ValueError("evolve_cell_population_markov: initial_counts 长度必须为 4")
    if trans_matrix.shape != (4, 4):
        raise ValueError("evolve_cell_population_markov: 转移矩阵必须是 4x4")
    if steps < 0:
        raise ValueError("evolve_cell_population_markov: steps >= 0")

    history = np.zeros((steps + 1, 4))
    history[0, :] = initial_counts
    state = initial_counts.copy()

    for t in range(1, steps + 1):
        state = trans_matrix.T @ state

        state = np.where(state < 0, 0.0, state)
        history[t, :] = state

    return history


def ca_contact_inhibition_update(
    cell_grid: np.ndarray, nutrient_grid: np.ndarray,
    threshold_nutrient: float = 0.1,
    inhibition_threshold: int = 5
) -> np.ndarray:
    H, W = cell_grid.shape
    if nutrient_grid.shape != (H, W):
        raise ValueError("ca_contact_inhibition_update: 网格尺寸不匹配")

    new_grid = cell_grid.copy()

    for i in range(H):
        for j in range(W):

            neighbor_count = 0
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    if di == 0 and dj == 0:
                        continue
                    ni, nj = i + di, j + dj
                    if 0 <= ni < H and 0 <= nj < W:
                        if cell_grid[ni, nj] in (STATE_PROLIFERATION, STATE_QUIESCENCE):
                            neighbor_count += 1

            current = cell_grid[i, j]


            if current == STATE_NECROSIS:
                new_grid[i, j] = STATE_NECROSIS
                continue


            if nutrient_grid[i, j] < threshold_nutrient * 0.3:
                if current == STATE_APOPTOSIS:
                    new_grid[i, j] = STATE_NECROSIS
                else:
                    new_grid[i, j] = STATE_APOPTOSIS
                continue


            if nutrient_grid[i, j] < threshold_nutrient:
                if current == STATE_PROLIFERATION:
                    new_grid[i, j] = STATE_APOPTOSIS
                else:
                    new_grid[i, j] = current
                continue


            if neighbor_count >= inhibition_threshold:
                if current == STATE_PROLIFERATION:
                    new_grid[i, j] = STATE_QUIESCENCE
                else:
                    new_grid[i, j] = current
                continue


            if current == STATE_QUIESCENCE and nutrient_grid[i, j] > threshold_nutrient * 2.0:
                new_grid[i, j] = STATE_PROLIFERATION
            elif current == STATE_APOPTOSIS and nutrient_grid[i, j] > threshold_nutrient * 1.5:

                new_grid[i, j] = STATE_QUIESCENCE
            else:
                new_grid[i, j] = current

    return new_grid


def ca_proliferation_step(
    cell_grid: np.ndarray, empty_probability: float = 0.1
) -> np.ndarray:
    H, W = cell_grid.shape
    new_grid = cell_grid.copy()
    rng = np.random.default_rng(seed=42)

    for i in range(H):
        for j in range(W):
            if cell_grid[i, j] == STATE_PROLIFERATION:
                if rng.random() < empty_probability:

                    candidates = []
                    for di in (-1, 0, 1):
                        for dj in (-1, 0, 1):
                            if di == 0 and dj == 0:
                                continue
                            ni, nj = i + di, j + dj
                            if 0 <= ni < H and 0 <= nj < W:
                                if cell_grid[ni, nj] in (STATE_APOPTOSIS, STATE_NECROSIS):
                                    candidates.append((ni, nj))
                    if candidates:
                        ni, nj = candidates[rng.integers(len(candidates))]
                        new_grid[ni, nj] = STATE_PROLIFERATION
    return new_grid


def compute_tumor_cellularity(cell_grid: np.ndarray) -> Tuple[float, float, float, float]:
    total = cell_grid.size
    if total == 0:
        return 0.0, 0.0, 0.0, 0.0

    frac_P = np.sum(cell_grid == STATE_PROLIFERATION) / total
    frac_Q = np.sum(cell_grid == STATE_QUIESCENCE) / total
    frac_A = np.sum(cell_grid == STATE_APOPTOSIS) / total
    frac_N = np.sum(cell_grid == STATE_NECROSIS) / total
    return float(frac_P), float(frac_Q), float(frac_A), float(frac_N)


def compute_doubling_time(population_history: np.ndarray) -> float:
    if population_history.shape[0] < 2:
        return np.inf

    total = population_history.sum(axis=1)
    proliferative = population_history[:, STATE_PROLIFERATION]


    valid = total > 1e-12
    if np.sum(valid) < 2:
        return np.inf

    t_vals = np.arange(population_history.shape[0])[valid]
    y_vals = np.log(total[valid] + 1e-12)


    n = t_vals.shape[0]
    sum_t = np.sum(t_vals)
    sum_y = np.sum(y_vals)
    sum_tt = np.sum(t_vals ** 2)
    sum_ty = np.sum(t_vals * y_vals)
    denom = n * sum_tt - sum_t ** 2
    if abs(denom) < 1e-15:
        return np.inf

    lam = (n * sum_ty - sum_t * sum_y) / denom
    if lam <= 1e-15:
        return np.inf

    T_d = np.log(2.0) / lam
    return float(T_d)
