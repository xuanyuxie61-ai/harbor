
import numpy as np
from typing import Tuple, List, Optional


class ENSOState:
    STRONG_NINO = 2
    WEAK_NINO = 1
    NEUTRAL = 0
    WEAK_NINA = -1
    STRONG_NINA = -2


def classify_enso_state(nino34: float) -> int:
    if nino34 > 1.5:
        return ENSOState.STRONG_NINO
    elif nino34 > 0.5:
        return ENSOState.WEAK_NINO
    elif nino34 > -0.5:
        return ENSOState.NEUTRAL
    elif nino34 > -1.5:
        return ENSOState.WEAK_NINA
    else:
        return ENSOState.STRONG_NINA


def state_name(state: int) -> str:
    names = {
        ENSOState.STRONG_NINO: "Strong El Nino",
        ENSOState.WEAK_NINO: "Weak El Nino",
        ENSOState.NEUTRAL: "Neutral",
        ENSOState.WEAK_NINA: "Weak La Nina",
        ENSOState.STRONG_NINA: "Strong La Nina",
    }
    return names.get(state, "Unknown")


def transition_probability(current_state: int,
                           month: int,
                           transition_matrix: Optional[np.ndarray] = None) -> np.ndarray:
    if transition_matrix is None:

        base_matrix = np.array([

            [0.50, 0.30, 0.15, 0.04, 0.01],
            [0.20, 0.45, 0.25, 0.08, 0.02],
            [0.08, 0.18, 0.48, 0.18, 0.08],
            [0.02, 0.08, 0.25, 0.45, 0.20],
            [0.01, 0.04, 0.15, 0.30, 0.50],
        ])


        if month in [11, 12, 1]:
            season_factor = 1.3
        elif month in [5, 6, 7]:
            season_factor = 0.8
        else:
            season_factor = 1.0


        for i in range(5):
            base_matrix[i, i] *= season_factor

            base_matrix[i] /= np.sum(base_matrix[i])

        transition_matrix = base_matrix

    state_idx = current_state + 2
    state_idx = max(0, min(4, state_idx))
    return transition_matrix[state_idx]


def monte_carlo_enso_forecast(nino34_current: float,
                              month_current: int,
                              n_ensemble: int = 1000,
                              n_months: int = 12,
                              noise_std: float = 0.3) -> dict:
    if n_ensemble < 1:
        raise ValueError("n_ensemble must be positive")


    rho = 0.85

    forecasts = np.zeros((n_ensemble, n_months))
    states = np.zeros((n_ensemble, n_months), dtype=int)

    for e in range(n_ensemble):
        nino = nino34_current
        month = month_current
        for m in range(n_months):
            noise = np.random.normal(0.0, noise_std)
            nino = rho * nino + noise

            nino = np.clip(nino, -4.0, 4.0)
            forecasts[e, m] = nino
            states[e, m] = classify_enso_state(nino)
            month = month % 12 + 1


    mean_trajectory = np.mean(forecasts, axis=0)
    std_trajectory = np.std(forecasts, axis=0)


    state_probs = np.zeros((5, n_months))
    state_labels = [ENSOState.STRONG_NINA, ENSOState.WEAK_NINA,
                    ENSOState.NEUTRAL, ENSOState.WEAK_NINO,
                    ENSOState.STRONG_NINO]
    for i, s in enumerate(state_labels):
        state_probs[i] = np.mean(states == s, axis=0)

    return {
        "mean_trajectory": mean_trajectory,
        "std_trajectory": std_trajectory,
        "ensemble": forecasts,
        "state_probabilities": state_probs,
        "state_labels": [state_name(s) for s in state_labels],
    }


def forecast_skill(forecasts: np.ndarray, observations: np.ndarray) -> dict:
    if forecasts.shape != observations.shape:
        raise ValueError("Shape mismatch")

    diff = forecasts - observations
    rmse = np.sqrt(np.mean(diff ** 2))
    mae = np.mean(np.abs(diff))


    f_mean, o_mean = np.mean(forecasts), np.mean(observations)
    numerator = np.sum((forecasts - f_mean) * (observations - o_mean))
    denom = np.sqrt(np.sum((forecasts - f_mean) ** 2) * np.sum((observations - o_mean) ** 2))
    correlation = numerator / denom if denom > 1e-14 else 0.0


    persistence = observations[:-1]
    persist_obs = observations[1:]
    persist_rmse = np.sqrt(np.mean((persistence - persist_obs) ** 2))
    skill_score = 1.0 - rmse / persist_rmse if persist_rmse > 1e-14 else 0.0

    return {
        "rmse": float(rmse),
        "mae": float(mae),
        "correlation": float(correlation),
        "skill_score": float(skill_score),
    }


def probabilistic_event_forecast(nino34_current: float,
                                 month_current: int,
                                 target_state: int,
                                 lead_months: int = 6,
                                 n_trials: int = 10000) -> float:
    rho = 0.85
    noise_std = 0.3
    count = 0

    for _ in range(n_trials):
        nino = nino34_current
        month = month_current
        for _ in range(lead_months):
            nino = rho * nino + np.random.normal(0.0, noise_std)
            nino = np.clip(nino, -4.0, 4.0)
            month = month % 12 + 1
        if classify_enso_state(nino) == target_state:
            count += 1

    return count / n_trials
