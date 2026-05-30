
import numpy as np
from utils_numerical import safe_divide


def cavitation_probability_local(mean_p: float, p_vapor: float, std_p: float) -> float:
    if std_p < 1e-14:
        return 1.0 if mean_p < p_vapor else 0.0

    z = (p_vapor - mean_p) / (np.sqrt(2.0) * std_p)

    z = np.clip(z, -5.0, 5.0)


    prob = 0.5 * (1.0 + erf_approx(z))
    return float(np.clip(prob, 0.0, 1.0))


def erf_approx(x: float) -> float:
    p = 0.3275911
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429

    sign_x = np.sign(x)
    x_abs = abs(x)

    t = 1.0 / (1.0 + p * x_abs)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * np.exp(-x_abs ** 2)

    return sign_x * y


def joint_cavitation_probability(probabilities: np.ndarray, independence: bool = True) -> float:
    probs = np.clip(np.asarray(probabilities), 0.0, 1.0)

    if len(probs) == 0:
        return 0.0

    if independence:
        p_no_cav = np.prod(1.0 - probs)
        p_union = 1.0 - p_no_cav
    else:

        p_union = float(np.max(probs))

    return float(np.clip(p_union, 0.0, 1.0))


def cavitation_inception_criterion(Re: float, sigma: float, roughness: float = 1e-5) -> dict:

    C_crit = 0.5
    sigma_critical = C_crit * (Re * roughness) ** (-0.2)

    margin = sigma - sigma_critical
    inception_risk = 1.0 / (1.0 + np.exp(5.0 * margin))

    return {
        'cavitation_inception': sigma < sigma_critical,
        'sigma_critical': float(sigma_critical),
        'sigma_actual': float(sigma),
        'safety_margin': float(margin),
        'inception_risk': float(inception_risk)
    }


def analyze_pressure_field_for_cavitation(p_field: np.ndarray, p_vapor: float,
                                          u_field: np.ndarray = None, rho: float = 1.0) -> dict:
    mean_p = np.mean(p_field)
    min_p = np.min(p_field)
    std_p = np.std(p_field)


    if u_field is not None:
        u_max = np.max(np.abs(u_field))
        sigma_global = (mean_p - p_vapor) / (0.5 * rho * u_max ** 2 + 1e-14)
    else:
        sigma_global = (mean_p - p_vapor) / (0.5 * rho + 1e-14)


    prob_field = np.zeros_like(p_field)
    for j in range(p_field.shape[0]):
        for i in range(p_field.shape[1]):
            prob_field[j, i] = cavitation_probability_local(p_field[j, i], p_vapor, std_p)


    high_risk_mask = prob_field > 0.1
    high_risk_fraction = np.sum(high_risk_mask) / p_field.size


    high_risk_probs = prob_field[high_risk_mask]
    if len(high_risk_probs) > 0:
        joint_prob = joint_cavitation_probability(high_risk_probs[:100])
    else:
        joint_prob = 0.0

    return {
        'mean_pressure': float(mean_p),
        'min_pressure': float(min_p),
        'pressure_std': float(std_p),
        'cavitation_number': float(sigma_global),
        'probability_field': prob_field,
        'high_risk_fraction': float(high_risk_fraction),
        'joint_cavitation_probability': float(joint_prob),
        'max_local_probability': float(np.max(prob_field))
    }


def compute_nucleation_rate(p: float, p_vapor: float, T: float, surface_tension: float = 0.072) -> float:
    k_b = 1.380649e-23
    delta_p = max(p_vapor - p, 1e-10)


    r_star = 2.0 * surface_tension / delta_p


    delta_g_star = (16.0 * np.pi * surface_tension ** 3) / (3.0 * delta_p ** 2)


    j0 = 1e33


    J = j0 * np.exp(-delta_g_star / max(T, 1e-10))

    return float(J)
