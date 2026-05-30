
import numpy as np
from typing import List, Tuple, Dict





def luhn_checksum(digits: List[int]) -> int:
    if not digits:
        return 0
    total = 0
    reverse = digits[::-1]
    for i, d in enumerate(reverse):
        if i % 2 == 1:
            d *= 2
            if d >= 10:
                d -= 9
        total += d
    return (10 - total % 10) % 10


def luhn_is_valid(digits: List[int]) -> bool:
    if len(digits) < 2:
        return False
    check = luhn_checksum(digits[:-1])
    return check == digits[-1]


def luhn_check_digit_calculate(digits: List[int]) -> int:
    return luhn_checksum(digits)


def validate_patient_id(patient_id_str: str) -> bool:
    digits = [int(c) for c in patient_id_str if c.isdigit()]
    return luhn_is_valid(digits)






def concentration_binning(concentrations: np.ndarray,
                           bin_edges: np.ndarray = None,
                           n_bins: int = 10) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    C = np.asarray(concentrations, dtype=float)
    if len(C) == 0:
        raise ValueError("concentrations array is empty")
    if bin_edges is None:
        cmin, cmax = np.min(C), np.max(C)
        if cmin == cmax:
            cmax = cmin + 1.0
        bin_edges = np.linspace(cmin, cmax, n_bins + 1)
    counts, edges = np.histogram(C, bins=bin_edges)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return counts, edges, centers


def summary_statistics(data: np.ndarray) -> Dict[str, float]:
    x = np.asarray(data, dtype=float)
    if len(x) == 0:
        raise ValueError("data is empty")
    q25, q50, q75 = np.percentile(x, [25.0, 50.0, 75.0])
    iqr = q75 - q25
    return {
        "n": float(len(x)),
        "mean": float(np.mean(x)),
        "std": float(np.std(x, ddof=1)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "median": float(q50),
        "q25": float(q25),
        "q75": float(q75),
        "iqr": float(iqr),
        "cv": float(np.std(x, ddof=1) / max(abs(np.mean(x)), 1e-20)),
    }






def safe_divide(a: float, b: float, default: float = 0.0) -> float:
    if abs(b) < 1e-300:
        return default
    return a / b


def safe_log(x: float, min_val: float = 1e-300) -> float:
    return np.log(max(x, min_val))


def safe_exp(x: float, max_val: float = 700.0) -> float:
    if x > max_val:
        return np.exp(max_val)
    if x < -max_val:
        return 0.0
    return np.exp(x)


def softplus(x: float) -> float:
    if x > 20.0:
        return x
    return np.log1p(np.exp(x))


def clip_concentration(C: float, C_min: float = 1e-12, C_max: float = 1e6) -> float:
    return max(C_min, min(C, C_max))






def ml_to_L(x: float) -> float:
    return x / 1000.0


def L_to_ml(x: float) -> float:
    return x * 1000.0


def mg_to_ng(x: float) -> float:
    return x * 1e6


def ng_to_mg(x: float) -> float:
    return x / 1e6


def min_to_hr(x: float) -> float:
    return x / 60.0


def hr_to_min(x: float) -> float:
    return x * 60.0


PHYSIOLOGICAL_CONSTANTS = {
    "BLOOD_DENSITY": 1.055,
    "PLASMA_VOLUME_70KG": 3.0,
    "CARDIAC_OUTPUT_REST": 5.0,
    "HEPATIC_BLOOD_FLOW": 1.5,
    "RENAL_BLOOD_FLOW": 1.2,
    "GFR_STANDARD": 0.125,
    "BODY_WEIGHT_STANDARD": 70.0,
    "AVOGADRO": 6.02214076e23,
    "BOLTZMANN": 1.380649e-23,
    "GAS_CONSTANT": 8.314462618,
    "TEMP_BODY": 310.15,
}


def scale_by_body_weight(value_70kg: float, actual_bw: float) -> float:
    if actual_bw <= 0:
        raise ValueError("Body weight must be positive")


    raise NotImplementedError("Hole 2: Allometric scaling formula not implemented")






if __name__ == "__main__":
    digits = [4, 5, 3, 2, 0, 1, 9, 4, 9, 6, 2]
    check = luhn_check_digit_calculate(digits)
    print(f"Luhn check digit: {check}")
    valid = luhn_is_valid(digits + [check])
    print(f"Luhn valid: {valid}")
    data = np.random.lognormal(0, 1, 1000)
    counts, edges, centers = concentration_binning(data, n_bins=10)
    print(f"Bin counts: {counts}")
    stats = summary_statistics(data)
    print(f"Summary: mean={stats['mean']:.3f}, median={stats['median']:.3f}, cv={stats['cv']:.3f}")
    print(f"Safe divide 5/0: {safe_divide(5.0, 0.0, default=999.0)}")
    print(f"Softplus(5): {softplus(5.0):.4f}")
    print(f"Scale Q by BW 50kg: {scale_by_body_weight(5.0, 50.0):.3f}")
