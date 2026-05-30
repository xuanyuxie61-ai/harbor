
import sys
import numpy as np


def r1mach(i: int) -> float:
    if i < 1 or i > 5:
        raise ValueError(f"R1MACH: 参数 i={i} 越界，合法范围为 1~5")
    values = {
        1: np.float32(1.1754944e-38),
        2: np.float32(3.4028235e38),
        3: np.float32(5.9604645e-08),
        4: np.float32(1.1920929e-07),
        5: np.float32(0.3010300),
    }
    return float(values[i])


def d1mach(i: int) -> float:
    if i < 1 or i > 5:
        raise ValueError(f"D1MACH: 参数 i={i} 越界，合法范围为 1~5")
    values = {
        1: np.finfo(float).tiny,
        2: np.finfo(float).max,
        3: np.finfo(float).eps * 0.5,
        4: np.finfo(float).eps,
        5: np.log10(2.0),
    }
    return float(values[i])


def i1mach(i: int) -> int:
    if i < 1 or i > 16:
        raise ValueError(f"I1MACH: 参数 i={i} 越界，合法范围为 1~16")
    values = {
        1: 5,
        2: 6,
        4: 32,
        5: 9,
        6: 2,
        7: 31,
        8: 127,
        11: 52,
        12: -1021,
        13: 1024,
    }
    return values.get(i, 0)


def get_machine_epsilon() -> float:
    return d1mach(4)


def get_safe_tol(scale: float = 1.0) -> float:
    eps = d1mach(4)
    return scale * np.sqrt(eps)
