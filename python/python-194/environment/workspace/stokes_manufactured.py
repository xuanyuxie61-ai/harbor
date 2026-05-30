
import numpy as np
from typing import Tuple


def stokes_solution_polynomial(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    u = 2.0 * x ** 2 * (x - 1.0) ** 2 * y * (2.0 * y - 1.0) * (y - 1.0)
    v = -2.0 * x * (2.0 * x - 1.0) * (x - 1.0) * y ** 2 * (y - 1.0) ** 2
    p = x * (1.0 - x) * y * (1.0 - y)
    return u, v, p


def stokes_rhs_polynomial(x: np.ndarray, y: np.ndarray, nu: float = 1.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:









    raise NotImplementedError("Hole 1: stokes_rhs_polynomial 需要补全制造解的解析推导")


def stokes_solution_trigonometric(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    u = np.sin(np.pi * x) * np.cos(np.pi * y)
    v = -np.cos(np.pi * x) * np.sin(np.pi * y)
    p = np.sin(np.pi * x) * np.sin(np.pi * y)
    return u, v, p


def stokes_rhs_trigonometric(x: np.ndarray, y: np.ndarray, nu: float = 1.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    pi2 = np.pi ** 2
    d2u = -2.0 * pi2 * np.sin(np.pi * x) * np.cos(np.pi * y)
    d2v = 2.0 * pi2 * np.cos(np.pi * x) * np.sin(np.pi * y)
    dpdx = np.pi * np.cos(np.pi * x) * np.sin(np.pi * y)
    dpdy = np.pi * np.sin(np.pi * x) * np.cos(np.pi * y)
    fx = -nu * d2u + dpdx
    fy = -nu * d2v + dpdy
    h = np.zeros_like(x)
    return fx, fy, h


def stokes_solution_kovasznay(x: np.ndarray, y: np.ndarray, Re: float = 40.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if Re <= 0:
        Re = 40.0
    lambda_k = Re / 2.0 - np.sqrt(Re ** 2 / 4.0 + 4.0 * np.pi ** 2)
    u = 1.0 - np.exp(lambda_k * x) * np.cos(2.0 * np.pi * y)
    v = lambda_k / (2.0 * np.pi) * np.exp(lambda_k * x) * np.sin(2.0 * np.pi * y)
    p = 0.5 * (1.0 - np.exp(2.0 * lambda_k * x))
    return u, v, p


def evaluate_solution(
    x: np.ndarray, y: np.ndarray,
    sol_type: str = "polynomial",
    nu: float = 1.0, Re: float = 40.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if sol_type == "polynomial":
        u, v, p = stokes_solution_polynomial(x, y)
        fx, fy, h = stokes_rhs_polynomial(x, y, nu)
    elif sol_type == "trigonometric":
        u, v, p = stokes_solution_trigonometric(x, y)
        fx, fy, h = stokes_rhs_trigonometric(x, y, nu)
    elif sol_type == "kovasznay":
        u, v, p = stokes_solution_kovasznay(x, y, Re)

        fx = np.zeros_like(x)
        fy = np.zeros_like(x)
        h = np.zeros_like(x)
    else:
        raise ValueError(f"Unknown solution type: {sol_type}")
    return u, v, p, fx, fy, h


def compute_discrete_residual(
    u_h: np.ndarray, v_h: np.ndarray, p_h: np.ndarray,
    x: np.ndarray, y: np.ndarray,
    nu: float = 1.0,
    sol_type: str = "polynomial"
) -> float:
    u_ex, v_ex, p_ex, _, _, _ = evaluate_solution(x, y, sol_type, nu)
    err_u = np.linalg.norm(u_h - u_ex) / max(1.0, np.linalg.norm(u_ex))
    err_v = np.linalg.norm(v_h - v_ex) / max(1.0, np.linalg.norm(v_ex))
    err_p = np.linalg.norm(p_h - p_ex) / max(1.0, np.linalg.norm(p_ex))
    return float(np.sqrt(err_u ** 2 + err_v ** 2 + err_p ** 2))
