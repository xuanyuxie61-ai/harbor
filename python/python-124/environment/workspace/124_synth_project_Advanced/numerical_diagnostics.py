
import numpy as np
from typing import Tuple, List, Dict





def matrix_exponential_pade(A: np.ndarray, order: int = 3) -> np.ndarray:
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be a square matrix")

    n = A.shape[0]
    I = np.eye(n)


    norm_A = np.linalg.norm(A, ord=1)
    if norm_A < 1e-15:
        return I.copy()

    s = max(0, int(np.ceil(np.log2(norm_A))))
    if s > 50:
        s = 50
    A_scaled = A / (2.0 ** s)


    if order == 1:
        N = I + 0.5 * A_scaled
        D = I - 0.5 * A_scaled
    elif order == 2:
        A2 = A_scaled @ A_scaled
        N = I + 0.5 * A_scaled + (1.0 / 12.0) * A2
        D = I - 0.5 * A_scaled + (1.0 / 12.0) * A2
    elif order == 3:
        A2 = A_scaled @ A_scaled
        A3 = A2 @ A_scaled
        N = I + 0.5 * A_scaled + (1.0 / 9.0) * A2 + (1.0 / 72.0) * A3
        D = I - 0.5 * A_scaled + (1.0 / 9.0) * A2 - (1.0 / 72.0) * A3
    else:
        raise ValueError("Unsupported Padé order")


    E = np.linalg.solve(D, N)


    for _ in range(s):
        E = E @ E

    return E


def matrix_exponential_taylor(A: np.ndarray, terms: int = 20) -> np.ndarray:
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be a square matrix")

    n = A.shape[0]
    I = np.eye(n)
    E = I.copy()
    term = I.copy()

    for k in range(1, terms):
        term = term @ A / k
        E += term
        if np.linalg.norm(term, ord='fro') < 1e-15:
            break

    return E





def polynomial_value_direct(x: float, coeffs: np.ndarray) -> float:
    val = 0.0
    for i, c in enumerate(coeffs):
        val += c * (x ** i)
    return val


def polynomial_value_horner(x: float, coeffs: np.ndarray) -> float:
    val = coeffs[-1]
    for i in range(len(coeffs) - 2, -1, -1):
        val = val * x + coeffs[i]
    return val


def compare_polynomial_evaluation(x: float, coeffs: np.ndarray) -> Dict[str, float]:
    val_direct = polynomial_value_direct(x, coeffs)
    val_horner = polynomial_value_horner(x, coeffs)


    coeffs_hp = np.array(coeffs, dtype=np.longdouble)
    x_hp = np.longdouble(x)
    val_ref = polynomial_value_horner(float(x_hp), coeffs_hp.astype(float))

    err_direct = abs(val_direct - val_ref) / max(abs(val_ref), 1e-15)
    err_horner = abs(val_horner - val_ref) / max(abs(val_ref), 1e-15)

    return {
        'direct': val_direct,
        'horner': val_horner,
        'reference': float(val_ref),
        'rel_err_direct': float(err_direct),
        'rel_err_horner': float(err_horner),
    }





def quadratic_roots_standard(a: float, b: float, c: float) -> Tuple[complex, complex]:
    disc = b * b - 4.0 * a * c
    sqrt_disc = np.sqrt(disc + 0j)

    r1 = (-b + sqrt_disc) / (2.0 * a)
    r2 = (-b - sqrt_disc) / (2.0 * a)
    return r1, r2


def quadratic_roots_stable(a: float, b: float, c: float) -> Tuple[complex, complex]:
    disc = b * b - 4.0 * a * c
    if disc < 0:
        sqrt_disc = np.sqrt(-disc) * 1j
        q = -0.5 * (b + np.sign(b) * sqrt_disc * 1j)
    else:
        sqrt_disc = np.sqrt(disc)
        if b >= 0:
            q = -0.5 * (b + sqrt_disc)
        else:
            q = -0.5 * (b - sqrt_disc)

    if abs(q) < 1e-15:
        r1 = (-b + sqrt_disc) / (2.0 * a)
        r2 = (-b - sqrt_disc) / (2.0 * a)
    else:
        r1 = q / a
        r2 = c / q

    return complex(r1), complex(r2)





def condition_number_analysis(A: np.ndarray) -> Dict[str, float]:
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be square")

    cond_2 = np.linalg.cond(A)
    cond_1 = np.linalg.cond(A, 1)
    cond_inf = np.linalg.cond(A, np.inf)
    rank = np.linalg.matrix_rank(A)

    sign, logdet = np.linalg.slogdet(A)
    det_val = sign * np.exp(logdet) if abs(logdet) < 700 else float('inf')

    return {
        'cond_2': cond_2,
        'cond_1': cond_1,
        'cond_inf': cond_inf,
        'determinant': det_val,
        'rank': rank,
        'is_singular': rank < A.shape[0],
        'is_ill_conditioned': cond_2 > 1e12,
    }





class NumericalDiagnostics:

    def __init__(self):
        self.reports: List[Dict] = []

    def test_matrix_exponential(self, A: np.ndarray) -> Dict:
        E_pade = matrix_exponential_pade(A, order=3)
        E_taylor = matrix_exponential_taylor(A, terms=50)


        try:
            from scipy.linalg import expm
            E_ref = expm(A)
        except ImportError:
            E_ref = E_taylor

        err_pade = np.linalg.norm(E_pade - E_ref, ord='fro') / np.linalg.norm(E_ref, ord='fro')
        err_taylor = np.linalg.norm(E_taylor - E_ref, ord='fro') / np.linalg.norm(E_ref, ord='fro')

        report = {
            'test': 'matrix_exponential',
            'matrix_norm': np.linalg.norm(A, ord='fro'),
            'err_pade': err_pade,
            'err_taylor': err_taylor,
        }
        self.reports.append(report)
        return report

    def test_polynomial_evaluation(self, degree: int = 15) -> Dict:

        coeffs = np.random.randn(degree + 1)
        coeffs[0] = 1e10
        x = 1.0001

        result = compare_polynomial_evaluation(x, coeffs)
        report = {
            'test': 'polynomial_evaluation',
            'degree': degree,
            'x': x,
            **result,
        }
        self.reports.append(report)
        return report

    def test_quadratic_roots(self) -> Dict:


        a, b, c = 1.0, -2.0, 1.0 - 1e-12

        r1_std, r2_std = quadratic_roots_standard(a, b, c)
        r1_stb, r2_stb = quadratic_roots_stable(a, b, c)

        true_r1 = 1.0 + 1e-6
        true_r2 = 1.0 - 1e-6

        err_std = max(abs(r1_std - true_r1), abs(r2_std - true_r2))
        err_stb = max(abs(r1_stb - true_r1), abs(r2_stb - true_r2))

        report = {
            'test': 'quadratic_roots',
            'standard_roots': (complex(r1_std), complex(r2_std)),
            'stable_roots': (complex(r1_stb), complex(r2_stb)),
            'true_roots': (true_r1, true_r2),
            'err_standard': float(err_std),
            'err_stable': float(err_stb),
            'improvement_factor': float(err_std / max(err_stb, 1e-15)),
        }
        self.reports.append(report)
        return report

    def test_matrix_condition(self, K: np.ndarray) -> Dict:
        analysis = condition_number_analysis(K)
        report = {
            'test': 'matrix_condition',
            'matrix_size': K.shape[0],
            **analysis,
        }
        self.reports.append(report)
        return report

    def generate_report(self) -> str:
        lines = ["=" * 60,
                 "NUMERICAL DIAGNOSTICS REPORT",
                 "=" * 60]

        for r in self.reports:
            lines.append(f"\nTest: {r['test']}")
            for k, v in r.items():
                if k != 'test':
                    lines.append(f"  {k}: {v}")

        lines.append("=" * 60)
        return "\n".join(lines)
