
import numpy as np
from typing import Callable, Tuple, Optional


def unstable_ode_system(t: float, y: np.ndarray, mu: float = 0.1) -> np.ndarray:
    if mu <= 0:
        raise ValueError("mu must be positive")
    if len(y) != 2:
        raise ValueError("State vector must have dimension 2")

    A = np.array([[mu, 1.0 / mu], [-1.0 / mu, mu]], dtype=np.float64)
    return A @ y


def unstable_exact_solution(t: float, mu: float = 0.1) -> np.ndarray:
    if mu <= 0:
        raise ValueError("mu must be positive")

    exp_term = np.exp(mu * t)
    cos_term = np.cos(t / mu)
    sin_term = np.sin(t / mu)

    y1 = exp_term * (cos_term - mu * mu * sin_term)
    y2 = mu * exp_term * (cos_term - mu * mu * sin_term)
    y2 += exp_term * (-sin_term / mu - mu * cos_term)

    return np.array([y1, y2])


def broyden_quasi_newton(
    F: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray,
    atol: float = 1e-8,
    rtol: float = 1e-6,
    maxit: int = 100,
    maxdim: int = 10
) -> Tuple[np.ndarray, int]:
    x = x0.copy().astype(np.float64)
    n = len(x)

    fc = F(x)
    fnrm = np.linalg.norm(fc) / np.sqrt(n)
    if fnrm < 1e-15:
        return x, 0

    stop_tol = atol + rtol * fnrm


    stp = np.zeros((n, maxdim))
    stp[:, 0] = -fc
    stp_nrm = np.zeros(maxdim)
    stp_nrm[0] = np.dot(stp[:, 0], stp[:, 0])

    nbroy = 0
    itc = 0

    while itc < maxit:
        fnrmo = fnrm
        nbroy += 1
        itc += 1


        if nbroy < maxdim:
            x = x + stp[:, nbroy - 1]
        else:
            x = x + stp[:, maxdim - 1]

        fc = F(x)
        fnrm = np.linalg.norm(fc) / np.sqrt(n)


        if fnrm <= stop_tol:
            return x, 0


        if fnrmo <= fnrm and itc > 1:

            x = x - stp[:, min(nbroy - 1, maxdim - 1)]
            nbroy = 0
            stp[:, 0] = -fc
            stp_nrm[0] = np.dot(stp[:, 0], stp[:, 0])
            continue


        if nbroy + 1 < maxdim:
            z = -fc
            for kbr in range(nbroy - 1):
                if stp_nrm[kbr] < 1e-15:
                    continue
                z = z + stp[:, kbr + 1] * np.dot(stp[:, kbr], z) / stp_nrm[kbr]

            if stp_nrm[nbroy - 1] > 1e-15:
                zz = np.dot(stp[:, nbroy - 1], z) / stp_nrm[nbroy - 1]
                denom = 1.0 - zz
                if abs(denom) > 1e-15:
                    stp[:, nbroy] = z / denom
                    stp_nrm[nbroy] = np.dot(stp[:, nbroy], stp[:, nbroy])
                else:

                    nbroy = 0
                    stp[:, 0] = -fc
                    stp_nrm[0] = np.dot(stp[:, 0], stp[:, 0])
            else:
                nbroy = 0
                stp[:, 0] = -fc
                stp_nrm[0] = np.dot(stp[:, 0], stp[:, 0])
        else:

            nbroy = 0
            stp[:, 0] = -fc
            stp_nrm[0] = np.dot(stp[:, 0], stp[:, 0])


    fc = F(x)
    fnrm = np.linalg.norm(fc) / np.sqrt(n)
    if fnrm <= stop_tol:
        return x, 0
    return x, 1


class VariationalQuantumOptimizer:

    def __init__(
        self,
        energy_func: Optional[Callable[[np.ndarray], float]] = None,
        gradient_func: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        atol: float = 1e-7,
        rtol: float = 1e-5,
        maxit: int = 200,
        maxdim: int = 15
    ):
        self.energy_func = energy_func
        self.gradient_func = gradient_func
        self.atol = atol
        self.rtol = rtol
        self.maxit = maxit
        self.maxdim = maxdim
        self.history: list = []

    def _numerical_gradient(self, theta: np.ndarray, eps: float = 1e-7) -> np.ndarray:
        grad = np.zeros_like(theta)
        for i in range(len(theta)):
            theta_plus = theta.copy()
            theta_minus = theta.copy()
            theta_plus[i] += eps
            theta_minus[i] -= eps
            grad[i] = (self.energy_func(theta_plus) - self.energy_func(theta_minus)) / (2.0 * eps)
        return grad

    def optimize(self, theta0: np.ndarray) -> Tuple[np.ndarray, float, int]:
        theta0 = np.array(theta0, dtype=np.float64)

        def F(theta):
            if self.gradient_func is not None:
                return self.gradient_func(theta)
            return self._numerical_gradient(theta)

        theta_opt, ierr = broyden_quasi_newton(
            F, theta0, self.atol, self.rtol, self.maxit, self.maxdim
        )

        energy_opt = self.energy_func(theta_opt)
        self.history.append({"theta": theta_opt.copy(), "energy": energy_opt, "ierr": ierr})

        return theta_opt, energy_opt, ierr

    def vqe_minimize(
        self,
        hamiltonian: np.ndarray,
        ansatz_func: Callable[[np.ndarray], np.ndarray],
        theta0: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        H = np.array(hamiltonian, dtype=np.complex128)
        if not np.allclose(H, H.conj().T, atol=1e-10):
            raise ValueError("Hamiltonian must be Hermitian")

        def energy(theta):
            psi = ansatz_func(theta)
            psi = psi / (np.linalg.norm(psi) + 1e-15)
            E = np.vdot(psi, H @ psi).real
            return E

        self.energy_func = energy
        self.gradient_func = None

        theta_opt, E_opt, ierr = self.optimize(theta0)
        return theta_opt, E_opt
