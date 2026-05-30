
import numpy as np
from lattice_gauge import Lattice, GaugeConfig, su2_dagger
from matrix_algebra import r8pp_fa, r8pp_sl, dense_to_packed



_GAMMA = [
    np.array([[0, 1], [1, 0]], dtype=complex),
    np.array([[0, -1j], [1j, 0]], dtype=complex),
    np.array([[1, 0], [0, -1]], dtype=complex),
    np.array([[0, 1], [1, 0]], dtype=complex),
]

_GAMMA5 = np.array([[1, 0], [0, -1]], dtype=complex)


def gamma_mu(mu: int) -> np.ndarray:
    return _GAMMA[mu % 4]


class WilsonDiracOperator:

    def __init__(self, lattice: Lattice, gauge: GaugeConfig,
                 mass: float = 0.1, boundary_phase: np.ndarray = None):
        self.lat = lattice
        self.gauge = gauge
        self.mass = mass
        self.kappa = 1.0 / (2.0 * mass + 8.0)
        if boundary_phase is None:

            self.boundary_phase = np.array([1.0, 1.0, 1.0, -1.0])
        else:
            self.boundary_phase = np.array(boundary_phase)

    def apply(self, psi: np.ndarray) -> np.ndarray:

        raise NotImplementedError("Hole 1: implement Wilson-Dirac apply")

    def apply_dagger(self, psi: np.ndarray) -> np.ndarray:

        raise NotImplementedError("Hole 2: implement apply_dagger using gamma_5 hermiticity")

    def apply_hermitian(self, psi: np.ndarray) -> np.ndarray:
        tmp = self.apply(psi)
        return self.apply_dagger(tmp)


def solve_propagator_cg(wd: WilsonDiracOperator, source: np.ndarray,
                        max_iter: int = 80, tol: float = 1e-6) -> np.ndarray:
    lat = wd.lat

    b = wd.apply_dagger(source).reshape(-1, 2)

    def matvec(v_flat):
        v = v_flat.reshape(*lat.shape, 2)
        Av = wd.apply_hermitian(v)
        return Av.reshape(-1, 2)

    def dot(a, b_vec):
        return np.sum(a.conj() * b_vec).real

    x = np.zeros_like(b)
    r = b.copy()
    p = r.copy()
    rsold = dot(r, r)
    if rsold < tol ** 2:
        return x.reshape(*lat.shape, 2)

    for k in range(max_iter):
        Ap = matvec(p)
        pAp = dot(p, Ap)
        if abs(pAp) < 1e-30:
            break
        alpha = rsold / pAp
        x += alpha * p
        r -= alpha * Ap
        rsnew = dot(r, r)
        if np.sqrt(rsnew) < tol:
            break
        beta = rsnew / rsold
        p = r + beta * p
        rsold = rsnew

    sol = x.reshape(*lat.shape, 2)
    return sol


def point_source(lat: Lattice, x0: np.ndarray, spin: int = 0) -> np.ndarray:
    src = np.zeros((*lat.shape, 2), dtype=complex)
    x0 = np.mod(x0, lat.dims)
    src[(x0[0], x0[1], x0[2], x0[3], spin)] = 1.0
    return src


def solve_all_propagators(wd: WilsonDiracOperator,
                          sources: list) -> list:
    props = []
    for src in sources:
        prop = solve_propagator_cg(wd, src, max_iter=80, tol=1e-6)
        props.append(prop)
    return props
