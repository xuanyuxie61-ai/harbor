
import numpy as np
from lattice_gauge import Lattice, GaugeConfig, su2_dagger, su2_trace, su2_stereographic_project, su2_stereographic_inverse



_RK4A = np.array([0.0,
                  -567301805773.0 / 1357537059087.0,
                  -2404267990393.0 / 2016746695238.0,
                  -3550918686646.0 / 2091501179385.0,
                  -1275806237668.0 / 842570457699.0])
_RK4B = np.array([1432997174477.0 / 9575080441755.0,
                  5161836677717.0 / 13612068292357.0,
                  1720146321549.0 / 2090206949498.0,
                  3134564353537.0 / 4481467310338.0,
                  2277821191437.0 / 14882151754819.0])
_RK4C = np.array([0.0,
                  1432997174477.0 / 9575080441755.0,
                  2526269341429.0 / 6820363962896.0,
                  2006345519317.0 / 3224310063776.0,
                  2802321613138.0 / 2924317926251.0])


def compute_staple(gauge: GaugeConfig, x: np.ndarray, mu: int) -> np.ndarray:
    lat = gauge.lat
    staple = np.zeros((2, 2), dtype=complex)
    for nu in range(4):
        if nu == mu:
            continue

        s1 = gauge.get_link(nu, x)
        s2 = gauge.get_link(mu, lat.neighbor(x, nu, 1))
        s3 = su2_dagger(gauge.get_link(nu, lat.neighbor(x, mu, 1)))
        staple += s1 @ s2 @ s3

        r1 = su2_dagger(gauge.get_link(nu, lat.neighbor(x, nu, -1)))
        r2 = gauge.get_link(mu, lat.neighbor(x, nu, -1))
        r3 = gauge.get_link(nu, lat.neighbor(lat.neighbor(x, nu, -1), mu, 1))
        staple += r1 @ r2 @ r3
    return staple


def compute_force(gauge: GaugeConfig, x: np.ndarray, mu: int) -> np.ndarray:
    u = gauge.get_link(mu, x)
    st = compute_staple(gauge, x, mu)

    tmp = st @ su2_dagger(u)
    force = 0.5 * (tmp - su2_dagger(tmp))
    return force


def dg_gauge_rhs(gauge: GaugeConfig, beta: float = 2.4) -> np.ndarray:
    lat = gauge.lat
    rhs = np.zeros((4, *lat.shape, 3), dtype=float)

    for mu in range(4):
        for idx in range(lat.vol):
            x = lat.index_to_site(idx)
            force = compute_force(gauge, x, mu)


            fvec = np.array([force[0, 1].real,
                             force[0, 1].imag,
                             force[0, 0].imag])

            x_plus = lat.neighbor(x, mu, 1)
            force_plus = compute_force(gauge, x_plus, mu)
            fvec_plus = np.array([force_plus[0, 1].real,
                                  force_plus[0, 1].imag,
                                  force_plus[0, 0].imag])

            flux = 0.5 * (fvec_plus - fvec)
            rhs[(mu, *x)] = (beta / 6.0) * fvec + flux

    return rhs


def dg_gauge_evolve(gauge: GaugeConfig, beta: float = 2.4,
                    final_time: float = 1.0, cfl: float = 0.5) -> GaugeConfig:
    lat = gauge.lat

    dt = cfl * 0.5
    nsteps = max(1, int(np.ceil(final_time / dt)))
    dt = final_time / nsteps


    res = np.zeros((4, *lat.shape, 3), dtype=float)

    for _ in range(nsteps):
        for intrk in range(5):
            rhs = dg_gauge_rhs(gauge, beta)
            res = _RK4A[intrk] * res + dt * rhs

            for mu in range(4):
                for idx in range(lat.vol):
                    x = lat.index_to_site(idx)
                    dq = _RK4B[intrk] * res[(mu, *x)]

                    norm_dq = np.linalg.norm(dq)
                    if norm_dq > 0.5:
                        dq = dq * (0.5 / norm_dq)
                    u_old = gauge.get_link(mu, x)
                    q_old = su2_stereographic_project(u_old)
                    q_new = q_old + dq

                    norm_q = np.linalg.norm(q_new)
                    if norm_q > 10.0:
                        q_new = q_new * (10.0 / norm_q)
                    u_new = su2_stereographic_inverse(q_new)
                    gauge.set_link(mu, x, u_new)

    return gauge
