
import numpy as np
from lattice_gauge import Lattice, GaugeConfig, su2_dagger, su2_stereographic_project, su2_stereographic_inverse
from gauge_dg_update import compute_force


def apply_flow_step_euler_backward(gauge: GaugeConfig, dt: float,
                                   max_iter: int = 6) -> GaugeConfig:
    lat = gauge.lat
    for idx in range(lat.vol):
        x = lat.index_to_site(idx)
        for mu in range(4):
            u_n = gauge.get_link(mu, x)
            q_n = su2_stereographic_project(u_n)
            q = q_n.copy()
            for _ in range(max_iter):
                u_trial = su2_stereographic_inverse(q)
                gauge.set_link(mu, x, u_trial)
                force = compute_force(gauge, x, mu)
                fvec = np.array([force[0, 1].real,
                                 force[0, 1].imag,
                                 force[0, 0].imag])
                q_new = q_n + dt * fvec

                q = 0.5 * q + 0.5 * q_new
                if np.linalg.norm(q_new - q_n) < 1e-10:
                    break
            gauge.set_link(mu, x, su2_stereographic_inverse(q))
    return gauge


def adaptive_midpoint_step(gauge: GaugeConfig, dt: float,
                           theta: float = 0.5) -> tuple:
    lat = gauge.lat
    state_n = gauge.U.copy()


    for idx in range(lat.vol):
        x = lat.index_to_site(idx)
        for mu in range(4):
            q_n = su2_stereographic_project(state_n[(mu, *x)])
            q = q_n.copy()
            for _ in range(8):
                u_tmp = su2_stereographic_inverse(q)
                gauge.set_link(mu, x, u_tmp)
                force = compute_force(gauge, x, mu)
                fvec = np.array([force[0, 1].real,
                                 force[0, 1].imag,
                                 force[0, 0].imag])
                q_new = q_n + theta * dt * fvec
                delta = np.linalg.norm(q_new - q)
                q = q_new
                if delta < 1e-12:
                    break
            q_m = q
            q_np1 = (1.0 / theta) * q_m + (1.0 - 1.0 / theta) * q_n
            gauge.set_link(mu, x, su2_stereographic_inverse(q_np1))


    lte = 0.0
    return gauge, dt, lte


def wilson_flow_run(gauge: GaugeConfig, flow_time: float = 1.0,
                    dt_init: float = 0.05, method: str = "midpoint") -> GaugeConfig:
    t = 0.0
    dt = dt_init
    n_steps = 0
    while t < flow_time and n_steps < 200:
        dt = min(dt, flow_time - t)
        if method == "euler":
            gauge = apply_flow_step_euler_backward(gauge, dt)
            t += dt
        else:
            gauge, dt_acc, _ = adaptive_midpoint_step(gauge, dt)
            t += dt_acc
            dt = min(2.0 * dt_acc, dt_init)
        n_steps += 1
    return gauge
